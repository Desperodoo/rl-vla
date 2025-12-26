#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARX5 Real Robot Policy Inference Script

This script runs trained diffusion policy / flow matching models on the
physical ARX5 robot using real-time camera observations.

Features:
- Loads checkpoint from rlft training
- Dual-thread architecture: control (500Hz) + inference (~30Hz)
- Dual RealSense camera visualization (wrist + external)
- EMA low-pass filtering for smoother output
- Keyboard control: start/pause, reset, quit

Architecture:
    Control Thread (500Hz):
        - Executes interpolated actions
        - Reads robot state
        - No blocking operations
    
    Inference Thread (~30Hz):
        - Runs policy inference
        - Gets camera images
        - Updates action buffer

Usage:
    python -m data_collection.arx5_policy_inference \\
        --checkpoint ~/rlft/runs/.../checkpoints/final.pt \\
        --max-steps 1000
        
    # Dry run (no robot execution)
    python -m data_collection.arx5_policy_inference \\
        --checkpoint ~/rlft/runs/.../checkpoints/final.pt \\
        --dry-run
        
    # Adjust filter smoothness
    python -m data_collection.arx5_policy_inference \\
        --checkpoint ~/rlft/runs/.../checkpoints/final.pt \\
        --filter-alpha 0.2  # More smooth

Controls:
    [Space]     - Start/Pause inference
    [R]         - Reset robot to home
    [Q]         - Quit
    Or type in terminal: s, r, q, h

Author: Generated for ARX5 teleoperation project
"""

import argparse
import os
import sys
import time
import select
import threading
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

import numpy as np
import cv2
import torch
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d

# 轨迹记录器 (用于对比分析)
try:
    from inference.trajectory_logger import TrajectoryLogger
    TRAJECTORY_LOGGER_AVAILABLE = True
except ImportError:
    import sys
    sys.path.insert(0, os.path.expanduser("~/rl-vla"))
    try:
        from inference.trajectory_logger import TrajectoryLogger
        TRAJECTORY_LOGGER_AVAILABLE = True
    except ImportError:
        TRAJECTORY_LOGGER_AVAILABLE = False
        print("Warning: TrajectoryLogger not available")

# Add rlft to path
RLFT_PATH = os.path.expanduser("~/rlft/diffusion_policy")
if RLFT_PATH not in sys.path:
    sys.path.insert(0, RLFT_PATH)


@dataclass
class InferenceConfig:
    """Configuration for real robot inference."""
    
    # Timing
    obs_horizon: int = 2          # Number of observation frames to stack
    act_horizon: int = 8          # Number of action frames to execute
    pred_horizon: int = 16        # Total predicted action frames (must match training)
    
    # Control
    control_freq: float = 500.0   # Target control frequency (Hz)
    inference_freq: float = 30.0  # Policy inference frequency (Hz)
    record_freq: float = 30.0     # Original data recording frequency (Hz)
    
    # Inference speed (flow/diffusion steps - lower = faster but less accurate)
    num_flow_steps: int = 10      # Number of ODE steps for flow matching (default: 10)
    
    # Camera
    image_size: Tuple[int, int] = (128, 128)  # (H, W) for model input
    
    # Smoothing
    enable_filter: bool = True    # Enable low-pass filter on output
    filter_alpha: float = 0.3     # EMA filter coefficient (lower = more smooth)
    smooth_window: int = 3        # Pre-interpolation smoothing window
    
    # Safety
    max_joint_delta: float = 0.1   # Max joint position change per step (rad)
    max_gripper_delta: float = 0.02  # Max gripper change per step (m)
    joint_limits_low: Tuple[float, ...] = (-3.14, -3.14, -3.14, -3.14, -3.14, -3.14)
    joint_limits_high: Tuple[float, ...] = (3.14, 3.14, 3.14, 3.14, 3.14, 3.14)
    gripper_limits: Tuple[float, float] = (0.0, 0.08)  # Gripper limits (m)
    
    # Initialization safety
    init_move_duration: float = 3.5    # Duration for moving to initial pose (seconds)
    require_init_confirm: bool = True  # Require user confirmation before init move
    
    # Task initial pose (from training data - pick_cube task)
    # Set to None to use home position, or specify joint positions
    task_initial_pose: Optional[Tuple[float, ...]] = None


class InferenceState(Enum):
    """State machine for inference control."""
    IDLE = "idle"           # Waiting to start
    RUNNING = "running"     # Actively executing policy
    PAUSED = "paused"       # Paused, holding position
    RESETTING = "resetting" # Resetting to home


class ActionFilter:
    """
    Low-pass filter for smoothing action outputs.
    
    Implements exponential moving average (EMA) filtering to reduce
    high-frequency noise and jitter in predicted actions.
    """
    
    def __init__(self, action_dim: int, alpha: float = 0.3):
        """
        Args:
            action_dim: Dimension of action vector
            alpha: EMA coefficient (0 < alpha <= 1)
                   Lower values = more smoothing
                   alpha=1 means no filtering
        """
        self.action_dim = action_dim
        self.alpha = alpha
        self.prev_action: Optional[np.ndarray] = None
        
    def reset(self, initial_action: Optional[np.ndarray] = None):
        """Reset filter state."""
        self.prev_action = initial_action.copy() if initial_action is not None else None
        
    def filter(self, action: np.ndarray) -> np.ndarray:
        """
        Apply EMA filter to action.
        
        filtered = alpha * current + (1 - alpha) * previous
        """
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action
        
        filtered = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = filtered.copy()
        return filtered


class ARX5PolicyInference:
    """
    ARX5 Real Robot Policy Inference Controller
    
    Dual-thread architecture:
    - Control Thread (500Hz): Executes interpolated actions, no blocking
    - Inference Thread (~30Hz): Runs policy, captures cameras
    
    Features:
    - rlft policy inference (directly load models)
    - data_collection camera manager (dual cameras)
    - ARX5 robot control interface
    - Action interpolation and EMA filtering
    - Keyboard control (start/pause/reset/quit)
    """
    
    # Constants for ARX5 real robot
    ACTION_DIM = 7   # 6 joints + 1 gripper
    STATE_DIM = 13   # 6 joint_pos + 6 joint_vel + 1 gripper_pos
    
    def __init__(
        self,
        checkpoint_path: str,
        config: InferenceConfig = None,
        device: str = "cuda",
        use_ema: bool = True,
        verbose: bool = True,
        headless: bool = False
    ):
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.config = config or InferenceConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_ema = use_ema
        self.verbose = verbose
        self.headless = headless
        
        # Model components (initialized in _load_policy())
        self.visual_encoder = None
        self.state_encoder = None
        self.agent = None
        self.algorithm = None
        
        # Hardware components (initialized in setup())
        self.camera_manager = None
        self.robot_ctrl = None
        self.controller_dt = 1.0 / self.config.control_freq
        
        # Observation history
        self.obs_buffer: deque = deque(maxlen=self.config.obs_horizon)
        
        # Action processing
        self.action_filter = ActionFilter(
            action_dim=self.ACTION_DIM,
            alpha=self.config.filter_alpha
        )
        
        # Thread-safe shared state
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()
        
        # Shared data between threads
        self._latest_robot_state: Optional[Dict] = None  # From control thread
        self._latest_camera_frames: Optional[Dict] = None  # From inference thread
        self._current_action_buffer: Optional[np.ndarray] = None  # Interpolated actions
        self._action_buffer_idx: int = 0
        self._prev_action: Optional[np.ndarray] = None
        self._pause_position: Optional[Dict] = None  # Position when paused
        
        # Control state
        self.state = InferenceState.IDLE
        self._running = False
        self.step_count = 0
        self.inference_count = 0
        
        # 轨迹记录器 (用于对比分析)
        self.trajectory_logger = None
        self._log_trajectory = False
        
    def enable_trajectory_logging(self, output_path: str = None):
        """启用轨迹记录"""
        if not TRAJECTORY_LOGGER_AVAILABLE:
            print("Warning: TrajectoryLogger not available, skipping")
            return
        
        self.trajectory_logger = TrajectoryLogger(name="single_process")
        self._log_trajectory = True
        self._trajectory_output_path = output_path or "trajectory_single_process.npz"
        print(f"[TrajectoryLogger] 已启用, 输出: {self._trajectory_output_path}")
        
    def setup(self):
        """Initialize all components."""
        print("\n" + "=" * 60)
        print("ARX5 Policy Inference Setup")
        print("=" * 60)
        
        # 1. Load policy from rlft
        self._load_policy()
        
        # 2. Warm up GPU (first inference is slow due to JIT compilation)
        self._warmup_gpu()
        
        # 3. Initialize cameras
        self._setup_cameras()
        
        # 4. Initialize robot
        self._setup_robot()
        
        print("\n✓ Setup complete!")
        
    def _load_policy(self):
        """Load trained policy directly from checkpoint."""
        print(f"\nLoading policy from: {self.checkpoint_path}")
        
        try:
            # Import rlft components
            from diffusion_policy.plain_conv import PlainConv
            from diffusion_policy.utils import StateEncoder
            from diffusion_policy.algorithms import (
                DiffusionPolicyAgent,
                FlowMatchingAgent,
                ConsistencyFlowAgent,
                ReflectedFlowAgent,
                ShortCutFlowAgent,
                ShortCutVelocityUNet1D,
            )
            from diffusion_policy.algorithms.networks import VelocityUNet1D
            from diffusion_policy.conditional_unet1d import ConditionalUnet1D
            
            # Load checkpoint
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Auto-detect algorithm from checkpoint content (more reliable than path)
            agent_keys = list(ckpt.get('agent', ckpt.get('ema_agent', {})).keys())
            if any('velocity_net' in k for k in agent_keys):
                # Flow-based models have velocity_net
                if any('velocity_net_ema' in k for k in agent_keys):
                    self.algorithm = "consistency_flow"
                else:
                    self.algorithm = "flow_matching"
            elif any('noise_pred_net' in k for k in agent_keys):
                self.algorithm = "diffusion_policy"
            else:
                # Fallback to path-based detection
                path_lower = self.checkpoint_path.lower()
                if "diffusion_policy" in path_lower:
                    self.algorithm = "diffusion_policy"
                elif "consistency_flow" in path_lower:
                    self.algorithm = "consistency_flow"
                elif "shortcut_flow" in path_lower:
                    self.algorithm = "shortcut_flow"
                elif "reflected_flow" in path_lower:
                    self.algorithm = "reflected_flow"
                else:
                    self.algorithm = "flow_matching"
            
            print(f"  Algorithm: {self.algorithm}")
            
            # Model architecture parameters (from rlft defaults)
            visual_feature_dim = 256
            state_encoder_hidden_dim = 128
            state_encoder_out_dim = 256
            diffusion_step_embed_dim = 64
            unet_dims = (64, 128, 256)
            n_groups = 8
            
            # Global conditioning dimension
            global_cond_dim = self.config.obs_horizon * (visual_feature_dim + state_encoder_out_dim)
            
            # Create visual encoder
            self.visual_encoder = PlainConv(
                in_channels=3,  # RGB only
                out_dim=visual_feature_dim,
                pool_feature_map=True,
            ).to(self.device)
            
            # Create state encoder
            self.state_encoder = StateEncoder(
                state_dim=self.STATE_DIM,
                hidden_dim=state_encoder_hidden_dim,
                out_dim=state_encoder_out_dim,
            ).to(self.device)
            
            # Create agent based on algorithm
            if self.algorithm == "diffusion_policy":
                noise_pred_net = ConditionalUnet1D(
                    input_dim=self.ACTION_DIM,
                    global_cond_dim=global_cond_dim,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=unet_dims,
                    n_groups=n_groups,
                )
                self.agent = DiffusionPolicyAgent(
                    noise_pred_net=noise_pred_net,
                    action_dim=self.ACTION_DIM,
                    obs_horizon=self.config.obs_horizon,
                    pred_horizon=self.config.pred_horizon,
                    num_diffusion_iters=100,
                    device=str(self.device),
                )
            elif self.algorithm == "consistency_flow":
                velocity_net = VelocityUNet1D(
                    input_dim=self.ACTION_DIM,
                    global_cond_dim=global_cond_dim,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=unet_dims,
                    n_groups=n_groups,
                )
                self.agent = ConsistencyFlowAgent(
                    velocity_net=velocity_net,
                    action_dim=self.ACTION_DIM,
                    obs_horizon=self.config.obs_horizon,
                    pred_horizon=self.config.pred_horizon,
                    num_flow_steps=self.config.num_flow_steps,
                    ema_decay=0.999,
                    action_bounds=None,  # No clipping for real robot
                    device=str(self.device),
                )
            elif self.algorithm == "flow_matching":
                velocity_net = VelocityUNet1D(
                    input_dim=self.ACTION_DIM,
                    global_cond_dim=global_cond_dim,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=unet_dims,
                    n_groups=n_groups,
                )
                self.agent = FlowMatchingAgent(
                    velocity_net=velocity_net,
                    action_dim=self.ACTION_DIM,
                    obs_horizon=self.config.obs_horizon,
                    pred_horizon=self.config.pred_horizon,
                    num_flow_steps=self.config.num_flow_steps,
                    action_bounds=None,
                    device=str(self.device),
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
            self.agent = self.agent.to(self.device)
            
            # Load weights
            agent_key = "ema_agent" if self.use_ema else "agent"
            if agent_key not in ckpt:
                agent_key = "agent"
                print(f"  Warning: EMA weights not found, using regular weights")
            
            self.agent.load_state_dict(ckpt[agent_key])
            self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
            self.state_encoder.load_state_dict(ckpt["state_encoder"])
            
            # Set to eval mode
            self.agent.eval()
            self.visual_encoder.eval()
            self.state_encoder.eval()
            
            # Update obs buffer size
            self.obs_buffer = deque(maxlen=self.config.obs_horizon)
            
            print(f"  obs_horizon: {self.config.obs_horizon}")
            print(f"  act_horizon: {self.config.act_horizon}")
            print(f"  pred_horizon: {self.config.pred_horizon}")
            print(f"  EMA weights: {agent_key == 'ema_agent'}")
            
        except Exception as e:
            print(f"Error loading policy: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    def _setup_cameras(self):
        """Initialize RealSense cameras (dual cameras: wrist + external)."""
        print("\nInitializing cameras...")
        
        try:
            # 修复导入路径：确保 data_collection 目录在 sys.path 中
            # 这样 camera_manager.py 内部的导入也能正常工作
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            
            # 直接从当前目录导入 (因为我们已经在 data_collection 目录中)
            from camera_manager import CameraManager, CameraConfig
            
            # Use both cameras like data collection
            camera_configs = {
                "wrist": CameraConfig(
                    name="wrist",
                    serial_number="036222071712",  # D435i
                    resolution=(640, 480),
                    fps=30,
                    enable_depth=False
                ),
                "external": CameraConfig(
                    name="external",
                    serial_number="037522250003",  # D455
                    resolution=(640, 480),
                    fps=30,
                    enable_depth=False
                )
            }
            
            self.camera_manager = CameraManager(camera_configs)
            self.camera_manager.initialize(auto_assign=True)
            self.camera_manager.start()
            
            print(f"  Cameras initialized: wrist (D435i), external (D455)")
            
        except Exception as e:
            print(f"Warning: Camera initialization failed: {e}")
            import traceback
            traceback.print_exc()
            print("  Running without cameras (state-only mode)")
            self.camera_manager = None
            
    def _setup_robot(self):
        """Initialize ARX5 robot controller."""
        print("\nInitializing robot...")
        
        try:
            sys.path.insert(0, "/home/lizh/arx5-sdk/python")
            os.chdir("/home/lizh/arx5-sdk/python")
            import arx5_interface as arx5
            
            # Use full initialization like ArxTeleop (NOT simplified constructor!)
            self.robot_cfg = arx5.RobotConfigFactory.get_instance().get_config("X5")
            self.ctrl_cfg = arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", self.robot_cfg.joint_dof
            )
            
            self.robot_ctrl = arx5.Arx5JointController(
                self.robot_cfg, self.ctrl_cfg, "can0"
            )
            
            # Set PID gains (same as ArxTeleop)
            gain = arx5.Gain(self.robot_cfg.joint_dof)
            gain.kd()[:] = 0.01
            self.robot_ctrl.set_gain(gain)
            
            # Get actual controller dt for precise timing
            self.controller_dt = float(self.ctrl_cfg.controller_dt)
            
            print(f"  Robot model: X5")
            print(f"  DOF: {self.robot_cfg.joint_dof}")
            print(f"  Controller dt: {self.controller_dt*1000:.1f}ms ({1.0/self.controller_dt:.1f}Hz)")
            
        except Exception as e:
            print(f"Warning: Robot initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.robot_ctrl = None
            self.controller_dt = 1.0 / self.config.control_freq
            
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation from cameras and robot.
        
        Returns:
            Dict with:
                - "rgb": [H, W, 3] uint8 RGB image
                - "state": [13] float32 state vector
                    - joint_pos[0:6]: rad
                    - joint_vel[6:12]: rad/s
                    - gripper_pos[12]: meters [0, 0.08]
        """
        obs = {}
        
        # Get camera image
        if self.camera_manager is not None:
            frames = self.camera_manager.get_frames()
            if "wrist" in frames:
                frame = frames["wrist"]
                rgb = frame.rgb  # [H, W, 3] uint8
                
                # Resize to model input size
                rgb = cv2.resize(rgb, 
                               (self.config.image_size[1], self.config.image_size[0]),
                               interpolation=cv2.INTER_LINEAR)
                obs["rgb"] = rgb
            else:
                obs["rgb"] = np.zeros((*self.config.image_size, 3), dtype=np.uint8)
        else:
            obs["rgb"] = np.zeros((*self.config.image_size, 3), dtype=np.uint8)
        
        # Get robot state
        if self.robot_ctrl is not None:
            state = self.robot_ctrl.get_state()
            joint_pos = np.array(state.pos(), dtype=np.float32)  # [6]
            joint_vel = np.array(state.vel(), dtype=np.float32)  # [6]
            gripper_pos = np.array([state.gripper_pos], dtype=np.float32)  # [1]
            
            # Stack state: [joint_pos(6), joint_vel(6), gripper_pos(1)] = 13D
            obs["state"] = np.concatenate([joint_pos, joint_vel, gripper_pos])
        else:
            obs["state"] = np.zeros(13, dtype=np.float32)
        
        obs["timestamp"] = time.time()
        
        return obs
    
    def prepare_obs_for_policy(self) -> Dict[str, np.ndarray]:
        """
        Prepare observation buffer for policy inference.
        
        Returns:
            Dict with:
                - "rgb": [obs_horizon, C, H, W] float32 NCHW format, normalized
                - "state": [obs_horizon, 13] float32
        """
        # Pad buffer if not full
        obs_list = list(self.obs_buffer)
        if len(obs_list) == 0:
            # Return dummy observation (should not happen in normal operation)
            dummy_obs = {
                "rgb": np.zeros((*self.config.image_size, 3), dtype=np.uint8),
                "state": np.zeros(13, dtype=np.float32)
            }
            obs_list = [dummy_obs]
        
        while len(obs_list) < self.config.obs_horizon:
            obs_list.insert(0, obs_list[0])
        
        # Stack observations
        rgb_stack = np.stack([o["rgb"] for o in obs_list])  # [T, H, W, 3]
        state_stack = np.stack([o["state"] for o in obs_list])  # [T, 13]
        
        # Convert RGB to NCHW and normalize
        rgb_nchw = np.transpose(rgb_stack, (0, 3, 1, 2))  # [T, 3, H, W]
        rgb_nchw = rgb_nchw.astype(np.float32) / 255.0
        
        return {"rgb": rgb_nchw, "state": state_stack}
    
    def predict_action(self, profile: bool = False) -> np.ndarray:
        """
        Run policy inference to get action sequence.
        
        Args:
            profile: If True, print detailed timing breakdown
        
        Returns:
            action_seq: [act_horizon, 7] predicted actions
        """
        if profile:
            t0 = time.time()
        
        # Prepare observation
        obs_dict = self.prepare_obs_for_policy()
        
        if profile:
            t1 = time.time()
        
        # Convert to tensors
        rgb = torch.from_numpy(obs_dict["rgb"]).unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        state = torch.from_numpy(obs_dict["state"]).unsqueeze(0).to(self.device)  # [1, T, 13]
        
        if profile:
            t2 = time.time()
        
        # CRITICAL: All inference must be in no_grad for performance!
        with torch.no_grad():
            # Encode observations
            B, T = rgb.shape[0], rgb.shape[1]
            
            # Flatten for batch processing
            rgb_flat = rgb.view(B * T, *rgb.shape[2:])  # [B*T, C, H, W]
            state_flat = state.view(B * T, -1)  # [B*T, 13]
            
            # Visual encoding
            if profile:
                torch.cuda.synchronize()
                t3 = time.time()
            
            visual_feat = self.visual_encoder(rgb_flat)  # [B*T, 256]
            visual_feat = visual_feat.view(B, T, -1)  # [B, T, 256]
            
            if profile:
                torch.cuda.synchronize()
                t4 = time.time()
            
            # State encoding
            state_feat = self.state_encoder(state_flat)  # [B*T, 256]
            state_feat = state_feat.view(B, T, -1)  # [B, T, 256]
            
            if profile:
                torch.cuda.synchronize()
                t5 = time.time()
            
            # Concatenate and flatten for global conditioning
            obs_features = torch.cat([visual_feat, state_feat], dim=-1)  # [B, T, 512]
            obs_cond = obs_features.view(B, -1)  # [B, T*512]
            
            # Run policy inference (this is the main cost - flow/diffusion steps)
            if profile:
                torch.cuda.synchronize()
                t6 = time.time()
            
            action_seq = self.agent.get_action(obs_cond)
            # action_seq: [B, pred_horizon, 7]
            
            if profile:
                torch.cuda.synchronize()
                t7 = time.time()
            
            if isinstance(action_seq, tuple):
                action_seq = action_seq[0]
        
        # Extract executable actions
        action_seq = action_seq[0].cpu().numpy()  # [pred_horizon, 7]
        
        if profile:
            t8 = time.time()
            print(f"  [Profile] prepare_obs: {(t1-t0)*1000:.1f}ms")
            print(f"  [Profile] to_tensor:   {(t2-t1)*1000:.1f}ms")
            print(f"  [Profile] visual_enc:  {(t4-t3)*1000:.1f}ms")
            print(f"  [Profile] state_enc:   {(t5-t4)*1000:.1f}ms")
            print(f"  [Profile] policy:      {(t7-t6)*1000:.1f}ms  <-- main cost")
            print(f"  [Profile] to_numpy:    {(t8-t7)*1000:.1f}ms")
            print(f"  [Profile] TOTAL:       {(t8-t0)*1000:.1f}ms")
        
        # Get act_horizon actions starting from obs_horizon - 1
        start = self.config.obs_horizon - 1
        end = start + self.config.act_horizon
        executable_actions = action_seq[start:end]  # [act_horizon, 7]
        
        return executable_actions
    
    def interpolate_actions(
        self,
        actions: np.ndarray,
        record_dt: float = None,
        target_dt: float = None,
        smooth: bool = True
    ) -> np.ndarray:
        """
        Interpolate action sequence to higher frequency.
        
        Uses cubic spline for joints (smooth) and linear for gripper (no overshoot).
        
        Args:
            actions: [N, 7] action sequence at record_freq
            record_dt: Original time interval (default: 1/30Hz)
            target_dt: Target time interval (default: 1/500Hz)
            smooth: Apply smoothing before interpolation
            
        Returns:
            interp_actions: [M, 7] interpolated actions at target_freq
        """
        record_dt = record_dt or (1.0 / self.config.record_freq)
        target_dt = target_dt or (1.0 / self.config.control_freq)
        
        num_frames = len(actions)
        if num_frames < 2:
            return actions
        
        # Apply smoothing before interpolation
        if smooth and num_frames > self.config.smooth_window:
            smoothed = np.zeros_like(actions)
            for j in range(actions.shape[1]):
                smoothed[:, j] = uniform_filter1d(
                    actions[:, j], 
                    size=self.config.smooth_window, 
                    mode='nearest'
                )
            actions = smoothed
        
        # Create time arrays
        t_orig = np.arange(num_frames) * record_dt
        total_time = t_orig[-1]
        t_interp = np.arange(0, total_time + target_dt, target_dt)
        
        # Interpolate each dimension
        interp_actions = np.zeros((len(t_interp), actions.shape[1]))
        
        for j in range(actions.shape[1]):
            if j == 6:
                # Linear interpolation for gripper (avoid overshoot)
                interp_actions[:, j] = np.interp(t_interp, t_orig, actions[:, j])
            else:
                # Cubic spline for joints (smooth)
                cs = CubicSpline(t_orig, actions[:, j], bc_type='clamped')
                interp_actions[:, j] = cs(t_interp)
        
        # Clamp gripper
        interp_actions[:, 6] = np.clip(
            interp_actions[:, 6],
            self.config.gripper_limits[0],
            self.config.gripper_limits[1]
        )
        
        return interp_actions
    
    def apply_safety_limits(self, action: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
        """
        Apply safety limits to action.
        
        Args:
            action: [7] current action
            prev_action: [7] previous action
            
        Returns:
            safe_action: [7] clamped action
        """
        safe_action = action.copy()
        
        # Clamp joint delta
        joint_delta = safe_action[:6] - prev_action[:6]
        joint_delta = np.clip(joint_delta, 
                             -self.config.max_joint_delta, 
                             self.config.max_joint_delta)
        safe_action[:6] = prev_action[:6] + joint_delta
        
        # Clamp joint limits
        safe_action[:6] = np.clip(
            safe_action[:6],
            self.config.joint_limits_low,
            self.config.joint_limits_high
        )
        
        # Clamp gripper delta
        gripper_delta = safe_action[6] - prev_action[6]
        gripper_delta = np.clip(gripper_delta,
                               -self.config.max_gripper_delta,
                               self.config.max_gripper_delta)
        safe_action[6] = prev_action[6] + gripper_delta
        
        # Clamp gripper limits
        safe_action[6] = np.clip(
            safe_action[6],
            self.config.gripper_limits[0],
            self.config.gripper_limits[1]
        )
        
        return safe_action
    
    def execute_action(self, action: np.ndarray):
        """
        Send action command to robot.
        
        Args:
            action: [7] action vector (6 joints + 1 gripper)
        """
        if self.robot_ctrl is None:
            return
        
        import arx5_interface as arx5
        
        js = arx5.JointState(self.robot_cfg.joint_dof)
        js.pos()[:] = action[:6]
        js.gripper_pos = float(action[6])
        
        # Debug: print actual command being sent
        if self.verbose and self.step_count % 100 == 0:
            print(f"[Execute] Sending: pos={action[:3]}, gripper={action[6]:.4f}")
        
        self.robot_ctrl.set_joint_cmd(js)
        self.robot_ctrl.send_recv_once()  # CRITICAL: must call this!
    
    def _hold_current_position(self):
        """
        Hold the robot at its paused position.
        
        CRITICAL: Must continuously send position commands to maintain torque!
        Without this, the robot will lose position control and fall due to gravity.
        
        Uses the position saved when entering PAUSED state to avoid drift.
        """
        if self.robot_ctrl is None:
            return
        
        import arx5_interface as arx5
        
        # Use saved pause position if available, otherwise get current
        if self._pause_position is not None:
            target_joints = self._pause_position['joints']
            target_gripper = self._pause_position['gripper']
        else:
            current_state = self.robot_ctrl.get_state()
            target_joints = np.array(current_state.pos())
            target_gripper = current_state.gripper_pos
        
        js = arx5.JointState(self.robot_cfg.joint_dof)
        js.pos()[:] = target_joints
        js.gripper_pos = target_gripper
        
        self.robot_ctrl.set_joint_cmd(js)
        self.robot_ctrl.send_recv_once()
    
    def _enter_damping_mode(self):
        """
        Enter damping mode for safe shutdown.
        
        Damping mode allows the robot to be backdriven while providing
        resistance to rapid movements. Used for emergency stop.
        """
        if self.robot_ctrl is None:
            return
        
        print("[Safety] Entering damping mode...")
        try:
            self.robot_ctrl.set_to_damping()
            print("[Safety] Robot is now in damping mode (can be manually moved)")
        except Exception as e:
            print(f"[Safety] Warning: Failed to enter damping mode: {e}")
    
    def _warmup_gpu(self):
        """
        Warm up GPU by running a dummy inference.
        
        The first CUDA call is slow due to JIT compilation and memory allocation.
        Running a warmup ensures consistent inference timing during actual use.
        """
        if self.agent is None:
            return
        
        print("\nWarming up GPU (first inference is slow due to JIT compilation)...")
        
        # Create dummy observation
        dummy_rgb = np.random.randint(0, 255, (self.config.obs_horizon, 3, *self.config.image_size), dtype=np.uint8)
        dummy_state = np.random.randn(self.config.obs_horizon, self.STATE_DIM).astype(np.float32)
        
        rgb = torch.from_numpy(dummy_rgb.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
        state = torch.from_numpy(dummy_state).unsqueeze(0).to(self.device)
        
        # Run warmup inference
        with torch.no_grad():
            B, T = rgb.shape[0], rgb.shape[1]
            rgb_flat = rgb.view(B * T, *rgb.shape[2:])
            state_flat = state.view(B * T, -1)
            
            # Warmup visual encoder
            for _ in range(2):
                visual_feat = self.visual_encoder(rgb_flat)
            visual_feat = visual_feat.view(B, T, -1)
            
            # Warmup state encoder
            state_feat = self.state_encoder(state_flat)
            state_feat = state_feat.view(B, T, -1)
            
            # Warmup policy
            obs_features = torch.cat([visual_feat, state_feat], dim=-1)
            obs_cond = obs_features.view(B, -1)
            
            for _ in range(2):
                _ = self.agent.get_action(obs_cond)
        
        # Ensure CUDA operations complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print("  GPU warmup complete!")

    def _check_stdin(self) -> Optional[str]:
        """Check for stdin input (non-blocking)."""
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.readline().strip().lower()
        except:
            pass
        return None
    
    def _handle_key(self, key: int):
        """Handle OpenCV key press."""
        if key == ord(' '):  # Space - Start/Pause
            self._toggle_running()
        elif key == ord('r'):  # R - Reset
            self._request_reset()
        elif key == ord('q'):  # Q - Quit with emergency stop
            self._request_emergency_stop()
    
    def _handle_stdin_key(self, cmd: str):
        """Handle terminal command."""
        if cmd in ['s', 'start', 'pause']:
            self._toggle_running()
        elif cmd in ['r', 'reset']:
            self._request_reset()
        elif cmd in ['q', 'quit', 'exit']:
            self._request_emergency_stop()
        elif cmd in ['h', 'help']:
            self._print_help()
    
    def _request_emergency_stop(self):
        """
        Request emergency stop - enters damping mode before quitting.
        
        This allows the robot to be safely backdriven after the program exits,
        preventing sudden drops or damage.
        """
        print("\n[Emergency] Initiating safe shutdown...")
        
        # First signal control loop to stop (it will stop sending commands)
        self._running = False
        
        # Wait a bit for control loop to finish current iteration
        time.sleep(0.1)
        
        # Then enter damping mode for safe manual movement
        self._enter_damping_mode()
        print("[Emergency] Quit requested - robot in damping mode")
    
    def _toggle_running(self):
        """Toggle between running and paused states."""
        if self.state == InferenceState.IDLE:
            self.state = InferenceState.RUNNING
            self._pause_position = None
            print("\n[State] Started inference")
        elif self.state == InferenceState.RUNNING:
            # Save current position when pausing
            if self.robot_ctrl is not None:
                try:
                    current_state = self.robot_ctrl.get_state()
                    self._pause_position = {
                        'joints': np.array(current_state.pos()),
                        'gripper': current_state.gripper_pos
                    }
                    print(f"\\n[State] Paused at position: {self._pause_position['joints'][:3]}...")
                except:
                    self._pause_position = None
                    print("\n[State] Paused (failed to save position)")
            self.state = InferenceState.PAUSED
        elif self.state == InferenceState.PAUSED:
            self._pause_position = None
            self.state = InferenceState.RUNNING
            print("\n[State] Resumed")
    
    def _request_reset(self):
        """Request robot reset to home position."""
        if self.state != InferenceState.RESETTING:
            self.state = InferenceState.RESETTING
            print("\n[State] Resetting to home...")
    
    def _print_help(self):
        """Print control help."""
        print("\n" + "=" * 50)
        print("Controls:")
        print("  [Space] / s  - Start/Pause inference")
        print("  [R] / r      - Reset robot to home")
        print("  [Q] / q      - Quit")
        print("=" * 50)
    
    def run(
        self,
        max_steps: int = 1000,
        dry_run: bool = False,
        visualize: bool = True
    ):
        """
        Main inference loop with dual-thread architecture.
        
        Thread 1 (Control): 500Hz - Executes actions, no blocking
        Thread 2 (Main/UI): ~30Hz - Inference, cameras, visualization
        
        Args:
            max_steps: Maximum number of control steps to run
            dry_run: If True, don't execute on robot
            visualize: Show camera feed
        """
        print("\n" + "=" * 60)
        print("Policy Inference Controls" + (" (Headless)" if self.headless else ""))
        print("=" * 60)
        if self.headless:
            print("  Type command + Enter:")
            print("    s         - Start/Pause inference")
            print("    r         - Reset robot to home")
            print("    q         - Quit")
            print("    h         - Show help")
        else:
            print("  In OpenCV Window (click window first):")
            print("    [Space]   - Start/Pause inference")
            print("    [R]       - Reset robot to home")
            print("    [Q]       - Quit")
            print("  Or type in terminal: s, r, q, h")
        print("=" * 60)
        print(f"\nMax steps: {max_steps}")
        print(f"Dry run: {dry_run}")
        print(f"Control freq: {1.0/self.controller_dt:.1f}Hz")
        print(f"Inference freq: {self.config.inference_freq}Hz")
        print(f"Filter: {self.config.enable_filter}, alpha={self.config.filter_alpha}")
        
        # Reset robot if connected - ALWAYS reset first to set proper gains
        if not dry_run and self.robot_ctrl is not None:
            print("\nResetting robot to home first (sets kp/kd gains)...")
            self.robot_ctrl.reset_to_home()
            time.sleep(0.5)
            
            if self.config.task_initial_pose is not None:
                # Move to task initial pose with confirmation
                print(f"\n" + "=" * 60)
                print("WARNING: Robot will move to task initial position!")
                print("=" * 60)
                print(f"  Target joints: {np.array(self.config.task_initial_pose[:6])}")
                print(f"  Target gripper: {self.config.task_initial_pose[6] if len(self.config.task_initial_pose) > 6 else 0.0:.4f}")
                print(f"  Movement duration: {self.config.init_move_duration:.1f}s")
                print("\n  Please ensure:")
                print("    1. The robot workspace is clear")
                print("    2. No obstacles in the movement path")
                print("    3. You are ready to press emergency stop if needed")
                
                if self.config.require_init_confirm:
                    print("\n  Press ENTER to proceed, or Ctrl+C to abort...")
                    try:
                        input()
                    except KeyboardInterrupt:
                        print("\n[Abort] User cancelled initialization")
                        self._running = False
                        return
                else:
                    print("\n  [No confirmation required, moving in 2 seconds...]")
                    time.sleep(2.0)
                
                print(f"\nMoving robot to task initial pose...")
                self._move_to_pose(
                    np.array(self.config.task_initial_pose),
                    duration=self.config.init_move_duration
                )
            time.sleep(0.5)
        
        # Initialize state
        self.obs_buffer.clear()
        self.action_filter.reset()
        self._running = True
        self.state = InferenceState.IDLE
        self.step_count = 0
        self.inference_count = 0
        
        # Get initial robot state
        initial_state = self._get_robot_state_dict()
        if initial_state:
            self._prev_action = np.concatenate([
                initial_state["joint_pos"],
                [initial_state["gripper_pos"]]
            ])
            self.action_filter.reset(self._prev_action)
            print(f"\nInitial pose: {initial_state['joint_pos'][:3]}...")
        else:
            self._prev_action = np.zeros(7, dtype=np.float32)
            self.action_filter.reset(self._prev_action)
        
        # Start control thread (high frequency, no blocking)
        control_thread = threading.Thread(
            target=self._control_loop,
            args=(max_steps, dry_run),
            daemon=True
        )
        control_thread.start()
        
        # Main thread: inference + UI
        self._inference_and_ui_loop(visualize)
        
        # Wait for control thread to finish
        control_thread.join(timeout=1.0)
    
    def _move_to_pose(self, target_pose: np.ndarray, duration: float = 2.0):
        """
        Smoothly move robot to target pose.
        
        Args:
            target_pose: [7] target joint positions (6 joints + gripper)
            duration: Time to complete movement (seconds)
        """
        if self.robot_ctrl is None:
            return
        
        import arx5_interface as arx5
        
        # Get current pose
        current_state = self.robot_ctrl.get_state()
        current_pos = np.array(current_state.pos())
        current_gripper = current_state.gripper_pos
        
        target_joints = target_pose[:6]
        target_gripper = target_pose[6] if len(target_pose) > 6 else current_gripper
        
        # Interpolate
        n_steps = int(duration / self.controller_dt)
        
        print(f"  Moving in {n_steps} steps over {duration}s...")
        print(f"  From: {current_pos[:3]}...")
        print(f"  To:   {target_joints[:3]}...")
        
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            # Smooth interpolation (ease in/out)
            alpha = 0.5 * (1 - np.cos(np.pi * alpha))
            
            interp_pos = current_pos + alpha * (target_joints - current_pos)
            interp_gripper = current_gripper + alpha * (target_gripper - current_gripper)
            
            js = arx5.JointState(self.robot_cfg.joint_dof)
            js.pos()[:] = interp_pos
            js.gripper_pos = float(interp_gripper)
            
            self.robot_ctrl.set_joint_cmd(js)
            self.robot_ctrl.send_recv_once()  # CRITICAL: must call this!
            
            # Print progress periodically
            if (i + 1) % 200 == 0:
                cur_state = self.robot_ctrl.get_state()
                print(f"  Progress {i+1}/{n_steps}: current={np.array(cur_state.pos())[:3]}")
        
        # Verify final position
        time.sleep(0.1)  # Wait for robot to settle
        final_state = self.robot_ctrl.get_state()
        final_pos = np.array(final_state.pos())
        error = np.linalg.norm(final_pos - target_joints)
        print(f"  Final position: {final_pos[:3]}...")
        print(f"  Final position error: {error:.4f} rad")
    
    def _get_robot_state_dict(self) -> Optional[Dict]:
        """Get current robot state as dictionary."""
        if self.robot_ctrl is None:
            return None
        
        try:
            state = self.robot_ctrl.get_state()
            return {
                "joint_pos": np.array(state.pos(), dtype=np.float32),
                "joint_vel": np.array(state.vel(), dtype=np.float32),
                "gripper_pos": float(state.gripper_pos),
                "timestamp": time.time()
            }
        except Exception as e:
            return None
    
    def _control_loop(self, max_steps: int, dry_run: bool):
        """
        Dedicated control thread - runs at ARX5 controller frequency.
        
        CRITICAL: No blocking operations allowed here:
        - No camera access
        - No file I/O
        - No cv2 operations
        - No inference
        """
        loop_count = 0
        last_warn_time = 0
        
        print(f"[Control] Starting at {1.0/self.controller_dt:.1f}Hz")
        
        while self._running and self.step_count < max_steps:
            loop_start = time.time()
            
            # Get robot state (fast operation)
            robot_state = self._get_robot_state_dict()
            if robot_state:
                with self._state_lock:
                    self._latest_robot_state = robot_state
            
            # Execute action based on state
            if self.state == InferenceState.RUNNING:
                # Get current action from buffer
                action = None
                with self._action_lock:
                    if self._current_action_buffer is not None:
                        idx = min(self._action_buffer_idx, len(self._current_action_buffer) - 1)
                        action = self._current_action_buffer[idx].copy()
                        self._action_buffer_idx += 1
                        
                        # Debug: print every 100 steps
                        if self.step_count % 100 == 0:
                            print(f"[Control] Step {self.step_count}: action={action[:3]}, idx={idx}")
                
                if action is not None:
                    raw_action = action.copy()  # 保存原始动作
                    
                    # Apply EMA filter
                    if self.config.enable_filter:
                        action = self.action_filter.filter(action)
                    filtered_action = action.copy()  # 保存滤波后动作
                    
                    # Apply safety limits
                    if self._prev_action is not None:
                        original_action = action.copy()
                        action = self.apply_safety_limits(action, self._prev_action)
                        
                        # Debug: check if clipped significantly
                        if self.step_count % 100 == 0:
                            delta = action - original_action
                            if np.abs(delta).max() > 0.001:
                                print(f"[Control] Safety clipped: max_delta={np.abs(delta).max():.4f}")
                    
                    executed_action = action.copy()  # 保存实际执行动作
                    
                    # Execute
                    if not dry_run:
                        self.execute_action(action)
                    
                    # 记录控制数据
                    if self._log_trajectory and self.trajectory_logger:
                        robot_state = np.zeros(7)
                        with self._state_lock:
                            if self._latest_robot_state:
                                robot_state[:6] = self._latest_robot_state['joint_pos']
                                robot_state[6] = self._latest_robot_state['gripper_pos']
                        self.trajectory_logger.log_control_step(
                            step=self.step_count,
                            buffer_idx=idx,
                            raw_action=raw_action,
                            filtered_action=filtered_action,
                            executed_action=executed_action,
                            robot_state=robot_state,
                        )
                    
                    self._prev_action = action.copy()
                    self.step_count += 1
                else:
                    # Buffer empty - waiting for inference
                    if loop_count % 500 == 0:
                        print(f"[Control] Waiting for action buffer...")
            
            elif self.state == InferenceState.RESETTING:
                # Reset to home
                if not dry_run and self.robot_ctrl is not None:
                    self.robot_ctrl.reset_to_home()
                    time.sleep(0.5)
                self.state = InferenceState.IDLE
                self.action_filter.reset()
                with self._action_lock:
                    self._current_action_buffer = None
                    self._action_buffer_idx = 0
                print("[Control] Reset complete")
            
            elif self.state == InferenceState.PAUSED:
                # CRITICAL: Hold current position by continuously sending position commands
                # Without this, the robot will lose torque and fall due to gravity!
                if not dry_run and self.robot_ctrl is not None:
                    self._hold_current_position()
                
                if loop_count % 1000 == 0:  # Log every 2 seconds at 500Hz
                    print("[Control] Paused - holding position")
            
            loop_count += 1
            
            # Precise timing
            elapsed = time.time() - loop_start
            sleep_time = self.controller_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > self.controller_dt * 2 and time.time() - last_warn_time > 1.0:
                print(f"[Control] Warning: Loop overrun {elapsed*1000:.1f}ms")
                last_warn_time = time.time()
        
        print(f"[Control] Stopped after {self.step_count} steps")
    
    def _inference_and_ui_loop(self, visualize: bool):
        """
        Main thread: policy inference, camera capture, and UI.
        Runs at inference_freq (~30Hz).
        """
        # Create preview window
        if visualize and not self.headless:
            cv2.namedWindow("Policy Inference", cv2.WINDOW_NORMAL)
        
        inference_dt = 1.0 / self.config.inference_freq
        steps_per_chunk = int(self.config.act_horizon / self.config.record_freq * self.config.control_freq)
        
        last_status_time = time.time()
        inference_start = time.time()
        
        try:
            while self._running:
                loop_start = time.time()
                
                # Get camera frames
                camera_frames = {}
                if self.camera_manager:
                    camera_frames = self.camera_manager.get_frames()
                    with self._state_lock:
                        self._latest_camera_frames = camera_frames
                
                # Run inference if running
                if self.state == InferenceState.RUNNING:
                    # Check if need new action chunk
                    need_new_chunk = False
                    with self._action_lock:
                        if self._current_action_buffer is None:
                            need_new_chunk = True
                            if self.verbose:
                                print("[Inference] Need new chunk: buffer is None")
                        elif self._action_buffer_idx >= steps_per_chunk:
                            need_new_chunk = True
                            if self.verbose:
                                print(f"[Inference] Need new chunk: idx={self._action_buffer_idx} >= {steps_per_chunk}")
                    
                    if need_new_chunk:
                        # Build observation
                        obs = self._build_observation(camera_frames)
                        if obs:
                            self.obs_buffer.append(obs)
                            
                            # Profile first few inferences to diagnose slowness
                            should_profile = self.verbose and self.inference_count < 10
                            
                            if should_profile:
                                print(f"\n[Inference #{self.inference_count + 1}] Profiling...")
                            
                            # Run policy inference
                            pred_start = time.time()
                            raw_actions = self.predict_action(profile=should_profile)
                            pred_time = time.time() - pred_start
                            
                            if self.verbose:
                                print(f"[Inference] Policy output: shape={raw_actions.shape}, "
                                      f"range=[{raw_actions.min():.4f}, {raw_actions.max():.4f}]")
                            
                            # Interpolate to control frequency
                            interp_actions = self.interpolate_actions(raw_actions)
                            
                            if self.verbose:
                                print(f"[Inference] Interpolated: shape={interp_actions.shape}, "
                                      f"first={interp_actions[0, :3]}")
                            
                            # Update action buffer (thread-safe)
                            with self._action_lock:
                                self._current_action_buffer = interp_actions
                                self._action_buffer_idx = 0
                            
                            # 记录推理数据
                            if self._log_trajectory and self.trajectory_logger:
                                current_pos = np.zeros(7)
                                with self._state_lock:
                                    if self._latest_robot_state:
                                        current_pos[:6] = self._latest_robot_state['joint_pos']
                                        current_pos[6] = self._latest_robot_state['gripper_pos']
                                self.trajectory_logger.log_inference(
                                    raw_actions=raw_actions,
                                    interp_actions=interp_actions,
                                    current_position=current_pos,
                                    inference_time=pred_time,
                                )
                            
                            self.inference_count += 1
                            
                            if self.verbose:
                                print(f"\r[Inference #{self.inference_count}] "
                                      f"Pred: {pred_time*1000:.0f}ms, "
                                      f"Steps: {self.step_count}", end="", flush=True)
                
                # UI updates
                if visualize and not self.headless:
                    self._update_preview(camera_frames)
                    
                    # Handle OpenCV key
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        self._handle_key(key)
                    
                    # Also check stdin
                    stdin_cmd = self._check_stdin()
                    if stdin_cmd:
                        self._handle_stdin_key(stdin_cmd)
                
                elif self.headless:
                    # Headless: status output + stdin check
                    if time.time() - last_status_time > 1.0:
                        print(f"[Status] {self.state.value.upper()} | "
                              f"Steps: {self.step_count} | "
                              f"Inferences: {self.inference_count}")
                        last_status_time = time.time()
                    
                    stdin_cmd = self._check_stdin()
                    if stdin_cmd:
                        self._handle_stdin_key(stdin_cmd)
                
                # Maintain inference frequency
                elapsed = time.time() - loop_start
                if elapsed < inference_dt:
                    time.sleep(inference_dt - elapsed)
        
        except KeyboardInterrupt:
            print("\n[Main] Interrupted by Ctrl+C")
        
        finally:
            self._running = False
            total_time = time.time() - inference_start
            
            print(f"\n" + "=" * 60)
            print("Inference Complete")
            print("=" * 60)
            print(f"Total control steps: {self.step_count}")
            print(f"Total inferences: {self.inference_count}")
            print(f"Total time: {total_time:.1f}s")
            if total_time > 0:
                print(f"Average control freq: {self.step_count / total_time:.1f}Hz")
                print(f"Average inference freq: {self.inference_count / total_time:.1f}Hz")
            
            if visualize and not self.headless:
                cv2.destroyAllWindows()
    
    def _build_observation(self, camera_frames: Dict) -> Optional[Dict]:
        """Build observation dict from camera frames and robot state."""
        obs = {}
        
        # Get wrist camera image for policy (resized)
        if "wrist" in camera_frames and camera_frames["wrist"] is not None:
            # 注意: RealSense 返回 BGR 格式，需要转换为 RGB
            bgr = camera_frames["wrist"].rgb.copy()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb,
                           (self.config.image_size[1], self.config.image_size[0]),
                           interpolation=cv2.INTER_LINEAR)
            obs["rgb"] = rgb
            if self.verbose and self.step_count % 100 == 0:
                print(f"[Obs] Camera OK, rgb shape={rgb.shape}, mean={rgb.mean():.1f}")
        else:
            obs["rgb"] = np.zeros((*self.config.image_size, 3), dtype=np.uint8)
            if self.verbose and self.step_count % 100 == 0:
                print("[Obs] WARNING: No wrist camera, using zeros!")
        
        # Get robot state
        with self._state_lock:
            robot_state = self._latest_robot_state
        
        if robot_state:
            obs["state"] = np.concatenate([
                robot_state["joint_pos"],
                robot_state["joint_vel"],
                [robot_state["gripper_pos"]]
            ]).astype(np.float32)
            if self.verbose and self.step_count % 100 == 0:
                print(f"[Obs] Robot state OK: joint_pos={robot_state['joint_pos'][:3]}")
        else:
            obs["state"] = np.zeros(13, dtype=np.float32)
            if self.verbose and self.step_count % 100 == 0:
                print("[Obs] WARNING: No robot state, using zeros!")
        
        obs["timestamp"] = time.time()
        return obs
    
    def _update_preview(self, camera_frames: Dict):
        """Update preview window with dual camera view."""
        preview_images = []
        
        for name in ["wrist", "external"]:
            if name in camera_frames and camera_frames[name] is not None:
                img = camera_frames[name].rgb.copy()
                
                # Add state indicator
                if self.state == InferenceState.RUNNING:
                    cv2.circle(img, (30, 30), 15, (0, 255, 0), -1)  # Green
                    cv2.putText(img, "RUN", (50, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif self.state == InferenceState.PAUSED:
                    cv2.circle(img, (30, 30), 15, (0, 255, 255), -1)  # Yellow
                    cv2.putText(img, "PAUSE", (50, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.circle(img, (30, 30), 15, (128, 128, 128), -1)  # Gray
                    cv2.putText(img, "IDLE", (50, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                
                # Add camera name
                cv2.putText(img, name, (10, img.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add step count
                cv2.putText(img, f"Steps: {self.step_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                preview_images.append(img)
            else:
                # Placeholder for missing camera
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"{name}: N/A", (200, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
                preview_images.append(placeholder)
        
        # Combine images side by side
        if preview_images:
            combined = np.hstack(preview_images)
            cv2.imshow("Policy Inference", combined)
    
    def cleanup(self):
        """
        Clean up resources and safely shutdown robot.
        
        On cleanup, we reset to home position for safe shutdown.
        """
        self._running = False
        
        # 保存轨迹数据
        if self._log_trajectory and self.trajectory_logger:
            self.trajectory_logger.stop_logging()
            self.trajectory_logger.save(self._trajectory_output_path)
        
        # Stop camera
        if self.camera_manager is not None:
            self.camera_manager.stop()
        
        # Safely shutdown robot
        if self.robot_ctrl is not None:
            try:
                print("[Cleanup] Reset to home for safe shutdown...")
                self.robot_ctrl.reset_to_home()
            except Exception as e:
                print(f"[Cleanup] Warning: Error during robot cleanup: {e}")
        
        print("[Cleanup] Done")


def main():
    parser = argparse.ArgumentParser(
        description="ARX5 Real Robot Policy Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python -m data_collection.arx5_policy_inference -c ~/rlft/runs/exp/checkpoints/final.pt
  
  # Dry run (no robot)
  python -m data_collection.arx5_policy_inference -c checkpoint.pt --dry-run
  
  # More smoothing
  python -m data_collection.arx5_policy_inference -c checkpoint.pt --filter-alpha 0.2
  
  # No filtering
  python -m data_collection.arx5_policy_inference -c checkpoint.pt --no-filter
        """
    )
    
    # Required
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Path to checkpoint file (.pt)")
    
    # Mode
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run without robot execution")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualization")
    
    # Initial pose (IMPORTANT for correct inference)
    parser.add_argument("--init-pose", type=str, default=None,
                       help="Initial pose: 'dataset:<path>' to load from dataset, "
                            "or comma-separated joint values")
    parser.add_argument("--init-frame", type=int, default=0,
                       help="Frame index to use when loading init pose from dataset (default: 0)")
    parser.add_argument("--no-confirm", action="store_true",
                       help="Skip confirmation prompt before moving to initial pose")
    parser.add_argument("--init-duration", type=float, default=3.5,
                       help="Duration for moving to initial pose in seconds (default: 3.5)")
    
    # Control
    parser.add_argument("--max-steps", type=int, default=1000000000,
                       help="Maximum number of steps (default: 1000)")
    parser.add_argument("--control-freq", type=float, default=500.0,
                       help="Control frequency in Hz (default: 500)")
    
    # Policy timing (for smooth inference)
    parser.add_argument("--act-horizon", type=int, default=8,
                       help="Action horizon - frames per inference chunk (default: 8)")
    parser.add_argument("--flow-steps", type=int, default=10,
                       help="Number of flow/diffusion steps (default: 10, lower=faster)")
    
    # Smoothing
    parser.add_argument("--no-filter", action="store_true",
                       help="Disable EMA action filtering")
    parser.add_argument("--filter-alpha", type=float, default=0.3,
                       help="EMA filter coefficient, lower=smoother (default: 0.3)")
    
    # Model
    parser.add_argument("--no-ema", action="store_true",
                       help="Don't use EMA weights from checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for inference (default: cuda)")
    
    # Verbosity
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Reduce output verbosity")
    
    # 轨迹记录 (用于对比分析)
    parser.add_argument("--log-trajectory", action="store_true",
                       help="启用轨迹记录 (用于对比分析)")
    parser.add_argument("--trajectory-output", type=str, default="trajectory_single_process.npz",
                       help="轨迹记录输出文件 (默认: trajectory_single_process.npz)")
    
    args = parser.parse_args()
    
    # Parse initial pose
    task_initial_pose = None
    if args.init_pose:
        if args.init_pose.startswith("dataset:"):
            # Load from dataset
            dataset_path = os.path.expanduser(args.init_pose[8:])
            print(f"\nLoading initial pose from dataset: {dataset_path}")
            try:
                import h5py
                with h5py.File(dataset_path, 'r') as f:
                    traj = f['traj_0']
                    joint_pos = np.array(traj['obs']['joint_pos'][args.init_frame])
                    gripper_pos = np.array(traj['obs']['gripper_pos'][args.init_frame])
                    task_initial_pose = tuple(np.concatenate([joint_pos, gripper_pos]))
                    print(f"  Frame {args.init_frame}: joints={joint_pos[:3]}..., gripper={gripper_pos[0]:.4f}")
            except Exception as e:
                print(f"  Error loading dataset: {e}")
                print("  Will use home position instead")
        else:
            # Parse comma-separated values
            try:
                values = [float(x.strip()) for x in args.init_pose.split(',')]
                if len(values) == 6:
                    values.append(0.0)  # Default gripper
                task_initial_pose = tuple(values)
                print(f"\nUsing specified initial pose: {task_initial_pose[:3]}...")
            except Exception as e:
                print(f"Error parsing init-pose: {e}")
    
    # Create config
    config = InferenceConfig(
        control_freq=args.control_freq,
        act_horizon=args.act_horizon,
        num_flow_steps=args.flow_steps,
        enable_filter=not args.no_filter,
        filter_alpha=args.filter_alpha,
        task_initial_pose=task_initial_pose,
        init_move_duration=args.init_duration,
        require_init_confirm=not args.no_confirm,
    )
    
    # Print timing analysis
    steps_per_chunk = int(config.act_horizon / config.record_freq * config.control_freq)
    chunk_duration_ms = steps_per_chunk / config.control_freq * 1000
    print(f"\n" + "=" * 60)
    print("Timing Analysis")
    print("=" * 60)
    print(f"  Action horizon: {config.act_horizon} frames")
    print(f"  Flow/diffusion steps: {config.num_flow_steps}")
    print(f"  Control steps per chunk: {steps_per_chunk}")
    print(f"  Chunk duration: {chunk_duration_ms:.0f}ms")
    print(f"  (Inference must complete within {chunk_duration_ms:.0f}ms for smooth motion)")
    print(f"  TIP: Reduce --flow-steps (e.g., 5) if inference is too slow")
    print("=" * 60)
    
    # Create inference runner
    runner = ARX5PolicyInference(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
        use_ema=not args.no_ema,
        verbose=not args.quiet
    )
    
    try:
        # Setup
        runner.setup()
        
        # 启用轨迹记录
        if args.log_trajectory:
            runner.enable_trajectory_logging(args.trajectory_output)
            if runner.trajectory_logger:
                runner.trajectory_logger.start_logging()
        
        # Run
        runner.run(
            max_steps=args.max_steps,
            dry_run=args.dry_run,
            visualize=not args.no_viz
        )
        
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()