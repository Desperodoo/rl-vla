"""
Real Robot Policy Inference Script

Load trained checkpoint and run inference on real robot data.
Supports visualization of predicted actions vs ground truth.

Usage:
    # Single checkpoint, single step inference
    python inference_real_robot.py --checkpoint_path <path> --mode single
    
    # Single checkpoint, full trajectory rollout
    python inference_real_robot.py --checkpoint_path <path> --mode trajectory
    
    # Compare multiple checkpoints
    python inference_real_robot.py --checkpoint_dir <dir> --mode compare
    
Example:
    python inference_real_robot.py \
        --checkpoint_path runs/consistency_flow-pick_cube-real__1__1766127719/checkpoints/iter_10000.pt \
        --mode trajectory
    
    python inference_real_robot.py \
        --checkpoint_dir runs/consistency_flow-pick_cube-real__1__1766127719/checkpoints \
        --mode compare
"""

import os
import glob
import re
import argparse
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusion_policy.utils import (
    StateEncoder,
    create_real_robot_obs_process_fn,
    get_real_robot_data_info,
    load_traj_hdf5,
)
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.algorithms import (
    DiffusionPolicyAgent,
    FlowMatchingAgent,
    ReflectedFlowAgent,
    ConsistencyFlowAgent,
    ShortCutFlowAgent,
    ShortCutVelocityUNet1D,
)
from diffusion_policy.algorithms.networks import VelocityUNet1D
from diffusion_policy.conditional_unet1d import ConditionalUnet1D


@dataclass
class InferenceConfig:
    """Inference configuration matching training defaults."""
    # Model architecture
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: Tuple[int, ...] = (64, 128, 256)
    n_groups: int = 8
    visual_feature_dim: int = 256
    state_encoder_hidden_dim: int = 128
    state_encoder_out_dim: int = 256
    
    # Algorithm specific
    num_diffusion_iters: int = 100
    num_flow_steps: int = 10
    ema_decay: float = 0.999
    
    # Data processing
    camera_name: str = "wrist"
    include_depth: bool = False
    target_image_size: Tuple[int, int] = (128, 128)


def detect_algorithm_from_checkpoint(checkpoint_path: str) -> str:
    """Detect algorithm type from checkpoint path or content."""
    path_lower = checkpoint_path.lower()
    
    if "diffusion_policy" in path_lower:
        return "diffusion_policy"
    elif "consistency_flow" in path_lower:
        return "consistency_flow"
    elif "shortcut_flow" in path_lower:
        return "shortcut_flow"
    elif "reflected_flow" in path_lower:
        return "reflected_flow"
    elif "flow_matching" in path_lower:
        return "flow_matching"
    
    # Default fallback - try to detect from checkpoint structure
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    agent_keys = list(ckpt.get('agent', {}).keys())
    
    if any('noise_pred_net' in k for k in agent_keys):
        return "diffusion_policy"
    elif any('velocity_net_ema' in k for k in agent_keys):
        return "consistency_flow"
    else:
        return "flow_matching"


def create_agent_for_inference(
    algorithm: str,
    action_dim: int,
    global_cond_dim: int,
    config: InferenceConfig,
    device: str = "cuda",
    action_bounds: Optional[tuple] = None,  # (min, max) or None to disable clipping
) -> nn.Module:
    """Create agent module for inference.
    
    Args:
        action_bounds: Tuple of (min, max) for action clipping. Set to None to disable
                      clipping (recommended for real robot data where actions may exceed [-1, 1]).
    """
    
    if algorithm == "diffusion_policy":
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.unet_dims,
            n_groups=config.n_groups,
        )
        return DiffusionPolicyAgent(
            noise_pred_net=noise_pred_net,
            action_dim=action_dim,
            obs_horizon=config.obs_horizon,
            pred_horizon=config.pred_horizon,
            num_diffusion_iters=config.num_diffusion_iters,
            device=device,
        )
    
    elif algorithm == "flow_matching":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.unet_dims,
            n_groups=config.n_groups,
        )
        return FlowMatchingAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=config.obs_horizon,
            pred_horizon=config.pred_horizon,
            num_flow_steps=config.num_flow_steps,
            action_bounds=action_bounds,
            device=device,
        )
    
    elif algorithm == "reflected_flow":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.unet_dims,
            n_groups=config.n_groups,
        )
        # ReflectedFlow uses action_low/action_high, set wide range if no bounds
        if action_bounds is None:
            action_low, action_high = -10.0, 10.0
        else:
            action_low, action_high = action_bounds
        return ReflectedFlowAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=config.obs_horizon,
            pred_horizon=config.pred_horizon,
            num_flow_steps=config.num_flow_steps,
            action_low=action_low,
            action_high=action_high,
            device=device,
        )
    
    elif algorithm == "consistency_flow":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.unet_dims,
            n_groups=config.n_groups,
        )
        return ConsistencyFlowAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=config.obs_horizon,
            pred_horizon=config.pred_horizon,
            num_flow_steps=config.num_flow_steps,
            ema_decay=config.ema_decay,
            action_bounds=action_bounds,
            device=device,
        )
    
    elif algorithm == "shortcut_flow":
        shortcut_velocity_net = ShortCutVelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.unet_dims,
            n_groups=config.n_groups,
        )
        return ShortCutFlowAgent(
            velocity_net=shortcut_velocity_net,
            action_dim=action_dim,
            obs_horizon=config.obs_horizon,
            pred_horizon=config.pred_horizon,
            device=device,
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


class RealRobotPolicyInference:
    """Real robot policy inference wrapper."""
    
    def __init__(
        self,
        checkpoint_path: str,
        demo_path: str,
        algorithm: Optional[str] = None,
        device: str = "cuda",
        use_ema: bool = True,
        verbose: bool = True,
    ):
        self.device = device
        self.use_ema = use_ema
        self.config = InferenceConfig()
        self.verbose = verbose
        
        # Get data info
        self.data_info = get_real_robot_data_info(demo_path)
        self.action_dim = self.data_info["action_dim"]
        self.state_dim = self.data_info["state_dim"]
        
        # Auto-detect algorithm if not specified
        if algorithm is None:
            algorithm = detect_algorithm_from_checkpoint(checkpoint_path)
        self.algorithm = algorithm
        if verbose:
            print(f"Using algorithm: {algorithm}")
        
        # Create observation processing function
        self.obs_process_fn = create_real_robot_obs_process_fn(
            output_format="NCHW",
            camera_name=self.config.camera_name,
            include_depth=self.config.include_depth,
            target_size=self.config.target_image_size,
        )
        
        # Compute dimensions
        self.visual_feature_dim = self.config.visual_feature_dim
        self.encoded_state_dim = self.config.state_encoder_out_dim
        self.global_cond_dim = self.config.obs_horizon * (self.visual_feature_dim + self.encoded_state_dim)
        
        # Load models
        self._load_checkpoint(checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and create models."""
        if self.verbose:
            print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        # Create visual encoder
        in_channels = 4 if self.config.include_depth else 3
        self.visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=self.visual_feature_dim,
            pool_feature_map=True,
        ).to(self.device)
        
        # Create state encoder
        self.state_encoder = StateEncoder(
            state_dim=self.state_dim,
            hidden_dim=self.config.state_encoder_hidden_dim,
            out_dim=self.config.state_encoder_out_dim,
        ).to(self.device)
        
        # Create agent (action_bounds=None to disable clipping for real robot)
        self.agent = create_agent_for_inference(
            self.algorithm,
            self.action_dim,
            self.global_cond_dim,
            self.config,
            self.device,
            action_bounds=None,  # No clipping for real robot data
        ).to(self.device)
        
        # Load weights
        agent_key = "ema_agent" if self.use_ema else "agent"
        self.agent.load_state_dict(ckpt[agent_key])
        self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        
        # Set to eval mode
        self.agent.eval()
        self.visual_encoder.eval()
        self.state_encoder.eval()
    
    def encode_observation(
        self,
        rgb: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observations to get global conditioning."""
        B = rgb.shape[0]
        T = rgb.shape[1]
        
        rgb_flat = rgb.view(B * T, *rgb.shape[2:]).float() / 255.0
        visual_feat = self.visual_encoder(rgb_flat)
        visual_feat = visual_feat.view(B, T, -1)
        
        state_flat = state.view(B * T, -1).float()
        state_feat = self.state_encoder(state_flat)
        state_feat = state_feat.view(B, T, -1)
        
        obs_features = torch.cat([visual_feat, state_feat], dim=-1)
        obs_cond = obs_features.view(B, -1)
        
        return obs_cond
    
    @torch.no_grad()
    def predict(
        self,
        rgb: torch.Tensor,
        state: torch.Tensor,
        return_full_sequence: bool = False,
    ) -> torch.Tensor:
        """Predict action sequence from observations."""
        obs_cond = self.encode_observation(rgb, state)
        action_seq = self.agent.get_action(obs_cond)
        
        if isinstance(action_seq, tuple):
            action_seq = action_seq[0]
        
        if return_full_sequence:
            return action_seq
        else:
            start = self.config.obs_horizon - 1
            end = start + self.config.act_horizon
            return action_seq[:, start:end]
    
    @torch.no_grad()
    def rollout_trajectory(
        self,
        processed_obs: Dict[str, np.ndarray],
        step_size: int = 1,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Roll out the policy along the entire trajectory.
        
        At each timestep t, use observations [t-obs_horizon+1, t] to predict
        the action at timestep t (first action of the predicted sequence).
        
        Args:
            processed_obs: Dict with 'rgb' [T, C, H, W] and 'state' [T, state_dim]
            step_size: Step size for rollout (1 = every frame)
            show_progress: Whether to show progress bar
            
        Returns:
            predictions: Predicted actions [T, action_dim]
        """
        rgb_all = processed_obs["rgb"]
        state_all = processed_obs["state"]
        T = len(rgb_all)
        
        obs_horizon = self.config.obs_horizon
        predictions = np.zeros((T, self.action_dim), dtype=np.float32)
        
        # Iterator with optional progress bar
        indices = range(0, T, step_size)
        if show_progress:
            indices = tqdm(indices, desc="Rolling out trajectory")
        
        for t in indices:
            # Get observation indices
            obs_start = max(0, t - obs_horizon + 1)
            obs_indices = list(range(obs_start, t + 1))
            
            # Pad if needed
            while len(obs_indices) < obs_horizon:
                obs_indices.insert(0, obs_indices[0])
            
            # Prepare input
            rgb = torch.from_numpy(rgb_all[obs_indices]).unsqueeze(0).to(self.device)
            state = torch.from_numpy(state_all[obs_indices]).unsqueeze(0).to(self.device)
            
            # Predict
            action_seq = self.predict(rgb, state, return_full_sequence=True)
            
            # Take the first action (corresponding to current timestep)
            # The first action in pred_horizon corresponds to obs_horizon - 1 index
            first_action = action_seq[0, self.config.obs_horizon - 1].cpu().numpy()
            predictions[t] = first_action
            
            # Fill in skipped steps with same prediction
            for i in range(1, step_size):
                if t + i < T:
                    predictions[t + i] = first_action
        
        return predictions


def visualize_trajectory_comparison(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    action_names: Optional[List[str]] = None,
    title: str = "Trajectory Rollout: Prediction vs Ground Truth",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 12),
):
    """Visualize full trajectory prediction vs ground truth.
    
    Args:
        ground_truth: Ground truth actions [T, action_dim]
        predictions: Predicted actions [T, action_dim]
        action_names: Names for each action dimension
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    T, action_dim = ground_truth.shape
    
    if action_names is None:
        action_names = [f"Joint {i}" for i in range(action_dim - 1)] + ["Gripper"]
    
    fig, axes = plt.subplots(action_dim, 1, figsize=figsize)
    if action_dim == 1:
        axes = [axes]
    
    timesteps = np.arange(T)
    
    for i, (ax, name) in enumerate(zip(axes, action_names)):
        ax.plot(timesteps, ground_truth[:, i], 'b-', label='Ground Truth', linewidth=1.5, alpha=0.8)
        ax.plot(timesteps, predictions[:, i], 'r-', label='Prediction', linewidth=1.5, alpha=0.8)
        ax.set_ylabel(name, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add error shading
        error = np.abs(predictions[:, i] - ground_truth[:, i])
        ax.fill_between(timesteps, ground_truth[:, i] - error, ground_truth[:, i] + error,
                       alpha=0.2, color='red', label='_nolegend_')
    
    axes[-1].set_xlabel('Timestep', fontsize=10)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    plt.close()


def visualize_multi_checkpoint_comparison(
    ground_truth: np.ndarray,
    checkpoint_predictions: Dict[str, np.ndarray],
    action_names: Optional[List[str]] = None,
    title: str = "Multi-Checkpoint Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 14),
):
    """Visualize predictions from multiple checkpoints against ground truth.
    
    Args:
        ground_truth: Ground truth actions [T, action_dim]
        checkpoint_predictions: Dict mapping checkpoint name to predictions [T, action_dim]
        action_names: Names for each action dimension
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    T, action_dim = ground_truth.shape
    
    if action_names is None:
        action_names = [f"Joint {i}" for i in range(action_dim - 1)] + ["Gripper"]
    
    # Color palette for checkpoints
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(checkpoint_predictions)))
    
    fig, axes = plt.subplots(action_dim, 1, figsize=figsize)
    if action_dim == 1:
        axes = [axes]
    
    timesteps = np.arange(T)
    
    for i, (ax, name) in enumerate(zip(axes, action_names)):
        # Plot ground truth
        ax.plot(timesteps, ground_truth[:, i], 'k-', label='Ground Truth', 
                linewidth=2, alpha=0.9)
        
        # Plot each checkpoint prediction
        for (ckpt_name, pred), color in zip(checkpoint_predictions.items(), colors):
            ax.plot(timesteps, pred[:, i], '-', label=ckpt_name, 
                   linewidth=1.2, alpha=0.7, color=color)
        
        ax.set_ylabel(name, fontsize=10)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Timestep', fontsize=10)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    plt.close()


def visualize_error_over_checkpoints(
    ground_truth: np.ndarray,
    checkpoint_predictions: Dict[str, np.ndarray],
    action_names: Optional[List[str]] = None,
    title: str = "Error Over Training Iterations",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """Visualize how error changes across checkpoints.
    
    Args:
        ground_truth: Ground truth actions [T, action_dim]
        checkpoint_predictions: Dict mapping checkpoint name to predictions
        action_names: Names for each action dimension
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    T, action_dim = ground_truth.shape
    
    if action_names is None:
        action_names = [f"Joint {i}" for i in range(action_dim - 1)] + ["Gripper"]
    
    # Extract iteration numbers and sort
    def extract_iter(name):
        match = re.search(r'iter[_]?(\d+)', name)
        return int(match.group(1)) if match else 0
    
    sorted_names = sorted(checkpoint_predictions.keys(), key=extract_iter)
    iterations = [extract_iter(name) for name in sorted_names]
    
    # Compute errors
    mae_per_dim = np.zeros((len(sorted_names), action_dim))
    mse_total = np.zeros(len(sorted_names))
    mae_total = np.zeros(len(sorted_names))
    
    for idx, name in enumerate(sorted_names):
        pred = checkpoint_predictions[name]
        mae_per_dim[idx] = np.mean(np.abs(pred - ground_truth), axis=0)
        mse_total[idx] = np.mean((pred - ground_truth) ** 2)
        mae_total[idx] = np.mean(np.abs(pred - ground_truth))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot per-dimension MAE
    for i, name in enumerate(action_names):
        ax1.plot(iterations, mae_per_dim[:, i], 'o-', label=name, linewidth=1.5, markersize=4)
    
    ax1.set_xlabel('Training Iteration', fontsize=10)
    ax1.set_ylabel('MAE', fontsize=10)
    ax1.set_title('Per-Dimension MAE vs Training Iteration', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot total MAE and MSE
    ax2.plot(iterations, mae_total, 'b-o', label='MAE', linewidth=2, markersize=5)
    ax2.plot(iterations, mse_total, 'r-s', label='MSE', linewidth=2, markersize=5)
    ax2.set_xlabel('Training Iteration', fontsize=10)
    ax2.set_ylabel('Error', fontsize=10)
    ax2.set_title('Total Error vs Training Iteration', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    plt.close()


def find_checkpoints(checkpoint_dir: str, pattern: str = "iter_*.pt") -> List[str]:
    """Find all checkpoint files in directory."""
    search_pattern = os.path.join(checkpoint_dir, pattern)
    checkpoints = glob.glob(search_pattern)
    
    # Sort by iteration number
    def extract_iter(path):
        match = re.search(r'iter[_]?(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else 0
    
    checkpoints = sorted(checkpoints, key=extract_iter)
    return checkpoints


def run_single_step_inference(args):
    """Run single step inference (original functionality)."""
    use_ema = not args.no_ema
    
    inference = RealRobotPolicyInference(
        checkpoint_path=args.checkpoint_path,
        demo_path=args.demo_path,
        algorithm=args.algorithm,
        device=args.device,
        use_ema=use_ema,
    )
    
    print(f"\nLoading demo data from: {args.demo_path}")
    raw_data = load_traj_hdf5(args.demo_path, num_traj=args.traj_idx + 1)
    traj_key = f"traj_{args.traj_idx}"
    traj = raw_data[traj_key]
    
    print(f"Trajectory {args.traj_idx} length: {len(traj['actions'])} steps")
    
    config = inference.config
    obs_horizon = config.obs_horizon
    pred_horizon = config.pred_horizon
    
    start_idx = args.start_idx
    obs_indices = list(range(max(0, start_idx - obs_horizon + 1), start_idx + 1))
    while len(obs_indices) < obs_horizon:
        obs_indices.insert(0, obs_indices[0])
    
    processed = inference.obs_process_fn(traj["obs"])
    
    rgb_obs = torch.from_numpy(processed["rgb"][obs_indices]).unsqueeze(0).to(args.device)
    state_obs = torch.from_numpy(processed["state"][obs_indices]).unsqueeze(0).to(args.device)
    
    with torch.no_grad():
        full_prediction = inference.predict(rgb_obs, state_obs, return_full_sequence=True)
    
    gt_start = start_idx
    gt_end = min(start_idx + pred_horizon, len(traj["actions"]))
    ground_truth = traj["actions"][gt_start:gt_end]
    
    if len(ground_truth) < pred_horizon:
        pad_len = pred_horizon - len(ground_truth)
        ground_truth = np.concatenate([ground_truth, np.tile(ground_truth[-1:], (pad_len, 1))], axis=0)
    
    pred_np = full_prediction[0].cpu().numpy()
    action_names = [f"Joint{i}" for i in range(6)] + ["Gripper"]
    
    # Compute metrics
    mse = np.mean((pred_np - ground_truth) ** 2)
    mae = np.mean(np.abs(pred_np - ground_truth))
    print(f"\nMSE: {mse:.6f}, MAE: {mae:.6f}")
    
    save_path = args.save_fig
    if save_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint_path)
        save_path = os.path.join(ckpt_dir, f"inference_single_traj{args.traj_idx}_step{start_idx}.png")
    
    visualize_trajectory_comparison(
        ground_truth, pred_np,
        action_names=action_names,
        title=f"Single Step Inference: Traj {args.traj_idx}, Step {start_idx}",
        save_path=save_path,
    )


def run_trajectory_rollout(args):
    """Run full trajectory rollout."""
    use_ema = not args.no_ema
    
    inference = RealRobotPolicyInference(
        checkpoint_path=args.checkpoint_path,
        demo_path=args.demo_path,
        algorithm=args.algorithm,
        device=args.device,
        use_ema=use_ema,
    )
    
    print(f"\nLoading demo data from: {args.demo_path}")
    raw_data = load_traj_hdf5(args.demo_path, num_traj=args.traj_idx + 1)
    traj_key = f"traj_{args.traj_idx}"
    traj = raw_data[traj_key]
    
    T = len(traj['actions'])
    print(f"Trajectory {args.traj_idx} length: {T} steps")
    
    # Process observations
    processed = inference.obs_process_fn(traj["obs"])
    
    # Rollout along trajectory
    print(f"\nRolling out policy along trajectory...")
    predictions = inference.rollout_trajectory(
        processed, 
        step_size=args.step_size,
        show_progress=True
    )
    
    ground_truth = traj["actions"]
    
    # Compute metrics
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    print(f"\n{'='*50}")
    print(f"Trajectory Rollout Results")
    print(f"{'='*50}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    action_names = [f"Joint{i}" for i in range(6)] + ["Gripper"]
    print(f"\nPer-Dimension MAE:")
    for i, name in enumerate(action_names):
        dim_mae = np.mean(np.abs(predictions[:, i] - ground_truth[:, i]))
        print(f"  {name}: {dim_mae:.6f}")
    
    # Save results
    save_path = args.save_fig
    if save_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint_path)
        ckpt_name = os.path.basename(args.checkpoint_path).replace('.pt', '')
        save_path = os.path.join(ckpt_dir, f"trajectory_rollout_{ckpt_name}_traj{args.traj_idx}.png")
    
    visualize_trajectory_comparison(
        ground_truth, predictions,
        action_names=action_names,
        title=f"Trajectory Rollout: {os.path.basename(args.checkpoint_path)} - Traj {args.traj_idx}",
        save_path=save_path,
    )


def run_multi_checkpoint_comparison(args):
    """Compare predictions from multiple checkpoints."""
    # Find all checkpoints
    checkpoint_dir = args.checkpoint_dir
    checkpoints = find_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {os.path.basename(ckpt)}")
    
    # Load demo data
    print(f"\nLoading demo data from: {args.demo_path}")
    raw_data = load_traj_hdf5(args.demo_path, num_traj=args.traj_idx + 1)
    traj_key = f"traj_{args.traj_idx}"
    traj = raw_data[traj_key]
    
    T = len(traj['actions'])
    print(f"Trajectory {args.traj_idx} length: {T} steps")
    
    ground_truth = traj["actions"]
    
    # Run inference for each checkpoint
    checkpoint_predictions = {}
    use_ema = not args.no_ema
    
    print(f"\nRunning inference for each checkpoint...")
    for ckpt_path in tqdm(checkpoints, desc="Processing checkpoints"):
        ckpt_name = os.path.basename(ckpt_path).replace('.pt', '')
        
        inference = RealRobotPolicyInference(
            checkpoint_path=ckpt_path,
            demo_path=args.demo_path,
            algorithm=args.algorithm,
            device=args.device,
            use_ema=use_ema,
            verbose=False,
        )
        
        processed = inference.obs_process_fn(traj["obs"])
        predictions = inference.rollout_trajectory(
            processed,
            step_size=args.step_size,
            show_progress=False
        )
        
        checkpoint_predictions[ckpt_name] = predictions
        
        # Clean up
        del inference
        torch.cuda.empty_cache()
    
    # Print metrics for each checkpoint
    print(f"\n{'='*60}")
    print("Checkpoint Comparison Results")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<20} {'MSE':>12} {'MAE':>12}")
    print(f"{'-'*44}")
    
    for ckpt_name, pred in checkpoint_predictions.items():
        mse = np.mean((pred - ground_truth) ** 2)
        mae = np.mean(np.abs(pred - ground_truth))
        print(f"{ckpt_name:<20} {mse:>12.6f} {mae:>12.6f}")
    
    action_names = [f"Joint{i}" for i in range(6)] + ["Gripper"]
    
    # Save trajectory comparison
    save_path = args.save_fig
    if save_path is None:
        save_path = os.path.join(checkpoint_dir, f"checkpoint_comparison_traj{args.traj_idx}.png")
    
    visualize_multi_checkpoint_comparison(
        ground_truth, checkpoint_predictions,
        action_names=action_names,
        title=f"Multi-Checkpoint Comparison - Trajectory {args.traj_idx}",
        save_path=save_path,
    )
    
    # Save error progression
    error_save_path = save_path.replace('.png', '_error.png')
    visualize_error_over_checkpoints(
        ground_truth, checkpoint_predictions,
        action_names=action_names,
        title=f"Error Progression Over Training - Trajectory {args.traj_idx}",
        save_path=error_save_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Real Robot Policy Inference")
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="trajectory",
        choices=["single", "trajectory", "compare"],
        help="Inference mode: single (one timestep), trajectory (full rollout), compare (multi-checkpoint)",
    )
    
    # Checkpoint paths
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint file (.pt) for single/trajectory mode",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing checkpoints for compare mode",
    )
    
    # Data paths
    parser.add_argument(
        "--demo_path",
        type=str,
        default="~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5",
        help="Path to demo data (HDF5 file)",
    )
    
    # Inference settings
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Algorithm type (auto-detect if not specified)",
    )
    parser.add_argument(
        "--traj_idx",
        type=int,
        default=0,
        help="Trajectory index to test",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=100,
        help="Starting timestep for single mode",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size for trajectory rollout (1 = every frame)",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Don't use EMA weights",
    )
    parser.add_argument(
        "--save_fig",
        type=str,
        default=None,
        help="Path to save visualization figure",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    
    args = parser.parse_args()
    
    # Expand paths
    args.demo_path = os.path.expanduser(args.demo_path)
    
    # Validate arguments
    if args.mode in ["single", "trajectory"]:
        if args.checkpoint_path is None:
            parser.error("--checkpoint_path is required for single/trajectory mode")
    elif args.mode == "compare":
        if args.checkpoint_dir is None:
            parser.error("--checkpoint_dir is required for compare mode")
    
    # Run appropriate mode
    if args.mode == "single":
        run_single_step_inference(args)
    elif args.mode == "trajectory":
        run_trajectory_rollout(args)
    elif args.mode == "compare":
        run_multi_checkpoint_comparison(args)


if __name__ == "__main__":
    main()
