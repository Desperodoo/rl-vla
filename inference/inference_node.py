#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARX5 推理进程

负责策略推理、相机采集和用户交互。
通过共享内存与控制进程通信，不直接控制机器人。

特性:
- GPU 策略推理 (~30Hz)
- 双 RealSense 相机采集
- OpenCV 可视化
- 键盘控制

用法:
    # 先启动控制节点
    python -m inference.control_node --model X5 --interface can0
    
    # 再启动推理节点
    python -m inference.inference_node -c checkpoint.pt
    
    # 模拟模式
    python -m inference.inference_node -c checkpoint.pt --dry-run
    
控制:
    [Space] - 开始/暂停推理
    [R]     - 复位机器人
    [Q]     - 退出
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

import cv2
import torch
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.config import RL_VLA_CONFIG, setup_arx5, setup_rlft
from inference.shared_state import SharedState, ControlFlags
from inference.trajectory_logger import TrajectoryLogger


@dataclass
class InferenceConfig:
    """推理配置"""
    
    # 时序参数
    obs_horizon: int = 2          # 观测帧堆叠数
    act_horizon: int = 8          # 执行的动作帧数
    pred_horizon: int = 16        # 预测的总动作帧数
    
    # 控制参数
    control_freq: float = 500.0   # 控制频率 (Hz)
    inference_freq: float = 30.0  # 推理频率 (Hz)
    record_freq: float = 30.0     # 原始数据采集频率 (Hz)
    
    # 推理速度 (flow/diffusion 步数)
    num_flow_steps: int = 10
    
    # 相机
    image_size: Tuple[int, int] = (128, 128)  # 模型输入尺寸 (H, W)
    
    # 平滑 (注意: 实际滤波在控制节点中进行)
    smooth_window: int = 3        # 插值前平滑窗口
    
    # Chunk 过渡
    # 注意: 原版没有过渡帧逻辑，设为 False 以保持一致
    # 设为 True 时会在 chunk 边界添加平滑过渡帧
    enable_transition_frames: bool = False
    transition_frame_count: int = 10  # 过渡帧数量


class InferenceNode:
    """
    ARX5 推理节点
    
    负责:
    1. 加载和运行策略模型
    2. 相机数据采集
    3. 用户交互
    4. 通过共享内存与控制进程通信
    """
    
    ACTION_DIM = 7   # 6 关节 + 1 夹爪
    STATE_DIM = 13   # 6 joint_pos + 6 joint_vel + 1 gripper_pos
    SHM_NAME = "arx5_control"
    
    def __init__(
        self,
        checkpoint_path: str,
        config: InferenceConfig = None,
        device: str = "cuda",
        use_ema: bool = True,
        verbose: bool = True,
        headless: bool = False,
        dry_run: bool = False,
        initial_pose: Optional[np.ndarray] = None,
    ):
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.config = config or InferenceConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_ema = use_ema
        self.verbose = verbose
        self.headless = headless
        self.dry_run = dry_run
        self.initial_pose = initial_pose
        
        # 模型组件
        self.visual_encoder = None
        self.state_encoder = None
        self.agent = None
        self.algorithm = None
        
        # 硬件组件
        self.camera_manager = None
        
        # 共享内存
        self.shared_state: Optional[SharedState] = None
        
        # 观测历史
        self.obs_buffer: deque = deque(maxlen=self.config.obs_horizon)
        
        # 注意: EMA 滤波已移到控制节点中
        # 这样与原版行为一致 (在 500Hz 控制循环中滤波)
        
        # 状态
        self._running = False
        self._is_inferencing = False
        self.inference_count = 0
        
        # 轨迹记录器 (用于对比分析)
        self.trajectory_logger: Optional[TrajectoryLogger] = None
        self._log_trajectory = False
        self._trajectory_output_path = "trajectory_multi_process_inference.npz"
    
    def enable_trajectory_logging(self, output_path: str = None):
        """启用轨迹记录"""
        self.trajectory_logger = TrajectoryLogger(name="multi_process_inference")
        self._log_trajectory = True
        self._trajectory_output_path = output_path or "trajectory_multi_process_inference.npz"
        print(f"[TrajectoryLogger] 已启用, 输出: {self._trajectory_output_path}")
    
    def setup(self):
        """初始化"""
        print("\n" + "=" * 60)
        print("ARX5 推理节点")
        print("=" * 60)
        
        # 1. 加载策略
        self._load_policy()
        
        # 2. GPU 预热
        self._warmup_gpu()
        
        # 3. 初始化相机
        self._setup_cameras()
        
        # 4. 连接控制进程
        self._connect_control_node()
        
        print("\n✓ 推理节点初始化完成")
    
    def _load_policy(self):
        """加载训练好的策略"""
        print(f"\n加载策略: {self.checkpoint_path}")
        
        setup_rlft()
        
        try:
            from diffusion_policy.plain_conv import PlainConv
            from diffusion_policy.utils import StateEncoder
            from diffusion_policy.algorithms import (
                DiffusionPolicyAgent,
                FlowMatchingAgent,
                ConsistencyFlowAgent,
            )
            from diffusion_policy.algorithms.networks import VelocityUNet1D
            from diffusion_policy.conditional_unet1d import ConditionalUnet1D
            
            # 加载 checkpoint
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 自动检测算法类型
            agent_keys = list(ckpt.get('agent', ckpt.get('ema_agent', {})).keys())
            if any('velocity_net' in k for k in agent_keys):
                if any('velocity_net_ema' in k for k in agent_keys):
                    self.algorithm = "consistency_flow"
                else:
                    self.algorithm = "flow_matching"
            elif any('noise_pred_net' in k for k in agent_keys):
                self.algorithm = "diffusion_policy"
            else:
                path_lower = self.checkpoint_path.lower()
                if "diffusion_policy" in path_lower:
                    self.algorithm = "diffusion_policy"
                elif "consistency_flow" in path_lower:
                    self.algorithm = "consistency_flow"
                else:
                    self.algorithm = "flow_matching"
            
            print(f"  算法: {self.algorithm}")
            
            # 模型架构参数
            visual_feature_dim = 256
            state_encoder_hidden_dim = 128
            state_encoder_out_dim = 256
            diffusion_step_embed_dim = 64
            unet_dims = (64, 128, 256)
            n_groups = 8
            
            global_cond_dim = self.config.obs_horizon * (visual_feature_dim + state_encoder_out_dim)
            
            # 创建视觉编码器
            self.visual_encoder = PlainConv(
                in_channels=3,
                out_dim=visual_feature_dim,
                pool_feature_map=True,
            ).to(self.device)
            
            # 创建状态编码器
            self.state_encoder = StateEncoder(
                state_dim=self.STATE_DIM,
                hidden_dim=state_encoder_hidden_dim,
                out_dim=state_encoder_out_dim,
            ).to(self.device)
            
            # 创建 agent
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
                    action_bounds=None,
                    device=str(self.device),
                )
            else:  # flow_matching
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
            
            self.agent = self.agent.to(self.device)
            
            # 加载权重
            agent_key = "ema_agent" if self.use_ema else "agent"
            if agent_key not in ckpt:
                agent_key = "agent"
                print(f"  警告: 未找到 EMA 权重，使用普通权重")
            
            self.agent.load_state_dict(ckpt[agent_key])
            self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
            self.state_encoder.load_state_dict(ckpt["state_encoder"])
            
            # 设置为评估模式
            self.agent.eval()
            self.visual_encoder.eval()
            self.state_encoder.eval()
            
            print(f"  obs_horizon: {self.config.obs_horizon}")
            print(f"  act_horizon: {self.config.act_horizon}")
            print(f"  pred_horizon: {self.config.pred_horizon}")
            print(f"  num_flow_steps: {self.config.num_flow_steps}")
            print(f"  EMA 权重: {agent_key == 'ema_agent'}")
            
        except Exception as e:
            print(f"加载策略失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _warmup_gpu(self):
        """GPU 预热"""
        if self.agent is None:
            return
        
        print("\nGPU 预热中...")
        
        dummy_rgb = np.random.randint(
            0, 255, 
            (self.config.obs_horizon, 3, *self.config.image_size), 
            dtype=np.uint8
        )
        dummy_state = np.random.randn(
            self.config.obs_horizon, self.STATE_DIM
        ).astype(np.float32)
        
        rgb = torch.from_numpy(dummy_rgb.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
        state = torch.from_numpy(dummy_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            B, T = rgb.shape[0], rgb.shape[1]
            rgb_flat = rgb.view(B * T, *rgb.shape[2:])
            state_flat = state.view(B * T, -1)
            
            for _ in range(3):
                visual_feat = self.visual_encoder(rgb_flat)
                visual_feat = visual_feat.view(B, T, -1)
                
                state_feat = self.state_encoder(state_flat)
                state_feat = state_feat.view(B, T, -1)
                
                obs_features = torch.cat([visual_feat, state_feat], dim=-1)
                obs_cond = obs_features.view(B, -1)
                
                _ = self.agent.get_action(obs_cond)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print("  GPU 预热完成!")
    
    def _setup_cameras(self):
        """初始化 RealSense 相机"""
        print("\n初始化相机...")
        
        try:
            from .camera_manager import CameraManager, DEFAULT_CAMERA_CONFIGS
            
            self.camera_manager = CameraManager(DEFAULT_CAMERA_CONFIGS)
            self.camera_manager.initialize(auto_assign=True)
            self.camera_manager.start()
            
            print("  相机初始化完成")
            
        except Exception as e:
            print(f"警告: 相机初始化失败: {e}")
            print("  将以纯状态模式运行")
            self.camera_manager = None
    
    def _connect_control_node(self):
        """连接控制进程"""
        print("\n连接控制进程...")
        
        if self.dry_run:
            print("  模拟模式 - 创建本地共享内存")
            self.shared_state = SharedState.create(self.SHM_NAME)
        else:
            try:
                self.shared_state = SharedState.connect(self.SHM_NAME, timeout=10.0)
                print(f"  已连接到控制进程")
            except TimeoutError:
                print("  [错误] 无法连接到控制进程")
                print("  请先启动控制节点: python -m inference.control_node")
                raise
        
        # 读取初始状态
        pos, vel, ts = self.shared_state.read_robot_state()
        print(f"  机器人位置: {pos[:3]}...")
        print(f"  时间戳: {ts:.3f}")
        
        # 如果指定了初始位姿，写入共享内存
        if self.initial_pose is not None:
            print(f"\n设置初始位姿:")
            print(f"  关节: {self.initial_pose[:3]}...")
            print(f"  夹爪: {self.initial_pose[6]:.4f}")
            self.shared_state.write_initial_pose(self.initial_pose)
    
    def prepare_obs_for_policy(self) -> Dict[str, np.ndarray]:
        """准备观测数据供策略推理"""
        obs_list = list(self.obs_buffer)
        if len(obs_list) == 0:
            dummy_obs = {
                "rgb": np.zeros((*self.config.image_size, 3), dtype=np.uint8),
                "state": np.zeros(self.STATE_DIM, dtype=np.float32)
            }
            obs_list = [dummy_obs]
        
        while len(obs_list) < self.config.obs_horizon:
            obs_list.insert(0, obs_list[0])
        
        rgb_stack = np.stack([o["rgb"] for o in obs_list])
        state_stack = np.stack([o["state"] for o in obs_list])
        
        rgb_nchw = np.transpose(rgb_stack, (0, 3, 1, 2))
        rgb_nchw = rgb_nchw.astype(np.float32) / 255.0
        
        return {"rgb": rgb_nchw, "state": state_stack}
    
    def predict_action(self) -> np.ndarray:
        """执行策略推理获取动作序列"""
        obs_dict = self.prepare_obs_for_policy()
        
        rgb = torch.from_numpy(obs_dict["rgb"]).unsqueeze(0).to(self.device)
        state = torch.from_numpy(obs_dict["state"]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            B, T = rgb.shape[0], rgb.shape[1]
            rgb_flat = rgb.view(B * T, *rgb.shape[2:])
            state_flat = state.view(B * T, -1)
            
            visual_feat = self.visual_encoder(rgb_flat)
            visual_feat = visual_feat.view(B, T, -1)
            
            state_feat = self.state_encoder(state_flat)
            state_feat = state_feat.view(B, T, -1)
            
            obs_features = torch.cat([visual_feat, state_feat], dim=-1)
            obs_cond = obs_features.view(B, -1)
            
            action_seq = self.agent.get_action(obs_cond)
            
            if isinstance(action_seq, tuple):
                action_seq = action_seq[0]
        
        action_seq = action_seq[0].cpu().numpy()
        
        start = self.config.obs_horizon - 1
        end = start + self.config.act_horizon
        executable_actions = action_seq[start:end]
        
        return executable_actions
    
    def interpolate_actions(self, actions: np.ndarray) -> np.ndarray:
        """将动作序列插值到更高频率"""
        record_dt = 1.0 / self.config.record_freq
        target_dt = 1.0 / self.config.control_freq
        
        num_frames = len(actions)
        if num_frames < 2:
            return actions
        
        # 插值前平滑
        if num_frames > self.config.smooth_window:
            smoothed = np.zeros_like(actions)
            for j in range(actions.shape[1]):
                smoothed[:, j] = uniform_filter1d(
                    actions[:, j], 
                    size=self.config.smooth_window, 
                    mode='nearest'
                )
            actions = smoothed
        
        t_orig = np.arange(num_frames) * record_dt
        total_time = t_orig[-1]
        t_interp = np.arange(0, total_time + target_dt, target_dt)
        
        interp_actions = np.zeros((len(t_interp), actions.shape[1]))
        
        for j in range(actions.shape[1]):
            if j == 6:  # 夹爪使用线性插值
                interp_actions[:, j] = np.interp(t_interp, t_orig, actions[:, j])
            else:  # 关节使用三次样条
                cs = CubicSpline(t_orig, actions[:, j], bc_type='clamped')
                interp_actions[:, j] = cs(t_interp)
        
        return interp_actions
    
    def run(self, visualize: bool = True):
        """主推理循环"""
        print("\n" + "=" * 60)
        print("推理循环启动")
        print("=" * 60)
        
        if self.headless:
            print("  [无头模式] 使用键盘输入:")
            print("    s - 开始/暂停")
            print("    r - 复位")
            print("    q - 退出")
        else:
            print("  [OpenCV 窗口]")
            print("    [Space] - 开始/暂停")
            print("    [R]     - 复位")
            print("    [Q]     - 退出")
        
        if visualize and not self.headless:
            cv2.namedWindow("ARX5 Inference", cv2.WINDOW_NORMAL)
        
        self._running = True
        self._is_inferencing = False
        self.obs_buffer.clear()
        
        inference_dt = 1.0 / self.config.inference_freq
        last_inference_time = 0.0
        
        try:
            while self._running:
                loop_start = time.time()
                
                # 获取相机帧
                camera_frames = self._get_camera_frames()
                
                # 获取机器人状态
                robot_pos, robot_vel, _ = self.shared_state.read_robot_state()
                
                # 构建观测
                obs = self._build_observation(camera_frames, robot_pos, robot_vel)
                if obs is not None:
                    self.obs_buffer.append(obs)
                
                # 检查控制状态
                control_state = self.shared_state.get_state()
                
                # 推理
                if control_state == ControlFlags.RUNNING and self._is_inferencing:
                    # 重要修复：必须等缓冲区完全执行完再替换，否则会导致动作跳变！
                    # 原版 arx5_policy_inference 也是这样做的：idx >= steps_per_chunk 时才触发
                    # 不能用提前触发(70%)，因为 write_actions 会重置 idx 为 0
                    if self.shared_state.is_action_buffer_empty():
                        if time.time() - last_inference_time >= inference_dt:
                            self._do_inference()
                            last_inference_time = time.time()
                
                # 更新可视化
                if visualize and not self.headless:
                    self._update_preview(camera_frames, control_state)
                    
                    # 处理按键
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key)
                else:
                    # 无头模式下检查标准输入
                    self._check_stdin()
                
                # 更新推理时间戳
                self.shared_state.update_inference_timestamp()
                
                # 频率控制
                elapsed = time.time() - loop_start
                sleep_time = inference_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time * 0.5)  # 留余量给 UI
        
        except KeyboardInterrupt:
            print("\n[推理节点] Ctrl+C 中断")
        
        finally:
            self._cleanup()
    
    def _get_camera_frames(self) -> Dict:
        """获取相机帧"""
        if self.camera_manager is None:
            return {}
        
        try:
            return self.camera_manager.get_frames()
        except:
            return {}
    
    def _build_observation(self, camera_frames: Dict, 
                          robot_pos: np.ndarray, 
                          robot_vel: np.ndarray) -> Optional[Dict]:
        """构建观测数据"""
        obs = {}
        
        # RGB
        if "wrist" in camera_frames and camera_frames["wrist"] is not None:
            rgb = camera_frames["wrist"].rgb.copy()
            rgb = cv2.resize(
                rgb,
                (self.config.image_size[1], self.config.image_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            obs["rgb"] = rgb
        else:
            obs["rgb"] = np.zeros((*self.config.image_size, 3), dtype=np.uint8)
        
        # State
        obs["state"] = np.concatenate([
            robot_pos[:6],   # joint_pos
            robot_vel[:6],   # joint_vel
            [robot_pos[6]]   # gripper_pos
        ]).astype(np.float32)
        
        obs["timestamp"] = time.time()
        return obs
    
    def _do_inference(self):
        """执行一次推理"""
        start = time.time()
        
        # 获取当前控制目标位置（用于平滑过渡）
        current_target = self.shared_state.read_target_pose()
        
        # 预测动作
        actions = self.predict_action()
        
        # 注意: EMA 滤波已移到控制节点 (500Hz) 中应用
        # 这里不再做批量滤波，因为：
        # 1. 原版在 500Hz 控制循环中对每个动作滤波
        # 2. 在推理时批量滤波会改变时序特性
        
        # 插值到控制频率
        interp_actions = self.interpolate_actions(actions)
        
        # 关键：检查新 chunk 首帧与当前位置的跳变
        if np.any(current_target != 0):  # 确保 current_target 有效
            first_action = interp_actions[0]
            delta = first_action - current_target
            max_joint_jump = np.abs(delta[:6]).max()
            gripper_jump = np.abs(delta[6])
            
            if self.verbose:
                print(f"[推理] Chunk 过渡检查:")
                print(f"  当前位置: joints={current_target[:3]}, gripper={current_target[6]:.4f}")
                print(f"  新chunk首帧: joints={first_action[:3]}, gripper={first_action[6]:.4f}")
                print(f"  跳变量: 关节={max_joint_jump:.4f}rad, 夹爪={gripper_jump:.4f}m")
            
            # 如果启用过渡帧且跳变太大，添加平滑过渡帧
            # 注意: 原版没有这个逻辑，默认关闭以保持一致
            if self.config.enable_transition_frames:
                if max_joint_jump > 0.02 or gripper_jump > 0.005:
                    print(f"[推理] ⚠ 检测到较大跳变，添加过渡帧...")
                    # 在开头插入从当前位置到第一帧的过渡
                    n_transition = self.config.transition_frame_count
                    transition = np.zeros((n_transition, 7))
                    for i in range(n_transition):
                        alpha = (i + 1) / (n_transition + 1)
                        transition[i] = current_target + alpha * delta
                    interp_actions = np.vstack([transition, interp_actions])
        
        # 调试：记录动作序列的统计信息
        if self.verbose:
            # 计算动作的变化量
            action_diff = np.diff(interp_actions, axis=0)
            max_diff = np.abs(action_diff).max(axis=0)
            print(f"[推理] 动作统计:")
            print(f"  原始帧数: {len(actions)}, 最终帧数: {len(interp_actions)}")
            print(f"  首帧: joints={interp_actions[0, :3]}, gripper={interp_actions[0, 6]:.4f}")
            print(f"  末帧: joints={interp_actions[-1, :3]}, gripper={interp_actions[-1, 6]:.4f}")
            print(f"  最大帧间变化: joints={max_diff[:6].max():.6f}, gripper={max_diff[6]:.6f}")
        
        # 写入共享内存
        self.shared_state.write_actions(interp_actions)
        self.shared_state.update_inference_timestamp()  # 更新心跳
        
        elapsed = time.time() - start
        self.inference_count += 1
        
        # 记录推理数据
        if self._log_trajectory and self.trajectory_logger:
            self.trajectory_logger.log_inference(
                raw_actions=actions,
                interp_actions=interp_actions,
                current_position=current_target,
                inference_time=elapsed,
            )
        
        if self.verbose:
            print(f"[推理] #{self.inference_count}, 耗时: {elapsed*1000:.1f}ms")
    
    def _handle_key(self, key: int):
        """处理 OpenCV 按键"""
        if key == ord(' '):
            self._toggle_running()
        elif key == ord('r') or key == ord('R'):
            self._request_reset()
        elif key == ord('q') or key == ord('Q'):
            self._request_stop()
    
    def _check_stdin(self):
        """检查标准输入 (无头模式)"""
        import select
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip().lower()
                if cmd in ['s', 'start', 'pause']:
                    self._toggle_running()
                elif cmd in ['r', 'reset']:
                    self._request_reset()
                elif cmd in ['q', 'quit', 'exit']:
                    self._request_stop()
        except:
            pass
    
    def _toggle_running(self):
        """切换运行/暂停状态"""
        current_state = self.shared_state.get_state()
        
        if current_state == ControlFlags.IDLE or current_state == ControlFlags.PAUSED:
            print("\n[推理] 发送 START 请求...")
            self.shared_state.request_start()
            self._is_inferencing = True
            
            # 等待确认
            for _ in range(100):
                if self.shared_state.has_ack(ControlFlags.ACK_START):
                    self.shared_state.clear_acks()
                    print("[推理] START 已确认")
                    break
                time.sleep(0.01)
        
        elif current_state == ControlFlags.RUNNING:
            print("\n[推理] 发送 PAUSE 请求...")
            self.shared_state.request_pause()
            self._is_inferencing = False
            
            # 等待确认
            for _ in range(100):
                if self.shared_state.has_ack(ControlFlags.ACK_PAUSE):
                    self.shared_state.clear_acks()
                    print("[推理] PAUSE 已确认")
                    break
                time.sleep(0.01)
    
    def _request_reset(self):
        """请求复位"""
        print("\n[推理] 发送 RESET 请求...")
        self._is_inferencing = False
        self.shared_state.request_reset()
        
        # 等待确认
        for _ in range(500):  # 5秒超时
            if self.shared_state.has_ack(ControlFlags.ACK_RESET):
                self.shared_state.clear_acks()
                print("[推理] RESET 已确认")
                
                # 清空观测和滤波器
                self.obs_buffer.clear()
                
                # 如果有初始位姿，重新写入共享内存以便 control_node 再次应用
                if self.initial_pose is not None:
                    print(f"[推理] 重新设置初始位姿:")
                    print(f"  关节: {self.initial_pose[:3]}...")
                    print(f"  夹爪: {self.initial_pose[6]:.4f}")
                    self.shared_state.write_initial_pose(self.initial_pose)
                break
            time.sleep(0.01)
    
    def _request_stop(self):
        """请求停止"""
        print("\n[推理] 发送 STOP 请求...")
        self._is_inferencing = False
        self.shared_state.request_stop()
        
        # 等待确认
        for _ in range(100):
            if self.shared_state.has_ack(ControlFlags.ACK_STOP):
                self.shared_state.clear_acks()
                print("[推理] STOP 已确认")
                break
            time.sleep(0.01)
        
        self._running = False
    
    def _update_preview(self, camera_frames: Dict, control_state: int):
        """更新预览窗口"""
        preview_images = []
        
        for name in ["wrist", "external"]:
            if name in camera_frames and camera_frames[name] is not None:
                frame = camera_frames[name].rgb.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 添加标签
                cv2.putText(frame, name, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                preview_images.append(frame)
            else:
                # 占位图
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"{name}: N/A", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                preview_images.append(placeholder)
        
        if preview_images:
            combined = np.hstack(preview_images)
            
            # 状态指示
            state_names = {
                ControlFlags.IDLE: "IDLE",
                ControlFlags.RUNNING: "RUNNING",
                ControlFlags.PAUSED: "PAUSED",
                ControlFlags.RESETTING: "RESETTING",
                ControlFlags.EMERGENCY_STOP: "E-STOP",
                ControlFlags.DAMPING: "DAMPING",
            }
            state_name = state_names.get(control_state, f"UNKNOWN({control_state})")
            
            color = (0, 255, 0) if control_state == ControlFlags.RUNNING else (0, 165, 255)
            cv2.putText(combined, f"State: {state_name}", (10, combined.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(combined, f"Inference: {self.inference_count}", (10, combined.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("ARX5 Inference", combined)
    
    def _cleanup(self):
        """清理资源"""
        print("\n[推理节点] 清理资源...")
        
        self._running = False
        
        # 保存轨迹数据
        if self._log_trajectory and self.trajectory_logger:
            self.trajectory_logger.stop_logging()
            self.trajectory_logger.save(self._trajectory_output_path)
        
        if self.camera_manager is not None:
            self.camera_manager.stop()
            print("  相机已停止")
        
        if self.shared_state is not None:
            if self.dry_run:
                self.shared_state.close()
            # 非 dry_run 模式下不关闭共享内存，由控制进程管理
        
        cv2.destroyAllWindows()
        
        print(f"  总推理次数: {self.inference_count}")
        print("[推理节点] 退出")


def main():
    parser = argparse.ArgumentParser(
        description="ARX5 推理节点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Checkpoint 文件路径 (.pt)")
    parser.add_argument("--dry-run", action="store_true",
                       help="模拟模式，不连接控制进程")
    parser.add_argument("--no-viz", action="store_true",
                       help="禁用可视化")
    parser.add_argument("--headless", action="store_true",
                       help="无头模式 (无 OpenCV 窗口)")
    parser.add_argument("--act-horizon", type=int, default=8,
                       help="动作 horizon")
    parser.add_argument("--flow-steps", type=int, default=10,
                       help="Flow/diffusion 步数")
    parser.add_argument("--no-ema", action="store_true",
                       help="不使用 EMA 权重")
    parser.add_argument("--device", type=str, default="cuda",
                       help="推理设备")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="减少输出")
    
    # 初始位姿 (从数据集加载)
    parser.add_argument("--init-pose", type=str, default=None,
                       help="初始位姿: 'dataset:<path>' 从数据集加载, 或逗号分隔的关节值")
    parser.add_argument("--init-frame", type=int, default=0,
                       help="从数据集加载时使用的帧索引 (默认: 0)")
    
    # 轨迹记录 (用于对比分析)
    parser.add_argument("--log-trajectory", action="store_true",
                       help="启用轨迹记录 (用于对比分析)")
    parser.add_argument("--trajectory-output", type=str, default="trajectory_multi_process_inference.npz",
                       help="轨迹记录输出文件 (默认: trajectory_multi_process_inference.npz)")
    
    # 过渡帧控制 (用于对比测试)
    parser.add_argument("--enable-transition", action="store_true",
                       help="启用 chunk 边界过渡帧 (默认禁用，与原版一致)")
    parser.add_argument("--transition-frames", type=int, default=10,
                       help="过渡帧数量 (默认: 10)")
    
    args = parser.parse_args()
    
    # 解析初始位姿
    task_initial_pose = None
    if args.init_pose:
        task_initial_pose = parse_init_pose(args.init_pose, args.init_frame)
    
    config = InferenceConfig(
        act_horizon=args.act_horizon,
        num_flow_steps=args.flow_steps,
        enable_transition_frames=args.enable_transition,
        transition_frame_count=args.transition_frames,
    )
    
    node = InferenceNode(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
        use_ema=not args.no_ema,
        verbose=not args.quiet,
        headless=args.headless,
        dry_run=args.dry_run,
        initial_pose=task_initial_pose,
    )
    
    try:
        node.setup()
        
        # 启用轨迹记录
        if args.log_trajectory:
            node.enable_trajectory_logging(args.trajectory_output)
            if node.trajectory_logger:
                node.trajectory_logger.start_logging()
        
        node.run(visualize=not args.no_viz)
    except KeyboardInterrupt:
        print("\n[推理节点] 被用户中断")
    except Exception as e:
        print(f"\n[推理节点] 错误: {e}")
        import traceback
        traceback.print_exc()


def parse_init_pose(init_pose_str: str, frame_idx: int = 0) -> Optional[np.ndarray]:
    """
    解析初始位姿参数
    
    支持格式:
    - "dataset:<path>" : 从 HDF5 数据集加载
    - "0.1,0.2,0.3,0.4,0.5,0.6,0.07" : 逗号分隔的关节值
    """
    if init_pose_str.startswith("dataset:"):
        # 从数据集加载
        dataset_path = os.path.expanduser(init_pose_str[8:])
        print(f"\n从数据集加载初始位姿: {dataset_path}")
        
        try:
            import h5py
            with h5py.File(dataset_path, 'r') as f:
                # 查找轨迹
                traj_keys = [k for k in f.keys() if k.startswith('traj_')]
                if not traj_keys:
                    print(f"  错误: 数据集中没有找到轨迹")
                    return None
                
                traj = f[traj_keys[0]]
                
                # 读取关节位置
                if 'obs' in traj and 'joint_pos' in traj['obs']:
                    joint_pos = np.array(traj['obs']['joint_pos'][frame_idx])
                elif 'joint_pos' in traj:
                    joint_pos = np.array(traj['joint_pos'][frame_idx])
                else:
                    print(f"  错误: 数据集中没有找到 joint_pos")
                    return None
                
                # 读取夹爪位置
                if 'obs' in traj and 'gripper_pos' in traj['obs']:
                    gripper_pos = np.array(traj['obs']['gripper_pos'][frame_idx])
                elif 'gripper_pos' in traj:
                    gripper_pos = np.array(traj['gripper_pos'][frame_idx])
                else:
                    gripper_pos = np.array([0.074])  # 默认张开
                
                # 确保 gripper_pos 是标量
                if isinstance(gripper_pos, np.ndarray):
                    gripper_pos = float(gripper_pos.flatten()[0])
                
                pose = np.concatenate([joint_pos.flatten()[:6], [gripper_pos]])
                print(f"  帧 {frame_idx}: joints={pose[:3]}..., gripper={pose[6]:.4f}")
                return pose
                
        except Exception as e:
            print(f"  加载数据集出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        # 解析逗号分隔的值
        try:
            values = [float(x.strip()) for x in init_pose_str.split(',')]
            if len(values) == 6:
                values.append(0.074)  # 默认夹爪位置
            pose = np.array(values[:7])
            print(f"\n使用指定的初始位姿: joints={pose[:3]}..., gripper={pose[6]:.4f}")
            return pose
        except Exception as e:
            print(f"解析初始位姿出错: {e}")
            return None


if __name__ == "__main__":
    main()
