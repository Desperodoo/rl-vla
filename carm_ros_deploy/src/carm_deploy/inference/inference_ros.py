#!/usr/bin/env python3
"""
CARM 机械臂 ROS 策略推理主程序
基于 carm_real/infer_g3_api.py 重构，将 svar 通信替换为 ROS1 原生通信

支持的算法:
    - consistency_flow: Consistency Flow Matching (推荐)
    - flow_matching: Flow Matching Policy
    - diffusion_policy: DDPM-based Diffusion Policy
    - reflected_flow: Reflected Flow Matching
    - shortcut_flow: Shortcut Flow Matching

使用方法:
    # 正常推理 (30Hz)
    rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt
    
    # 启用干预和采集
    rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt --intervention --record_inference
"""

import argparse
import threading
import time
import json
import signal
import numpy as np
import cv2
import rospy
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from einops import rearrange
from typing import Optional, Dict, List, Any

# 本地模块
import sys
import os

# 添加 carm_deploy 根目录到路径
carm_deploy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, carm_deploy_root)

# 添加 rl-vla 根目录到路径（使 rlft 包可用）
rl_vla_root = os.path.dirname(os.path.dirname(os.path.dirname(carm_deploy_root)))
sys.path.insert(0, rl_vla_root)

# PyTorch
import torch

# rlft 训练框架中的共享模块
from rlft.utils.pose_utils import (
    pose_to_transform_matrix, apply_relative_transform,
    quaternion_slerp, apply_teleop_scale,
)

# 策略加载（从 policy_loader.py 导入）
from inference.policy_loader import PolicyInterface, RealPolicy
from rlft.utils.model_factory import SUPPORTED_ALGORITHMS

# 安全控制和日志
from core.safety_controller import SafetyController
from inference.inference_logger import InferenceLogger
from inference.inference_recorder import InferenceRecorder

from core.env_ros import RealEnvironment
from utils.trajectory_interpolator import VecTF, ActionChunkManager
from utils.timeline_logger import TimelineLogger
from utils.keyboard_intervention import KeyboardInterventionHandler, InterventionApplier


class InferenceNode:
    """
    ROS 推理节点
    
    支持人工干预和数据采集
    """
    
    def __init__(self, config):
        """
        初始化推理节点
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 参数
        self.temporal_factor_k = config.get('temporal_factor_k', 0.05)  # 默认 0.05
        self.desire_inference_freq = config.get('desire_inference_freq', 30)  # 默认 30Hz
        self.pos_lookahead_step = config.get('pos_lookahead_step', 1)
        self.pos_lookahead_duration = config.get('pos_lookahead_duration', 0.015)
        self.check_workspace = True  # 默认开启 workspace 检测
        
        # Teleop 对齐模式参数
        # teleop_scale 固定为 1.0: 训练数据的 action 已包含 scale 效果（GAP-2 修复）
        self.teleop_scale = 1.0
        # inference_speed_scale: 推理时可选调速（独立于 teleop scale 语义）
        self.inference_speed_scale = config.get('inference_speed_scale', 1.0)
        self.control_freq = config.get('control_freq', 50)   # 默认 50Hz 对齐遥操
        
        # Action Chunk 执行模式参数
        # execution_mode: 'temporal_ensemble' (原始) 或 'receding_horizon' (标准 action chunking)
        self.execution_mode = config.get('execution_mode', 'temporal_ensemble')
        self.max_active_chunks = config.get('max_active_chunks', None)  # None = 不限制
        self.crossfade_steps = config.get('crossfade_steps', 0)  # 0 = 无 crossfade
        self.truncate_at_act_horizon = config.get('truncate_at_act_horizon', False)  # 是否截断到 act_horizon
        
        # 初始化环境
        rospy.loginfo("Initializing environment...")
        self.env = RealEnvironment(config)
        
        # 初始化策略
        rospy.loginfo("Initializing policy...")
        self.policy = self._create_policy(config)
        
        # 初始化安全控制器
        self.safety_controller = self._create_safety_controller(config)
        
        # 初始化推理日志记录器
        self.logger = self._create_logger(config)
        self.episode_started = False

        # 时间线日志（用于分析 chunking 时间语义）
        self.timeline_enabled = config.get('timeline_enabled', True)
        self.timeline_control_stride = config.get('timeline_control_stride', 10)
        self.chunk_time_base = config.get('chunk_time_base', 'sys_time')
        self.timeline_logger = None
        
        # 从策略获取 horizon 参数（如果已加载）
        self._act_horizon = getattr(self.policy, 'pred_horizon', 8)  # 默认与 pred_horizon 相同
        self._pred_horizon = getattr(self.policy, 'pred_horizon', 16)
        self._obs_horizon = getattr(self.policy, 'obs_horizon', 2)
        # 允许通过 config 覆盖 act_horizon
        self._act_horizon = config.get('act_horizon', self._act_horizon)
        
        # 从策略获取 action_dim_full（用于后处理）
        # full mode: 15D = joint(6) + gripper(1) + rel_pose(7) + gripper(1)
        # ee_only mode: 8D = rel_pose(7) + gripper(1)
        self._action_dim_full = getattr(self.policy, 'action_dim_full', 15)
        
        if self.timeline_enabled:
            timeline_path = config.get('timeline_log', '')
            if not timeline_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                timeline_path = os.path.join(self.logger.log_dir, f'timeline_{timestamp}.jsonl')
            self.timeline_logger = TimelineLogger(timeline_path, control_log_interval=self.timeline_control_stride * 10)
            self._timeline_path = timeline_path  # 保存供 logger 使用
            self.timeline_logger.log(
                'init',
                desire_inference_freq=self.desire_inference_freq,
                temporal_factor_k=self.temporal_factor_k,
                pos_lookahead_step=self.pos_lookahead_step,
                pos_lookahead_duration=self.pos_lookahead_duration,
                chunk_time_base=self.chunk_time_base,
                act_horizon=self._act_horizon,
                pred_horizon=self._pred_horizon,
                obs_horizon=self._obs_horizon,
                # 执行模式参数
                execution_mode=self.execution_mode,
                max_active_chunks=self.max_active_chunks,
                crossfade_steps=self.crossfade_steps,
                truncate_at_act_horizon=self.truncate_at_act_horizon,
                # 新增：关键控制参数
                teleop_scale=self.teleop_scale,
                control_freq=self.control_freq,
            )
        else:
            self._timeline_path = None
        
        # 设置 logger metadata（包含完整的运行配置）
        self._setup_logger_metadata(config)
        
        # 动作管理器
        self.action_manager = ActionChunkManager(
            temporal_factor_k=self.temporal_factor_k,
            execution_mode=self.execution_mode,
            max_active_chunks=self.max_active_chunks,
            crossfade_steps=self.crossfade_steps,
        )
        self.lock_tfs = threading.Lock()
        
        rospy.loginfo(f"ActionChunkManager: mode={self.execution_mode}, "
                      f"max_active_chunks={self.max_active_chunks}, "
                      f"crossfade_steps={self.crossfade_steps}, "
                      f"truncate_at_act_horizon={self.truncate_at_act_horizon}")
        
        # 控制变量
        self.running = True
        self.latest_obs = None
        self.pos_lookahead_step_start_idx = 0
        self.step_count = 0
        self.last_action = None
        self.control_step_count = 0
        self._last_control_time = None
        self._control_hz_ema = None
        self._last_gripper_value = None
        self._last_gripper_log_time = 0.0
        
        # Episode 状态（用于多 episode 采集）
        # 只有启用 record_inference 时才等待按键开始
        self.record_inference_enabled = config.get('record_inference', False)
        self.waiting_start = self.record_inference_enabled  # 启用采集时等待 R 键开始
        self.episode_paused = self.waiting_start  # 与 waiting_start 一致
        self.pending_save = False  # 等待保存确认
        self.max_steps = config.get('max_steps', 99999)  # 每个 episode 最大步数
        
        # 干预和采集模块
        self.intervention_enabled = config.get('intervention', False)
        self.intervention_handler = None
        self.inference_recorder = None
        
        if self.intervention_enabled or self.record_inference_enabled:
            self._init_intervention_and_recording(config)
        
        # 启动推理线程
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        rospy.loginfo("InferenceNode initialized")
    
    def _init_intervention_and_recording(self, config):
        """初始化干预和数据采集模块"""
        # 干预处理器
        if self.intervention_enabled:
            self.intervention_handler = KeyboardInterventionHandler(
                xyz_scale=config.get('intervention_xyz_scale', 0.005),
                gripper_open=config.get('intervention_gripper_open', 1.0),
                gripper_close=config.get('intervention_gripper_close', 0.0),
                mode=config.get('intervention_mode', 'replace'),
            )
            
            # 设置录制控制回调
            def on_record_action(action):
                self._handle_record_action(action)
            
            def on_quit():
                rospy.loginfo("Quit requested via keyboard")
                self.running = False
            
            self.intervention_handler.set_record_callback(on_record_action)
            self.intervention_handler.set_quit_callback(on_quit)
            self.intervention_handler.start()
            rospy.loginfo("Keyboard intervention enabled")
        
        # 数据采集记录器
        if self.record_inference_enabled:
            record_dir = config.get('record_dir', '')
            if not record_dir:
                record_dir = config.get('log_dir', '')
            if not record_dir:
                from utils.paths import get_inference_logs_dir
                record_dir = get_inference_logs_dir()
            
            # 获取 action_dim
            action_dim = getattr(self.policy, 'action_dim_full', 15)
            
            self.inference_recorder = InferenceRecorder(
                output_dir=record_dir,
                pred_horizon=self._pred_horizon,
                action_dim=action_dim,
            )
            rospy.loginfo(f"Inference recording enabled, output_dir: {record_dir}")
            
            # 启动时等待按 R 开始第一个 episode
            rospy.loginfo("=" * 60)
            rospy.loginfo("Multi-episode recording mode enabled")
            rospy.loginfo("Press 'R' to start recording an episode")
            rospy.loginfo("Press 'R' again to stop and choose to save (Y) or discard (N)")
            rospy.loginfo("After save/discard, arm will return to init position")
            rospy.loginfo("Press 'R' to start next episode")
            rospy.loginfo("Press Ctrl+C to quit")
            rospy.loginfo("=" * 60)
    
    def _handle_record_action(self, action: str):
        """
        处理录制相关的键盘动作
        
        状态机:
        1. waiting_start=True, pending_save=False: 等待按 R 开始
        2. waiting_start=False, pending_save=False: 正在录制，按 R 停止
        3. waiting_start=False, pending_save=True: 等待 Y/N 确认保存
        """
        if action == 'toggle':  # R 键
            if self.pending_save:
                # 正在等待保存确认，忽略 R 键
                rospy.logwarn("Please confirm save first (Y/N)")
                return
            
            if self.waiting_start:
                # 开始新 episode
                self._start_new_episode()
            else:
                # 停止当前 episode，等待确认
                self._stop_current_episode()
                
        elif action == 'confirm':  # Y 键
            if self.pending_save:
                self._confirm_save_episode(save=True)
            else:
                rospy.logwarn("No episode waiting for save")
                
        elif action == 'discard':  # N 键
            if self.pending_save:
                self._confirm_save_episode(save=False)
            else:
                rospy.logwarn("No episode to discard")
    
    def _start_new_episode(self):
        """开始新的 episode"""
        self.pending_save = False
        
        # 清空 action chunk 管理器（推理线程仍被 episode_paused 阻塞）
        with self.lock_tfs:
            self.action_manager.clear()
        
        # 重置策略状态（观测历史、gripper 历史等）
        # 必须在 episode_paused = False 之前完成，否则推理线程可能访问空的 obs_history
        if hasattr(self, 'policy') and self.policy is not None:
            self.policy.reset()
            rospy.loginfo("Policy state reset")
        
        # 最后才解除暂停，让推理线程开始工作
        self.waiting_start = False
        self.episode_paused = False
        
        # 开始录制
        if self.inference_recorder:
            self.inference_recorder.start_recording()
        
        self.step_count = 0
        rospy.loginfo("=" * 60)
        rospy.loginfo("Episode started! Robot is now under policy control.")
        rospy.loginfo("Press 'R' to stop recording")
        rospy.loginfo("=" * 60)
    
    def _stop_current_episode(self):
        """停止当前 episode，等待保存确认"""
        self.episode_paused = True
        self.pending_save = True
        
        # 停止录制
        if self.inference_recorder:
            self.inference_recorder.stop_recording()
        
        step_count = self.step_count
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Episode stopped - {step_count} steps recorded")
        rospy.loginfo("Save this episode? Press 'Y' to save, 'N' to discard")
        rospy.loginfo("=" * 60)
    
    def _confirm_save_episode(self, save: bool):
        """确认保存或丢弃 episode，然后初始化机械臂等待下一个 episode"""
        if save:
            # 保存
            if self.inference_recorder:
                filepath = self.inference_recorder.confirm_save()
                if filepath:
                    rospy.loginfo(f"Episode saved to: {filepath}")
        else:
            # 丢弃
            if self.inference_recorder:
                self.inference_recorder.discard()
            rospy.loginfo("Episode discarded")
        
        self.pending_save = False
        
        # 初始化机械臂回到初始位置
        rospy.loginfo("Returning to initial position...")
        self._reinitialize_arm()
        
        # 进入等待开始状态
        self.waiting_start = True
        self.episode_paused = True
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("Ready for next episode. Press 'R' to start recording")
        rospy.loginfo("=" * 60)
    
    def _reinitialize_arm(self):
        """重新初始化机械臂到初始位置"""
        try:
            # 使用 env 的 init_status 方法
            self.env.init_status()
            rospy.loginfo("Arm returned to initial position")
        except Exception as e:
            rospy.logerr(f"Failed to reinitialize arm: {e}")
    
    def _create_policy(self, config):
        """
        创建策略实例
        """
        pretrain_path = config.get('pretrain', '')
        
        if not pretrain_path:
            rospy.logerr("No pretrain model specified! Use --pretrain to specify model path.")
            raise SystemExit(1)
        
        if not os.path.exists(pretrain_path):
            rospy.logerr(f"Pretrain model not found: {pretrain_path}")
            raise SystemExit(1)
        
        rospy.loginfo(f"Loading policy from: {pretrain_path}")
        policy = RealPolicy(config)
        policy.load_model(pretrain_path)
        return policy
    
    def _create_safety_controller(self, config):
        """
        创建安全控制器
        
        优先从 dataset_info.json 加载配置
        """
        safety_config_path = config.get('safety_config', '')
        data_dir = config.get('data_dir', '')
        
        if safety_config_path and os.path.exists(safety_config_path):
            rospy.loginfo(f"Loading safety config from: {safety_config_path}")
            return SafetyController.from_config(safety_config_path)
        else:
            # 使用默认参数
            rospy.logwarn("No safety config or dataset stats found, using default safety limits")
            return SafetyController()
    
    def _create_logger(self, config):
        """
        创建推理日志记录器
        """
        log_dir = config.get('log_dir', '')
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            return InferenceLogger(log_dir=log_dir)
        else:
            # 使用路径模块获取默认日志目录
            from utils.paths import get_inference_logs_dir, ensure_dir
            default_log_dir = ensure_dir(get_inference_logs_dir())
            return InferenceLogger(log_dir=default_log_dir)
    
    def _setup_logger_metadata(self, config):
        """
        设置 logger 的完整 metadata（用于生成 run_info.json）
        """
        pretrain_path = config.get('pretrain', '')
        
        # 模型配置
        model_config = {
            'path': pretrain_path,
            'algorithm': getattr(self.policy, 'algorithm', 'unknown'),
            'action_mode': 'full' if getattr(self.policy, 'action_dim_full', 15) == 15 else 'ee_only',
            'state_mode': getattr(self.policy, 'state_mode', 'joint_only'),
            'obs_horizon': getattr(self.policy, 'obs_horizon', 2),
            'pred_horizon': getattr(self.policy, 'pred_horizon', 16),
            'action_dim': getattr(self.policy, 'action_dim', 13),
            'action_dim_full': getattr(self.policy, 'action_dim_full', 15),
            'visual_encoder_type': config.get('visual_encoder_type', 'unknown'),
            'use_ema': getattr(self.policy, 'use_ema', False),
            'num_inference_steps': getattr(self.policy, 'num_inference_steps', 10),
        }
        
        # Normalizer 配置
        normalizer_config = {
            'enabled': getattr(self.policy, 'normalize_actions', False),
            'mode': getattr(self.policy, 'action_norm_mode', 'standard'),
        }
        if hasattr(self.policy, 'action_normalizer') and self.policy.action_normalizer is not None:
            normalizer = self.policy.action_normalizer
            if hasattr(normalizer, 'stats') and normalizer.stats:
                normalizer_config['action_stats'] = {
                    'mean': normalizer.stats.get('mean', []),
                    'std': normalizer.stats.get('std', []),
                }
        
        # 控制配置
        control_config = {
            'control_freq': self.control_freq,
            'teleop_scale': self.teleop_scale,
            'gripper_hysteresis_window': getattr(self.policy, 'gripper_hysteresis_window', 1),
        }
        
        # 执行配置
        execution_config = {
            'mode': self.execution_mode,
            'act_horizon': self._act_horizon,
            'max_active_chunks': self.max_active_chunks,
            'crossfade_steps': self.crossfade_steps,
            'truncate_at_act_horizon': self.truncate_at_act_horizon,
            'temporal_factor_k': self.temporal_factor_k,
            'pos_lookahead_step': self.pos_lookahead_step,
            'chunk_time_base': self.chunk_time_base,
            'desire_inference_freq': self.desire_inference_freq,
        }
        
        # 安全配置
        safety_config = {
            'config_path': config.get('safety_config', ''),
            'check_workspace': self.check_workspace,
            'max_relative_translation': 0.1,  # 硬编码在代码中的值
        }
        
        # 调用 logger 的 set_metadata
        self.logger.set_metadata(
            model_path=pretrain_path,
            model_config=model_config,
            normalizer_config=normalizer_config,
            control_config=control_config,
            execution_config=execution_config,
            safety_config=safety_config,
        )
        
        rospy.loginfo("Logger metadata configured for run_info.json")
    
    def _preprocess_image(self, image: np.ndarray, target_size=(128, 128)) -> np.ndarray:
        """
        预处理单张图像: resize
        
        Args:
            image: RGB 图像 [H, W, C]
            target_size: 目标尺寸 (H, W)
            
        Returns:
            处理后的图像 [H, W, C]
        """
        h, w = target_size
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def _normalize_images(self, obs, target_size=(128, 128)):
        """
        归一化图像（对齐训练代码）
        
        Args:
            obs: 观测字典
            target_size: 目标图像尺寸 (H, W)
            
        Returns:
            torch.Tensor: 预处理后的图像 [C, H, W] (未归一化，RealPolicy 内部会归一化)
        """
        # 只使用第一个相机
        image = obs["images"][0]  # [H, W, C] RGB 格式
        
        # Resize 到目标尺寸
        image = self._preprocess_image(image, target_size)
        
        # HWC -> CHW
        image = rearrange(image, 'h w c -> c h w')
        
        return image
    
    def _inference_loop(self):
        """推理线程主循环"""
        rospy.loginfo("Inference thread started")
        
        # 初始化时 reset 策略状态
        if hasattr(self, 'policy') and self.policy is not None:
            self.policy.reset()
            rospy.loginfo("Policy state initialized (reset)")
        
        desire_period = 1.0 / self.desire_inference_freq
        
        with torch.inference_mode():
            while self.running and not rospy.is_shutdown():
                # 如果 episode 暂停（等待保存确认或等待开始），不执行推理
                if self.episode_paused or self.waiting_start:
                    time.sleep(0.1)
                    continue
                
                # 获取观测
                self.latest_obs = self.env.get_observation()
                if self.latest_obs is None:
                    time.sleep(0.5)
                    rospy.loginfo_throttle(5.0, "Waiting for observation...")
                    continue

                t_obs_ready_sys = time.time()
                obs_stamp_ros = self.latest_obs.get('stamp', None)
                if self.timeline_logger is not None:
                    delta_obs = None
                    if obs_stamp_ros is not None:
                        delta_obs = t_obs_ready_sys - obs_stamp_ros
                    self.timeline_logger.log(
                        'obs',
                        obs_stamp_ros=obs_stamp_ros,
                        t_obs_ready_sys=t_obs_ready_sys,
                        delta_obs=delta_obs,
                    )
                
                # 启动 episode（如果尚未启动）
                if not self.episode_started:
                    self.logger.start_episode(timeline_path=self._timeline_path)
                    self.episode_started = True
                    rospy.loginfo("Episode started, logging enabled")
                
                last_start = time.time()
                
                try:
                    # 准备输入
                    qpos_joint = np.array(self.latest_obs['qpos_joint'])  # [7]
                    qpos_end = np.array(self.latest_obs['qpos_end'])  # [8]
                    
                    # 构建 state 向量（基于 state_mode 配置）
                    # joint_only: qpos_joint [7]
                    # ee_only: qpos_end [8]
                    # both: concat [14]
                    if hasattr(self.policy, 'build_state_from_obs'):
                        state = self.policy.build_state_from_obs(qpos_joint, qpos_end)
                    else:
                        # Fallback for old policies without state_mode support
                        state = qpos_joint.astype(np.float32)
                    qpos = torch.from_numpy(state).float().cuda().unsqueeze(0)  # [1, state_dim]
                    
                    # 保存 qpos_end 用于后续安全检查（转回 list 以兼容后续代码）
                    qpos_end = qpos_end.tolist()
                    
                    # 图像预处理: resize + HWC->CHW
                    # 获取目标尺寸（从 policy 获取，如果是 RealPolicy）
                    target_size = (128, 128)
                    if hasattr(self.policy, 'target_image_size') and self.policy.target_image_size:
                        target_size = self.policy.target_image_size
                    
                    curr_image = self._normalize_images(self.latest_obs, target_size)  # [C, H, W]
                    
                    # 转换为 torch tensor
                    curr_image = torch.from_numpy(curr_image).float().cuda()  # [C, H, W]
                    
                    # 推理计时
                    inference_start = time.time()
                    ret = self.policy({"qpos": qpos, "image": curr_image})
                    inference_time = time.time() - inference_start
                    inference_end = inference_start + inference_time

                    if self.timeline_logger is not None:
                        self.timeline_logger.log(
                            'inference',
                            t_infer_start=inference_start,
                            t_infer_end=inference_end,
                            inference_time=inference_time,
                        )
                    
                    all_actions = ret["a_hat"].squeeze(0).cpu().numpy()  # [pred_horizon, action_dim]
                    
                    # ============================================================
                    # 对齐顺序: 1) inverse-normalize (模型内部已做)
                    #          2) teleop_scale (模拟遥操调速)
                    #          3) safety clip (安全层)
                    # ============================================================
                    
                    # 应用 inference_speed_scale 缩放 (推理时可选调速)
                    if self.inference_speed_scale != 1.0:
                        # 只对 ee_delta_pose 模式应用 scale
                        is_full_mode = (self._action_dim_full == 15)
                        rel_pose_start = 7 if is_full_mode else 0
                        rel_pose_end = 14 if is_full_mode else 7

                        for i in range(len(all_actions)):
                            # 提取 rel_pose [7]，应用缩放，写回
                            rel_pose = all_actions[i, rel_pose_start:rel_pose_end].copy()
                            scaled_rel_pose = apply_teleop_scale(rel_pose, self.inference_speed_scale)
                            all_actions[i, rel_pose_start:rel_pose_end] = scaled_rel_pose
                    
                    # 安全检查和裁剪
                    safety_events = []
                    safety_clipped = False
                    
                    # 末端位姿模式：
                    # 1. 检查相对位移是否过大
                    # 2. 计算绝对位姿并检查工作空间边界
                    # 
                    # 根据 action_dim_full 确定索引：
                    # - full mode (15D): [joint(6), gripper(1), rel_pose(7), gripper(1)]
                    #   rel_pose at [7:14], gripper at [14]
                    # - ee_only mode (8D): [rel_pose(7), gripper(1)]
                    #   rel_pose at [0:7], gripper at [7]
                    is_full_mode = (self._action_dim_full == 15)
                    rel_pose_start = 7 if is_full_mode else 0
                    rel_pose_end = 14 if is_full_mode else 7
                    gripper_idx = 14 if is_full_mode else 7
                    
                    for i in range(len(all_actions)):
                        relative_pose = all_actions[i, rel_pose_start:rel_pose_end]  # [7] 相对位姿
                        grip = all_actions[i, gripper_idx]  # 夹爪
                        
                        # 检查相对位移是否过大
                        max_trans = 0.1  # 10cm
                        trans_norm = np.linalg.norm(relative_pose[:3])
                        if trans_norm > max_trans:
                            # 缩放位移到安全范围
                            scale = max_trans / trans_norm
                            all_actions[i, rel_pose_start:rel_pose_start+3] *= scale
                            relative_pose = all_actions[i, rel_pose_start:rel_pose_end]  # 更新
                            if i == 0:
                                safety_events.append(f"Translation scaled: {trans_norm:.3f}m -> {max_trans}m")
                                rospy.logwarn(f"Safety: Translation scaled from {trans_norm:.3f}m to {max_trans}m")
                            safety_clipped = True
                        
                        # 计算目标绝对位姿
                        target_pose = apply_relative_transform(relative_pose, qpos_end[:7], grip)
                        target_pose_np = np.array(target_pose[:7])  # [x,y,z,qx,qy,qz,qw]
                        
                        # 检查工作空间边界 (如果启用)
                        if self.check_workspace:
                            clipped_pose, ws_warnings = self.safety_controller.check_workspace(target_pose_np)
                            if ws_warnings:
                                safety_clipped = True
                                if i == 0:
                                    safety_events.extend(ws_warnings)
                                    for w in ws_warnings:
                                        rospy.logwarn(f"Workspace clip: {w}")
                                
                                # 重新计算相对位姿：clipped_target = current @ new_relative
                                # => new_relative = current^-1 @ clipped_target
                                T_current = pose_to_transform_matrix(qpos_end[:3], qpos_end[3:7])
                                T_clipped = pose_to_transform_matrix(clipped_pose[:3], clipped_pose[3:7])
                                T_relative_new = np.linalg.inv(T_current) @ T_clipped
                                new_relative_pos = T_relative_new[:3, 3]
                                new_relative_quat = R.from_matrix(T_relative_new[:3, :3]).as_quat()
                                all_actions[i, rel_pose_start:rel_pose_start+3] = new_relative_pos
                                all_actions[i, rel_pose_start+3:rel_pose_end] = new_relative_quat
                        
                        # 检查并裁剪夹爪限位
                        gripper_action = np.array([0, 0, 0, 0, 0, 0, grip])  # dummy joints + gripper
                        clipped_gripper, grip_warnings = self.safety_controller.check_joint_limits(gripper_action)
                        if grip_warnings:
                            all_actions[i, gripper_idx] = clipped_gripper[6]
                            if is_full_mode:
                                all_actions[i, 6] = clipped_gripper[6]  # 第一个 gripper (full mode only)
                            if i == 0:
                                safety_events.extend(grip_warnings)
                                safety_clipped = True
                    
                    # 保存模型原始输出（安全检查后，干预前）
                    action_model = all_actions.copy()
                    
                    # 应用键盘干预（如果启用）
                    intervention_mask = None
                    if self.intervention_enabled and self.intervention_handler is not None:
                        intervention = self.intervention_handler.get_intervention()
                        if intervention is not None:
                            # 确定 action 格式
                            action_format = 'ee_delta'
                            all_actions, intervention_mask = InterventionApplier.apply_to_action_chunk(
                                all_actions, intervention, action_format=action_format
                            )
                            rospy.loginfo_throttle(2.0, f"Intervention applied: mask={intervention_mask[0].sum()} dims")
                    
                    # 保存干预后的 action
                    action_intervened = all_actions.copy()
                    
                    # 记录数据（如果启用采集）
                    if self.record_inference_enabled and self.inference_recorder is not None:
                        if self.inference_recorder.is_recording:
                            self.inference_recorder.record_step(
                                obs=self.latest_obs,
                                action_model=action_model,
                                action_intervened=action_intervened,
                                intervention_mask=intervention_mask,
                                timestamp=time.time(),
                            )
                    
                    # 记录第一个动作用于下一次参考
                    self.last_action = all_actions[0].copy()
                    
                    # 保存 raw_action（模型输出，相对位姿）用于后续日志记录
                    raw_action_for_log = all_actions[0].copy()
                    
                    # 转换动作空间
                    # full mode (15D): [joint(6), gripper(1), relative_end_pose(7), gripper(1)]
                    # ee_only mode (8D): [relative_end_pose(7), gripper(1)]
                    # 根据 action_dim_full 确定索引
                    is_full_mode = (self._action_dim_full == 15)
                    rel_pose_start = 7 if is_full_mode else 0
                    rel_pose_end = 14 if is_full_mode else 7
                    gripper_idx = 14 if is_full_mode else 7
                    
                    all_endactions = []
                    for i in range(all_actions.shape[0]):
                        relative_pose = all_actions[i][rel_pose_start:rel_pose_end]  # [7] 相对位姿
                        grip = all_actions[i][gripper_idx]  # gripper
                        # 将相对位姿变换应用到当前位姿，得到目标绝对位姿
                        target_pose = apply_relative_transform(relative_pose, qpos_end[:7], grip)
                        all_endactions.append(target_pose)
                    all_actions = np.array(all_endactions)
                    
                    # 创建轨迹并添加到管理器
                    obs_stamp_ros = self.latest_obs.get("stamp", None)
                    if self.chunk_time_base == 'obs_stamp' and obs_stamp_ros is not None:
                        chunk_base_time = obs_stamp_ros
                    else:
                        chunk_base_time = time.time()
                    tf = VecTF({})
                    
                    # 动作执行间隔: 使用 control_freq (默认 50Hz 对齐 teleop)
                    action_interval = 1.0 / self.control_freq
                    
                    # 根据 truncate_at_act_horizon 决定添加多少步动作到 chunk
                    # 标准 action chunking: 只添加前 act_horizon 步
                    # 旧行为: 添加所有 pred_horizon 步
                    if self.truncate_at_act_horizon:
                        num_actions_to_add = min(self._act_horizon, len(all_actions))
                    else:
                        num_actions_to_add = len(all_actions)
                    
                    self.pos_lookahead_step_start_idx += 1
                    chunk_targets = []
                    for i in range(num_actions_to_add):
                        if self.pos_lookahead_step == 1:
                            target_time = chunk_base_time + i * action_interval
                            tf.append(target_time, all_actions[i].tolist())
                        else:
                            if self.pos_lookahead_step_start_idx % self.pos_lookahead_step == 0:
                                target_time = chunk_base_time + i * action_interval
                                tf.append(target_time, all_actions[i].tolist())
                            else:
                                target_time = chunk_base_time + i * self.pos_lookahead_duration
                                tf.append(target_time, all_actions[i].tolist())

                        chunk_targets.append(target_time)

                    if self.timeline_logger is not None:
                        delta_chunk_obs = None
                        if obs_stamp_ros is not None:
                            delta_chunk_obs = chunk_base_time - obs_stamp_ros
                        # chunk_id 在下方 add_trajectory 后获取
                        
                    with self.lock_tfs:
                        chunk_id = self.action_manager.add_trajectory(tf)
                    
                    if self.timeline_logger is not None:
                        self.timeline_logger.log(
                            'chunk',
                            chunk_id=chunk_id,
                            chunk_base_time=chunk_base_time,
                            obs_stamp_ros=obs_stamp_ros,
                            t_obs_ready_sys=t_obs_ready_sys,
                            action_interval=action_interval,
                            pred_horizon=len(all_actions),
                            act_horizon=self._act_horizon,
                            num_actions_added=num_actions_to_add,  # 实际添加的动作数
                            truncated=self.truncate_at_act_horizon,
                            delta_chunk_obs=delta_chunk_obs,
                            chunk_targets=chunk_targets,
                        )
                    
                    # 记录步骤日志（在动作转换后，包含 raw_action 和 executed_action）
                    self.logger.log_step(
                        timestamp=time.time(),
                        obs=self.latest_obs,  # 包含 images, qpos_joint, qpos_end
                        raw_action=raw_action_for_log,  # 模型输出的相对位姿
                        executed_action=all_actions[0],  # 转换后的绝对位姿/关节角度
                        inference_time=inference_time,
                        safety_clipped=safety_clipped,
                        safety_warnings=safety_events if safety_events else None,
                    )
                    
                    self.step_count += 1
                    rospy.loginfo_throttle(5.0, 
                        f"Step {self.step_count}, Inference: {inference_time:.4f}s, "
                        f"Actions: {all_actions.shape}")
                    
                    # 检查是否达到最大步数
                    if self.step_count >= self.max_steps:
                        rospy.logwarn(f"Reached max_steps ({self.max_steps}), auto-stopping episode...")
                        self._stop_current_episode()
                    
                except Exception as e:
                    import traceback
                    rospy.logerr(f"Error in inference: {e}")
                    rospy.logerr(traceback.format_exc())
                
                # 等待下一个周期
                wait_tm = desire_period - (time.time() - last_start)
                if wait_tm > 0:
                    time.sleep(wait_tm)
    
    def control_loop(self):
        """控制主循环"""
        rospy.loginfo("Control loop started")
        
        # 控制频率: 默认 200Hz，teleop 模式 50Hz
        control_period = 1.0 / self.control_freq
        rospy.loginfo(f"Control frequency: {self.control_freq}Hz (period={control_period:.4f}s)")
        
        while self.running and not rospy.is_shutdown():
            # 如果 episode 暂停，不执行控制
            if self.episode_paused or self.waiting_start:
                time.sleep(0.05)
                continue
            
            # 获取融合后的动作
            tm = time.time()
            meta = None
            with self.lock_tfs:
                if self.timeline_logger is not None:
                    action, meta = self.action_manager.get_fused_action_with_meta(tm)
                else:
                    action = self.action_manager.get_fused_action(tm)
            
            if action is None:
                time.sleep(0.02)
                continue

            # 估计控制频率 (EMA)
            if self._last_control_time is not None:
                dt = tm - self._last_control_time
                if dt > 0:
                    inst_hz = 1.0 / dt
                    if self._control_hz_ema is None:
                        self._control_hz_ema = inst_hz
                    else:
                        self._control_hz_ema = 0.2 * inst_hz + 0.8 * self._control_hz_ema
            self._last_control_time = tm

            # 打印夹爪下发值与频率（节流）
            grip_val = None
            if len(action) > 0:
                grip_val = float(action[-1])

            now = time.time()
            if grip_val is not None and (now - self._last_gripper_log_time) >= 5.0:
                delta = None if self._last_gripper_value is None else (grip_val - self._last_gripper_value)
                hz_str = f"{self._control_hz_ema:.1f}Hz" if self._control_hz_ema is not None else "n/a"
                rospy.loginfo(
                    f"Gripper cmd: {grip_val:.4f}, delta: {delta if delta is not None else 'n/a'}, control_hz: {hz_str}"
                )
                self._last_gripper_value = grip_val
                self._last_gripper_log_time = now
            
            # 执行控制 (末端位姿模式)
            rospy.logdebug("End pose control")
            self.env.end_control_nostep(action)

            if self.timeline_logger is not None and (self.control_step_count % self.timeline_control_stride == 0):
                self.timeline_logger.log(
                    'control',
                    query_time=tm,
                    t_send_sys=time.time(),
                    candidate_timestamps=meta.get('candidate_timestamps', []) if meta else [],
                    weights=meta.get('weights', []) if meta else [],
                    num_candidates=meta.get('num_candidates', 0) if meta else 0,
                    used_chunk_ids=meta.get('used_chunk_ids', []) if meta else [],
                )
            self.control_step_count += 1
            
            time.sleep(control_period)
    
    def shutdown(self):
        """关闭节点"""
        # 防止重复调用
        if hasattr(self, '_shutdown_called') and self._shutdown_called:
            return
        self._shutdown_called = True
        
        rospy.loginfo("Shutting down InferenceNode...")
        self.running = False
        
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        
        # 停止键盘干预
        if self.intervention_handler is not None:
            self.intervention_handler.stop()
        
        # 处理未保存的录制数据
        if self.inference_recorder is not None:
            if self.inference_recorder.is_recording:
                self.inference_recorder.stop_recording()
            if self.inference_recorder.is_pending_save:
                rospy.logwarn("Discarding unsaved recording data on shutdown")
                self.inference_recorder.discard()
        
        # 结束并保存日志
        if self.episode_started:
            log_path = self.logger.end_episode()
            if log_path:
                rospy.loginfo(f"Inference log saved to: {log_path}")

        if self.timeline_logger is not None:
            self.timeline_logger.close()
        
        self.env.shutdown()
        rospy.loginfo("InferenceNode shutdown complete")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARM Robot Policy Inference (ROS)')
    
    # 机械臂参数
    parser.add_argument('--robot_ip', type=str, default='10.42.0.101',
                        help='Robot IP address')
    parser.add_argument('--robot_mode', type=int, default=4,
                        help='Control mode (0=IDLE, 1=POSITION, 2=MIT, 3=DRAG, 4=PF)')
    parser.add_argument('--robot_tau', type=float, default=10,
                        help='Gripper torque')
    
    # 初始位置 (从实际机械臂读取 2026-01-13)
    parser.add_argument('--arm_init_pose', type=float, nargs=7,
                        default=[0.2475, 0.0014, 0.3251, 0.9996, -0.0034, 0.0255, -0.0074],
                        help='Initial end effector pose [x,y,z,qx,qy,qz,qw]')
    parser.add_argument('--arm_init_gripper', type=float, default=0.078,
                        help='Initial gripper position')
    
    # 相机参数
    parser.add_argument('--camera_topics', type=str,
                        default='/camera/color/image_raw',
                        help='Camera topic(s), comma separated')
    parser.add_argument('--sync_slop', type=float, default=0.02,
                        help='Image sync tolerance in seconds')
    
    # 时间线与 chunking 诊断
    parser.add_argument('--timeline_enabled', action='store_true',
                        help='Enable timeline logging (default: enabled)')
    parser.add_argument('--timeline_disabled', action='store_true',
                        help='Disable timeline logging')
    parser.add_argument('--timeline_log', type=str, default='',
                        help='Timeline log path (JSONL). Empty uses log_dir')
    parser.add_argument('--timeline_control_stride', type=int, default=10,
                        help='Log every N control steps (control loop)')
    parser.add_argument('--chunk_time_base', type=str, default='sys_time',
                        choices=['sys_time', 'obs_stamp'],
                        help='Chunk base time: sys_time (recommended) or obs_stamp')
    
    # 策略参数
    parser.add_argument('--pretrain', type=str, default='',
                        help='Path to pretrained model checkpoint (e.g., runs/exp/checkpoints/latest.pt)')
    parser.add_argument('--algorithm', type=str, default='consistency_flow',
                        choices=SUPPORTED_ALGORITHMS,
                        help='Algorithm type (auto-detected from args.json if available)')
    parser.add_argument('--desire_inference_freq', type=float, default=30,
                        help='Desired inference frequency')
    parser.add_argument('--temporal_factor_k', type=float, default=0.05,
                        help='Temporal factor for action fusion')
    parser.add_argument('--num_inference_steps', type=int, default=10,
                        help='Number of flow/diffusion steps for inference (default: 10, more steps = better quality but slower)')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA model for inference (recommended only for 1-step inference, otherwise Non-EMA is better)')
    
    # Action Chunk 执行模式参数
    parser.add_argument('--execution_mode', type=str, default='receding_horizon',
                        choices=['temporal_ensemble', 'receding_horizon'],
                        help='Action chunk execution mode: '
                             'temporal_ensemble (original, multi-chunk time-weighted fusion) or '
                             'receding_horizon (standard action chunking, only use latest chunk)')
    parser.add_argument('--max_active_chunks', type=int, default=None,
                        help='Max active chunks in manager (default: None for temporal_ensemble, 2 for receding_horizon)')
    parser.add_argument('--crossfade_steps', type=int, default=0,
                        help='Number of steps for crossfade smoothing when switching chunks (receding_horizon mode only)')
    parser.add_argument('--truncate_at_act_horizon', action='store_true', default=True,
                        help='Truncate action chunk at act_horizon (standard action chunking behavior)')
    parser.add_argument('--act_horizon', type=int, default=8,
                        help='Action horizon for chunk truncation (default: same as pred_horizon)')
    
    # 控制参数
    parser.add_argument('--pos_lookahead_step', type=int, default=1,
                        help='Position lookahead step')
    parser.add_argument('--pos_lookahead_duration', type=float, default=0.015,
                        help='Position lookahead duration')
    parser.add_argument('--joint_cmd_mode', action='store_true',
                        help='[DEPRECATED] Joint command mode is no longer supported. Will raise error if used.')
    
    # Teleop 对齐模式参数
    parser.add_argument('--teleop_scale', type=float, default=1.0,
                        help='[DEPRECATED] Fixed to 1.0. Use --inference_speed_scale for runtime speed control.')
    parser.add_argument('--inference_speed_scale', type=float, default=1.0,
                        help='Runtime speed scaling for predicted actions (default: 1.0 = no scaling)')
    parser.add_argument('--control_freq', type=int, default=50,
                        help='Control loop frequency in Hz (default: 50, aligned with joystick)')
    parser.add_argument('--gripper_hysteresis_window', type=int, default=1,
                        help='Gripper hysteresis window size for voting (default: 1 = no hysteresis)')
    
    # 安全控制参数
    parser.add_argument('--safety_config', type=str, default='',
                        help='Path to safety config JSON file (required)')
    parser.add_argument('--init_speed', type=float, default=2.0,
                        help='Speed level for initialization movement (0-10, default: 2.0 = slow)')
    
    # 日志参数
    parser.add_argument('--log_dir', type=str, default='',
                        help='Directory to save inference logs (default: ~/rl-vla/inference_logs)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save images in inference log (increases file size)')
    
    # 可视化
    parser.add_argument('--vis', action='store_true', default=True,
                        help='Visualize images in OpenCV window')
    
    # 干预和数据采集参数
    parser.add_argument('--record_inference', action='store_true',
                        help='Enable inference data recording')
    parser.add_argument('--intervention', action='store_true',
                        help='Enable keyboard intervention during inference')
    parser.add_argument('--intervention_mode', type=str, default='replace',
                        choices=['replace', 'additive'],
                        help='Intervention mode: replace (override) or additive (delta)')
    parser.add_argument('--intervention_xyz_scale', type=float, default=0.01,
                        help='XYZ movement scale per keypress in meters (default: 10mm)')
    parser.add_argument('--intervention_gripper_open', type=float, default=1.0,
                        help='Gripper open value for intervention')
    parser.add_argument('--intervention_gripper_close', type=float, default=0.0,
                        help='Gripper close value for intervention')
    parser.add_argument('--record_dir', type=str, default='',
                        help='Directory to save recorded inference data (default: log_dir)')
    parser.add_argument('--max_steps', type=int, default=99999,
                        help='Maximum steps per episode (default: 99999). Episode auto-stops when exceeded.')
    
    # 兼容 roslaunch remap 参数
    return parser.parse_args(args=rospy.myargv()[1:])


def main():
    """主函数"""
    # 初始化 ROS 节点
    rospy.init_node('carm_inference', anonymous=True)
    
    # 解析参数
    args = parse_args()
    
    # 转换为配置字典
    config = vars(args)

    # 从 ROS 参数覆盖（支持 roslaunch <param> 方式）
    for key in [
        'robot_ip', 'robot_mode', 'robot_tau', 'arm_init_pose', 'arm_init_gripper',
        'camera_topics', 'sync_slop', 'timeline_log', 'timeline_enabled',
        'timeline_disabled', 'timeline_control_stride', 'chunk_time_base',
        'pretrain', 'algorithm', 'desire_inference_freq', 'temporal_factor_k',
        'num_inference_steps', 'use_ema', 'pos_lookahead_step', 'pos_lookahead_duration',
        'safety_config', 'data_dir',
        'log_dir', 'save_images', 'vis',
        # Action chunk 执行模式参数
        'execution_mode', 'max_active_chunks', 'crossfade_steps', 
        'truncate_at_act_horizon', 'act_horizon',
        # Teleop 对齐模式参数
        'teleop_scale', 'inference_speed_scale', 'control_freq', 'gripper_hysteresis_window',
        # 干预和采集参数
        'record_inference', 'intervention', 'intervention_mode',
        'intervention_xyz_scale', 'intervention_gripper_open', 'intervention_gripper_close',
        'record_dir', 'max_steps',
    ]:
        if rospy.has_param(f'~{key}'):
            config[key] = rospy.get_param(f'~{key}')

    # 时间线日志开关：默认开启，除非显式禁用
    if config.get('timeline_disabled', False):
        config['timeline_enabled'] = False
    else:
        config['timeline_enabled'] = True
    
    # 处理相机话题
    if isinstance(config['camera_topics'], str):
        config['camera_topics'] = config['camera_topics'].split(',')

    # 规范化 arm_init_pose / arm_init_gripper（roslaunch 传入可能是字符串）
    if isinstance(config.get('arm_init_pose'), str):
        config['arm_init_pose'] = [float(x) for x in config['arm_init_pose'].split()]
    if isinstance(config.get('arm_init_gripper'), str):
        config['arm_init_gripper'] = float(config['arm_init_gripper'])

    # 安全配置：默认使用 carm_deploy 目录下的 safety_config.json，且必须存在
    if not config.get('safety_config'):
        default_safety = os.path.join(carm_deploy_root, 'safety_config.json')
        config['safety_config'] = default_safety
    config['safety_config'] = os.path.expandvars(os.path.expanduser(config['safety_config']))
    if not os.path.exists(config['safety_config']):
        rospy.logfatal("=" * 60)
        rospy.logfatal("安全配置文件不存在: %s", config['safety_config'])
        rospy.logfatal("")
        rospy.logfatal("首次使用必须先录制安全边界，请执行:")
        rospy.logfatal("  cd carm_ros_deploy/src/carm_deploy/tools")
        rospy.logfatal("  python record_workspace.py")
        rospy.logfatal("")
        rospy.logfatal("录制完成后重新启动推理。")
        rospy.logfatal("=" * 60)
        raise SystemExit(1)
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("CARM Policy Inference Node")
    rospy.loginfo("=" * 60)
    rospy.loginfo(f"Robot IP: {config['robot_ip']}")
    rospy.loginfo(f"Camera topics: {config['camera_topics']}")
    rospy.loginfo(f"Pretrain: {config['pretrain']}")
    rospy.loginfo("-" * 60)
    
    # 禁止使用 joint_cmd_mode
    if config.get('joint_cmd_mode', False):
        rospy.logfatal("=" * 60)
        rospy.logfatal("joint_cmd_mode 当前不支持！")
        rospy.logfatal("请移除 --joint_cmd_mode 参数，使用默认的末端位姿控制模式。")
        rospy.logfatal("=" * 60)
        raise SystemExit(1)
    
    rospy.loginfo("-" * 60)
    rospy.loginfo("Inference Configuration:")
    rospy.loginfo(f"  num_inference_steps: {config['num_inference_steps']} (more = better quality)")
    rospy.loginfo(f"  use_ema: {config['use_ema']} (EMA better for 1-step, Non-EMA better for multi-step)")
    rospy.loginfo("-" * 60)
    rospy.loginfo("Action Chunk Execution Mode:")
    rospy.loginfo(f"  execution_mode: {config.get('execution_mode', 'temporal_ensemble')}")
    rospy.loginfo(f"  max_active_chunks: {config.get('max_active_chunks', 'auto')}")
    rospy.loginfo(f"  crossfade_steps: {config.get('crossfade_steps', 0)}")
    rospy.loginfo(f"  truncate_at_act_horizon: {config.get('truncate_at_act_horizon', False)}")
    rospy.loginfo(f"  act_horizon: {config.get('act_horizon', 'same as pred_horizon')}")
    rospy.loginfo("-" * 60)
    rospy.loginfo("Teleop Alignment Parameters:")
    rospy.loginfo(f"  teleop_scale: 1.0 (fixed, GAP-2 fix)")
    rospy.loginfo(f"  inference_speed_scale: {config.get('inference_speed_scale', 1.0)} (runtime speed control)")
    rospy.loginfo(f"  control_freq: {config.get('control_freq', 50)}Hz")
    rospy.loginfo(f"  gripper_hysteresis_window: {config.get('gripper_hysteresis_window', 1)}")
    rospy.loginfo("-" * 60)
    rospy.loginfo(f"  log_dir: {config['log_dir'] or '~/rl-vla/inference_logs'}")
    rospy.loginfo("-" * 60)
    rospy.loginfo("Intervention & Recording:")
    rospy.loginfo(f"  record_inference: {config.get('record_inference', False)}")
    rospy.loginfo(f"  intervention: {config.get('intervention', False)}")
    rospy.loginfo(f"  intervention_mode: {config.get('intervention_mode', 'replace')}")
    rospy.loginfo(f"  max_steps: {config.get('max_steps', 99999)} (auto-stop episode when reached)")
    rospy.loginfo("-" * 60)
    rospy.loginfo("Safety Configuration:")
    rospy.loginfo(f"  safety_config: {config['safety_config'] or 'default'}")
    rospy.loginfo("=" * 60)
    
    # 创建推理节点
    node = InferenceNode(config)
    
    # 全局变量用于信号处理
    shutdown_in_progress = False
    
    def signal_handler(signum, frame):
        """处理 Ctrl+C 信号，确保安全退出"""
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            rospy.logwarn("Force exit requested, exiting immediately...")
            sys.exit(1)
        shutdown_in_progress = True
        rospy.loginfo("\nReceived shutdown signal, cleaning up...")
        node.shutdown()
        rospy.signal_shutdown("User interrupted")
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 注册 ROS 关闭回调
    rospy.on_shutdown(node.shutdown)
    
    try:
        # 运行控制循环
        node.control_loop()
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
    finally:
        if not shutdown_in_progress:
            node.shutdown()


if __name__ == '__main__':
    main()
