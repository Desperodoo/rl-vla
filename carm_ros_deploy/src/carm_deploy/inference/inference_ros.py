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
    
    # 启用推理回流采集
    rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt --record_inference
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
from inference.hitl_bridge import HumanChunkProposalBuilder, HitlArbitrationBridge
from rlft.utils.model_factory import SUPPORTED_ALGORITHMS

# 安全控制和日志
from core.camera_config import resolve_camera_config
from core.safety_controller import SafetyController
from inference.inference_logger import InferenceLogger
from inference.inference_recorder import InferenceRecorder

from core.env_ros import RealEnvironment
from utils.trajectory_interpolator import VecTF, ActionChunkManager
from utils.timeline_logger import TimelineLogger
from utils.episode_keyboard import EpisodeKeyboardHandler
from data.teleop_bridge import TeleopSignalClient


class InferenceNode:
    """
    ROS 推理节点
    
    支持推理执行与 inference rollout 采集
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
        self.hitl_mode = config.get('hitl_mode', 'disabled')
        self.hitl_enabled = self.hitl_mode != 'disabled'
        self.hitl_signal_source = config.get('hitl_signal_source', 'teleop_v2_processed')
        self.hitl_stale_timeout_ms = float(config.get('hitl_stale_timeout_ms', 150.0))
        self.hitl_require_active = bool(config.get('hitl_require_active', True))
        self.hitl_record_full_provenance = bool(config.get('hitl_record_full_provenance', True))
        self.hitl_human_execute_mode = str(config.get('hitl_human_execute_mode', 'direct'))
        if self.hitl_human_execute_mode not in ('direct', 'scheduled'):
            raise ValueError(f"Unsupported hitl_human_execute_mode: {self.hitl_human_execute_mode}")
        
        # Action Chunk 执行模式参数
        # execution_mode: 'temporal_ensemble' (原始) 或 'receding_horizon' (标准 action chunking)
        self.execution_mode = config.get('execution_mode', 'temporal_ensemble')
        self.max_active_chunks = config.get('max_active_chunks', None)  # None = 不限制
        self.crossfade_steps = config.get('crossfade_steps', 0)  # 0 = 无 crossfade
        self.truncate_at_act_horizon = config.get('truncate_at_act_horizon', False)  # 是否截断到 act_horizon
        
        self.camera_config = config['camera_config']

        # 初始化环境
        rospy.loginfo("Initializing environment...")
        self.env = RealEnvironment(config)
        self._init_camera_metadata(config)

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
        self.hitl_candidate_manager = None
        if self.hitl_mode == 'candidate':
            self.hitl_candidate_manager = ActionChunkManager(
                temporal_factor_k=self.temporal_factor_k,
                execution_mode=self.execution_mode,
                max_active_chunks=self.max_active_chunks,
                crossfade_steps=self.crossfade_steps,
            )
        self.lock_tfs = threading.Lock()
        self.hitl_state_lock = threading.Lock()
        self.hitl_signal_client = None
        self.hitl_human_builder = None
        self.hitl_arbitration_bridge = None
        self.hitl_policy_sequence = 0
        self.hitl_candidate_state = {
            'sched_action': None,
            'exec_action': None,
            'shared_source': 'policy',
            'shared_source_code': 0,
            'num_candidates': 0,
        }
        self.hitl_live_owner_active = False
        self.hitl_live_last_error = None
        self.hitl_live_state = {
            'human_direct_target_abs': None,
            'human_sched_target_abs': None,
            'human_exec_target_abs': None,
            'human_execute_source': 'policy',
            'human_active': False,
            'human_valid': False,
            'human_stale': False,
            'signal_age_ms': None,
            'shared_source': 'policy',
            'shared_source_code': 0,
            'human_execute_mode': self.hitl_human_execute_mode,
        }
        if self.hitl_enabled:
            self.hitl_signal_client = TeleopSignalClient(
                robot_ip=config.get('robot_ip', '10.42.0.101'),
                backend_url_v2=config.get('backend_url_v2', ''),
                events_v2_url=config.get('events_v2_url', ''),
                enable_sse=False,
            )
            self.hitl_human_builder = HumanChunkProposalBuilder(
                pred_horizon=self._pred_horizon,
                control_freq=self.control_freq,
                act_horizon=self._act_horizon,
                stale_timeout_ms=self.hitl_stale_timeout_ms,
                require_active=self.hitl_require_active,
            )
            self.hitl_arbitration_bridge = HitlArbitrationBridge()
        
        rospy.loginfo(f"ActionChunkManager: mode={self.execution_mode}, "
                      f"max_active_chunks={self.max_active_chunks}, "
                      f"crossfade_steps={self.crossfade_steps}, "
                      f"truncate_at_act_horizon={self.truncate_at_act_horizon}")
        rospy.loginfo(
            f"HITL mode: {self.hitl_mode}, signal_source={self.hitl_signal_source}, "
            f"stale_timeout_ms={self.hitl_stale_timeout_ms}, require_active={self.hitl_require_active}, "
            f"human_execute_mode={self.hitl_human_execute_mode}"
        )
        
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
        self.pending_outcome_label = False  # 等待 success/failure 标注
        self.max_steps = config.get('max_steps', 900)  # 每个 episode 最大步数
        
        # episode 控制与采集模块
        self.episode_keyboard = None
        self.inference_recorder = None
        
        if self.record_inference_enabled:
            self._init_recording_controls(config)
        
        # 启动推理线程
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        rospy.loginfo("InferenceNode initialized")
    
    def _topic_to_camera_name(self, topic, index):
        """从 topic 自动生成稳定的相机名"""
        name = topic.strip('/').replace('/', '_').replace('-', '_')
        if not name:
            name = f"camera_{index}"
        return name

    def _init_camera_metadata(self, config):
        """初始化相机元数据，用于推理输入和录制对齐 teleop setting"""
        self.camera_topics = list(self.camera_config.topics)
        self.camera_names = list(self.camera_config.names)
        self.primary_camera = self.camera_config.primary_name
        self.primary_camera_idx = self.camera_config.primary_index

    def _init_recording_controls(self, config):
        """初始化 inference rollout 采集与 episode 键盘控制。"""
        self.episode_keyboard = EpisodeKeyboardHandler()

        def on_record_action(action):
            self._handle_record_action(action)

        def on_quit():
            rospy.loginfo("Quit requested via keyboard")
            self.running = False

        self.episode_keyboard.set_record_callback(on_record_action)
        self.episode_keyboard.set_quit_callback(on_quit)
        self.episode_keyboard.start()
        rospy.loginfo("Episode keyboard controls enabled for inference recording")

        # 数据采集记录器
        record_dir = config.get('record_dir', '')
        if not record_dir:
            record_dir = config.get('log_dir', '')
        if not record_dir:
            from utils.paths import get_inference_logs_dir
            record_dir = get_inference_logs_dir()

        action_dim = getattr(self.policy, 'action_dim_full', 15)

        self.inference_recorder = InferenceRecorder(
            output_dir=record_dir,
            pred_horizon=self._pred_horizon,
            action_dim=action_dim,
            camera_topics=self.camera_topics,
            camera_names=self.camera_names,
            primary_camera=self.primary_camera,
            hitl_enabled=self.hitl_enabled,
            hitl_mode=self.hitl_mode,
            hitl_signal_source=self.hitl_signal_source,
            hitl_arbitration_mode='source_select',
            hitl_record_full_provenance=self.hitl_record_full_provenance,
        )
        rospy.loginfo(f"Inference recording enabled, output_dir: {record_dir}")

        rospy.loginfo("=" * 60)
        rospy.loginfo("Multi-episode recording mode enabled")
        rospy.loginfo("Press 'R' to start recording an episode")
        rospy.loginfo("Press 'R' again to stop and choose to save (Y) or discard (N)")
        rospy.loginfo("If saved, press 'S' for success or 'F' for failure")
        rospy.loginfo("After save/discard, arm will return to init position")
        rospy.loginfo("Press 'R' to start next episode")
        rospy.loginfo("Press Ctrl+C to quit")
        rospy.loginfo("=" * 60)
    
    def _handle_record_action(self, action: str):
        """
        处理录制相关的键盘动作
        
        状态机:
        1. waiting_start=True, pending_save=False, pending_outcome_label=False: 等待按 R 开始
        2. waiting_start=False, pending_save=False, pending_outcome_label=False: 正在录制，按 R 停止
        3. waiting_start=False, pending_save=True: 等待 Y/N 确认是否保存
        4. waiting_start=False, pending_outcome_label=True: 等待 S/F 标注 success/failure
        """
        if action == 'toggle':  # R 键
            if self.pending_save or self.pending_outcome_label:
                # 正在等待保存确认或 outcome 标注，忽略 R 键
                rospy.logwarn("Please finish save/outcome confirmation first")
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

        elif action == 'mark_success':  # S 键
            if self.pending_outcome_label:
                self._finalize_episode_outcome(success=True)
            else:
                rospy.logwarn("No episode waiting for success/failure label")

        elif action == 'mark_failure':  # F 键
            if self.pending_outcome_label:
                self._finalize_episode_outcome(success=False)
            else:
                rospy.logwarn("No episode waiting for success/failure label")
    
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

        if self.hitl_mode == 'live' and not self._activate_hitl_live_owner():
            self.waiting_start = True
            self.episode_paused = True
            rospy.logerr("HITL live owner acquire failed, episode start aborted")
            return
        
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
        self._restore_hitl_live_owner()
        
        # 停止录制
        if self.inference_recorder:
            self.inference_recorder.stop_recording()
        
        step_count = self.step_count
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Episode stopped - {step_count} steps recorded")
        rospy.loginfo("Save this episode? Press 'Y' to save, 'N' to discard")
        rospy.loginfo("=" * 60)
    
    def _confirm_save_episode(self, save: bool):
        """确认保存或丢弃 episode。保存时进入 success/failure 标注阶段。"""
        if save:
            self.pending_save = False
            self.pending_outcome_label = True
            rospy.loginfo("Mark episode outcome: press 'S' for success or 'F' for failure")
            return

        # 丢弃
        if self.inference_recorder:
            self.inference_recorder.discard()
        rospy.loginfo("Episode discarded")

        self.pending_save = False
        self.pending_outcome_label = False
        self._finish_episode_cycle()

    def _finalize_episode_outcome(self, success: bool):
        """在保存前补充 success/failure 标签并真正保存 episode。"""
        outcome_label = 'success' if success else 'failure'
        if self.inference_recorder:
            filepath = self.inference_recorder.confirm_save(success=success, outcome_label=outcome_label)
            if filepath:
                self.logger.record_episode_file(filepath, success=success, outcome_label=outcome_label)
                rospy.loginfo(f"Episode saved to: {filepath} ({outcome_label})")

        self.pending_outcome_label = False
        self._finish_episode_cycle()

    def _finish_episode_cycle(self):
        """结束当前 episode 周期，机械臂回初始位并等待下一条。"""
        self._restore_hitl_live_owner()
        rospy.loginfo("Returning to initial position...")
        self._reinitialize_arm()

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
        创建推理运行信息记录器
        """
        log_dir = config.get('log_dir', '')

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            return InferenceLogger(log_dir=log_dir)
        else:
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
            'init_speed': config.get('init_speed', 2.0),
            'normal_speed_level': config.get('normal_speed_level', 10.0),
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

        hitl_config = {
            'hitl_mode': self.hitl_mode,
            'hitl_enabled': self.hitl_enabled,
            'hitl_policy_version': 'hitl_inference_v1',
            'hitl_human_signal_source': self.hitl_signal_source,
            'hitl_arbitration_mode': 'source_select',
            'hitl_live_execute_enabled': self.hitl_mode == 'live',
            'hitl_human_execute_mode': self.hitl_human_execute_mode,
            'hitl_stale_timeout_ms': self.hitl_stale_timeout_ms,
            'hitl_require_active': self.hitl_require_active,
            'hitl_record_full_provenance': self.hitl_record_full_provenance,
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
            hitl_config=hitl_config,
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
        # 只使用 primary camera，与 teleop 的 primary_camera 语义对齐
        image_idx = getattr(self, 'primary_camera_idx', 0)
        if image_idx >= len(obs["images"]):
            image_idx = 0
        image = obs["images"][image_idx]  # [H, W, C] RGB 格式
        
        # Resize 到目标尺寸
        image = self._preprocess_image(image, target_size)
        
        # HWC -> CHW
        image = rearrange(image, 'h w c -> c h w')
        
        return image

    def _build_hitl_from_snapshot(self, policy_chunk_abs: np.ndarray, qpos_end: np.ndarray) -> Dict[str, Any]:
        if not self.hitl_enabled or self.hitl_signal_client is None or self.hitl_human_builder is None:
            return {
                'teleop_snapshot': None,
                'human_proposal': None,
                'arbitration': {
                    'shared_chunk': np.asarray(policy_chunk_abs, dtype=np.float64),
                    'shared_source': 'policy',
                    'shared_source_code': 0,
                    'human_selected': False,
                    'fallback_reason': '',
                    'shared_valid': True,
                },
            }

        teleop_snapshot = self.hitl_signal_client.fetch_snapshot()
        human_proposal = self.hitl_human_builder.build(np.asarray(qpos_end, dtype=np.float64), teleop_snapshot)
        arbitration = self.hitl_arbitration_bridge.arbitrate(policy_chunk_abs, human_proposal)
        return {
            'teleop_snapshot': teleop_snapshot,
            'human_proposal': human_proposal,
            'arbitration': arbitration,
        }

    def _activate_hitl_live_owner(self) -> bool:
        if self.hitl_mode != 'live' or self.hitl_signal_client is None:
            return True
        try:
            self.hitl_signal_client.set_control_state(False, "upper_machine", timeout_s=0.2)
            with self.hitl_state_lock:
                self.hitl_live_owner_active = True
                self.hitl_live_last_error = None
            return True
        except Exception as exc:
            with self.hitl_state_lock:
                self.hitl_live_owner_active = False
                self.hitl_live_last_error = str(exc)
            rospy.logerr(f"Failed to acquire HITL live upper owner: {exc}")
            return False

    def _restore_hitl_live_owner(self):
        if self.hitl_mode != 'live' or self.hitl_signal_client is None:
            return
        with self.hitl_state_lock:
            had_owner = self.hitl_live_owner_active
            self.hitl_live_owner_active = False
        if not had_owner:
            return
        last_exc = None
        for attempt in range(3):
            try:
                self.hitl_signal_client.set_control_state(True, "lower_machine", timeout_s=0.2)
                with self.hitl_state_lock:
                    self.hitl_live_last_error = None
                return
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.05)
        with self.hitl_state_lock:
            self.hitl_live_last_error = str(last_exc)
        rospy.logerr(f"Failed to restore HITL live lower owner: {last_exc}")

    def _update_hitl_live_state(self, human_proposal: Optional[Dict[str, Any]], arbitration: Dict[str, Any]):
        direct_target = None
        human_active = False
        human_valid = False
        human_stale = False
        signal_age_ms = None
        if human_proposal is not None:
            processed_target_abs = human_proposal.get('processed_target_abs')
            if processed_target_abs is not None:
                direct_target = np.asarray(processed_target_abs, dtype=np.float64)
            human_active = bool(human_proposal.get('human_active'))
            human_valid = bool(human_proposal.get('human_valid'))
            human_stale = bool(human_proposal.get('human_stale'))
            signal_age_ms = human_proposal.get('signal_age_ms')
        with self.hitl_state_lock:
            self.hitl_live_state = {
                'human_direct_target_abs': direct_target,
                'human_sched_target_abs': None,
                'human_exec_target_abs': None,
                'human_execute_source': 'policy',
                'human_active': human_active,
                'human_valid': human_valid,
                'human_stale': human_stale,
                'signal_age_ms': None if signal_age_ms is None else float(signal_age_ms),
                'shared_source': arbitration['shared_source'],
                'shared_source_code': arbitration['shared_source_code'],
                'human_execute_mode': self.hitl_human_execute_mode,
            }

    def _add_chunk_to_manager(self, manager, chunk_abs: np.ndarray, chunk_base_time: float):
        tf = VecTF({})
        action_interval = 1.0 / self.control_freq
        if self.truncate_at_act_horizon:
            num_actions_to_add = min(self._act_horizon, len(chunk_abs))
        else:
            num_actions_to_add = len(chunk_abs)

        self.pos_lookahead_step_start_idx += 1
        chunk_targets = []
        for i in range(num_actions_to_add):
            if self.pos_lookahead_step == 1:
                target_time = chunk_base_time + i * action_interval
                tf.append(target_time, chunk_abs[i].tolist())
            else:
                if self.pos_lookahead_step_start_idx % self.pos_lookahead_step == 0:
                    target_time = chunk_base_time + i * action_interval
                    tf.append(target_time, chunk_abs[i].tolist())
                else:
                    target_time = chunk_base_time + i * self.pos_lookahead_duration
                    tf.append(target_time, chunk_abs[i].tolist())
            chunk_targets.append(target_time)
        chunk_id = manager.add_trajectory(tf)
        return chunk_id, chunk_targets, action_interval, num_actions_to_add
    
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
                    
                    # 保存 qpos_end 用于后续安全检查与绝对位姿重建
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
                    safety_reason_counts = {}
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
                            safety_reason_counts['translation_scaled'] = safety_reason_counts.get('translation_scaled', 0) + 1
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
                                safety_reason_counts['workspace_clip'] = safety_reason_counts.get('workspace_clip', 0) + 1
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
                            safety_reason_counts['gripper_clip'] = safety_reason_counts.get('gripper_clip', 0) + 1
                            if i == 0:
                                safety_events.extend(grip_warnings)
                                safety_clipped = True
                    
                    # 保存模型输出（相对位姿）用于后续日志记录
                    raw_action_for_log = all_actions[0].copy()

                    # 保存安全检查后的相对动作调试快照
                    debug_action_model_relative = all_actions.copy()

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

                    # 生成 absolute target pose 语义的 policy chunk
                    if safety_reason_counts:
                        safety_clipped = True
                    policy_chunk_abs = np.array([
                        apply_relative_transform(step[rel_pose_start:rel_pose_end], qpos_end[:7], step[gripper_idx])
                        for step in debug_action_model_relative
                    ])

                    hitl_bundle = self._build_hitl_from_snapshot(policy_chunk_abs, np.asarray(qpos_end, dtype=np.float64))
                    teleop_snapshot = hitl_bundle['teleop_snapshot']
                    human_proposal = hitl_bundle['human_proposal']
                    arbitration = hitl_bundle['arbitration']
                    self._update_hitl_live_state(human_proposal, arbitration)
                    shared_chunk_abs = np.asarray(arbitration['shared_chunk'], dtype=np.float64)
                    human_direct_target_abs = (
                        None if human_proposal is None else np.asarray(human_proposal['processed_target_abs'], dtype=np.float64)
                    )
                    live_execute_target_abs = policy_chunk_abs[0].copy()
                    if (
                        self.hitl_mode == 'live'
                        and arbitration['shared_source'] == 'human'
                    ):
                        if self.hitl_human_execute_mode == 'scheduled':
                            live_execute_target_abs = shared_chunk_abs[0].copy()
                        elif human_direct_target_abs is not None:
                            live_execute_target_abs = human_direct_target_abs.copy()

                    action_model = policy_chunk_abs.copy()
                    action_executed = shared_chunk_abs.copy() if self.hitl_mode == 'live' else policy_chunk_abs.copy()
                    self.hitl_policy_sequence += 1

                    recorded_step = not self.record_inference_enabled
                    with self.hitl_state_lock:
                        sched_candidate = None if self.hitl_candidate_state.get('sched_action') is None else self.hitl_candidate_state['sched_action'].copy()
                        exec_candidate = None if self.hitl_candidate_state.get('exec_action') is None else self.hitl_candidate_state['exec_action'].copy()
                        human_sched_target_abs = None if self.hitl_live_state.get('human_sched_target_abs') is None else self.hitl_live_state['human_sched_target_abs'].copy()
                        human_exec_target_abs = None if self.hitl_live_state.get('human_exec_target_abs') is None else self.hitl_live_state['human_exec_target_abs'].copy()

                    # 记录数据（如果启用采集）
                    if self.record_inference_enabled and self.inference_recorder is not None:
                        if self.inference_recorder.is_recording:
                            obs_stamp_ros = self.latest_obs.get('stamp', None)
                            if obs_stamp_ros is None:
                                obs_stamp_ros = time.time()
                            recorded_step = self.inference_recorder.record_step(
                                obs=self.latest_obs,
                                action_model=action_model,
                                action_executed=action_executed,
                                timestamp=obs_stamp_ros,
                                hitl_data={
                                    'action_policy_chunk': policy_chunk_abs,
                                    'action_human_chunk': None if human_proposal is None else human_proposal['human_chunk_proposal'],
                                    'action_shared_chunk': shared_chunk_abs,
                                    'action_sched_candidate': sched_candidate,
                                    'action_exec_candidate': exec_candidate,
                                    'action_human_direct_target': human_direct_target_abs,
                                    'action_human_sched_target': human_sched_target_abs,
                                    'action_human_exec_target': human_exec_target_abs,
                                    'action_live_execute_target': live_execute_target_abs,
                                    'hitl_human_active': False if human_proposal is None else human_proposal['human_active'],
                                    'hitl_human_valid': False if human_proposal is None else human_proposal['human_valid'],
                                    'hitl_signal_age_ms': None if human_proposal is None else human_proposal['signal_age_ms'],
                                    'hitl_human_history_count': 0 if human_proposal is None else human_proposal['history_count'],
                                    'hitl_human_history_span_ms': None if human_proposal is None else human_proposal['history_span_ms'],
                                    'hitl_human_history_usable': False if human_proposal is None else human_proposal['history_usable'],
                                    'hitl_human_rollout_step_count': 0 if human_proposal is None else human_proposal['rollout_step_count'],
                                    'hitl_human_rollout_dt_ms': None if human_proposal is None else human_proposal['rollout_dt_ms'],
                                    'hitl_human_linear_velocity': np.zeros((3,), dtype=np.float64) if human_proposal is None else human_proposal['linear_velocity'],
                                    'hitl_human_angular_velocity': np.zeros((3,), dtype=np.float64) if human_proposal is None else human_proposal['angular_velocity'],
                                    'hitl_human_gripper_velocity': None if human_proposal is None else human_proposal['gripper_velocity'],
                                    'hitl_policy_sequence': self.hitl_policy_sequence,
                                    'hitl_human_sequence': -1 if human_proposal is None else human_proposal['processed_sequence'],
                                    'hitl_shared_source': arbitration['shared_source_code'],
                                    'hitl_shared_valid_mask': arbitration['shared_valid'],
                                    'hitl_live_execute_source': arbitration['shared_source_code'] if self.hitl_mode == 'live' else 0,
                                },
                            )

                    # 记录第一个动作用于下一次参考
                    self.last_action = action_executed[0].copy()
                    
                    # 创建 policy 轨迹并添加到主管理器
                    obs_stamp_ros = self.latest_obs.get("stamp", None)
                    if self.chunk_time_base == 'obs_stamp' and obs_stamp_ros is not None:
                        chunk_base_time = obs_stamp_ros
                    else:
                        chunk_base_time = time.time()
                    chunk_id, chunk_targets, action_interval, num_actions_to_add = self._add_chunk_to_manager(
                        self.action_manager,
                        action_executed,
                        chunk_base_time,
                    )

                    if self.timeline_logger is not None:
                        delta_chunk_obs = None
                        if obs_stamp_ros is not None:
                            delta_chunk_obs = chunk_base_time - obs_stamp_ros
                        if teleop_snapshot is not None:
                            self.timeline_logger.log(
                                'hitl_signal',
                                hitl_mode=self.hitl_mode,
                                teleop_active=bool(teleop_snapshot.get('teleop_active')),
                                teleop_processed_sequence=teleop_snapshot.get('processed_sequence'),
                                teleop_raw_sequence=teleop_snapshot.get('raw_sequence'),
                                teleop_signal_age_ms=teleop_snapshot.get('signal_age_ms'),
                            )
                        if human_proposal is not None:
                            self.timeline_logger.log(
                                'hitl_human_chunk',
                                hitl_mode=self.hitl_mode,
                                human_valid=bool(human_proposal['human_valid']),
                                human_active=bool(human_proposal['human_active']),
                                human_stale=bool(human_proposal['human_stale']),
                                signal_age_ms=human_proposal['signal_age_ms'],
                                processed_sequence=human_proposal['processed_sequence'],
                                abs_reconstruction_pos_error=human_proposal['abs_reconstruction_pos_error'],
                                abs_reconstruction_rot_error=human_proposal['abs_reconstruction_rot_error'],
                                history_count=int(human_proposal['history_count']),
                                history_span_ms=float(human_proposal['history_span_ms']),
                                history_usable=bool(human_proposal['history_usable']),
                                rollout_step_count=int(human_proposal['rollout_step_count']),
                                rollout_dt_ms=float(human_proposal['rollout_dt_ms']),
                                linear_velocity=np.asarray(human_proposal['linear_velocity'], dtype=np.float64).tolist(),
                                angular_velocity=np.asarray(human_proposal['angular_velocity'], dtype=np.float64).tolist(),
                                gripper_velocity=float(human_proposal['gripper_velocity']),
                                horizon=int(len(human_proposal['human_chunk_proposal'])),
                            )
                        self.timeline_logger.log(
                            'hitl_arbitration',
                            hitl_mode=self.hitl_mode,
                            shared_source=arbitration['shared_source'],
                            fallback_reason=arbitration['fallback_reason'],
                            human_selected=bool(arbitration['human_selected']),
                            human_execute_mode=self.hitl_human_execute_mode,
                        )
                        if self.hitl_mode == 'live':
                            self.timeline_logger.log(
                                'hitl_live_execute',
                                hitl_mode=self.hitl_mode,
                                shared_source=arbitration['shared_source'],
                                human_execute_mode=self.hitl_human_execute_mode,
                                human_direct_target_abs=None if human_direct_target_abs is None else human_direct_target_abs.tolist(),
                                live_execute_target_abs=live_execute_target_abs.tolist(),
                            )
                    
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
                    if self.hitl_mode == 'candidate' and self.hitl_candidate_manager is not None:
                        candidate_chunk_id, candidate_chunk_targets, _, candidate_actions_added = self._add_chunk_to_manager(
                            self.hitl_candidate_manager,
                            shared_chunk_abs,
                            chunk_base_time,
                        )
                        if self.timeline_logger is not None:
                            self.timeline_logger.log(
                                'hitl_candidate_chunk',
                                hitl_mode=self.hitl_mode,
                                chunk_id=candidate_chunk_id,
                                chunk_base_time=chunk_base_time,
                                shared_source=arbitration['shared_source'],
                                num_actions_added=candidate_actions_added,
                                chunk_targets=candidate_chunk_targets,
                            )
                    
                    if recorded_step:
                        # 记录步骤日志（在动作转换后，包含 raw_action 和 executed_action）
                        self.logger.log_step(
                            timestamp=time.time(),
                            obs=self.latest_obs,  # 包含 images, qpos_joint, qpos_end
                            raw_action=raw_action_for_log,  # 模型输出的相对位姿
                            executed_action=action_executed[0],
                            inference_time=inference_time,
                            safety_clipped=safety_clipped,
                            safety_warnings=safety_events if safety_events else None,
                            safety_reason_counts=safety_reason_counts if safety_reason_counts else None,
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

            execute_action = action
            execute_source = 'policy'
            if self.hitl_mode == 'live':
                with self.hitl_state_lock:
                    live_owner_active = self.hitl_live_owner_active
                    live_state = dict(self.hitl_live_state)
                if (
                    live_owner_active
                    and live_state.get('shared_source') == 'human'
                    and live_state.get('human_valid')
                ):
                    if self.hitl_human_execute_mode == 'scheduled':
                        execute_action = np.asarray(action, dtype=np.float64)
                        execute_source = 'human_scheduled'
                    elif live_state.get('human_direct_target_abs') is not None:
                        execute_action = np.asarray(live_state['human_direct_target_abs'], dtype=np.float64)
                        execute_source = 'human_direct'
                    with self.hitl_state_lock:
                        self.hitl_live_state['human_sched_target_abs'] = np.asarray(action, dtype=np.float64).copy()
                        self.hitl_live_state['human_exec_target_abs'] = np.asarray(execute_action, dtype=np.float64).copy()
                        self.hitl_live_state['human_execute_source'] = execute_source
                else:
                    with self.hitl_state_lock:
                        self.hitl_live_state['human_sched_target_abs'] = None
                        self.hitl_live_state['human_exec_target_abs'] = None
                        self.hitl_live_state['human_execute_source'] = 'policy'

            # 打印夹爪下发值与频率（节流）
            grip_val = None
            if len(execute_action) > 0:
                grip_val = float(execute_action[-1])

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
            self.env.end_control_nostep(execute_action)

            if (
                self.record_inference_enabled
                and self.inference_recorder is not None
                and self.inference_recorder.is_recording
            ):
                human_direct_target = None
                human_sched_target = None
                human_exec_target = None
                shared_source = None
                human_execute_mode = None
                if self.hitl_mode == 'live':
                    with self.hitl_state_lock:
                        live_state = dict(self.hitl_live_state)
                    human_direct_target = live_state.get('human_direct_target_abs')
                    human_sched_target = live_state.get('human_sched_target_abs')
                    human_exec_target = live_state.get('human_exec_target_abs')
                    shared_source = live_state.get('shared_source')
                    human_execute_mode = self.hitl_human_execute_mode
                self.inference_recorder.record_control_step(
                    query_time=tm,
                    t_send_sys=time.time(),
                    execute_source=execute_source,
                    human_execute_mode=human_execute_mode,
                    live_execute_target=np.asarray(execute_action, dtype=np.float64),
                    human_direct_target=human_direct_target,
                    human_sched_target=human_sched_target,
                    human_exec_target=human_exec_target,
                    shared_source=shared_source,
                )

            candidate_action = None
            candidate_meta = None
            if self.hitl_mode == 'candidate' and self.hitl_candidate_manager is not None:
                with self.lock_tfs:
                    candidate_action, candidate_meta = self.hitl_candidate_manager.get_fused_action_with_meta(tm)
                with self.hitl_state_lock:
                    self.hitl_candidate_state = {
                        'sched_action': None if candidate_action is None else np.asarray(candidate_action, dtype=np.float64),
                        'exec_action': None if candidate_action is None else np.asarray(candidate_action, dtype=np.float64),
                        'shared_source': 'policy' if candidate_meta is None else 'candidate',
                        'shared_source_code': 0,
                        'num_candidates': 0 if candidate_meta is None else int(candidate_meta.get('num_candidates', 0)),
                    }
                if self.timeline_logger is not None and candidate_action is not None and (
                    self.control_step_count % self.timeline_control_stride == 0
                ):
                    self.timeline_logger.log(
                        'hitl_candidate_control',
                        query_time=tm,
                        t_send_sys=time.time(),
                        num_candidates=0 if candidate_meta is None else int(candidate_meta.get('num_candidates', 0)),
                        used_chunk_ids=[] if candidate_meta is None else candidate_meta.get('used_chunk_ids', []),
                    )

            if self.timeline_logger is not None and (self.control_step_count % self.timeline_control_stride == 0):
                if self.hitl_mode == 'live' and execute_source.startswith('human'):
                    self.timeline_logger.log(
                        'hitl_human_control',
                        query_time=tm,
                        t_send_sys=time.time(),
                        human_execute_mode=self.hitl_human_execute_mode,
                        execute_source=execute_source,
                        human_sched_target_abs=np.asarray(action, dtype=np.float64).tolist(),
                        human_exec_target_abs=np.asarray(execute_action, dtype=np.float64).tolist(),
                    )
                self.timeline_logger.log(
                    'control',
                    query_time=tm,
                    t_send_sys=time.time(),
                    execute_source=execute_source,
                    hitl_human_execute_mode=self.hitl_human_execute_mode if self.hitl_mode == 'live' else None,
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
        
        if self.episode_keyboard is not None:
            self.episode_keyboard.stop()

        self._restore_hitl_live_owner()
        
        # 处理未保存的录制数据
        if self.inference_recorder is not None:
            if self.inference_recorder.is_recording:
                self.inference_recorder.stop_recording()
            if self.inference_recorder.is_pending_save:
                rospy.logwarn("Discarding unsaved recording data on shutdown")
                self.inference_recorder.discard()
        
        # 结束并保存 run_info
        if self.episode_started:
            run_info_path = self.logger.end_episode()
            if run_info_path:
                rospy.loginfo(f"Run info saved to: {run_info_path}")

        if self.timeline_logger is not None:
            self.timeline_logger.close()

        if self.hitl_signal_client is not None:
            self.hitl_signal_client.close()
        
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
    
    parser.add_argument('--camera_topics', type=str,
                        default='/camera/color/image_raw',
                        help='Camera topic(s), comma separated')
    parser.add_argument('--camera_names', type=str,
                        default='',
                        help='Camera name(s), comma separated (must align with camera_topics order)')
    parser.add_argument('--primary_camera', type=str,
                        default='',
                        help='Primary camera name used for observations/images canonical view (default: first camera)')
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

    # HITL inference 参数
    parser.add_argument('--hitl_mode', type=str, default='disabled',
                        choices=['disabled', 'shadow', 'candidate', 'live'],
                        help='Human-in-the-loop inference mode')
    parser.add_argument('--hitl_signal_source', type=str, default='teleop_v2_processed',
                        help='Human signal source for HITL proposal building')
    parser.add_argument('--hitl_stale_timeout_ms', type=float, default=150.0,
                        help='Signal stale timeout for HITL teleop snapshots')
    parser.add_argument('--hitl_require_active', action=argparse.BooleanOptionalAction, default=True,
                        help='Require teleop active=true before enabling human proposal')
    parser.add_argument('--hitl_record_full_provenance', action=argparse.BooleanOptionalAction, default=True,
                        help='Record full HITL chunk-level provenance into inference rollout HDF5')
    parser.add_argument('--hitl_human_execute_mode', type=str, default='direct',
                        choices=['direct', 'scheduled'],
                        help='Human live execute path: direct processed target or scheduled via ActionChunkManager')
    parser.add_argument('--backend_url_v2', type=str, default='',
                        help='Optional teleop_target_v2 URL override for HITL')
    parser.add_argument('--events_v2_url', type=str, default='',
                        help='Optional events_v2 URL override for HITL')
    
    # 安全控制参数
    parser.add_argument('--safety_config', type=str, default='',
                        help='Path to safety config JSON file (required)')
    parser.add_argument('--init_speed', type=float, default=2.0,
                        help='Speed level for initialization/home movement (0-10, default: 2.0 = slow)')
    parser.add_argument('--normal_speed_level', type=float, default=10.0,
                        help='Default runtime speed level restored after init/shutdown (0-10, default: 10.0 = teleop-aligned)')
    
    # 日志参数
    parser.add_argument('--log_dir', type=str, default='',
                        help='Directory to save inference logs (default: ~/rl-vla/inference_logs)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save images in inference log (increases file size)')
    
    # 可视化
    parser.add_argument('--vis', action='store_true', default=True,
                        help='Visualize images in OpenCV window')
    
    # inference rollout 采集参数
    parser.add_argument('--record_inference', action='store_true',
                        help='Enable inference data recording')
    parser.add_argument('--record_dir', type=str, default='',
                        help='Directory to save recorded inference data (default: log_dir)')
    parser.add_argument('--max_steps', type=int, default=900,
                        help='Maximum steps per episode (default: 900). Episode auto-stops when exceeded.')
    
    # 支持 roslaunch <param> 注入
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
        'camera_topics', 'camera_names', 'primary_camera', 'sync_slop', 'timeline_log', 'timeline_enabled',
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
        # HITL inference 参数
        'hitl_mode', 'hitl_signal_source', 'hitl_stale_timeout_ms', 'hitl_require_active',
        'hitl_record_full_provenance', 'hitl_human_execute_mode', 'backend_url_v2', 'events_v2_url',
        # inference rollout 采集参数
        'record_inference',
        'record_dir', 'max_steps',
        'init_speed', 'normal_speed_level',
    ]:
        if rospy.has_param(f'~{key}'):
            config[key] = rospy.get_param(f'~{key}')

    # 时间线日志开关：默认开启，除非显式禁用
    if config.get('timeline_disabled', False):
        config['timeline_enabled'] = False
    else:
        config['timeline_enabled'] = True
    
    camera_config = resolve_camera_config(
        config,
        ros_has_param=rospy.has_param,
        ros_get_param=rospy.get_param,
        logwarn=rospy.logwarn,
    )
    config.update(camera_config.to_runtime_dict())
    config['camera_config'] = camera_config

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
    rospy.loginfo(f"  hitl_mode: {config.get('hitl_mode', 'disabled')}")
    rospy.loginfo(f"  hitl_human_execute_mode: {config.get('hitl_human_execute_mode', 'direct')}")
    rospy.loginfo(f"  hitl_signal_source: {config.get('hitl_signal_source', 'teleop_v2_processed')}")
    rospy.loginfo(f"  hitl_stale_timeout_ms: {config.get('hitl_stale_timeout_ms', 150.0)}")
    rospy.loginfo("-" * 60)
    rospy.loginfo(f"  log_dir: {config['log_dir'] or '~/rl-vla/inference_logs'}")
    rospy.loginfo("-" * 60)
    rospy.loginfo("Inference Rollout Recording:")
    rospy.loginfo(f"  record_inference: {config.get('record_inference', False)}")
    rospy.loginfo(f"  max_steps: {config.get('max_steps', 900)} (auto-stop episode when reached)")
    rospy.loginfo("-" * 60)
    rospy.loginfo("Safety Configuration:")
    rospy.loginfo(f"  safety_config: {config['safety_config'] or 'default'}")
    rospy.loginfo(f"  init_speed: {config.get('init_speed', 2.0)}")
    rospy.loginfo(f"  normal_speed_level: {config.get('normal_speed_level', 10.0)}")
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
