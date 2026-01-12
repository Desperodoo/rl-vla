#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consistency Policy 真机评估脚本 (多进程版本)

整合 RealSense 相机、多进程机械臂控制器和策略推理，实现闭环控制。

架构:
┌─────────────────────────────────────────────────────────────────────┐
│                           主进程 (eval_real.py)                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ 键盘/UI 控制  │    │ 观测组装     │    │ 动作调度            │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                   │                       │
         │   ┌───────────────┼───────────────────────┼───────────┐
         │   │               ▼                       ▼           │
         │   │    SharedMemoryRingBuffer      SharedMemoryQueue  │
         │   │               │                       │           │
         │   └───────────────┼───────────────────────┼───────────┘
         │                   │                       │
         ▼                   ▼                       ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐
│ 相机进程       │  │ 策略推理进程   │  │ 控制器进程 (200Hz)          │
│ RealSenseCamera│  │ PolicyInference│  │ Arx5JointController        │
│ (30Hz)         │  │ (ZMQ Server)   │  │ JointTrajectoryInterpolator│
└────────────────┘  └────────────────┘  └────────────────────────────┘

用法:
    # 1. 启动策略推理节点 (在另一个终端)
    python -m consistency_policy.policy_inference
    
    # 2. 启动评估脚本
    python -m consistency_policy.eval_real \
        --output ./eval_output

键盘控制:
- 'q': 退出程序
- 'c': 开始策略控制
- 's': 停止策略控制
- 'r': 复位机械臂
- 'v': 开始/停止录制视频
"""

import os
import sys
import time
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import cv2
import zmq
from multiprocessing.managers import SharedMemoryManager

# 添加项目路径
CONSISTENCY_POLICY_PATH = os.path.dirname(os.path.abspath(__file__))
RL_VLA_PATH = os.path.dirname(CONSISTENCY_POLICY_PATH)
UMI_ARX_PATH = os.path.join(RL_VLA_PATH, 'umi-arx')


# ===================== 推理状态枚举 (流水线模式) =====================

class InferenceState(Enum):
    """推理状态 (用于流水线模式的非阻塞推理)"""
    IDLE = "idle"                    # 空闲，可以开始新推理
    WAITING_RESULT = "waiting"       # 已发送请求，等待结果

if RL_VLA_PATH not in sys.path:
    sys.path.insert(0, RL_VLA_PATH)
if UMI_ARX_PATH not in sys.path:
    sys.path.insert(0, UMI_ARX_PATH)

from consistency_policy.config import RL_VLA_CONFIG
from consistency_policy.robot_controller_mp import Arx5JointControllerProcess
from consistency_policy.realsense_camera import (
    RealSenseCameraProcess, 
    RealSenseCameraConfig,
    CAMERA_CONFIGS,
)
from consistency_policy.rtc_manager import (
    RTCConfig,
    ActionChunkManager,
    compute_soft_mask_weights,
)


# ===================== 默认配置 =====================

# 关节安全限位 (弧度) - ARX5 机械臂
JOINT_LIMITS = {
    'lower': np.array([-3.14, -1.57, -1.57, -1.57, -1.57, -1.57]),  # 下限
    'upper': np.array([3.14, 1.57, 1.57, 1.57, 1.57, 1.57]),         # 上限
}
GRIPPER_LIMITS = {'lower': 0.0, 'upper': 1.0}

DEFAULT_CONFIG = {
    # 策略
    'policy_endpoint': 'tcp://localhost:8766',  # 单机部署也可用 ipc:///tmp/policy.sock
    
    # 相机
    'wrist_camera_serial': '036222071712',  # D435i
    'external_camera_serial': '037522250003',  # D455
    'camera_resolution': (640, 480),
    'camera_fps': 30,
    
    # 机械臂
    'robot_model': 'X5',
    'robot_interface': 'can0',
    'control_frequency': 200,  # Hz
    
    # 策略参数
    'obs_horizon': 2,
    'pred_horizon': 16,
    'action_horizon': 8,
    'image_size': (128, 128),
    
    # 控制
    'eval_frequency': 10,  # Hz - 推理循环频率
    'action_frequency': 30,  # Hz - 训练数据的动作帧率
    
    # 初始姿态 (训练数据起始状态)
    'initial_gripper_pos': 0.08,           # 训练数据起始夹爪位置 (打开状态)
    'prepare_duration': 1.0,               # 准备阶段持续时间 (秒)
    
    # RTC (Real-Time Chunking) 参数
    'enable_rtc': True,                    # 是否启用 RTC
    'rtc_execute_horizon': 6,              # s: 每个 chunk 执行的动作数
    'rtc_inference_delay_percentile': 95,  # 推理延迟估计的百分位数
    'rtc_inference_delay_margin': 0.01,    # 推理延迟余量 (秒)
    'rtc_min_inference_delay_steps': 1,    # 最小 d 值
    'rtc_max_inference_delay_steps': 3,    # 最大 d 值
    'rtc_soft_mask_schedule': 'exp',       # 软掩码类型: 'exp' 或 'linear'
    'rtc_soft_mask_decay_rate': 2.0,       # 指数衰减速率
    'rtc_enable_soft_masking': True,       # 是否启用软掩码
}


# ===================== 时序日志记录 =====================

@dataclass
class TimingRecord:
    """单次推理的时序记录"""
    step: int                           # 步骤编号
    t_loop_start: float                 # 主循环开始时间
    t_obs_get: float                    # 获取观测完成时间
    t_infer_start: float                # 推理开始时间
    t_infer_end: float                  # 推理结束时间
    t_schedule: float                   # 动作调度时间
    chunk_start_time: float             # chunk 第一个动作的目标执行时间
    chunk_end_time: float               # chunk 最后一个动作的目标执行时间
    action_seq: np.ndarray              # 动作序列 (pred_horizon, action_dim)
    adaptive_delay: float               # 自适应延迟补偿值
    # RTC 相关 (可选)
    d_value: int = 0                    # 推理延迟步数 (d)
    actual_joint_pos: Optional[np.ndarray] = None  # 实际关节位置
    actual_gripper_pos: float = 0.0     # 实际夹爪位置


class TimingLogger:
    """
    时序日志记录器
    
    记录推理和动作调度的关键时间戳，用于分析 action chunking 时序问题。
    """
    
    def __init__(self, output_dir: str, downsample_hz: float = 20.0):
        """
        Args:
            output_dir: 日志输出目录
            downsample_hz: 控制器状态采样率 (Hz)
        """
        self.output_dir = output_dir
        self.downsample_hz = downsample_hz
        self.downsample_interval = 1.0 / downsample_hz
        
        # 推理时序记录
        self.inference_records: List[Dict] = []
        
        # 控制器状态记录 (降采样到 20Hz)
        self.controller_states: List[Dict] = []
        self.last_controller_sample_time = 0.0
        
        # 轨迹更新记录
        self.trajectory_updates: List[Dict] = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def log_inference(self, record: TimingRecord):
        """记录一次推理的时序信息"""
        self.inference_records.append({
            'step': record.step,
            't_loop_start': record.t_loop_start,
            't_obs_get': record.t_obs_get,
            't_infer_start': record.t_infer_start,
            't_infer_end': record.t_infer_end,
            't_schedule': record.t_schedule,
            'chunk_start_time': record.chunk_start_time,
            'chunk_end_time': record.chunk_end_time,
            'action_seq': record.action_seq.copy(),
            'adaptive_delay': record.adaptive_delay,
            'infer_duration': record.t_infer_end - record.t_infer_start,
            'obs_to_schedule': record.t_schedule - record.t_obs_get,
            'd_value': record.d_value,
            'actual_joint_pos': record.actual_joint_pos.copy() if record.actual_joint_pos is not None else None,
            'actual_gripper_pos': record.actual_gripper_pos,
        })
    
    def log_controller_state(self, state: Dict, t_now: float):
        """记录控制器状态 (降采样)"""
        if t_now - self.last_controller_sample_time >= self.downsample_interval:
            self.controller_states.append({
                't_sample': t_now,
                'joint_pos': state['joint_pos'].copy(),
                'joint_vel': state['joint_vel'].copy(),
                'gripper_pos': state['gripper_pos'],
            })
            self.last_controller_sample_time = t_now
    
    def log_trajectory_update(self, t_cmd_recv: float, input_times: np.ndarray, 
                               input_times_mono: np.ndarray, n_waypoints: int):
        """记录轨迹更新事件"""
        self.trajectory_updates.append({
            't_cmd_recv': t_cmd_recv,
            'input_times_start': input_times[0] if len(input_times) > 0 else 0.0,
            'input_times_end': input_times[-1] if len(input_times) > 0 else 0.0,
            'input_times_mono_start': input_times_mono[0] if len(input_times_mono) > 0 else 0.0,
            'input_times_mono_end': input_times_mono[-1] if len(input_times_mono) > 0 else 0.0,
            'n_waypoints': n_waypoints,
            'time_diff': t_cmd_recv - input_times[0] if len(input_times) > 0 else 0.0,
        })
    
    def save(self, filename: str = None):
        """保存日志到 NPZ 文件"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"timing_log_{timestamp}.npz"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 转换推理记录
        if len(self.inference_records) > 0:
            infer_data = {
                'step': np.array([r['step'] for r in self.inference_records]),
                't_loop_start': np.array([r['t_loop_start'] for r in self.inference_records]),
                't_obs_get': np.array([r['t_obs_get'] for r in self.inference_records]),
                't_infer_start': np.array([r['t_infer_start'] for r in self.inference_records]),
                't_infer_end': np.array([r['t_infer_end'] for r in self.inference_records]),
                't_schedule': np.array([r['t_schedule'] for r in self.inference_records]),
                'chunk_start_time': np.array([r['chunk_start_time'] for r in self.inference_records]),
                'chunk_end_time': np.array([r['chunk_end_time'] for r in self.inference_records]),
                'adaptive_delay': np.array([r['adaptive_delay'] for r in self.inference_records]),
                'infer_duration': np.array([r['infer_duration'] for r in self.inference_records]),
                'action_seqs': np.stack([r['action_seq'] for r in self.inference_records]),
                'd_values': np.array([r['d_value'] for r in self.inference_records]),
            }
            # 实际关节位置 (可能有 None)
            actual_positions = [r['actual_joint_pos'] for r in self.inference_records]
            if all(p is not None for p in actual_positions):
                infer_data['actual_joint_pos'] = np.stack(actual_positions)
                infer_data['actual_gripper_pos'] = np.array([r['actual_gripper_pos'] for r in self.inference_records])
        else:
            infer_data = {}
        
        # 转换控制器状态记录
        if len(self.controller_states) > 0:
            ctrl_data = {
                'ctrl_t_sample': np.array([s['t_sample'] for s in self.controller_states]),
                'ctrl_joint_pos': np.stack([s['joint_pos'] for s in self.controller_states]),
                'ctrl_joint_vel': np.stack([s['joint_vel'] for s in self.controller_states]),
                'ctrl_gripper_pos': np.array([s['gripper_pos'] for s in self.controller_states]),
            }
        else:
            ctrl_data = {}
        
        # 转换轨迹更新记录
        if len(self.trajectory_updates) > 0:
            traj_data = {
                'traj_t_cmd_recv': np.array([t['t_cmd_recv'] for t in self.trajectory_updates]),
                'traj_input_times_start': np.array([t['input_times_start'] for t in self.trajectory_updates]),
                'traj_input_times_end': np.array([t['input_times_end'] for t in self.trajectory_updates]),
                'traj_n_waypoints': np.array([t['n_waypoints'] for t in self.trajectory_updates]),
                'traj_time_diff': np.array([t['time_diff'] for t in self.trajectory_updates]),
            }
        else:
            traj_data = {}
        
        # 合并并保存
        all_data = {**infer_data, **ctrl_data, **traj_data}
        
        # 添加配置信息
        all_data['config_eval_frequency'] = DEFAULT_CONFIG['eval_frequency']
        all_data['config_action_frequency'] = DEFAULT_CONFIG['action_frequency']
        all_data['config_action_horizon'] = DEFAULT_CONFIG['action_horizon']
        all_data['config_pred_horizon'] = DEFAULT_CONFIG['pred_horizon']
        
        np.savez(filepath, **all_data)
        print(f"[TimingLogger] 时序日志已保存: {filepath}")
        print(f"  - 推理记录: {len(self.inference_records)} 条")
        print(f"  - 控制器状态: {len(self.controller_states)} 条")
        print(f"  - 轨迹更新: {len(self.trajectory_updates)} 条")
        
        return filepath
    
    def clear(self):
        """清空日志"""
        self.inference_records.clear()
        self.controller_states.clear()
        self.trajectory_updates.clear()
        self.last_controller_sample_time = 0.0


# ===================== 评估类 =====================

class RealEvaluation:
    """
    真机评估类 (多进程版本)
    
    整合 RealSense 相机、多进程控制器和策略推理
    支持 RTC (Real-Time Chunking) 模式
    """
    
    def __init__(
        self,
        output_dir: str,
        policy_endpoint: str = DEFAULT_CONFIG['policy_endpoint'],
        robot_model: str = DEFAULT_CONFIG['robot_model'],
        robot_interface: str = DEFAULT_CONFIG['robot_interface'],
        control_frequency: float = DEFAULT_CONFIG['control_frequency'],
        eval_frequency: float = DEFAULT_CONFIG['eval_frequency'],
        action_frequency: float = DEFAULT_CONFIG['action_frequency'],
        obs_horizon: int = DEFAULT_CONFIG['obs_horizon'],
        pred_horizon: int = DEFAULT_CONFIG['pred_horizon'],
        action_horizon: int = DEFAULT_CONFIG['action_horizon'],
        image_size: tuple = DEFAULT_CONFIG['image_size'],
        enable_external_camera: bool = True,  # 是否启用外部相机 (用于录制)
        enable_timing_log: bool = True,  # 是否启用时序日志
        enable_rtc: bool = DEFAULT_CONFIG['enable_rtc'],  # 是否启用 RTC
        rtc_config: Optional[RTCConfig] = None,  # RTC 配置
        verbose: bool = True,
    ):
        self.output_dir = output_dir
        self.policy_endpoint = policy_endpoint
        self.robot_model = robot_model
        self.robot_interface = robot_interface
        self.control_frequency = control_frequency
        self.eval_frequency = eval_frequency
        self.action_frequency = action_frequency  # 训练数据的动作帧率 (30Hz)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.enable_external_camera = enable_external_camera
        self.enable_timing_log = enable_timing_log
        self.enable_rtc = enable_rtc
        self.verbose = verbose
        
        self.eval_dt = 1.0 / eval_frequency
        self.action_dt = 1.0 / action_frequency  # 动作时间间隔 (~33.3ms @ 30Hz)
        
        # 组件 (稍后初始化)
        self.shm_manager: Optional[SharedMemoryManager] = None
        self.controller: Optional[Arx5JointControllerProcess] = None
        self.wrist_camera: Optional[RealSenseCameraProcess] = None
        self.external_camera: Optional[RealSenseCameraProcess] = None
        self.zmq_context: Optional[zmq.Context] = None
        self.zmq_socket: Optional[zmq.Socket] = None
        
        # 状态
        self.obs_buffer: List[Dict] = []
        self.episode_count = 0
        self.is_recording = False
        
        # 推理时间跟踪 (用于自适应延迟补偿)
        self.inference_times: List[float] = []
        self.max_inference_time_samples = 20
        
        # 时序日志记录器
        self.timing_logger: Optional[TimingLogger] = None
        if enable_timing_log:
            self.timing_logger = TimingLogger(output_dir, downsample_hz=20.0)
        
        # RTC (Real-Time Chunking) 管理器
        self.rtc_manager: Optional[ActionChunkManager] = None
        if enable_rtc:
            if rtc_config is None:
                rtc_config = RTCConfig(
                    prediction_horizon=pred_horizon,
                    execute_horizon=DEFAULT_CONFIG['rtc_execute_horizon'],
                    action_dim=7,  # 6 关节 + 1 夹爪
                    action_dt=self.action_dt,
                    inference_delay_percentile=DEFAULT_CONFIG['rtc_inference_delay_percentile'],
                    inference_delay_margin=DEFAULT_CONFIG['rtc_inference_delay_margin'],
                    min_inference_delay_steps=DEFAULT_CONFIG['rtc_min_inference_delay_steps'],
                    max_inference_delay_steps=DEFAULT_CONFIG['rtc_max_inference_delay_steps'],
                    soft_mask_schedule=DEFAULT_CONFIG['rtc_soft_mask_schedule'],
                    soft_mask_decay_rate=DEFAULT_CONFIG['rtc_soft_mask_decay_rate'],
                    enable_soft_masking=DEFAULT_CONFIG['rtc_enable_soft_masking'],
                )
            self.rtc_manager = ActionChunkManager(rtc_config, verbose=verbose)
            print(f"[Eval] RTC 已启用: H={rtc_config.prediction_horizon}, s={rtc_config.execute_horizon}")
        
        # 流水线模式状态 (非阻塞推理)
        self.inference_state = InferenceState.IDLE
        self.pending_obs_time: Optional[float] = None      # 待处理观测的时间
        self.pending_infer_start: Optional[float] = None   # 推理开始时间
        self.zmq_poller: Optional[zmq.Poller] = None       # ZMQ 轮询器
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _init_shm_manager(self):
        """初始化共享内存管理器"""
        print("[Eval] 初始化共享内存管理器...")
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        print("  ✓ 共享内存管理器就绪")
    
    def _init_controller(self):
        """初始化机械臂控制器"""
        print(f"[Eval] 初始化机械臂控制器 ({self.robot_model} @ {self.robot_interface})...")
        
        self.controller = Arx5JointControllerProcess(
            shm_manager=self.shm_manager,
            model=self.robot_model,
            interface=self.robot_interface,
            frequency=self.control_frequency,
            verbose=self.verbose,
        )
        self.controller.start(wait=True)
        print("  ✓ 机械臂控制器就绪")
    
    def _init_cameras(self):
        """初始化相机"""
        # 腕部相机 (用于推理)
        print("[Eval] 初始化腕部相机 (D435i)...")
        wrist_config = RealSenseCameraConfig(
            name='wrist',
            serial_number=DEFAULT_CONFIG['wrist_camera_serial'],
            resolution=DEFAULT_CONFIG['camera_resolution'],
            fps=DEFAULT_CONFIG['camera_fps'],
            enable_depth=False,
            enable_recording=False,
        )
        self.wrist_camera = RealSenseCameraProcess(
            shm_manager=self.shm_manager,
            config=wrist_config,
            verbose=self.verbose,
        )
        self.wrist_camera.start(wait=True)
        print("  ✓ 腕部相机就绪")
        
        # 外部相机 (用于录制)
        if self.enable_external_camera:
            print("[Eval] 初始化外部相机 (D455)...")
            external_config = RealSenseCameraConfig(
                name='external',
                serial_number=DEFAULT_CONFIG['external_camera_serial'],
                resolution=DEFAULT_CONFIG['camera_resolution'],
                fps=DEFAULT_CONFIG['camera_fps'],
                enable_depth=False,
                enable_recording=True,
            )
            self.external_camera = RealSenseCameraProcess(
                shm_manager=self.shm_manager,
                config=external_config,
                verbose=self.verbose,
            )
            self.external_camera.start(wait=True)
            print("  ✓ 外部相机就绪")
    
    def _init_policy_connection(self):
        """连接策略推理节点"""
        print(f"[Eval] 连接策略推理节点 {self.policy_endpoint}...")
        
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
        self.zmq_socket.setsockopt(zmq.LINGER, 0)  # 关闭时不等待未发送消息
        self.zmq_socket.setsockopt(zmq.SNDTIMEO, 5000)  # 发送超时
        self.zmq_socket.connect(self.policy_endpoint)
        
        # 创建 Poller (用于流水线模式的非阻塞检查)
        self.zmq_poller = zmq.Poller()
        self.zmq_poller.register(self.zmq_socket, zmq.POLLIN)
        
        print("  ✓ 策略连接就绪")
    
    def _reconnect_policy(self):
        """重连策略推理节点 (ZMQ REQ 超时后需要重建 socket)"""
        print(f"[Eval] 重连策略推理节点...")
        
        # 关闭旧 socket
        if self.zmq_socket is not None:
            # 从 poller 取消注册
            if self.zmq_poller is not None:
                try:
                    self.zmq_poller.unregister(self.zmq_socket)
                except:
                    pass
            self.zmq_socket.close()
        
        # 创建新 socket
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.zmq_socket.connect(self.policy_endpoint)
        
        # 重新注册 poller
        if self.zmq_poller is not None:
            self.zmq_poller.register(self.zmq_socket, zmq.POLLIN)
        
        # 重置推理状态
        self.inference_state = InferenceState.IDLE
        
        print("  ✓ 策略重连完成")
    
    def initialize(self):
        """初始化所有组件"""
        print("\n" + "=" * 60)
        print("初始化评估环境")
        print("=" * 60)
        
        self._init_shm_manager()
        self._init_controller()
        self._init_cameras()
        self._init_policy_connection()
        
        print("\n✓ 所有组件初始化完成")
    
    def shutdown(self):
        """关闭所有组件"""
        print("\n[Eval] 关闭组件...")
        
        # 停止录制
        if self.is_recording and self.external_camera is not None:
            self.external_camera.stop_recording()
        
        # 关闭策略连接
        if self.zmq_socket is not None:
            self.zmq_socket.close()
            self.zmq_socket = None
        if self.zmq_context is not None:
            self.zmq_context.term()
            self.zmq_context = None
        
        # 停止相机
        if self.wrist_camera is not None:
            self.wrist_camera.stop()
            self.wrist_camera = None
        if self.external_camera is not None:
            self.external_camera.stop()
            self.external_camera = None
        
        # 停止控制器
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        
        # 关闭共享内存管理器
        if self.shm_manager is not None:
            self.shm_manager.shutdown()
            self.shm_manager = None
        
        print("  ✓ 所有组件已关闭")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def prepare_for_policy(self, smooth_transition: bool = True):
        """
        准备策略控制: 将机器人移动到训练数据的初始状态
        
        训练数据起始时夹爪是打开的 (gripper_pos ≈ 0.08)，
        但 reset_to_home() 会将夹爪复位到闭合状态 (gripper_pos ≈ 0)。
        策略从未见过从夹爪闭合开始的场景，会导致错误输出。
        
        此方法在开始策略控制前将夹爪打开到训练数据的初始位置。
        
        Args:
            smooth_transition: 如果为 True，从当前位置平滑过渡（适用于示教模式后）
        """
        initial_gripper = DEFAULT_CONFIG.get('initial_gripper_pos', 0.08)
        duration = DEFAULT_CONFIG.get('prepare_duration', 1.0)
        
        print(f"[Prepare] 准备策略控制...")
        
        # 获取当前状态
        robot_state = self.controller.get_state()
        current_gripper = robot_state['gripper_pos']
        current_joints = robot_state['joint_pos']
        
        print(f"  当前关节位置: {current_joints[:3].round(3)}...")
        print(f"  当前夹爪位置: {current_gripper:.4f}m")
        print(f"  目标夹爪位置: {initial_gripper:.3f}m (打开)")
        
        # 检查夹爪是否已经接近目标
        gripper_ok = abs(current_gripper - initial_gripper) < 0.01
        
        if gripper_ok:
            print(f"  夹爪已在目标位置，跳过准备阶段")
            return
        
        # 逐步移动夹爪到目标位置 (保持关节位置不变)
        dt = 1.0 / self.action_frequency
        steps = int(duration / dt)
        
        print(f"  调整夹爪中 ({steps} 步, {duration:.1f}秒)...")
        
        t_start = time.time()
        for i in range(steps):
            alpha = (i + 1) / steps
            target_gripper = current_gripper + alpha * (initial_gripper - current_gripper)
            target_time = t_start + (i + 1) * dt
            
            self.controller.schedule_waypoint(
                joint_pos=current_joints,  # 保持当前关节位置
                gripper_pos=target_gripper,
                target_time=target_time,
            )
            
            # 等待到目标时间
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # 等待稳定
        time.sleep(0.3)
        
        # 验证最终状态
        robot_state = self.controller.get_state()
        final_gripper = robot_state['gripper_pos']
        print(f"  最终夹爪位置: {final_gripper:.4f}m")
        print(f"[Prepare] 准备完成!")
    
    def enable_teach_mode(self):
        """
        启用示教模式
        
        进入低阻尼模式，可以手动拖动机械臂到任意位置。
        重力补偿会让机械臂更容易保持姿态。
        """
        print("\n[Teach] 进入示教模式...")
        self.controller.enable_teach_mode()
        print("[Teach] 现在可以手动拖动机械臂")
        print("[Teach] 按 'c' 开始策略控制，按 'r' 复位")
    
    def disable_teach_mode(self):
        """
        禁用示教模式
        
        恢复正常控制增益，为策略控制做准备。
        机械臂会保持在当前位置。
        """
        print("\n[Teach] 退出示教模式...")
        self.controller.disable_teach_mode()
        time.sleep(0.2)  # 等待增益恢复
        print("[Teach] 已恢复正常控制")
    
    # ========= 观测与动作 =========
    
    def preprocess_image(self, rgb: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            rgb: (H, W, 3) RGB 图像
            
        Returns:
            processed: (C, H, W) RGB 图像, uint8
        """
        # [临时测试] RGB -> BGR (交换红蓝通道)
        rgb = rgb[:, :, ::-1].copy()
        
        # 调整尺寸
        h, w = self.image_size
        image = cv2.resize(rgb, (w, h))
        
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def get_obs(self) -> Dict[str, np.ndarray]:
        """
        获取当前观测
        
        Returns:
            obs_dict: 包含 'rgb' 和 'state' 的字典
        """
        # 获取相机帧
        camera_data = self.wrist_camera.get_frame()
        rgb = camera_data['rgb']  # (H, W, 3)
        
        # 预处理图像
        image = self.preprocess_image(rgb)
        
        # 获取机器人状态
        # state_dim = 13: 6 joint_pos + 6 joint_vel + 1 gripper_pos
        robot_state = self.controller.get_state()
        state = np.concatenate([
            robot_state['joint_pos'],  # 6 关节位置
            robot_state['joint_vel'],  # 6 关节速度
            [robot_state['gripper_pos']],  # 1 夹爪
        ])
        
        # 添加到缓冲区
        self.obs_buffer.append({
            'rgb': image,
            'state': state.astype(np.float32),
            'timestamp': time.time(),
        })
        
        # 保持缓冲区大小
        while len(self.obs_buffer) > self.obs_horizon:
            self.obs_buffer.pop(0)
        
        # 填充不足的观测
        while len(self.obs_buffer) < self.obs_horizon:
            self.obs_buffer.insert(0, self.obs_buffer[0].copy())
        
        # 堆叠观测
        rgb_stack = np.stack([obs['rgb'] for obs in self.obs_buffer[-self.obs_horizon:]])
        state_stack = np.stack([obs['state'] for obs in self.obs_buffer[-self.obs_horizon:]])
        
        return {
            'rgb': rgb_stack,  # (obs_horizon, C, H, W)
            'state': state_stack,  # (obs_horizon, state_dim)
        }
    
    def predict_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        调用策略推理 (阻塞模式)
        
        Args:
            obs_dict: 观测字典
            
        Returns:
            action: (pred_horizon, action_dim) 动作序列
        """
        self.zmq_socket.send_pyobj(obs_dict)
        action = self.zmq_socket.recv_pyobj()
        
        if isinstance(action, str):
            raise RuntimeError(f"策略推理错误: {action}")
        
        return action
    
    # ========= 流水线模式的非阻塞推理方法 =========
    
    def try_start_inference(self, obs_dict: Dict[str, np.ndarray], obs_time: float) -> bool:
        """
        尝试开始推理 (非阻塞发送)
        
        Args:
            obs_dict: 观测字典
            obs_time: 观测获取时间
            
        Returns:
            success: 是否成功发送请求
        """
        if self.inference_state != InferenceState.IDLE:
            return False
        
        try:
            self.zmq_socket.send_pyobj(obs_dict, zmq.NOBLOCK)
            self.inference_state = InferenceState.WAITING_RESULT
            self.pending_obs_time = obs_time
            self.pending_infer_start = time.time()
            return True
        except zmq.Again:
            # 发送缓冲区满，稍后重试
            return False
        except zmq.ZMQError as e:
            print(f"[Pipeline] 发送推理请求失败: {e}")
            return False
    
    def try_get_inference_result(self) -> Optional[tuple]:
        """
        尝试获取推理结果 (非阻塞接收)
        
        Returns:
            tuple(action_seq, obs_time, inference_time) 如果有结果
            None 如果还没有结果
        """
        if self.inference_state != InferenceState.WAITING_RESULT:
            return None
        
        # 使用 poller 检查是否有数据 (超时 0ms = 立即返回)
        events = dict(self.zmq_poller.poll(timeout=0))
        
        if self.zmq_socket not in events:
            return None  # 还没有结果
        
        try:
            action = self.zmq_socket.recv_pyobj(zmq.NOBLOCK)
            
            inference_time = time.time() - self.pending_infer_start
            obs_time = self.pending_obs_time
            
            # 重置状态
            self.inference_state = InferenceState.IDLE
            self.pending_obs_time = None
            self.pending_infer_start = None
            
            if isinstance(action, str):
                raise RuntimeError(f"策略推理错误: {action}")
            
            return (action, obs_time, inference_time)
            
        except zmq.Again:
            return None
        except zmq.ZMQError as e:
            print(f"[Pipeline] 接收推理结果失败: {e}")
            self.inference_state = InferenceState.IDLE
            return None

    def clip_action(self, joint_pos: np.ndarray, gripper_pos: float) -> tuple:
        """
        裁剪动作到安全范围
        
        Args:
            joint_pos: (6,) 关节位置
            gripper_pos: 夹爪位置
            
        Returns:
            (clipped_joint_pos, clipped_gripper_pos)
        """
        clipped_joint = np.clip(joint_pos, JOINT_LIMITS['lower'], JOINT_LIMITS['upper'])
        clipped_gripper = np.clip(gripper_pos, GRIPPER_LIMITS['lower'], GRIPPER_LIMITS['upper'])
        
        # 检查是否被裁剪
        if not np.allclose(joint_pos, clipped_joint):
            if self.verbose:
                diff = np.abs(joint_pos - clipped_joint)
                max_diff_idx = np.argmax(diff)
                print(f"[Eval] 警告: 关节 {max_diff_idx} 超出限位, 裁剪 {diff[max_diff_idx]:.4f} rad")
        
        return clipped_joint, clipped_gripper
    
    def get_adaptive_delay(self) -> float:
        """
        计算自适应延迟补偿
        
        基于最近推理时间的 95 百分位 + 固定余量
        
        Returns:
            delay: 延迟补偿时间 (秒)
        """
        if len(self.inference_times) < 3:
            return 0.08  # 默认 80ms
        
        # 使用 95 百分位 + 10ms 余量
        p95 = np.percentile(self.inference_times, 95)
        return p95 + 0.01
    
    def schedule_actions(self, action_seq: np.ndarray, start_time: float) -> tuple:
        """
        调度动作序列 (传统模式)
        
        使用 add_waypoint + update_trajectory 模式批量调度动作。
        
        Args:
            action_seq: (T, action_dim) 动作序列
            start_time: 第一个动作的执行时间
            
        Returns:
            (chunk_start_time, chunk_end_time): chunk 的时间范围
        """
        # 使用训练数据的动作帧率 (30Hz) 而不是推理频率 (10Hz)
        action_dt = self.action_dt  # 1.0 / 30 = 33.3ms
        
        chunk_start_time = start_time
        chunk_end_time = start_time
        
        # 添加航点到缓冲区
        for i, action in enumerate(action_seq[:self.action_horizon]):
            target_time = start_time + i * action_dt
            joint_pos = action[:6]
            gripper_pos = float(action[6]) if len(action) > 6 else 0.0
            
            # 动作边界检查
            joint_pos, gripper_pos = self.clip_action(joint_pos, gripper_pos)
            
            self.controller.add_waypoint(
                joint_pos=joint_pos,
                gripper_pos=gripper_pos,
                target_time=target_time,
            )
            chunk_end_time = target_time
        
        # 触发轨迹更新
        self.controller.update_trajectory()
        
        return chunk_start_time, chunk_end_time
    
    def schedule_actions_rtc(
        self, 
        action_seq: np.ndarray, 
        obs_time: float,
        inference_time: float,
        current_time: Optional[float] = None,
    ) -> tuple:
        """
        调度动作序列 (RTC 模式)
        
        使用 RTC 的 soft masking 和 chunk 拼接逻辑。
        
        Args:
            action_seq: (H, action_dim) 策略输出的完整动作序列
            obs_time: 观测获取时间
            inference_time: 本次推理耗时
            current_time: 当前时间 (可选)
            
        Returns:
            (chunk_start_time, chunk_end_time): chunk 的时间范围
        """
        if current_time is None:
            current_time = time.time()
        
        # 提交新 chunk 到 RTC 管理器 (会自动处理 soft masking)
        chunk = self.rtc_manager.submit_new_chunk(
            action_seq=action_seq,
            obs_time=obs_time,
            inference_time=inference_time,
            current_time=current_time,
        )
        
        # 获取调度用的动作和时间戳
        actions, timestamps, chunk_start, chunk_end = self.rtc_manager.get_scheduled_actions(
            start_time=current_time
        )
        
        # 添加航点到控制器
        for i, (action, target_time) in enumerate(zip(actions, timestamps)):
            joint_pos = action[:6]
            gripper_pos = float(action[6]) if len(action) > 6 else 0.0
            
            # 动作边界检查
            joint_pos, gripper_pos = self.clip_action(joint_pos, gripper_pos)
            
            self.controller.add_waypoint(
                joint_pos=joint_pos,
                gripper_pos=gripper_pos,
                target_time=target_time,
            )
        
        # 触发轨迹更新
        self.controller.update_trajectory()
        
        if self.verbose and chunk.chunk_id % 10 == 0:
            stats = self.rtc_manager.get_statistics()
            print(f"[RTC] Chunk #{chunk.chunk_id}: d={chunk.inference_delay}, "
                  f"s={chunk.execute_horizon}, "
                  f"infer_p95={stats['inference_time_p95']*1000:.1f}ms")
        
        return chunk_start, chunk_end
    
    # ========= 录制控制 =========
    
    def start_recording(self):
        """开始录制视频"""
        if self.external_camera is None:
            print("[Eval] 外部相机未启用，无法录制")
            return
        
        if self.is_recording:
            print("[Eval] 已在录制中")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(
            self.output_dir, 
            f"episode_{self.episode_count:03d}_{timestamp}.mp4"
        )
        
        self.external_camera.start_recording(video_path)
        self.is_recording = True
        print(f"[Eval] 开始录制: {video_path}")
    
    def stop_recording(self):
        """停止录制视频"""
        if not self.is_recording:
            return
        
        if self.external_camera is not None:
            self.external_camera.stop_recording()
        self.is_recording = False
        print("[Eval] 停止录制")
    
    # ========= 可视化 =========
    
    def visualize(self, info: dict):
        """
        可视化
        
        Args:
            info: 显示信息
        """
        # 获取腕部相机图像
        camera_data = self.wrist_camera.get_frame()
        frame = cv2.cvtColor(camera_data['rgb'], cv2.COLOR_RGB2BGR)
        
        # 获取外部相机图像 (如果可用)
        if self.external_camera is not None:
            ext_data = self.external_camera.get_frame()
            ext_frame = cv2.cvtColor(ext_data['rgb'], cv2.COLOR_RGB2BGR)
            # 缩放并拼接
            h, w = frame.shape[:2]
            ext_frame = cv2.resize(ext_frame, (w, h))
            frame = np.hstack([frame, ext_frame])
        
        # 添加文字信息
        text_lines = [
            f"Episode: {info.get('episode', 0)}",
            f"Step: {info.get('step', 0)}",
            f"Mode: {info.get('mode', 'idle')}",
            f"FPS: {info.get('fps', 0):.1f}",
            f"Recording: {'ON' if self.is_recording else 'OFF'}",
        ]
        
        y_offset = 30
        for line in text_lines:
            cv2.putText(
                frame, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2
            )
            y_offset += 25
        
        cv2.imshow('Consistency Policy Eval', frame)
    
    # ========= 主循环 =========
    
    def run(self, max_episodes: int = 10, max_steps_per_episode: int = 1000):
        """
        运行评估
        
        Args:
            max_episodes: 最大 episode 数
            max_steps_per_episode: 每个 episode 最大步数
        """
        print("\n" + "=" * 60)
        print("开始真机评估")
        print("=" * 60)
        print("\n键盘控制:")
        print("  'q' - 退出")
        print("  't' - 进入示教模式 (可手动拖动机械臂)")
        print("  'c' - 开始策略控制")
        print("  's' - 停止策略控制")
        print("  'r' - 复位机械臂")
        print("  'v' - 开始/停止录制")
        
        # 预热策略
        print("\n[Eval] 预热策略推理...")
        obs = self.get_obs()
        _ = self.predict_action(obs)
        print("  ✓ 策略预热完成")
        
        # 主循环
        mode = "human"  # human / teach / policy
        step = 0
        last_time = time.time()
        fps = 0.0
        
        try:
            while self.episode_count < max_episodes:
                loop_start = time.time()
                
                # 计算 FPS
                current_time = time.time()
                fps = 0.9 * fps + 0.1 / max(current_time - last_time, 0.001)
                last_time = current_time
                
                # 可视化
                self.visualize({
                    'episode': self.episode_count,
                    'step': step,
                    'mode': mode,
                    'fps': fps,
                })
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n[Eval] 退出...")
                    break
                
                elif key == ord('t'):
                    if mode == "policy":
                        print("\n[Eval] 请先停止策略控制 ('s') 再进入示教模式")
                    elif mode == "teach":
                        print("\n[Eval] 已在示教模式中")
                    else:
                        self.enable_teach_mode()
                        mode = "teach"
                
                elif key == ord('c'):
                    if mode == "teach":
                        # 从示教模式切换到策略控制
                        print("\n[Eval] 从示教模式开始策略控制")
                        self.disable_teach_mode()
                        # 准备阶段: 只调整夹爪，保持关节位置
                        self.prepare_for_policy(smooth_transition=True)
                    else:
                        print("\n[Eval] 开始策略控制")
                        # 从 home 位置开始: 调整夹爪
                        self.prepare_for_policy(smooth_transition=False)
                    
                    mode = "policy"
                    self.obs_buffer.clear()
                    step = 0
                    self.start_recording()
                
                elif key == ord('s'):
                    if mode == "policy":
                        print("\n[Eval] 停止策略控制")
                        mode = "human"
                        self.stop_recording()
                        self.episode_count += 1
                    elif mode == "teach":
                        print("\n[Eval] 退出示教模式")
                        self.disable_teach_mode()
                        mode = "human"
                
                elif key == ord('r'):
                    if mode == "teach":
                        print("\n[Eval] 请先退出示教模式 ('s') 再复位")
                    else:
                        print("\n[Eval] 复位机械臂...")
                        self.controller.reset_to_home()
                        self.obs_buffer.clear()
                
                elif key == ord('v'):
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                
                # 策略控制模式
                if mode == "policy":
                    try:
                        t_loop_start_record = time.time()
                        
                        # 获取观测
                        obs = self.get_obs()
                        t_obs_get = time.time()
                        
                        # 记录控制器状态 (降采样到 20Hz)
                        if self.timing_logger is not None:
                            robot_state = self.controller.get_state()
                            self.timing_logger.log_controller_state(robot_state, t_obs_get)
                        
                        # 推理
                        t_infer_start = time.time()
                        action_seq = self.predict_action(obs)
                        t_infer_end = time.time()
                        t_infer = t_infer_end - t_infer_start
                        
                        # 更新推理时间跟踪
                        self.inference_times.append(t_infer)
                        if len(self.inference_times) > self.max_inference_time_samples:
                            self.inference_times.pop(0)
                        
                        if self.verbose and step % 10 == 0:
                            avg_infer = np.mean(self.inference_times) * 1000
                            chunk_duration = self.action_horizon * self.action_dt * 1000
                            print(f"[Eval] Step {step}, 推理: {t_infer*1000:.1f}ms (avg: {avg_infer:.1f}ms), "
                                  f"chunk覆盖: {chunk_duration:.1f}ms")
                        
                        # 调度动作
                        if self.enable_rtc and self.rtc_manager is not None:
                            # RTC 模式: 使用 soft masking 和 chunk 拼接
                            chunk_start, chunk_end = self.schedule_actions_rtc(
                                action_seq=action_seq,
                                obs_time=t_obs_get,
                                inference_time=t_infer,
                                current_time=t_infer_end,
                            )
                            adaptive_delay = 0.0  # RTC 模式不使用额外延迟
                        else:
                            # 传统模式: 使用自适应延迟补偿
                            adaptive_delay = self.get_adaptive_delay()
                            action_start_time = time.time() + adaptive_delay
                            chunk_start, chunk_end = self.schedule_actions(action_seq, action_start_time)
                        
                        t_schedule = time.time()
                        
                        # 获取当前实际关节位置
                        robot_state = self.controller.get_state()
                        actual_joint_pos = robot_state['joint_pos'].copy()
                        actual_gripper_pos = robot_state['gripper_pos']
                        
                        # 获取 d 值
                        current_d = 0
                        if self.rtc_manager is not None:
                            stats = self.rtc_manager.get_statistics()
                            current_d = stats['current_d']
                        
                        # 记录时序日志
                        if self.timing_logger is not None:
                            record = TimingRecord(
                                step=step,
                                t_loop_start=t_loop_start_record,
                                t_obs_get=t_obs_get,
                                t_infer_start=t_infer_start,
                                t_infer_end=t_infer_end,
                                t_schedule=t_schedule,
                                chunk_start_time=chunk_start,
                                chunk_end_time=chunk_end,
                                action_seq=action_seq,
                                adaptive_delay=adaptive_delay,
                                d_value=current_d,
                                actual_joint_pos=actual_joint_pos,
                                actual_gripper_pos=actual_gripper_pos,
                            )
                            self.timing_logger.log_inference(record)
                        
                        step += 1
                        
                        if step >= max_steps_per_episode:
                            print(f"\n[Eval] Episode {self.episode_count} 完成 ({step} 步)")
                            mode = "human"
                            self.stop_recording()
                            self.episode_count += 1
                            step = 0
                            # 重置 RTC 管理器
                            if self.rtc_manager is not None:
                                self.rtc_manager.reset()
                    
                    except zmq.Again:
                        print("\n[Eval] 策略推理超时，尝试重连...")
                        self._reconnect_policy()
                        mode = "human"
                        self.stop_recording()
                    except zmq.ZMQError as e:
                        print(f"\n[Eval] ZMQ 错误: {e}，尝试重连...")
                        self._reconnect_policy()
                        mode = "human"
                        self.stop_recording()
                    except Exception as e:
                        print(f"\n[Eval] 策略执行错误: {e}")
                        import traceback
                        traceback.print_exc()
                        mode = "human"
                        self.stop_recording()
                
                # 频率控制
                elapsed = time.time() - loop_start
                if elapsed < self.eval_dt:
                    time.sleep(self.eval_dt - elapsed)
        
        except KeyboardInterrupt:
            print("\n[Eval] 收到中断信号")
        
        finally:
            cv2.destroyAllWindows()
            # 保存时序日志
            if self.timing_logger is not None and len(self.timing_logger.inference_records) > 0:
                self.timing_logger.save()
            # 打印 RTC 统计
            if self.rtc_manager is not None:
                stats = self.rtc_manager.get_statistics()
                print(f"\n[RTC] 统计信息:")
                print(f"  - Chunk 数量: {stats['chunk_count']}")
                print(f"  - 推理延迟 (p95): {stats['inference_time_p95']*1000:.1f}ms")
                print(f"  - 最终 d 值: {stats['current_d']}")
    
    # ========= 流水线模式主循环 =========
    
    def run_pipeline(self, max_episodes: int = 10, max_steps_per_episode: int = 1000):
        """
        流水线模式运行评估
        
        核心改进：推理与执行并行
        - 收到推理结果后立即发送下一个推理请求
        - 不等待推理完成，继续执行可视化和键盘处理
        - 实现 chunk 无缝衔接，消除顿挫
        
        Args:
            max_episodes: 最大 episode 数
            max_steps_per_episode: 每个 episode 最大步数
        """
        if self.rtc_manager is None:
            print("[Pipeline] 错误: 流水线模式需要启用 RTC")
            return
        
        print("\n" + "=" * 60)
        print("开始真机评估 (流水线模式)")
        print("=" * 60)
        print("\n流水线模式特点:")
        print("  - 推理与执行并行，无缝衔接")
        print("  - 收到结果后立即发送下一个请求")
        print("  - 运动更平滑，无顿挫")
        print("\n键盘控制:")
        print("  'q' - 退出")
        print("  't' - 进入示教模式 (可手动拖动机械臂)")
        print("  'c' - 开始策略控制")
        print("  's' - 停止策略控制")
        print("  'r' - 复位机械臂")
        print("  'v' - 开始/停止录制")
        
        # 预热策略 (同步模式)
        print("\n[Pipeline] 预热策略推理...")
        obs = self.get_obs()
        _ = self.predict_action(obs)
        print("  ✓ 策略预热完成")
        
        # 状态
        mode = "human"  # human / teach / policy
        step = 0
        last_time = time.time()
        fps = 0.0
        
        # 流水线统计
        pipeline_stats = {
            'inference_count': 0,
            'schedule_count': 0,
            'overlap_positive': 0,
            'overlap_negative': 0,
        }
        
        try:
            while self.episode_count < max_episodes:
                loop_start = time.time()
                
                # 计算 FPS
                current_time = time.time()
                fps = 0.9 * fps + 0.1 / max(current_time - last_time, 0.001)
                last_time = current_time
                
                # 可视化 (每次循环都执行，保持流畅)
                self.visualize({
                    'episode': self.episode_count,
                    'step': step,
                    'mode': f"{mode} (pipeline)",
                    'fps': fps,
                })
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n[Pipeline] 退出...")
                    break
                
                elif key == ord('t'):
                    if mode == "policy":
                        print("\n[Pipeline] 请先停止策略控制 ('s') 再进入示教模式")
                    elif mode == "teach":
                        print("\n[Pipeline] 已在示教模式中")
                    else:
                        self.enable_teach_mode()
                        mode = "teach"
                
                elif key == ord('c'):
                    if mode == "teach":
                        # 从示教模式切换到策略控制
                        print("\n[Pipeline] 从示教模式开始策略控制")
                        self.disable_teach_mode()
                        # 准备阶段: 只调整夹爪，保持关节位置
                        self.prepare_for_policy(smooth_transition=True)
                    else:
                        print("\n[Pipeline] 开始策略控制")
                        # 从 home 位置开始: 调整夹爪
                        self.prepare_for_policy(smooth_transition=False)
                    
                    mode = "policy"
                    self.obs_buffer.clear()
                    step = 0
                    self.inference_state = InferenceState.IDLE
                    self.start_recording()
                    
                    # 冷启动: 第一个 chunk 使用同步推理
                    print("[Pipeline] 冷启动: 同步推理第一个 chunk...")
                    obs = self.get_obs()
                    t_obs = time.time()
                    t_start = time.time()
                    action_seq = self.predict_action(obs)
                    t_infer = time.time() - t_start
                    
                    # 更新推理时间
                    self.inference_times.append(t_infer)
                    
                    # 调度第一个 chunk
                    chunk_start, chunk_end = self.schedule_actions_rtc(
                        action_seq=action_seq,
                        obs_time=t_obs,
                        inference_time=t_infer,
                        current_time=time.time(),
                    )
                    pipeline_stats['schedule_count'] += 1
                    print(f"[Pipeline] 冷启动完成, 推理耗时: {t_infer*1000:.1f}ms")
                    
                    # 立即发送下一个推理请求 (非阻塞)
                    obs = self.get_obs()
                    self.try_start_inference(obs, time.time())
                    pipeline_stats['inference_count'] += 1
                
                elif key == ord('s'):
                    if mode == "policy":
                        print("\n[Pipeline] 停止策略控制")
                        mode = "human"
                        self.inference_state = InferenceState.IDLE
                        self.stop_recording()
                        self.episode_count += 1
                    elif mode == "teach":
                        print("\n[Pipeline] 退出示教模式")
                        self.disable_teach_mode()
                        mode = "human"
                
                elif key == ord('r'):
                    if mode == "teach":
                        print("\n[Pipeline] 请先退出示教模式 ('s') 再复位")
                    else:
                        print("\n[Pipeline] 复位机械臂...")
                        self.controller.reset_to_home()
                        self.obs_buffer.clear()
                        self.inference_state = InferenceState.IDLE
                
                elif key == ord('v'):
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                
                # 策略控制模式 (流水线)
                if mode == "policy":
                    try:
                        # 1. 尝试获取推理结果 (非阻塞)
                        result = self.try_get_inference_result()
                        
                        if result is not None:
                            action_seq, obs_time, inference_time = result
                            t_now = time.time()
                            
                            # 更新推理时间统计
                            self.inference_times.append(inference_time)
                            if len(self.inference_times) > self.max_inference_time_samples:
                                self.inference_times.pop(0)
                            
                            # 获取当前实际关节位置 (用于误差分析)
                            robot_state = self.controller.get_state()
                            actual_joint_pos = robot_state['joint_pos'].copy()
                            actual_gripper_pos = robot_state['gripper_pos']
                            
                            # 调度新 chunk
                            chunk_start, chunk_end = self.schedule_actions_rtc(
                                action_seq=action_seq,
                                obs_time=obs_time,
                                inference_time=inference_time,
                                current_time=t_now,
                            )
                            pipeline_stats['schedule_count'] += 1
                            step += 1
                            
                            # 获取当前 d 值
                            current_d = 0
                            if self.rtc_manager is not None:
                                stats = self.rtc_manager.get_statistics()
                                current_d = stats['current_d']
                            
                            # 记录时序日志
                            if self.timing_logger is not None:
                                record = TimingRecord(
                                    step=step,
                                    t_loop_start=loop_start,
                                    t_obs_get=obs_time,
                                    t_infer_start=obs_time,
                                    t_infer_end=obs_time + inference_time,
                                    t_schedule=t_now,
                                    chunk_start_time=chunk_start,
                                    chunk_end_time=chunk_end,
                                    action_seq=action_seq,
                                    adaptive_delay=0.0,
                                    d_value=current_d,
                                    actual_joint_pos=actual_joint_pos,
                                    actual_gripper_pos=actual_gripper_pos,
                                )
                                self.timing_logger.log_inference(record)
                            
                            if self.verbose and step % 10 == 0:
                                avg_infer = np.mean(self.inference_times) * 1000
                                print(f"[Pipeline] Step {step}, 推理: {inference_time*1000:.1f}ms "
                                      f"(avg: {avg_infer:.1f}ms), d={current_d}")
                            
                            # 2. 立即发送下一个推理请求 (核心: 收到结果就发下一个)
                            obs = self.get_obs()
                            if self.try_start_inference(obs, time.time()):
                                pipeline_stats['inference_count'] += 1
                            
                            # 检查 episode 结束
                            if step >= max_steps_per_episode:
                                print(f"\n[Pipeline] Episode {self.episode_count} 完成 ({step} 步)")
                                mode = "human"
                                self.inference_state = InferenceState.IDLE
                                self.stop_recording()
                                self.episode_count += 1
                                step = 0
                                if self.rtc_manager is not None:
                                    self.rtc_manager.reset()
                        
                        # 3. 如果没有待处理的推理，且处于空闲状态，发送新请求
                        elif self.inference_state == InferenceState.IDLE:
                            obs = self.get_obs()
                            if self.try_start_inference(obs, time.time()):
                                pipeline_stats['inference_count'] += 1
                    
                    except zmq.ZMQError as e:
                        print(f"\n[Pipeline] ZMQ 错误: {e}，尝试重连...")
                        self._reconnect_policy()
                        mode = "human"
                        self.stop_recording()
                    except Exception as e:
                        print(f"\n[Pipeline] 执行错误: {e}")
                        import traceback
                        traceback.print_exc()
                        mode = "human"
                        self.inference_state = InferenceState.IDLE
                        self.stop_recording()
                
                # 短暂 sleep 避免空转 (1ms)
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\n[Pipeline] 收到中断信号")
        
        finally:
            cv2.destroyAllWindows()
            
            # 保存时序日志
            if self.timing_logger is not None and len(self.timing_logger.inference_records) > 0:
                self.timing_logger.save()
            
            # 打印统计
            print(f"\n[Pipeline] 统计信息:")
            print(f"  - 推理请求数: {pipeline_stats['inference_count']}")
            print(f"  - Chunk 调度数: {pipeline_stats['schedule_count']}")
            
            if self.rtc_manager is not None:
                stats = self.rtc_manager.get_statistics()
                print(f"  - RTC Chunk 数量: {stats['chunk_count']}")
                print(f"  - 推理延迟 (p95): {stats['inference_time_p95']*1000:.1f}ms")
                print(f"  - 最终 d 值: {stats['current_d']}")
                print(f"  - Splice 数量: {stats['splice_count']}")


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(description="Consistency Policy 真机评估 (多进程版本)")
    parser.add_argument("-o", "--output", default="./eval_output", help="输出目录")
    parser.add_argument("--policy-endpoint", default=DEFAULT_CONFIG['policy_endpoint'],
                        help="策略服务端点 (tcp://host:port 或 ipc:///path)")
    parser.add_argument("-m", "--model", default=DEFAULT_CONFIG['robot_model'], help="机械臂型号")
    parser.add_argument("-i", "--interface", default=DEFAULT_CONFIG['robot_interface'], help="CAN 接口")
    parser.add_argument("--control-freq", type=float, default=DEFAULT_CONFIG['control_frequency'],
                        help="控制器频率 Hz")
    parser.add_argument("--eval-freq", type=float, default=DEFAULT_CONFIG['eval_frequency'],
                        help="评估循环频率 Hz")
    parser.add_argument("--action-freq", type=float, default=DEFAULT_CONFIG['action_frequency'],
                        help="动作帧率 Hz (应与训练数据一致)")
    parser.add_argument("--no-external-camera", action="store_true", help="禁用外部相机")
    parser.add_argument("--no-timing-log", action="store_true", help="禁用时序日志")
    parser.add_argument("--max-episodes", type=int, default=2, help="最大 episode 数")
    parser.add_argument("--max-steps", type=int, default=10000, help="每 episode 最大步数")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_CONFIG['action_horizon'],
                        help="非 RTC 模式下每次执行的动作数 (默认 8)")
    
    # RTC 相关参数
    parser.add_argument("--no-rtc", action="store_true", help="禁用 RTC (Real-Time Chunking)")
    parser.add_argument("--pipeline", action="store_true", 
                        help="使用流水线模式 (推理与执行并行，运动更平滑)")
    parser.add_argument("--rtc-execute-horizon", type=int, default=DEFAULT_CONFIG['rtc_execute_horizon'],
                        help="RTC: 每个 chunk 执行的动作数 (s)")
    parser.add_argument("--rtc-min-d", type=int, default=DEFAULT_CONFIG['rtc_min_inference_delay_steps'],
                        help="RTC: 最小推理延迟步数 (d)")
    parser.add_argument("--rtc-max-d", type=int, default=DEFAULT_CONFIG['rtc_max_inference_delay_steps'],
                        help="RTC: 最大推理延迟步数 (d)")
    parser.add_argument("--rtc-mask-schedule", choices=['exp', 'linear'], 
                        default=DEFAULT_CONFIG['rtc_soft_mask_schedule'],
                        help="RTC: 软掩码衰减类型")
    parser.add_argument("--rtc-no-soft-mask", action="store_true", help="RTC: 禁用软掩码混合")
    
    args = parser.parse_args()
    
    # 构建 RTC 配置
    rtc_config = None
    if not args.no_rtc:
        rtc_config = RTCConfig(
            prediction_horizon=DEFAULT_CONFIG['pred_horizon'],
            execute_horizon=args.rtc_execute_horizon,
            action_dim=7,
            action_dt=1.0 / args.action_freq,
            inference_delay_percentile=DEFAULT_CONFIG['rtc_inference_delay_percentile'],
            inference_delay_margin=DEFAULT_CONFIG['rtc_inference_delay_margin'],
            min_inference_delay_steps=args.rtc_min_d,
            max_inference_delay_steps=args.rtc_max_d,
            soft_mask_schedule=args.rtc_mask_schedule,
            soft_mask_decay_rate=DEFAULT_CONFIG['rtc_soft_mask_decay_rate'],
            enable_soft_masking=not args.rtc_no_soft_mask,
        )
    
    print("=" * 60)
    print("Consistency Policy 真机评估 (多进程版本)")
    print("=" * 60)
    print(f"\n输出目录: {args.output}")
    print(f"策略端点: {args.policy_endpoint}")
    print(f"机械臂: {args.model} @ {args.interface}")
    print(f"控制频率: {args.control_freq} Hz")
    print(f"评估频率: {args.eval_freq} Hz")
    print(f"动作帧率: {args.action_freq} Hz")
    print(f"外部相机: {'禁用' if args.no_external_camera else '启用'}")
    print(f"时序日志: {'禁用' if args.no_timing_log else '启用'}")
    
    # 打印 RTC 配置
    if not args.no_rtc:
        print(f"\nRTC 配置:")
        print(f"  - 启用: True")
        print(f"  - 流水线模式: {'启用' if args.pipeline else '禁用'}")
        print(f"  - execute_horizon (s): {args.rtc_execute_horizon}")
        print(f"  - inference_delay 范围 (d): [{args.rtc_min_d}, {args.rtc_max_d}]")
        print(f"  - soft_mask_schedule: {args.rtc_mask_schedule}")
        print(f"  - soft_masking: {'禁用' if args.rtc_no_soft_mask else '启用'}")
    else:
        print(f"\nRTC: 禁用 (使用传统模式)")
    
    # 流水线模式需要 RTC
    if args.pipeline and args.no_rtc:
        print("\n⚠️  警告: --pipeline 需要 RTC 支持，但 --no-rtc 被设置")
        print("   将忽略 --pipeline 参数，使用传统模式")
        args.pipeline = False
    
    with RealEvaluation(
        output_dir=args.output,
        policy_endpoint=args.policy_endpoint,
        robot_model=args.model,
        robot_interface=args.interface,
        control_frequency=args.control_freq,
        eval_frequency=args.eval_freq,
        action_frequency=args.action_freq,
        action_horizon=args.action_horizon,
        enable_external_camera=not args.no_external_camera,
        enable_timing_log=not args.no_timing_log,
        enable_rtc=not args.no_rtc,
        rtc_config=rtc_config,
        verbose=args.verbose,
    ) as evaluator:
        if args.pipeline:
            evaluator.run_pipeline(
                max_episodes=args.max_episodes,
                max_steps_per_episode=args.max_steps,
            )
        else:
            evaluator.run(
                max_episodes=args.max_episodes,
                max_steps_per_episode=args.max_steps,
            )


if __name__ == "__main__":
    main()
