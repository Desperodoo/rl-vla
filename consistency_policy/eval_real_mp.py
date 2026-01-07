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
import cv2
import zmq
from multiprocessing.managers import SharedMemoryManager

# 添加项目路径
CONSISTENCY_POLICY_PATH = os.path.dirname(os.path.abspath(__file__))
RL_VLA_PATH = os.path.dirname(CONSISTENCY_POLICY_PATH)
UMI_ARX_PATH = os.path.join(RL_VLA_PATH, 'umi-arx')

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
            }
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
        
        print("  ✓ 策略连接就绪")
    
    def _reconnect_policy(self):
        """重连策略推理节点 (ZMQ REQ 超时后需要重建 socket)"""
        print(f"[Eval] 重连策略推理节点...")
        
        # 关闭旧 socket
        if self.zmq_socket is not None:
            self.zmq_socket.close()
        
        # 创建新 socket
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.zmq_socket.connect(self.policy_endpoint)
        
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
    
    # ========= 观测与动作 =========
    
    def preprocess_image(self, rgb: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            rgb: (H, W, 3) RGB 图像
            
        Returns:
            processed: (C, H, W) RGB 图像, uint8
        """
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
        调用策略推理
        
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
        调度动作序列
        
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
        mode = "human"  # human / policy
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
                
                elif key == ord('c'):
                    print("\n[Eval] 开始策略控制")
                    mode = "policy"
                    self.obs_buffer.clear()
                    step = 0
                    self.start_recording()
                
                elif key == ord('s'):
                    print("\n[Eval] 停止策略控制")
                    mode = "human"
                    self.stop_recording()
                    self.episode_count += 1
                
                elif key == ord('r'):
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
                        
                        # 调度动作 (使用自适应延迟补偿)
                        adaptive_delay = self.get_adaptive_delay()
                        action_start_time = time.time() + adaptive_delay
                        chunk_start, chunk_end = self.schedule_actions(action_seq, action_start_time)
                        t_schedule = time.time()
                        
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
                            )
                            self.timing_logger.log_inference(record)
                        
                        step += 1
                        
                        if step >= max_steps_per_episode:
                            print(f"\n[Eval] Episode {self.episode_count} 完成 ({step} 步)")
                            mode = "human"
                            self.stop_recording()
                            self.episode_count += 1
                            step = 0
                    
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
    
    args = parser.parse_args()
    
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
    
    with RealEvaluation(
        output_dir=args.output,
        policy_endpoint=args.policy_endpoint,
        robot_model=args.model,
        robot_interface=args.interface,
        control_frequency=args.control_freq,
        eval_frequency=args.eval_freq,
        action_frequency=args.action_freq,
        enable_external_camera=not args.no_external_camera,
        enable_timing_log=not args.no_timing_log,
        verbose=args.verbose,
    ) as evaluator:
        evaluator.run(
            max_episodes=args.max_episodes,
            max_steps_per_episode=args.max_steps,
        )


if __name__ == "__main__":
    main()
