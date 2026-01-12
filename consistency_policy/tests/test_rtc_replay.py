#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTC 轨迹回放对比测试

使用真机验证 RTC (Real-Time Chunking) 的效果:
1. 加载相同的 demo 数据
2. 分别以 RTC 模式和传统模式回放
3. 对比跟踪精度、动作连续性、chunk 重叠等指标
4. 可注入不同推理延迟测试 RTC 的鲁棒性

用法:
    # 基础对比测试 (真机)
    python -m consistency_policy.tests.test_rtc_replay \
        --demo ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5
    
    # 指定推理延迟
    python -m consistency_policy.tests.test_rtc_replay \
        --demo /path/to/demo.h5 \
        --inference-delay 0.08 \
        --inference-std 0.02
    
    # 仅运行 RTC 模式
    python -m consistency_policy.tests.test_rtc_replay \
        --demo /path/to/demo.h5 \
        --rtc-only

安全提示:
    - 首次运行请使用 --speed 0.3 慢速执行
    - 确保机械臂周围无障碍物
    - 随时准备按 Ctrl+C 停止

环境:
    conda activate arx-py310
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

# 添加项目路径
CONSISTENCY_POLICY_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_VLA_PATH = os.path.dirname(CONSISTENCY_POLICY_PATH)
if RL_VLA_PATH not in sys.path:
    sys.path.insert(0, RL_VLA_PATH)

from consistency_policy.config import setup_rlft
from consistency_policy.robot_controller_mp import Arx5JointControllerManager
from consistency_policy.rtc_manager import (
    RTCConfig,
    ActionChunkManager,
)


# ===================== 数据结构 =====================

@dataclass
class ReplayRecord:
    """单次回放记录"""
    batch_id: int
    batch_start_frame: int
    batch_end_frame: int
    scheduled_start: float
    scheduled_end: float
    target_joints: np.ndarray       # 目标关节位置
    actual_joints: np.ndarray       # 实际关节位置
    target_gripper: float
    actual_gripper: float
    timestamp: float
    inference_delay: int = 0        # RTC 的 d 值 (传统模式为 0)


@dataclass
class ReplayMetrics:
    """回放指标"""
    mode: str                       # "traditional" 或 "rtc"
    
    # 基础统计
    total_batches: int = 0
    total_frames: int = 0
    total_duration: float = 0.0
    
    # 跟踪误差
    joint_errors: List[float] = field(default_factory=list)
    joint_error_mean: float = 0.0
    joint_error_max: float = 0.0
    joint_error_std: float = 0.0
    
    gripper_errors: List[float] = field(default_factory=list)
    gripper_error_mean: float = 0.0
    gripper_error_max: float = 0.0
    
    # Chunk 重叠
    overlap_times: List[float] = field(default_factory=list)  # ms
    overlap_mean: float = 0.0
    overlap_max: float = 0.0
    positive_overlap_count: int = 0
    
    # 动作连续性
    action_discontinuities: List[float] = field(default_factory=list)
    discontinuity_mean: float = 0.0
    discontinuity_max: float = 0.0
    
    # RTC 特有
    d_values: List[int] = field(default_factory=list)
    d_mean: float = 0.0
    splice_count: int = 0
    
    # 推理延迟
    simulated_inference_times: List[float] = field(default_factory=list)
    inference_time_mean: float = 0.0
    inference_time_p95: float = 0.0


# ===================== RTCReplayComparator =====================

class RTCReplayComparator:
    """RTC 回放对比分析器"""
    
    def __init__(
        self,
        demo_path: str,
        model: str = "X5",
        interface: str = "can0",
        control_frequency: float = 200.0,
        speed_scale: float = 0.3,
        prediction_horizon: int = 16,
        execute_horizon: int = 8,
        action_horizon: int = 8,
        inference_delay_mean: float = 0.08,
        inference_delay_std: float = 0.02,
        # RTC 特有参数
        rtc_min_d: int = 2,
        rtc_max_d: int = 5,
        rtc_soft_mask_schedule: str = "exp",
        rtc_soft_mask_decay_rate: float = 2.0,
        rtc_enable_soft_masking: bool = True,
        verbose: bool = True,
    ):
        self.demo_path = os.path.expanduser(demo_path)
        self.model = model
        self.interface = interface
        self.control_frequency = control_frequency
        self.speed_scale = speed_scale
        self.prediction_horizon = prediction_horizon
        self.execute_horizon = execute_horizon
        self.action_horizon = action_horizon
        self.inference_delay_mean = inference_delay_mean
        self.inference_delay_std = inference_delay_std
        
        # RTC 特有参数
        self.rtc_min_d = rtc_min_d
        self.rtc_max_d = rtc_max_d
        self.rtc_soft_mask_schedule = rtc_soft_mask_schedule
        self.rtc_soft_mask_decay_rate = rtc_soft_mask_decay_rate
        self.rtc_enable_soft_masking = rtc_enable_soft_masking
        
        self.verbose = verbose
        
        self.controller_manager: Optional[Arx5JointControllerManager] = None
        self.controller = None
        self.actions: Optional[np.ndarray] = None
        self.obs: Optional[dict] = None
        self.T = 0
        
        # 结果
        self.traditional_metrics: Optional[ReplayMetrics] = None
        self.rtc_metrics: Optional[ReplayMetrics] = None
        self.traditional_records: List[ReplayRecord] = []
        self.rtc_records: List[ReplayRecord] = []
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    # ========= 初始化 =========
    
    def load_demo(self, traj_idx: int = 0) -> np.ndarray:
        """加载 demo 数据"""
        self.log(f"\n[1] 加载 demo 数据: {self.demo_path}")
        
        setup_rlft()
        from diffusion_policy.utils import load_traj_hdf5
        
        raw_data = load_traj_hdf5(self.demo_path, num_traj=traj_idx + 1)
        traj_key = f"traj_{traj_idx}"
        traj = raw_data[traj_key]
        
        self.actions = traj['actions']
        self.obs = traj['obs']
        self.T = len(self.actions)
        
        self.log(f"  轨迹长度: {self.T} 帧")
        self.log(f"  动作维度: {self.actions.shape}")
        self.log(f"  关节范围: [{self.actions[:, :6].min():.3f}, {self.actions[:, :6].max():.3f}]")
        
        return self.actions
    
    def connect_robot(self) -> bool:
        """连接机械臂"""
        self.log(f"\n[2] 连接机械臂 ({self.model} @ {self.interface})...")
        
        self.controller_manager = Arx5JointControllerManager(
            model=self.model,
            interface=self.interface,
            frequency=self.control_frequency,
            verbose=self.verbose,
        )
        self.controller_manager.start()
        self.controller = self.controller_manager.controller
        
        state = self.controller.get_state()
        self.log(f"  当前关节位置: {state['joint_pos']}")
        self.log(f"  ✓ 机械臂连接成功 (PID: {self.controller.pid})")
        
        return True
    
    def move_to_start(self, duration: float = 3.0) -> bool:
        """移动到起始位置"""
        self.log(f"\n[3] 移动到起始位置...")
        
        start_action = self.actions[0]
        target_joints = start_action[:6]
        target_gripper = float(start_action[6])
        
        self.log(f"  目标关节: {target_joints}")
        self.log(f"  移动时长: {duration:.1f}s")
        
        self.controller.servoL(
            joint_pos=target_joints,
            gripper_pos=target_gripper,
            duration=duration,
        )
        
        time.sleep(duration + 0.5)
        
        state = self.controller.get_state()
        error = np.linalg.norm(state['joint_pos'] - target_joints)
        self.log(f"  位置误差: {error:.4f} rad")
        
        return error < 0.1
    
    def return_to_home(self, duration: float = 3.0) -> bool:
        """安全回到 home 位置"""
        self.log(f"\n[安全复位] 回到 home 位置...")
        
        # home 位置 (全零)
        home_joints = np.zeros(6)
        home_gripper = 0.0
        
        self.log(f"  目标: home 位置")
        self.log(f"  移动时长: {duration:.1f}s")
        
        try:
            self.controller.servoL(
                joint_pos=home_joints,
                gripper_pos=home_gripper,
                duration=duration,
            )
            
            time.sleep(duration + 0.5)
            
            state = self.controller.get_state()
            error = np.linalg.norm(state['joint_pos'] - home_joints)
            self.log(f"  位置误差: {error:.4f} rad")
            self.log(f"  ✓ 已安全回到 home 位置")
            
            return error < 0.1
        except Exception as e:
            self.log(f"  ⚠ 复位失败: {e}")
            return False
    
    def cleanup(self, safe_return: bool = True):
        """清理资源"""
        self.log(f"\n[清理]...")
        
        if self.controller_manager is not None:
            # 先安全回到 home 位置
            if safe_return and self.controller is not None:
                try:
                    self.return_to_home(duration=3.0)
                except Exception as e:
                    self.log(f"  ⚠ 安全复位异常: {e}")
            
            self.controller_manager.stop()
            self.controller_manager = None
            self.controller = None
        self.log("  ✓ 清理完成")
    
    # ========= 模拟推理延迟 =========
    
    def simulate_inference_delay(self) -> float:
        """模拟推理延迟"""
        delay = np.random.normal(self.inference_delay_mean, self.inference_delay_std)
        return max(0.04, delay)  # 最小 40ms
    
    # ========= 传统模式回放 =========
    
    def replay_traditional(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        batch_size: int = 8,
    ) -> Tuple[ReplayMetrics, List[ReplayRecord]]:
        """
        传统模式回放
        
        使用 add_waypoint + update_trajectory，模拟推理延迟后调度。
        """
        if end_frame is None:
            end_frame = self.T
        
        self.log(f"\n[Traditional Mode] 开始回放...")
        self.log(f"  帧范围: [{start_frame}, {end_frame})")
        self.log(f"  batch_size (action_horizon): {batch_size}")
        self.log(f"  推理延迟: {self.inference_delay_mean*1000:.0f} ± {self.inference_delay_std*1000:.0f} ms")
        
        metrics = ReplayMetrics(mode="traditional")
        records: List[ReplayRecord] = []
        
        # 时间间隔
        dt = (1.0 / 30.0) / self.speed_scale
        
        start_time = time.time()
        current_frame = start_frame
        batch_id = 0
        prev_scheduled_end = None
        prev_last_action = None
        
        try:
            while current_frame < end_frame:
                batch_start_time = time.time()
                
                # 模拟推理延迟
                inference_delay = self.simulate_inference_delay()
                metrics.simulated_inference_times.append(inference_delay)
                time.sleep(inference_delay)
                
                infer_end_time = time.time()
                
                # 计算自适应延迟 (传统模式)
                if len(metrics.simulated_inference_times) >= 3:
                    adaptive_delay = np.percentile(metrics.simulated_inference_times, 95) + 0.01
                else:
                    adaptive_delay = 0.08
                
                # 调度开始时间
                schedule_start = infer_end_time + adaptive_delay
                
                # 确定本批次帧范围
                batch_end = min(current_frame + batch_size, end_frame)
                batch_frames = batch_end - current_frame
                
                # 添加航点
                for i, frame_idx in enumerate(range(current_frame, batch_end)):
                    action = self.actions[frame_idx]
                    target_joints = action[:6]
                    target_gripper = float(action[6])
                    target_time = schedule_start + (i + 1) * dt
                    
                    self.controller.add_waypoint(
                        joint_pos=target_joints,
                        gripper_pos=target_gripper,
                        target_time=target_time,
                    )
                
                # 触发轨迹更新
                self.controller.update_trajectory()
                
                scheduled_end = schedule_start + batch_frames * dt
                
                # 计算重叠
                if prev_scheduled_end is not None:
                    overlap = (prev_scheduled_end - schedule_start) * 1000  # ms
                    metrics.overlap_times.append(overlap)
                
                # 计算动作不连续性
                curr_first_action = self.actions[current_frame][:6]
                if prev_last_action is not None:
                    discontinuity = np.linalg.norm(curr_first_action - prev_last_action)
                    metrics.action_discontinuities.append(discontinuity)
                
                prev_scheduled_end = scheduled_end
                prev_last_action = self.actions[batch_end - 1][:6].copy()
                
                # 等待执行并采样状态
                batch_duration = batch_frames * dt
                time.sleep(batch_duration * 0.9)
                
                state = self.controller.get_state()
                target_action = self.actions[batch_end - 1]
                
                record = ReplayRecord(
                    batch_id=batch_id,
                    batch_start_frame=current_frame,
                    batch_end_frame=batch_end,
                    scheduled_start=schedule_start,
                    scheduled_end=scheduled_end,
                    target_joints=target_action[:6].copy(),
                    actual_joints=state['joint_pos'].copy(),
                    target_gripper=float(target_action[6]),
                    actual_gripper=state['gripper_pos'],
                    timestamp=state['timestamp'],
                    inference_delay=0,
                )
                records.append(record)
                
                # 计算误差
                joint_error = np.linalg.norm(record.actual_joints - record.target_joints)
                gripper_error = abs(record.actual_gripper - record.target_gripper)
                metrics.joint_errors.append(joint_error)
                metrics.gripper_errors.append(gripper_error)
                
                # 等待剩余时间
                time.sleep(batch_duration * 0.15)
                
                # 进度
                if self.verbose and batch_id % 5 == 0:
                    progress = (batch_end - start_frame) / (end_frame - start_frame) * 100
                    self.log(f"    [Traditional] 进度: {progress:.1f}% | 帧: {batch_end}/{end_frame}")
                
                current_frame = batch_end
                batch_id += 1
            
            metrics.total_duration = time.time() - start_time
            
        except KeyboardInterrupt:
            self.log("  ⚠ 回放被中断")
        
        # 计算统计
        metrics.total_batches = len(records)
        metrics.total_frames = end_frame - start_frame
        
        if metrics.joint_errors:
            metrics.joint_error_mean = np.mean(metrics.joint_errors)
            metrics.joint_error_max = np.max(metrics.joint_errors)
            metrics.joint_error_std = np.std(metrics.joint_errors)
        
        if metrics.gripper_errors:
            metrics.gripper_error_mean = np.mean(metrics.gripper_errors)
            metrics.gripper_error_max = np.max(metrics.gripper_errors)
        
        if metrics.overlap_times:
            metrics.overlap_mean = np.mean(metrics.overlap_times)
            metrics.overlap_max = np.max(metrics.overlap_times)
            metrics.positive_overlap_count = sum(1 for o in metrics.overlap_times if o > 0)
        
        if metrics.action_discontinuities:
            metrics.discontinuity_mean = np.mean(metrics.action_discontinuities)
            metrics.discontinuity_max = np.max(metrics.action_discontinuities)
        
        if metrics.simulated_inference_times:
            metrics.inference_time_mean = np.mean(metrics.simulated_inference_times)
            metrics.inference_time_p95 = np.percentile(metrics.simulated_inference_times, 95)
        
        self.log(f"  ✓ Traditional 回放完成! 时长: {metrics.total_duration:.2f}s")
        
        return metrics, records
    
    # ========= RTC 模式回放 =========
    
    def replay_rtc(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        soft_mask_schedule: Optional[str] = None,
    ) -> Tuple[ReplayMetrics, List[ReplayRecord]]:
        """
        RTC 模式回放
        
        使用 ActionChunkManager 进行 chunk 管理和 soft masking。
        """
        if end_frame is None:
            end_frame = self.T
        
        # 使用实例参数或传入参数
        schedule = soft_mask_schedule or self.rtc_soft_mask_schedule
        
        self.log(f"\n[RTC Mode] 开始回放...")
        self.log(f"  帧范围: [{start_frame}, {end_frame})")
        self.log(f"  execute_horizon (s): {self.execute_horizon}")
        self.log(f"  soft_mask_schedule: {schedule}")
        self.log(f"  soft_mask_decay_rate: {self.rtc_soft_mask_decay_rate}")
        self.log(f"  d 值范围: [{self.rtc_min_d}, {self.rtc_max_d}]")
        self.log(f"  enable_soft_masking: {self.rtc_enable_soft_masking}")
        
        # 创建 RTC 管理器
        rtc_config = RTCConfig(
            prediction_horizon=self.prediction_horizon,
            execute_horizon=self.execute_horizon,
            action_dim=7,
            action_dt=(1.0 / 30.0) / self.speed_scale,
            min_inference_delay_steps=self.rtc_min_d,
            max_inference_delay_steps=self.rtc_max_d,
            soft_mask_schedule=schedule,
            soft_mask_decay_rate=self.rtc_soft_mask_decay_rate,
            enable_soft_masking=self.rtc_enable_soft_masking,
        )
        rtc_manager = ActionChunkManager(rtc_config, verbose=False)
        
        metrics = ReplayMetrics(mode="rtc")
        records: List[ReplayRecord] = []
        
        # 时间间隔
        dt = (1.0 / 30.0) / self.speed_scale
        
        start_time = time.time()
        current_frame = start_frame
        batch_id = 0
        prev_scheduled_end = None
        prev_last_action = None
        
        try:
            while current_frame < end_frame:
                batch_start_time = time.time()
                
                # 模拟推理延迟
                inference_delay = self.simulate_inference_delay()
                metrics.simulated_inference_times.append(inference_delay)
                time.sleep(inference_delay)
                
                infer_end_time = time.time()
                
                # 准备动作序列 (pred_horizon 长度)
                action_start = min(current_frame, self.T - self.prediction_horizon)
                action_seq = self.actions[action_start:action_start + self.prediction_horizon].copy()
                if len(action_seq) < self.prediction_horizon:
                    padding = np.tile(action_seq[-1:], (self.prediction_horizon - len(action_seq), 1))
                    action_seq = np.vstack([action_seq, padding])
                
                # RTC 提交 chunk (自动 soft masking)
                chunk = rtc_manager.submit_new_chunk(
                    action_seq=action_seq,
                    obs_time=batch_start_time,
                    inference_time=inference_delay,
                    current_time=infer_end_time,
                )
                
                metrics.d_values.append(chunk.inference_delay)
                
                # 获取调度动作
                actions, timestamps, schedule_start, scheduled_end = rtc_manager.get_scheduled_actions(
                    start_time=infer_end_time
                )
                
                # 添加航点到控制器
                for i, (action, target_time) in enumerate(zip(actions, timestamps)):
                    target_joints = action[:6]
                    target_gripper = float(action[6])
                    
                    self.controller.add_waypoint(
                        joint_pos=target_joints,
                        gripper_pos=target_gripper,
                        target_time=target_time,
                    )
                
                # 触发轨迹更新
                self.controller.update_trajectory()
                
                # 计算重叠
                if prev_scheduled_end is not None:
                    overlap = (prev_scheduled_end - schedule_start) * 1000  # ms
                    metrics.overlap_times.append(overlap)
                
                # 计算动作不连续性 (使用 RTC 处理后的动作)
                curr_first_action = actions[0][:6]
                if prev_last_action is not None:
                    discontinuity = np.linalg.norm(curr_first_action - prev_last_action)
                    metrics.action_discontinuities.append(discontinuity)
                
                prev_scheduled_end = scheduled_end
                prev_last_action = actions[-1][:6].copy()
                
                # 等待执行并采样状态
                batch_duration = len(actions) * dt
                time.sleep(batch_duration * 0.9)
                
                state = self.controller.get_state()
                
                # 确定批次结束帧
                batch_end = min(current_frame + self.execute_horizon, end_frame)
                target_action = self.actions[min(batch_end - 1, self.T - 1)]
                
                record = ReplayRecord(
                    batch_id=batch_id,
                    batch_start_frame=current_frame,
                    batch_end_frame=batch_end,
                    scheduled_start=schedule_start,
                    scheduled_end=scheduled_end,
                    target_joints=target_action[:6].copy(),
                    actual_joints=state['joint_pos'].copy(),
                    target_gripper=float(target_action[6]),
                    actual_gripper=state['gripper_pos'],
                    timestamp=state['timestamp'],
                    inference_delay=chunk.inference_delay,
                )
                records.append(record)
                
                # 计算误差
                joint_error = np.linalg.norm(record.actual_joints - record.target_joints)
                gripper_error = abs(record.actual_gripper - record.target_gripper)
                metrics.joint_errors.append(joint_error)
                metrics.gripper_errors.append(gripper_error)
                
                # 等待剩余时间
                time.sleep(batch_duration * 0.15)
                
                # 进度
                if self.verbose and batch_id % 5 == 0:
                    progress = (batch_end - start_frame) / (end_frame - start_frame) * 100
                    self.log(f"    [RTC] 进度: {progress:.1f}% | 帧: {batch_end}/{end_frame} | d={chunk.inference_delay}")
                
                current_frame = batch_end
                batch_id += 1
            
            metrics.total_duration = time.time() - start_time
            metrics.splice_count = rtc_manager.get_statistics()['splice_count']
            
        except KeyboardInterrupt:
            self.log("  ⚠ 回放被中断")
        
        # 计算统计
        metrics.total_batches = len(records)
        metrics.total_frames = end_frame - start_frame
        
        if metrics.joint_errors:
            metrics.joint_error_mean = np.mean(metrics.joint_errors)
            metrics.joint_error_max = np.max(metrics.joint_errors)
            metrics.joint_error_std = np.std(metrics.joint_errors)
        
        if metrics.gripper_errors:
            metrics.gripper_error_mean = np.mean(metrics.gripper_errors)
            metrics.gripper_error_max = np.max(metrics.gripper_errors)
        
        if metrics.overlap_times:
            metrics.overlap_mean = np.mean(metrics.overlap_times)
            metrics.overlap_max = np.max(metrics.overlap_times)
            metrics.positive_overlap_count = sum(1 for o in metrics.overlap_times if o > 0)
        
        if metrics.action_discontinuities:
            metrics.discontinuity_mean = np.mean(metrics.action_discontinuities)
            metrics.discontinuity_max = np.max(metrics.action_discontinuities)
        
        if metrics.simulated_inference_times:
            metrics.inference_time_mean = np.mean(metrics.simulated_inference_times)
            metrics.inference_time_p95 = np.percentile(metrics.simulated_inference_times, 95)
        
        if metrics.d_values:
            metrics.d_mean = np.mean(metrics.d_values)
        
        self.log(f"  ✓ RTC 回放完成! 时长: {metrics.total_duration:.2f}s")
        
        return metrics, records
    
    # ========= RTC 流水线模式回放 =========
    
    def replay_rtc_pipeline(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        soft_mask_schedule: Optional[str] = None,
    ) -> Tuple[ReplayMetrics, List[ReplayRecord]]:
        """
        RTC 流水线模式回放
        
        关键改进：在执行当前 chunk 时，提前准备下一个 chunk，
        实现推理和执行的流水线并行，消除 chunk 间隙。
        
        Timeline:
            chunk1 执行: |==========|
            chunk2 推理:       |===|
            chunk2 执行:           |==========|  (无间隙衔接)
        """
        if end_frame is None:
            end_frame = self.T
        
        schedule = soft_mask_schedule or self.rtc_soft_mask_schedule
        
        self.log(f"\n[RTC Pipeline Mode] 开始回放...")
        self.log(f"  帧范围: [{start_frame}, {end_frame})")
        self.log(f"  execute_horizon (s): {self.execute_horizon}")
        self.log(f"  soft_mask_schedule: {schedule}")
        self.log(f"  d 值范围: [{self.rtc_min_d}, {self.rtc_max_d}]")
        self.log(f"  流水线模式: 推理与执行并行")
        
        # 创建 RTC 管理器
        rtc_config = RTCConfig(
            prediction_horizon=self.prediction_horizon,
            execute_horizon=self.execute_horizon,
            action_dim=7,
            action_dt=(1.0 / 30.0) / self.speed_scale,
            min_inference_delay_steps=self.rtc_min_d,
            max_inference_delay_steps=self.rtc_max_d,
            soft_mask_schedule=schedule,
            soft_mask_decay_rate=self.rtc_soft_mask_decay_rate,
            enable_soft_masking=self.rtc_enable_soft_masking,
        )
        rtc_manager = ActionChunkManager(rtc_config, verbose=False)
        
        metrics = ReplayMetrics(mode="rtc_pipeline")
        records: List[ReplayRecord] = []
        
        dt = (1.0 / 30.0) / self.speed_scale
        batch_duration = self.execute_horizon * dt
        
        start_time = time.time()
        current_frame = start_frame
        batch_id = 0
        prev_scheduled_end = None
        prev_last_action = None
        
        # 预计算推理开始时间点 (相对于 chunk 开始的偏移)
        # 在 chunk 执行到一半时开始下一轮推理
        inference_start_offset = batch_duration * 0.5
        
        try:
            # ===== 第一个 chunk (冷启动) =====
            first_batch_time = time.time()
            
            # 第一次推理（完整等待）
            inference_delay = self.simulate_inference_delay()
            metrics.simulated_inference_times.append(inference_delay)
            time.sleep(inference_delay)
            
            infer_end_time = time.time()
            
            # 准备第一个动作序列
            action_start = min(current_frame, self.T - self.prediction_horizon)
            action_seq = self.actions[action_start:action_start + self.prediction_horizon].copy()
            if len(action_seq) < self.prediction_horizon:
                padding = np.tile(action_seq[-1:], (self.prediction_horizon - len(action_seq), 1))
                action_seq = np.vstack([action_seq, padding])
            
            chunk = rtc_manager.submit_new_chunk(
                action_seq=action_seq,
                obs_time=first_batch_time,
                inference_time=inference_delay,
                current_time=infer_end_time,
            )
            metrics.d_values.append(chunk.inference_delay)
            
            actions, timestamps, schedule_start, scheduled_end = rtc_manager.get_scheduled_actions(
                start_time=infer_end_time
            )
            
            # 发送第一个 chunk
            for action, target_time in zip(actions, timestamps):
                self.controller.add_waypoint(
                    joint_pos=action[:6],
                    gripper_pos=float(action[6]),
                    target_time=target_time,
                )
            self.controller.update_trajectory()
            
            prev_scheduled_end = scheduled_end
            prev_last_action = actions[-1][:6].copy()
            
            # 记录第一个 chunk
            batch_end = min(current_frame + self.execute_horizon, end_frame)
            current_frame = batch_end
            batch_id = 1
            
            # ===== 流水线循环 =====
            while current_frame < end_frame:
                chunk_start_time = time.time()
                
                # 计算需要等待多久再开始推理
                # 目标：在上一个 chunk 执行完成前完成推理
                time_since_last_update = chunk_start_time - infer_end_time
                wait_before_inference = max(0, inference_start_offset - time_since_last_update)
                
                if wait_before_inference > 0:
                    time.sleep(wait_before_inference)
                
                # 开始推理
                inference_start = time.time()
                inference_delay = self.simulate_inference_delay()
                metrics.simulated_inference_times.append(inference_delay)
                time.sleep(inference_delay)
                infer_end_time = time.time()
                
                # 准备动作序列
                action_start = min(current_frame, self.T - self.prediction_horizon)
                action_seq = self.actions[action_start:action_start + self.prediction_horizon].copy()
                if len(action_seq) < self.prediction_horizon:
                    padding = np.tile(action_seq[-1:], (self.prediction_horizon - len(action_seq), 1))
                    action_seq = np.vstack([action_seq, padding])
                
                # 提交 chunk
                chunk = rtc_manager.submit_new_chunk(
                    action_seq=action_seq,
                    obs_time=inference_start,
                    inference_time=inference_delay,
                    current_time=infer_end_time,
                )
                metrics.d_values.append(chunk.inference_delay)
                
                # 获取调度动作
                actions, timestamps, schedule_start, scheduled_end = rtc_manager.get_scheduled_actions(
                    start_time=infer_end_time
                )
                
                # 发送航点
                for action, target_time in zip(actions, timestamps):
                    self.controller.add_waypoint(
                        joint_pos=action[:6],
                        gripper_pos=float(action[6]),
                        target_time=target_time,
                    )
                self.controller.update_trajectory()
                
                # 计算重叠
                if prev_scheduled_end is not None:
                    overlap = (prev_scheduled_end - schedule_start) * 1000
                    metrics.overlap_times.append(overlap)
                
                # 计算动作不连续性
                curr_first_action = actions[0][:6]
                if prev_last_action is not None:
                    discontinuity = np.linalg.norm(curr_first_action - prev_last_action)
                    metrics.action_discontinuities.append(discontinuity)
                
                prev_scheduled_end = scheduled_end
                prev_last_action = actions[-1][:6].copy()
                
                # 等待一小段时间后采样状态
                time.sleep(batch_duration * 0.3)
                state = self.controller.get_state()
                
                batch_end = min(current_frame + self.execute_horizon, end_frame)
                target_action = self.actions[min(batch_end - 1, self.T - 1)]
                
                record = ReplayRecord(
                    batch_id=batch_id,
                    batch_start_frame=current_frame,
                    batch_end_frame=batch_end,
                    scheduled_start=schedule_start,
                    scheduled_end=scheduled_end,
                    target_joints=target_action[:6].copy(),
                    actual_joints=state['joint_pos'].copy(),
                    target_gripper=float(target_action[6]),
                    actual_gripper=state['gripper_pos'],
                    timestamp=state['timestamp'],
                    inference_delay=chunk.inference_delay,
                )
                records.append(record)
                
                joint_error = np.linalg.norm(record.actual_joints - record.target_joints)
                gripper_error = abs(record.actual_gripper - record.target_gripper)
                metrics.joint_errors.append(joint_error)
                metrics.gripper_errors.append(gripper_error)
                
                if self.verbose and batch_id % 5 == 0:
                    progress = (batch_end - start_frame) / (end_frame - start_frame) * 100
                    self.log(f"    [Pipeline] 进度: {progress:.1f}% | d={chunk.inference_delay}")
                
                current_frame = batch_end
                batch_id += 1
            
            metrics.total_duration = time.time() - start_time
            metrics.splice_count = rtc_manager.get_statistics()['splice_count']
            
        except KeyboardInterrupt:
            self.log("  ⚠ 回放被中断")
        
        # 计算统计
        metrics.total_batches = len(records)
        metrics.total_frames = end_frame - start_frame
        
        if metrics.joint_errors:
            metrics.joint_error_mean = np.mean(metrics.joint_errors)
            metrics.joint_error_max = np.max(metrics.joint_errors)
            metrics.joint_error_std = np.std(metrics.joint_errors)
        
        if metrics.gripper_errors:
            metrics.gripper_error_mean = np.mean(metrics.gripper_errors)
            metrics.gripper_error_max = np.max(metrics.gripper_errors)
        
        if metrics.overlap_times:
            metrics.overlap_mean = np.mean(metrics.overlap_times)
            metrics.overlap_max = np.max(metrics.overlap_times)
            metrics.positive_overlap_count = sum(1 for o in metrics.overlap_times if o > 0)
        
        if metrics.action_discontinuities:
            metrics.discontinuity_mean = np.mean(metrics.action_discontinuities)
            metrics.discontinuity_max = np.max(metrics.action_discontinuities)
        
        if metrics.simulated_inference_times:
            metrics.inference_time_mean = np.mean(metrics.simulated_inference_times)
            metrics.inference_time_p95 = np.percentile(metrics.simulated_inference_times, 95)
        
        if metrics.d_values:
            metrics.d_mean = np.mean(metrics.d_values)
        
        self.log(f"  ✓ RTC Pipeline 回放完成! 时长: {metrics.total_duration:.2f}s")
        
        return metrics, records
    
    # ========= 对比分析 =========
    
    def compare_results(self) -> Dict:
        """对比分析两种模式的结果"""
        if self.traditional_metrics is None or self.rtc_metrics is None:
            self.log("请先运行两种模式的回放")
            return {}
        
        trad = self.traditional_metrics
        rtc = self.rtc_metrics
        
        self.log("\n" + "=" * 70)
        self.log("对比分析结果")
        self.log("=" * 70)
        
        self.log(f"\n{'指标':<30} {'Traditional':>15} {'RTC':>15} {'改进':>15}")
        self.log("-" * 75)
        
        # 跟踪误差
        self.log(f"{'关节误差均值 (rad)':<30} {trad.joint_error_mean:>15.4f} {rtc.joint_error_mean:>15.4f} {trad.joint_error_mean - rtc.joint_error_mean:>+15.4f}")
        self.log(f"{'关节误差最大 (rad)':<30} {trad.joint_error_max:>15.4f} {rtc.joint_error_max:>15.4f} {trad.joint_error_max - rtc.joint_error_max:>+15.4f}")
        self.log(f"{'关节误差标准差':<30} {trad.joint_error_std:>15.4f} {rtc.joint_error_std:>15.4f} {trad.joint_error_std - rtc.joint_error_std:>+15.4f}")
        
        # 夹爪误差
        self.log(f"{'夹爪误差均值':<30} {trad.gripper_error_mean:>15.4f} {rtc.gripper_error_mean:>15.4f} {trad.gripper_error_mean - rtc.gripper_error_mean:>+15.4f}")
        
        # Chunk 重叠
        self.log(f"{'Chunk 重叠均值 (ms)':<30} {trad.overlap_mean:>15.1f} {rtc.overlap_mean:>15.1f} {trad.overlap_mean - rtc.overlap_mean:>+15.1f}")
        self.log(f"{'Chunk 重叠最大 (ms)':<30} {trad.overlap_max:>15.1f} {rtc.overlap_max:>15.1f} {trad.overlap_max - rtc.overlap_max:>+15.1f}")
        self.log(f"{'重叠次数':<30} {trad.positive_overlap_count:>15} {rtc.positive_overlap_count:>15} {trad.positive_overlap_count - rtc.positive_overlap_count:>+15}")
        
        # 动作连续性
        self.log(f"{'动作不连续均值':<30} {trad.discontinuity_mean:>15.4f} {rtc.discontinuity_mean:>15.4f} {trad.discontinuity_mean - rtc.discontinuity_mean:>+15.4f}")
        self.log(f"{'动作不连续最大':<30} {trad.discontinuity_max:>15.4f} {rtc.discontinuity_max:>15.4f} {trad.discontinuity_max - rtc.discontinuity_max:>+15.4f}")
        
        # 时间
        self.log(f"{'总时长 (s)':<30} {trad.total_duration:>15.2f} {rtc.total_duration:>15.2f} {'-':>15}")
        
        # RTC 特有
        self.log(f"\nRTC 特有指标:")
        self.log(f"  d 值均值: {rtc.d_mean:.2f}")
        self.log(f"  d 值分布: {rtc.d_values[:10]}...")
        self.log(f"  拼接次数: {rtc.splice_count}")
        
        # 计算改进百分比
        comparison = {
            'joint_error_improvement': (trad.joint_error_mean - rtc.joint_error_mean) / trad.joint_error_mean * 100 if trad.joint_error_mean > 0 else 0,
            'overlap_improvement': (trad.overlap_mean - rtc.overlap_mean) / trad.overlap_mean * 100 if trad.overlap_mean > 0 else 0,
            'discontinuity_improvement': (trad.discontinuity_mean - rtc.discontinuity_mean) / trad.discontinuity_mean * 100 if trad.discontinuity_mean > 0 else 0,
        }
        
        self.log(f"\n改进百分比:")
        self.log(f"  关节误差: {comparison['joint_error_improvement']:+.1f}%")
        self.log(f"  重叠时间: {comparison['overlap_improvement']:+.1f}%")
        self.log(f"  动作不连续: {comparison['discontinuity_improvement']:+.1f}%")
        
        self.log("\n" + "=" * 70)
        
        return comparison
    
    # ========= 保存结果 =========
    
    def save_results(self, output_path: str):
        """保存结果到 NPZ 文件"""
        if self.traditional_metrics is None and self.rtc_metrics is None:
            self.log("无结果可保存")
            return
        
        data = {
            'config': {
                'prediction_horizon': self.prediction_horizon,
                'execute_horizon': self.execute_horizon,
                'action_horizon': self.action_horizon,
                'inference_delay_mean': self.inference_delay_mean,
                'inference_delay_std': self.inference_delay_std,
                'speed_scale': self.speed_scale,
            }
        }
        
        if self.traditional_metrics:
            data['trad_joint_errors'] = np.array(self.traditional_metrics.joint_errors)
            data['trad_overlap_times'] = np.array(self.traditional_metrics.overlap_times)
            data['trad_discontinuities'] = np.array(self.traditional_metrics.action_discontinuities)
            data['trad_inference_times'] = np.array(self.traditional_metrics.simulated_inference_times)
        
        if self.rtc_metrics:
            data['rtc_joint_errors'] = np.array(self.rtc_metrics.joint_errors)
            data['rtc_overlap_times'] = np.array(self.rtc_metrics.overlap_times)
            data['rtc_discontinuities'] = np.array(self.rtc_metrics.action_discontinuities)
            data['rtc_inference_times'] = np.array(self.rtc_metrics.simulated_inference_times)
            data['rtc_d_values'] = np.array(self.rtc_metrics.d_values)
        
        np.savez(output_path, **data)
        self.log(f"\n结果已保存: {output_path}")
    
    # ========= 完整测试流程 =========
    
    def run_comparison(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        traj_idx: int = 0,
        rtc_only: bool = False,
        traditional_only: bool = False,
        pipeline_mode: bool = False,
    ) -> Dict:
        """
        运行完整的对比测试
        
        Args:
            start_frame: 起始帧
            end_frame: 结束帧
            traj_idx: 轨迹索引
            rtc_only: 仅运行 RTC 模式
            traditional_only: 仅运行传统模式
            pipeline_mode: 使用流水线模式 (推理与执行并行)
        """
        self.log("\n" + "=" * 70)
        mode_str = "RTC Pipeline" if pipeline_mode else "RTC"
        self.log(f"{mode_str} vs Traditional 轨迹回放对比测试")
        self.log("=" * 70)
        
        # 加载数据
        self.load_demo(traj_idx=traj_idx)
        
        if end_frame is None:
            end_frame = min(self.T, start_frame + 300)  # 默认回放 300 帧
        
        # 连接机械臂
        self.connect_robot()
        
        try:
            # 运行传统模式
            if not rtc_only:
                self.log("\n" + "-" * 50)
                self.log("Phase 1: Traditional Mode")
                self.log("-" * 50)
                
                self.move_to_start()
                input("\n按 Enter 开始 Traditional 模式回放，Ctrl+C 取消...")
                
                self.traditional_metrics, self.traditional_records = self.replay_traditional(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    batch_size=self.action_horizon,
                )
            
            # 运行 RTC 模式
            if not traditional_only:
                self.log("\n" + "-" * 50)
                if pipeline_mode:
                    self.log("Phase 2: RTC Pipeline Mode (推理与执行并行)")
                else:
                    self.log("Phase 2: RTC Mode")
                self.log("-" * 50)
                
                self.move_to_start()
                input(f"\n按 Enter 开始 {mode_str} 模式回放，Ctrl+C 取消...")
                
                if pipeline_mode:
                    self.rtc_metrics, self.rtc_records = self.replay_rtc_pipeline(
                        start_frame=start_frame,
                        end_frame=end_frame,
                    )
                else:
                    self.rtc_metrics, self.rtc_records = self.replay_rtc(
                        start_frame=start_frame,
                        end_frame=end_frame,
                    )
            
            # 对比分析
            if self.traditional_metrics and self.rtc_metrics:
                comparison = self.compare_results()
            else:
                comparison = {}
            
            return comparison
        
        finally:
            self.cleanup()


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(description="RTC 轨迹回放对比测试")
    parser.add_argument("-d", "--demo", required=True, help="Demo 数据路径")
    parser.add_argument("--traj-idx", type=int, default=0, help="轨迹索引")
    parser.add_argument("-m", "--model", default="X5", help="机械臂型号")
    parser.add_argument("-i", "--interface", default="can0", help="CAN 接口")
    parser.add_argument("-f", "--frequency", type=float, default=200.0, help="控制器频率 Hz")
    parser.add_argument("-s", "--speed", type=float, default=0.3, help="速度缩放 (默认 0.3)")
    
    # 模式控制
    parser.add_argument("--rtc-only", action="store_true", help="仅运行 RTC 模式")
    parser.add_argument("--traditional-only", action="store_true", help="仅运行传统模式")
    parser.add_argument("--pipeline", action="store_true", 
                        help="使用流水线模式 (推理与执行并行，消除 chunk 间隙)")
    
    # Horizon 参数 (公平对比)
    parser.add_argument("--prediction-horizon", type=int, default=16, help="H 值")
    parser.add_argument("--execute-horizon", type=int, default=8, help="s 值 (RTC)")
    parser.add_argument("--action-horizon", type=int, default=8, help="传统模式 action_horizon")
    
    # 推理延迟
    parser.add_argument("--inference-delay", type=float, default=0.08, help="推理延迟均值 (s)")
    parser.add_argument("--inference-std", type=float, default=0.02, help="推理延迟标准差 (s)")
    
    # RTC 特有参数
    parser.add_argument("--rtc-min-d", type=int, default=2, 
                        help="最小推理延迟步数 (d 的下界)")
    parser.add_argument("--rtc-max-d", type=int, default=5, 
                        help="最大推理延迟步数 (d 的上界)")
    parser.add_argument("--rtc-mask-schedule", type=str, default="exp", 
                        choices=["exp", "linear"],
                        help="软掩码衰减类型: exp (指数) 或 linear (线性)")
    parser.add_argument("--rtc-decay-rate", type=float, default=2.0, 
                        help="指数衰减速率 α (仅 exp 模式有效)")
    parser.add_argument("--no-soft-mask", action="store_true", 
                        help="禁用软掩码，使用硬切换")
    
    # 帧范围
    parser.add_argument("--start", type=int, default=0, help="起始帧")
    parser.add_argument("--end", type=int, default=None, help="结束帧")
    
    # 输出
    parser.add_argument("-o", "--output", default=None, help="结果保存路径 (.npz)")
    parser.add_argument("-v", "--verbose", action="store_true", default=True, help="详细输出")
    
    args = parser.parse_args()
    
    # 限制速度范围
    args.speed = np.clip(args.speed, 0.1, 1.0)
    
    print("=" * 70)
    print("RTC 轨迹回放对比测试")
    print("=" * 70)
    print(f"\n⚠ 安全提示:")
    print(f"  - 速度: {args.speed:.0%}")
    print(f"  - 推理延迟: {args.inference_delay*1000:.0f} ± {args.inference_std*1000:.0f} ms")
    print(f"  - 确保机械臂周围无障碍物")
    print(f"  - 随时准备按 Ctrl+C 停止")
    
    print(f"\n配置:")
    print(f"  H (prediction_horizon): {args.prediction_horizon}")
    print(f"  s (execute_horizon): {args.execute_horizon}")
    print(f"  action_horizon: {args.action_horizon}")
    print(f"\nRTC 参数:")
    print(f"  d 值范围: [{args.rtc_min_d}, {args.rtc_max_d}]")
    print(f"  soft_mask_schedule: {args.rtc_mask_schedule}")
    print(f"  soft_mask_decay_rate: {args.rtc_decay_rate}")
    print(f"  enable_soft_masking: {not args.no_soft_mask}")
    print(f"  流水线模式: {args.pipeline}")
    
    comparator = RTCReplayComparator(
        demo_path=args.demo,
        model=args.model,
        interface=args.interface,
        control_frequency=args.frequency,
        speed_scale=args.speed,
        prediction_horizon=args.prediction_horizon,
        execute_horizon=args.execute_horizon,
        action_horizon=args.action_horizon,
        inference_delay_mean=args.inference_delay,
        inference_delay_std=args.inference_std,
        rtc_min_d=args.rtc_min_d,
        rtc_max_d=args.rtc_max_d,
        rtc_soft_mask_schedule=args.rtc_mask_schedule,
        rtc_soft_mask_decay_rate=args.rtc_decay_rate,
        rtc_enable_soft_masking=not args.no_soft_mask,
        verbose=args.verbose,
    )
    
    try:
        comparison = comparator.run_comparison(
            start_frame=args.start,
            end_frame=args.end,
            traj_idx=args.traj_idx,
            rtc_only=args.rtc_only,
            traditional_only=args.traditional_only,
            pipeline_mode=args.pipeline,
        )
        
        # 保存结果
        if args.output:
            comparator.save_results(args.output)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            mode_suffix = "_pipeline" if args.pipeline else ""
            output_path = f"./rtc_comparison{mode_suffix}_{timestamp}.npz"
            comparator.save_results(output_path)
        
        print("\n" + "=" * 70)
        if comparison:
            if comparison.get('joint_error_improvement', 0) > 0:
                mode_name = "RTC Pipeline" if args.pipeline else "RTC"
                print(f"✓ {mode_name} 模式表现更好!")
            else:
                print("⚠ 传统模式表现更好 (检查参数配置)")
        print("=" * 70)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠ 测试被中断")
        return 1
    
    except Exception as e:
        print(f"\n\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
