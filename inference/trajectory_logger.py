#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹数据记录器

用于记录推理过程中的动作和状态数据，以便对比分析不同方案的行为差异。

记录内容:
- 推理输出 (raw_actions): 策略模型的原始输出
- 插值动作 (interp_actions): 插值到 500Hz 后的动作序列
- 执行动作 (executed_actions): 实际发送给机器人的动作 (滤波+安全限制后)
- 机器人状态 (robot_states): 实际的关节位置和夹爪位置
- 时间戳 (timestamps): 各事件的精确时间

用法:
    logger = TrajectoryLogger("single_process")  # 或 "multi_process"
    logger.start_logging()
    
    # 在推理时
    logger.log_inference(chunk_id, raw_actions, interp_actions)
    
    # 在控制循环中
    logger.log_control_step(step, raw_action, filtered_action, executed_action, robot_state)
    
    # 结束
    logger.stop_logging()
    logger.save("trajectory_data.npz")
"""

import os
import time
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import threading


@dataclass
class InferenceRecord:
    """单次推理的记录"""
    chunk_id: int
    timestamp: float
    inference_time: float  # 推理耗时
    raw_actions: np.ndarray  # 策略原始输出 (act_horizon, 7)
    interp_actions: np.ndarray  # 插值后动作 (~133, 7)
    current_position: np.ndarray  # 推理时的机器人位置 (7,)


@dataclass
class ControlRecord:
    """单步控制的记录"""
    step: int
    timestamp: float
    chunk_id: int  # 属于哪个推理 chunk
    buffer_idx: int  # 在缓冲区中的索引
    raw_action: np.ndarray  # 从缓冲区读取的动作
    filtered_action: np.ndarray  # EMA 滤波后
    executed_action: np.ndarray  # 安全限制后 (实际发送)
    robot_state: np.ndarray  # 机器人实际状态 (7,)


class TrajectoryLogger:
    """
    轨迹数据记录器
    
    线程安全，可在多线程/多进程环境中使用。
    """
    
    def __init__(self, name: str = "trajectory", max_records: int = 100000):
        """
        Args:
            name: 记录名称 (用于区分不同方案)
            max_records: 最大记录数量
        """
        self.name = name
        self.max_records = max_records
        self._lock = threading.Lock()
        
        # 记录数据
        self.inference_records: List[InferenceRecord] = []
        self.control_records: List[ControlRecord] = []
        
        # 状态
        self._logging = False
        self._start_time = 0.0
        self._current_chunk_id = -1
    
    def start_logging(self):
        """开始记录"""
        with self._lock:
            self.inference_records.clear()
            self.control_records.clear()
            self._logging = True
            self._start_time = time.time()
            self._current_chunk_id = -1
        print(f"[TrajectoryLogger:{self.name}] 开始记录")
    
    def stop_logging(self):
        """停止记录"""
        with self._lock:
            self._logging = False
        print(f"[TrajectoryLogger:{self.name}] 停止记录, "
              f"推理记录: {len(self.inference_records)}, "
              f"控制记录: {len(self.control_records)}")
    
    @property
    def is_logging(self) -> bool:
        return self._logging
    
    def log_inference(
        self,
        raw_actions: np.ndarray,
        interp_actions: np.ndarray,
        current_position: np.ndarray,
        inference_time: float,
    ):
        """
        记录一次推理
        
        Args:
            raw_actions: 策略原始输出 (act_horizon, 7)
            interp_actions: 插值后动作 (~133, 7)
            current_position: 推理时机器人位置 (7,)
            inference_time: 推理耗时 (秒)
        """
        if not self._logging:
            return
        
        with self._lock:
            if len(self.inference_records) >= self.max_records:
                return
            
            self._current_chunk_id += 1
            
            record = InferenceRecord(
                chunk_id=self._current_chunk_id,
                timestamp=time.time() - self._start_time,
                inference_time=inference_time,
                raw_actions=raw_actions.copy(),
                interp_actions=interp_actions.copy(),
                current_position=current_position.copy(),
            )
            self.inference_records.append(record)
    
    def log_control_step(
        self,
        step: int,
        buffer_idx: int,
        raw_action: np.ndarray,
        filtered_action: np.ndarray,
        executed_action: np.ndarray,
        robot_state: np.ndarray,
    ):
        """
        记录一步控制
        
        Args:
            step: 控制步数
            buffer_idx: 在动作缓冲区中的索引
            raw_action: 从缓冲区读取的原始动作
            filtered_action: EMA 滤波后的动作
            executed_action: 安全限制后的动作 (实际发送)
            robot_state: 机器人实际状态 (7,)
        """
        if not self._logging:
            return
        
        with self._lock:
            if len(self.control_records) >= self.max_records:
                return
            
            record = ControlRecord(
                step=step,
                timestamp=time.time() - self._start_time,
                chunk_id=self._current_chunk_id,
                buffer_idx=buffer_idx,
                raw_action=raw_action.copy(),
                filtered_action=filtered_action.copy(),
                executed_action=executed_action.copy(),
                robot_state=robot_state.copy(),
            )
            self.control_records.append(record)
    
    def get_current_chunk_id(self) -> int:
        """获取当前 chunk ID"""
        with self._lock:
            return self._current_chunk_id
    
    def save(self, filepath: str):
        """
        保存记录到文件
        
        Args:
            filepath: 输出文件路径 (.npz)
        """
        with self._lock:
            # 转换推理记录
            if self.inference_records:
                inf_data = {
                    'inf_chunk_ids': np.array([r.chunk_id for r in self.inference_records]),
                    'inf_timestamps': np.array([r.timestamp for r in self.inference_records]),
                    'inf_times': np.array([r.inference_time for r in self.inference_records]),
                    'inf_raw_actions': np.array([r.raw_actions for r in self.inference_records]),
                    'inf_interp_actions_list': np.array([r.interp_actions for r in self.inference_records], dtype=object),
                    'inf_current_positions': np.array([r.current_position for r in self.inference_records]),
                }
            else:
                inf_data = {}
            
            # 转换控制记录
            if self.control_records:
                ctrl_data = {
                    'ctrl_steps': np.array([r.step for r in self.control_records]),
                    'ctrl_timestamps': np.array([r.timestamp for r in self.control_records]),
                    'ctrl_chunk_ids': np.array([r.chunk_id for r in self.control_records]),
                    'ctrl_buffer_idxs': np.array([r.buffer_idx for r in self.control_records]),
                    'ctrl_raw_actions': np.array([r.raw_action for r in self.control_records]),
                    'ctrl_filtered_actions': np.array([r.filtered_action for r in self.control_records]),
                    'ctrl_executed_actions': np.array([r.executed_action for r in self.control_records]),
                    'ctrl_robot_states': np.array([r.robot_state for r in self.control_records]),
                }
            else:
                ctrl_data = {}
            
            # 元数据
            meta = {
                'name': self.name,
                'num_inferences': len(self.inference_records),
                'num_control_steps': len(self.control_records),
            }
            
            # 保存
            np.savez_compressed(filepath, **inf_data, **ctrl_data, **meta)
            print(f"[TrajectoryLogger:{self.name}] 已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> Dict:
        """
        加载记录文件
        
        Returns:
            包含所有数据的字典
        """
        data = np.load(filepath, allow_pickle=True)
        return dict(data)


def compare_trajectories(single_process_file: str, multi_process_file: str):
    """
    对比两种方案的轨迹数据
    
    Args:
        single_process_file: 单进程方案的数据文件
        multi_process_file: 多进程方案的数据文件
    """
    print("\n" + "=" * 60)
    print("轨迹对比分析")
    print("=" * 60)
    
    # 加载数据
    sp = TrajectoryLogger.load(single_process_file)
    mp = TrajectoryLogger.load(multi_process_file)
    
    print(f"\n单进程方案: {sp.get('name', 'unknown')}")
    print(f"  推理次数: {sp.get('num_inferences', 0)}")
    print(f"  控制步数: {sp.get('num_control_steps', 0)}")
    
    print(f"\n多进程方案: {mp.get('name', 'unknown')}")
    print(f"  推理次数: {mp.get('num_inferences', 0)}")
    print(f"  控制步数: {mp.get('num_control_steps', 0)}")
    
    # 对比推理时间
    if 'inf_times' in sp and 'inf_times' in mp:
        sp_inf_times = sp['inf_times']
        mp_inf_times = mp['inf_times']
        print(f"\n推理时间对比:")
        print(f"  单进程: 平均 {sp_inf_times.mean()*1000:.1f}ms, 最大 {sp_inf_times.max()*1000:.1f}ms")
        print(f"  多进程: 平均 {mp_inf_times.mean()*1000:.1f}ms, 最大 {mp_inf_times.max()*1000:.1f}ms")
    
    # 对比执行动作
    if 'ctrl_executed_actions' in sp and 'ctrl_executed_actions' in mp:
        sp_actions = sp['ctrl_executed_actions']
        mp_actions = mp['ctrl_executed_actions']
        
        # 计算帧间变化
        sp_deltas = np.abs(np.diff(sp_actions, axis=0))
        mp_deltas = np.abs(np.diff(mp_actions, axis=0))
        
        print(f"\n执行动作帧间变化对比:")
        print(f"  单进程: 关节平均 {sp_deltas[:, :6].mean():.6f} rad, 最大 {sp_deltas[:, :6].max():.4f} rad")
        print(f"  多进程: 关节平均 {mp_deltas[:, :6].mean():.6f} rad, 最大 {mp_deltas[:, :6].max():.4f} rad")
        print(f"  单进程: 夹爪平均 {sp_deltas[:, 6].mean():.6f} m, 最大 {sp_deltas[:, 6].max():.4f} m")
        print(f"  多进程: 夹爪平均 {mp_deltas[:, 6].mean():.6f} m, 最大 {mp_deltas[:, 6].max():.4f} m")
    
    # 对比 chunk 切换处的跳变
    if 'ctrl_chunk_ids' in sp and 'ctrl_executed_actions' in sp:
        print(f"\nChunk 切换分析 (单进程):")
        _analyze_chunk_transitions(sp)
    
    if 'ctrl_chunk_ids' in mp and 'ctrl_executed_actions' in mp:
        print(f"\nChunk 切换分析 (多进程):")
        _analyze_chunk_transitions(mp)
    
    # 位置跟踪误差
    if 'ctrl_executed_actions' in sp and 'ctrl_robot_states' in sp:
        sp_tracking = np.abs(sp['ctrl_executed_actions'] - sp['ctrl_robot_states'])
        print(f"\n位置跟踪误差 (单进程):")
        print(f"  关节平均: {sp_tracking[:, :6].mean():.6f} rad")
        print(f"  夹爪平均: {sp_tracking[:, 6].mean():.6f} m")
    
    if 'ctrl_executed_actions' in mp and 'ctrl_robot_states' in mp:
        mp_tracking = np.abs(mp['ctrl_executed_actions'] - mp['ctrl_robot_states'])
        print(f"\n位置跟踪误差 (多进程):")
        print(f"  关节平均: {mp_tracking[:, :6].mean():.6f} rad")
        print(f"  夹爪平均: {mp_tracking[:, 6].mean():.6f} m")


def _analyze_chunk_transitions(data: Dict):
    """分析 chunk 切换处的跳变"""
    chunk_ids = data['ctrl_chunk_ids']
    actions = data['ctrl_executed_actions']
    
    # 找到 chunk 切换点
    transitions = np.where(np.diff(chunk_ids) > 0)[0]
    
    if len(transitions) == 0:
        print("  没有检测到 chunk 切换")
        return
    
    # 计算切换处的跳变
    jumps = []
    for idx in transitions:
        if idx + 1 < len(actions):
            delta = np.abs(actions[idx + 1] - actions[idx])
            jumps.append(delta)
    
    jumps = np.array(jumps)
    
    print(f"  切换次数: {len(transitions)}")
    print(f"  切换处关节跳变: 平均 {jumps[:, :6].mean():.6f} rad, 最大 {jumps[:, :6].max():.4f} rad")
    print(f"  切换处夹爪跳变: 平均 {jumps[:, 6].mean():.6f} m, 最大 {jumps[:, 6].max():.4f} m")
    
    # 对比非切换处
    non_transitions = np.ones(len(actions) - 1, dtype=bool)
    non_transitions[transitions] = False
    
    non_jumps = np.abs(np.diff(actions, axis=0)[non_transitions])
    if len(non_jumps) > 0:
        print(f"  非切换帧关节变化: 平均 {non_jumps[:, :6].mean():.6f} rad, 最大 {non_jumps[:, :6].max():.4f} rad")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="轨迹对比分析")
    parser.add_argument("--single", "-s", type=str, help="单进程方案数据文件")
    parser.add_argument("--multi", "-m", type=str, help="多进程方案数据文件")
    
    args = parser.parse_args()
    
    if args.single and args.multi:
        compare_trajectories(args.single, args.multi)
    else:
        print("用法: python trajectory_logger.py -s single.npz -m multi.npz")
