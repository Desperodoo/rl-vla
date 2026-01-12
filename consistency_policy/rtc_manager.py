#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Chunking (RTC) Manager

基于 Physical Intelligence 的 RTC 论文实现 (arXiv:2506.07339)
解决异步 action chunking 的时序问题。

核心概念:
- H (prediction_horizon): 每次预测的动作序列长度
- s (execute_horizon): 每个 chunk 实际执行的动作数
- d (inference_delay): 推理期间执行的动作数 (d = ceil(inference_time / action_dt))

算法核心:
1. 异步推理: 在执行当前 chunk 时开始下一次推理
2. Chunk 拼接: new_chunk 到达时, 实际执行 old_chunk[:d] + new_chunk[d:s]
3. Soft Masking: 对重叠区域使用软掩码进行平滑过渡

参考:
- https://github.com/Physical-Intelligence/real-time-chunking-kinetix
- https://arxiv.org/abs/2506.07339
"""

import os
import sys
import time
import threading
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

# ===================== 配置 =====================

@dataclass
class RTCConfig:
    """RTC 配置参数"""
    
    # 基础参数
    prediction_horizon: int = 16       # H: 策略预测的动作序列长度
    execute_horizon: int = 8           # s: 每个 chunk 执行的动作数
    action_dim: int = 7                # 动作维度 (6 关节 + 1 夹爪)
    action_dt: float = 1.0 / 30.0      # 动作时间间隔 (33.3ms @ 30Hz)
    
    # 推理延迟估计
    inference_delay_buffer_size: int = 20   # 推理时间历史缓冲区大小
    inference_delay_percentile: float = 95  # 用于估计延迟的百分位数
    inference_delay_margin: float = 0.01    # 延迟估计的额外余量 (秒)
    min_inference_delay_steps: int = 2      # 最小 d 值 (保守估计)
    max_inference_delay_steps: int = 5      # 最大 d 值 (避免过长)
    
    # Soft Masking 配置
    soft_mask_schedule: str = "exp"    # 软掩码类型: "exp" (指数衰减) 或 "linear"
    soft_mask_decay_rate: float = 2.0  # 指数衰减速率
    
    # 启用/禁用功能
    enable_soft_masking: bool = True   # 是否启用软掩码插值
    enable_async_inference: bool = True  # 是否启用异步推理
    
    def __post_init__(self):
        """验证配置"""
        assert self.execute_horizon <= self.prediction_horizon, \
            f"execute_horizon ({self.execute_horizon}) 必须 <= prediction_horizon ({self.prediction_horizon})"
        assert self.min_inference_delay_steps < self.execute_horizon, \
            f"min_inference_delay_steps ({self.min_inference_delay_steps}) 必须 < execute_horizon ({self.execute_horizon})"


# ===================== Soft Masking 工具函数 =====================

def compute_soft_mask_weights(
    H: int,
    s: int, 
    d: int,
    schedule: str = "exp",
    decay_rate: float = 2.0,
) -> np.ndarray:
    """
    计算 Soft Masking 权重 (对应论文 Eq. 5)
    
    W[i] = 1           if i < d       (冻结区域 - 保留旧 chunk)
    W[i] = decay(i)    if d <= i < H-s (重叠区域 - 混合)
    W[i] = 0           if i >= H-s     (新区域 - 使用新 chunk)
    
    注意: 权重用于旧 chunk，即 W=1 表示完全使用旧 chunk，W=0 表示完全使用新 chunk
    
    Args:
        H: prediction_horizon
        s: execute_horizon
        d: inference_delay (以 action steps 为单位)
        schedule: "exp" (指数衰减) 或 "linear"
        decay_rate: 指数衰减速率
        
    Returns:
        weights: (H,) 软掩码权重
    """
    weights = np.zeros(H, dtype=np.float32)
    
    # 冻结区域 (前 d 个动作)
    weights[:d] = 1.0
    
    # 重叠区域 (d 到 H-s)
    overlap_start = d
    overlap_end = H - s
    
    if overlap_end > overlap_start:
        overlap_length = overlap_end - overlap_start
        
        if schedule == "exp":
            # 指数衰减: exp(-decay_rate * relative_pos)
            for i in range(overlap_start, overlap_end):
                relative_pos = (i - overlap_start) / overlap_length
                weights[i] = np.exp(-decay_rate * relative_pos)
        elif schedule == "linear":
            # 线性衰减
            for i in range(overlap_start, overlap_end):
                relative_pos = (i - overlap_start) / overlap_length
                weights[i] = 1.0 - relative_pos
        else:
            raise ValueError(f"未知的 schedule 类型: {schedule}")
    
    # 新区域 (后 s 个动作) - 已经是 0
    
    return weights


def apply_soft_mask(
    old_chunk: np.ndarray,
    new_chunk: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    应用软掩码混合两个 chunk
    
    result[i] = weights[i] * old_chunk[i] + (1 - weights[i]) * new_chunk[i]
    
    Args:
        old_chunk: (H, action_dim) 旧的动作 chunk
        new_chunk: (H, action_dim) 新的动作 chunk
        weights: (H,) 软掩码权重 (用于旧 chunk)
        
    Returns:
        blended: (H, action_dim) 混合后的动作 chunk
    """
    # 扩展权重维度以支持广播
    weights_expanded = weights[:, np.newaxis]  # (H, 1)
    
    blended = weights_expanded * old_chunk + (1 - weights_expanded) * new_chunk
    return blended


# ===================== Action Chunk 数据结构 =====================

@dataclass
class ActionChunk:
    """动作 chunk 数据结构"""
    
    actions: np.ndarray              # (H, action_dim) 动作序列
    creation_time: float             # chunk 创建时间 (系统时间)
    scheduled_start_time: float      # chunk 计划开始执行的时间
    execute_horizon: int             # 实际执行的动作数 (s)
    inference_delay: int             # 推理延迟 (d)
    observation_time: float          # 用于生成此 chunk 的观测时间
    chunk_id: int = 0                # chunk ID (用于调试)
    
    @property
    def scheduled_end_time(self) -> float:
        """计划结束时间"""
        return self.scheduled_start_time + self.execute_horizon * (1.0 / 30.0)
    
    @property
    def action_times(self) -> np.ndarray:
        """每个动作的计划执行时间"""
        dt = 1.0 / 30.0
        return self.scheduled_start_time + np.arange(len(self.actions)) * dt


class ChunkState(Enum):
    """Chunk 执行状态"""
    PENDING = "pending"      # 等待执行
    EXECUTING = "executing"  # 正在执行
    COMPLETED = "completed"  # 执行完成


# ===================== RTC Manager 核心类 =====================

class ActionChunkManager:
    """
    RTC 动作 Chunk 管理器
    
    负责:
    1. 管理异步推理和动作执行的时序
    2. 实现 chunk 拼接逻辑
    3. 应用 soft masking 平滑过渡
    
    使用方法:
    ```python
    manager = ActionChunkManager(config)
    
    # 启动控制循环时
    manager.start()
    
    # 每次推理完成时
    manager.submit_new_chunk(action_seq, obs_time, inference_time)
    
    # 获取当前应该执行的动作
    action = manager.get_action(current_time)
    
    # 停止
    manager.stop()
    ```
    """
    
    def __init__(
        self,
        config: RTCConfig,
        verbose: bool = False,
    ):
        self.config = config
        self.verbose = verbose
        
        # 推理延迟估计
        self.inference_times: deque = deque(maxlen=config.inference_delay_buffer_size)
        self.current_d: int = config.min_inference_delay_steps
        
        # Chunk 管理
        self.current_chunk: Optional[ActionChunk] = None
        self.next_chunk: Optional[ActionChunk] = None
        self.chunk_counter: int = 0
        
        # 执行状态
        self.execution_start_time: Optional[float] = None
        self.execution_index: int = 0  # 当前执行到 chunk 的第几个动作
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 调试信息
        self.splice_history: List[Dict] = []
        
    # ========= 推理延迟估计 =========
    
    def update_inference_time(self, inference_time: float):
        """
        更新推理时间统计
        
        Args:
            inference_time: 最近一次推理耗时 (秒)
        """
        self.inference_times.append(inference_time)
    
    def estimate_inference_delay(self) -> int:
        """
        估计推理延迟 (以 action steps 为单位)
        
        d = ceil(inference_time / action_dt)
        
        Returns:
            d: 推理期间执行的动作步数
        """
        if len(self.inference_times) < 3:
            return self.config.min_inference_delay_steps
        
        # 使用百分位数估计
        p = self.config.inference_delay_percentile
        estimated_time = np.percentile(list(self.inference_times), p)
        estimated_time += self.config.inference_delay_margin
        
        # 转换为 action steps
        d = int(np.ceil(estimated_time / self.config.action_dt))
        
        # 限制范围
        d = max(self.config.min_inference_delay_steps, d)
        d = min(self.config.max_inference_delay_steps, d)
        
        self.current_d = d
        return d
    
    # ========= Chunk 管理 =========
    
    def submit_new_chunk(
        self,
        action_seq: np.ndarray,
        obs_time: float,
        inference_time: float,
        current_time: Optional[float] = None,
    ) -> ActionChunk:
        """
        提交新的动作 chunk
        
        实现 RTC 的核心拼接逻辑:
        1. 更新推理时间统计
        2. 计算新 chunk 的调度时间
        3. 如果存在当前 chunk，应用 soft masking
        
        Args:
            action_seq: (H, action_dim) 策略输出的动作序列
            obs_time: 用于推理的观测获取时间
            inference_time: 本次推理耗时
            current_time: 当前时间 (可选，默认使用 time.time())
            
        Returns:
            chunk: 创建的 ActionChunk
        """
        if current_time is None:
            current_time = time.time()
        
        # 更新推理时间统计
        self.update_inference_time(inference_time)
        
        # 估计推理延迟
        d = self.estimate_inference_delay()
        s = self.config.execute_horizon
        H = self.config.prediction_horizon
        
        with self._lock:
            self.chunk_counter += 1
            
            # 确定新 chunk 的开始时间
            # RTC 的关键: 新 chunk 应该从 "现在" 开始调度
            # 但前 d 个动作已经被旧 chunk 覆盖
            scheduled_start = current_time
            
            # 如果有当前 chunk，需要进行拼接
            if self.current_chunk is not None and self.config.enable_soft_masking:
                # 获取旧 chunk (用于混合)
                old_actions = self.current_chunk.actions.copy()
                new_actions = action_seq.copy()
                
                # 计算软掩码权重
                weights = compute_soft_mask_weights(
                    H=H, s=s, d=d,
                    schedule=self.config.soft_mask_schedule,
                    decay_rate=self.config.soft_mask_decay_rate,
                )
                
                # 时间对齐: 需要将 old_chunk 和 new_chunk 对齐到同一时间基准
                # old_chunk 的时间基准是 old_scheduled_start
                # new_chunk 的时间基准是 current_time (scheduled_start)
                
                # 计算时间偏移 (以 action steps 为单位)
                old_start = self.current_chunk.scheduled_start_time
                time_offset = (current_time - old_start) / self.config.action_dt
                time_offset_int = int(np.round(time_offset))
                
                if self.verbose:
                    print(f"[RTC] Chunk #{self.chunk_counter}: "
                          f"d={d}, time_offset={time_offset:.2f} steps")
                
                # 对齐旧 chunk 的动作 (shift by time_offset)
                if 0 < time_offset_int < H:
                    # 旧 chunk 需要前移 time_offset_int 步
                    aligned_old = np.zeros_like(old_actions)
                    remaining = H - time_offset_int
                    aligned_old[:remaining] = old_actions[time_offset_int:]
                    # 后面的用新 chunk 填充
                    aligned_old[remaining:] = new_actions[remaining:]
                else:
                    aligned_old = old_actions
                
                # 应用软掩码混合
                blended_actions = apply_soft_mask(aligned_old, new_actions, weights)
                
                # 记录拼接信息
                self.splice_history.append({
                    'chunk_id': self.chunk_counter,
                    'time': current_time,
                    'd': d,
                    's': s,
                    'time_offset': time_offset,
                    'weights_sum': weights.sum(),
                })
            else:
                # 没有旧 chunk，直接使用新动作
                blended_actions = action_seq.copy()
            
            # 创建新 chunk
            new_chunk = ActionChunk(
                actions=blended_actions,
                creation_time=current_time,
                scheduled_start_time=scheduled_start,
                execute_horizon=s,
                inference_delay=d,
                observation_time=obs_time,
                chunk_id=self.chunk_counter,
            )
            
            # 更新状态: 新 chunk 变成当前 chunk
            self.current_chunk = new_chunk
            self.execution_index = 0
            
            return new_chunk
    
    def get_scheduled_actions(
        self,
        start_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        获取用于调度的动作序列
        
        返回应该发送给控制器的动作和对应的时间戳。
        
        Args:
            start_time: 调度开始时间 (可选)
            
        Returns:
            actions: (s, action_dim) 要执行的动作序列
            timestamps: (s,) 每个动作的目标执行时间
            chunk_start: chunk 开始时间
            chunk_end: chunk 结束时间
        """
        with self._lock:
            if self.current_chunk is None:
                raise RuntimeError("没有可用的 chunk")
            
            chunk = self.current_chunk
            s = chunk.execute_horizon
            dt = self.config.action_dt
            
            if start_time is None:
                start_time = chunk.scheduled_start_time
            
            # 取前 s 个动作
            actions = chunk.actions[:s].copy()
            
            # 计算时间戳
            timestamps = start_time + np.arange(s) * dt
            
            chunk_start = timestamps[0]
            chunk_end = timestamps[-1]
            
            return actions, timestamps, chunk_start, chunk_end
    
    def get_action_at_time(self, t: float) -> Optional[np.ndarray]:
        """
        获取指定时间点的动作 (用于高频控制器)
        
        Args:
            t: 目标时间
            
        Returns:
            action: (action_dim,) 动作，或 None 如果没有有效 chunk
        """
        with self._lock:
            if self.current_chunk is None:
                return None
            
            chunk = self.current_chunk
            dt = self.config.action_dt
            
            # 计算时间索引
            idx = (t - chunk.scheduled_start_time) / dt
            idx_int = int(np.floor(idx))
            
            # 边界检查
            if idx_int < 0:
                return chunk.actions[0].copy()
            elif idx_int >= len(chunk.actions):
                return chunk.actions[-1].copy()
            
            # 线性插值
            alpha = idx - idx_int
            if idx_int + 1 < len(chunk.actions):
                action = (1 - alpha) * chunk.actions[idx_int] + \
                         alpha * chunk.actions[idx_int + 1]
            else:
                action = chunk.actions[idx_int].copy()
            
            return action
    
    def should_start_inference(self, t: float) -> bool:
        """
        判断是否应该开始新的推理
        
        RTC 的调度策略: 在执行完 d 个动作后开始新的推理
        
        Args:
            t: 当前时间
            
        Returns:
            should_start: 是否应该开始推理
        """
        with self._lock:
            if self.current_chunk is None:
                return True  # 没有 chunk 时立即推理
            
            chunk = self.current_chunk
            d = chunk.inference_delay
            dt = self.config.action_dt
            
            # 计算已执行的动作数
            elapsed = t - chunk.scheduled_start_time
            executed_steps = elapsed / dt
            
            # 当执行到第 d 个动作时，应该开始下一次推理
            # 这样当推理完成时，刚好执行完 d 个动作
            if not self.config.enable_async_inference:
                # 同步模式: 每个 chunk 执行完后才推理
                return executed_steps >= chunk.execute_horizon
            else:
                # 异步模式: 执行到第 d 步时开始推理
                return executed_steps >= d
    
    def get_time_until_next_inference(self, t: float) -> float:
        """
        计算距离下一次推理的时间
        
        Args:
            t: 当前时间
            
        Returns:
            wait_time: 等待时间 (秒)
        """
        with self._lock:
            if self.current_chunk is None:
                return 0.0
            
            chunk = self.current_chunk
            d = chunk.inference_delay
            dt = self.config.action_dt
            
            # 下一次推理时间
            next_inference_time = chunk.scheduled_start_time + d * dt
            wait_time = next_inference_time - t
            
            return max(0.0, wait_time)
    
    # ========= 状态查询 =========
    
    @property
    def has_chunk(self) -> bool:
        """是否有可用的 chunk"""
        return self.current_chunk is not None
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with self._lock:
            stats = {
                'chunk_count': self.chunk_counter,
                'current_d': self.current_d,
                'inference_time_mean': np.mean(list(self.inference_times)) if self.inference_times else 0,
                'inference_time_p95': np.percentile(list(self.inference_times), 95) if len(self.inference_times) >= 3 else 0,
                'splice_count': len(self.splice_history),
            }
            
            if self.current_chunk is not None:
                stats['current_chunk_id'] = self.current_chunk.chunk_id
                stats['current_chunk_start'] = self.current_chunk.scheduled_start_time
            
            return stats
    
    def reset(self):
        """重置管理器状态"""
        with self._lock:
            self.current_chunk = None
            self.next_chunk = None
            self.chunk_counter = 0
            self.execution_index = 0
            self.inference_times.clear()
            self.current_d = self.config.min_inference_delay_steps
            self.splice_history.clear()


# ===================== 测试函数 =====================

def test_soft_mask():
    """测试软掩码计算"""
    H, s, d = 16, 8, 3
    
    print(f"\n测试软掩码 (H={H}, s={s}, d={d}):")
    
    for schedule in ["exp", "linear"]:
        weights = compute_soft_mask_weights(H, s, d, schedule=schedule)
        print(f"\n{schedule} schedule:")
        print(f"  Frozen (0:{d}): {weights[:d]}")
        print(f"  Overlap ({d}:{H-s}): {weights[d:H-s]}")
        print(f"  New ({H-s}:{H}): {weights[H-s:]}")


def test_chunk_manager():
    """测试 ActionChunkManager"""
    config = RTCConfig(
        prediction_horizon=16,
        execute_horizon=8,
        action_dim=7,
    )
    
    manager = ActionChunkManager(config, verbose=True)
    
    print("\n测试 ActionChunkManager:")
    
    # 模拟推理序列
    for i in range(5):
        # 模拟动作序列
        action_seq = np.random.randn(16, 7) * 0.1
        
        # 模拟推理时间
        inference_time = np.random.uniform(0.06, 0.1)
        
        t = time.time()
        chunk = manager.submit_new_chunk(
            action_seq=action_seq,
            obs_time=t - inference_time,
            inference_time=inference_time,
            current_time=t,
        )
        
        print(f"\nChunk #{chunk.chunk_id}:")
        print(f"  d={chunk.inference_delay}, s={chunk.execute_horizon}")
        print(f"  scheduled_start={chunk.scheduled_start_time:.3f}")
        print(f"  scheduled_end={chunk.scheduled_end_time:.3f}")
        
        time.sleep(0.1)  # 模拟执行
    
    stats = manager.get_statistics()
    print(f"\n统计信息: {stats}")


if __name__ == "__main__":
    test_soft_mask()
    test_chunk_manager()
