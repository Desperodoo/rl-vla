#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTC 控制器对比测试

验证 RTC (Real-Time Chunking) 在控制器级别的性能:
1. ActionChunkManager 的 chunk 拼接逻辑
2. Soft Masking 权重计算和应用
3. d 值 (推理延迟) 自适应估计
4. RTC vs 传统模式的响应特性对比

用法:
    # 完整测试 (模拟推理，无真机)
    python -m consistency_policy.tests.test_rtc_controller
    
    # 真机测试
    python -m consistency_policy.tests.test_rtc_controller --real-robot
    
    # 测试不同 soft mask 配置
    python -m consistency_policy.tests.test_rtc_controller --mask-schedule linear

环境:
    conda activate arx-py310
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 添加项目路径
CONSISTENCY_POLICY_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_VLA_PATH = os.path.dirname(CONSISTENCY_POLICY_PATH)
if RL_VLA_PATH not in sys.path:
    sys.path.insert(0, RL_VLA_PATH)

from consistency_policy.rtc_manager import (
    RTCConfig,
    ActionChunkManager,
    compute_soft_mask_weights,
    apply_soft_mask,
)


# ===================== 数据结构 =====================

@dataclass
class ChunkRecord:
    """Chunk 记录"""
    chunk_id: int
    creation_time: float
    scheduled_start: float
    scheduled_end: float
    inference_delay: int  # d 值
    execute_horizon: int  # s 值
    actions: np.ndarray
    obs_time: float
    inference_time: float


@dataclass
class TestMetrics:
    """测试指标"""
    # Chunk 统计
    total_chunks: int = 0
    splice_count: int = 0
    
    # d 值统计
    d_values: List[int] = None
    d_mean: float = 0.0
    d_std: float = 0.0
    
    # 时间统计
    inference_times: List[float] = None
    inference_time_mean: float = 0.0
    inference_time_p95: float = 0.0
    
    # 重叠统计
    overlap_times: List[float] = None  # ms
    overlap_mean: float = 0.0
    overlap_max: float = 0.0
    positive_overlap_count: int = 0  # 有重叠的次数
    
    # 动作连续性
    action_discontinuities: List[float] = None  # 相邻 chunk 的动作跳变
    discontinuity_mean: float = 0.0
    discontinuity_max: float = 0.0
    
    def __post_init__(self):
        if self.d_values is None:
            self.d_values = []
        if self.inference_times is None:
            self.inference_times = []
        if self.overlap_times is None:
            self.overlap_times = []
        if self.action_discontinuities is None:
            self.action_discontinuities = []


# ===================== 测试器 =====================

class RTCControllerTester:
    """RTC 控制器测试器"""
    
    def __init__(
        self,
        prediction_horizon: int = 16,
        execute_horizon: int = 8,
        action_horizon: int = 8,  # 传统模式使用
        action_dim: int = 7,
        action_dt: float = 1.0 / 30.0,
        verbose: bool = True,
    ):
        self.prediction_horizon = prediction_horizon
        self.execute_horizon = execute_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.action_dt = action_dt
        self.verbose = verbose
        
        self.test_results: Dict[str, bool] = {}
        self.default_mask_schedule: str = "exp"  # 默认 soft mask 类型
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    # ========= Soft Masking 测试 =========
    
    def test_soft_mask_weights(self) -> bool:
        """测试软掩码权重计算"""
        self.log("\n" + "=" * 60)
        self.log("测试 1: Soft Masking 权重计算")
        self.log("=" * 60)
        
        try:
            H = self.prediction_horizon  # 16
            s = self.execute_horizon      # 8
            
            for d in [2, 3, 4, 5]:
                self.log(f"\n  d={d} (H={H}, s={s}):")
                
                # 指数衰减
                weights_exp = compute_soft_mask_weights(H, s, d, schedule="exp", decay_rate=2.0)
                # 线性衰减
                weights_linear = compute_soft_mask_weights(H, s, d, schedule="linear")
                
                # 验证权重范围
                assert np.all(weights_exp >= 0) and np.all(weights_exp <= 1), "exp 权重超出 [0,1]"
                assert np.all(weights_linear >= 0) and np.all(weights_linear <= 1), "linear 权重超出 [0,1]"
                
                # 验证分区
                # 冻结区: [0, d) 应该全为 1
                assert np.allclose(weights_exp[:d], 1.0), f"exp 冻结区 [0:{d}) 不全为 1"
                assert np.allclose(weights_linear[:d], 1.0), f"linear 冻结区 [0:{d}) 不全为 1"
                
                # 新区域: [H-s, H) 应该全为 0
                assert np.allclose(weights_exp[H-s:], 0.0), f"exp 新区域 [{H-s}:{H}) 不全为 0"
                assert np.allclose(weights_linear[H-s:], 0.0), f"linear 新区域 [{H-s}:{H}) 不全为 0"
                
                self.log(f"    exp    冻结区: {weights_exp[:d]}")
                self.log(f"    exp    重叠区: {weights_exp[d:H-s]}")
                self.log(f"    linear 冻结区: {weights_linear[:d]}")
                self.log(f"    linear 重叠区: {weights_linear[d:H-s]}")
            
            self.log("\n✓ Soft Masking 权重计算正确")
            return True
            
        except Exception as e:
            self.log(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_soft_mask_blending(self) -> bool:
        """测试软掩码混合"""
        self.log("\n" + "=" * 60)
        self.log("测试 2: Soft Masking 混合")
        self.log("=" * 60)
        
        try:
            H = self.prediction_horizon
            s = self.execute_horizon
            d = 3
            
            # 创建测试数据
            old_chunk = np.zeros((H, self.action_dim))
            new_chunk = np.ones((H, self.action_dim))
            
            weights = compute_soft_mask_weights(H, s, d, schedule="exp")
            blended = apply_soft_mask(old_chunk, new_chunk, weights)
            
            self.log(f"\n  权重分布 (H={H}, s={s}, d={d}):")
            self.log(f"    冻结区 [0:{d}): {weights[:d]}")
            self.log(f"    重叠区 [{d}:{H-s}): {weights[d:H-s]}")
            self.log(f"    新区域 [{H-s}:{H}): {weights[H-s:]}")
            
            # 验证混合结果
            # 冻结区: 应该接近 old_chunk (0)，因为 weights=1 保留旧值
            frozen_mean = np.mean(blended[:d])
            self.log(f"\n  冻结区 [0:{d}) 均值: {frozen_mean:.4f} (期望接近 0)")
            assert frozen_mean < 0.01, "冻结区混合错误"
            
            # 新区域: 应该接近 new_chunk (1)，因为 weights=0 使用新值
            new_mean = np.mean(blended[H-s:])
            self.log(f"  新区域 [{H-s}:{H}) 均值: {new_mean:.4f} (期望接近 1)")
            assert new_mean > 0.99, "新区域混合错误"
            
            # 重叠区: 应该在 0-1 之间渐变
            # 注意: 重叠区 [d, H-s) 的权重从 1 衰减到接近 0
            # blended = old * w + new * (1-w)
            # 当 w=1 时 blended=old=0, 当 w=0 时 blended=new=1
            overlap_values = np.mean(blended[d:H-s], axis=1)
            self.log(f"  重叠区 [{d}:{H-s}) 混合值: {overlap_values}")
            
            # 重叠区应该在 [0, 1] 范围内 (包含边界)
            assert np.all(overlap_values >= 0) and np.all(overlap_values <= 1), "重叠区混合值超出范围"
            
            # 验证单调性 (重叠区应该递增，从 0 向 1 过渡)
            is_monotonic = np.all(np.diff(overlap_values) >= -0.01)  # 允许小误差
            self.log(f"  重叠区单调递增: {is_monotonic}")
            assert is_monotonic, "重叠区应该单调递增"
            
            # 验证过渡的合理性: 至少有一个值在 (0.1, 0.9) 之间
            has_transition = np.any((overlap_values > 0.1) & (overlap_values < 0.9))
            self.log(f"  重叠区有过渡值: {has_transition}")
            
            self.log("\n✓ Soft Masking 混合正确")
            return True
            
        except Exception as e:
            self.log(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========= ActionChunkManager 测试 =========
    
    def test_chunk_manager_basic(self) -> bool:
        """测试 ActionChunkManager 基本功能"""
        self.log("\n" + "=" * 60)
        self.log("测试 3: ActionChunkManager 基本功能")
        self.log("=" * 60)
        
        try:
            config = RTCConfig(
                prediction_horizon=self.prediction_horizon,
                execute_horizon=self.execute_horizon,
                action_dim=self.action_dim,
                action_dt=self.action_dt,
                min_inference_delay_steps=2,
                max_inference_delay_steps=5,
            )
            manager = ActionChunkManager(config, verbose=False)
            
            # 提交第一个 chunk
            action_seq1 = np.random.randn(self.prediction_horizon, self.action_dim) * 0.1
            t1 = time.time()
            chunk1 = manager.submit_new_chunk(
                action_seq=action_seq1,
                obs_time=t1 - 0.08,
                inference_time=0.08,
                current_time=t1,
            )
            
            self.log(f"\n  Chunk #1:")
            self.log(f"    d={chunk1.inference_delay}, s={chunk1.execute_horizon}")
            self.log(f"    scheduled_start={chunk1.scheduled_start_time:.3f}")
            
            assert manager.has_chunk, "应该有可用 chunk"
            assert chunk1.chunk_id == 1, "第一个 chunk ID 应为 1"
            
            # 获取调度动作
            actions, timestamps, start, end = manager.get_scheduled_actions()
            self.log(f"    调度动作数: {len(actions)}")
            self.log(f"    时间范围: {start:.3f} - {end:.3f}")
            
            assert len(actions) == self.execute_horizon, f"应返回 {self.execute_horizon} 个动作"
            assert len(timestamps) == self.execute_horizon, "时间戳数量应匹配"
            
            # 提交第二个 chunk (测试拼接)
            time.sleep(0.1)
            action_seq2 = np.random.randn(self.prediction_horizon, self.action_dim) * 0.1
            t2 = time.time()
            chunk2 = manager.submit_new_chunk(
                action_seq=action_seq2,
                obs_time=t2 - 0.08,
                inference_time=0.08,
                current_time=t2,
            )
            
            self.log(f"\n  Chunk #2:")
            self.log(f"    d={chunk2.inference_delay}, s={chunk2.execute_horizon}")
            
            # 检查统计信息
            stats = manager.get_statistics()
            self.log(f"\n  统计信息:")
            self.log(f"    chunk_count: {stats['chunk_count']}")
            self.log(f"    splice_count: {stats['splice_count']}")
            self.log(f"    current_d: {stats['current_d']}")
            
            assert stats['chunk_count'] == 2, "应有 2 个 chunk"
            assert stats['splice_count'] >= 1, "应有至少 1 次拼接"
            
            self.log("\n✓ ActionChunkManager 基本功能正常")
            return True
            
        except Exception as e:
            self.log(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_inference_delay_adaptation(self) -> bool:
        """测试推理延迟 (d 值) 自适应"""
        self.log("\n" + "=" * 60)
        self.log("测试 4: 推理延迟自适应")
        self.log("=" * 60)
        
        try:
            config = RTCConfig(
                prediction_horizon=self.prediction_horizon,
                execute_horizon=self.execute_horizon,
                action_dim=self.action_dim,
                action_dt=self.action_dt,
                min_inference_delay_steps=2,
                max_inference_delay_steps=5,
                inference_delay_percentile=95,
                inference_delay_margin=0.01,
            )
            manager = ActionChunkManager(config, verbose=False)
            
            d_values = []
            
            # 阶段 1: 稳定的短推理时间 (50ms)
            self.log("\n  阶段 1: 短推理时间 (50ms)")
            for i in range(10):
                inference_time = np.random.normal(0.05, 0.005)  # 50±5ms
                inference_time = max(0.04, inference_time)
                
                action_seq = np.random.randn(self.prediction_horizon, self.action_dim) * 0.1
                t = time.time()
                chunk = manager.submit_new_chunk(
                    action_seq=action_seq,
                    obs_time=t - inference_time,
                    inference_time=inference_time,
                    current_time=t,
                )
                d_values.append(chunk.inference_delay)
            
            d_phase1 = d_values[-1]
            self.log(f"    最终 d 值: {d_phase1}")
            
            # 阶段 2: 增加到长推理时间 (100ms)
            self.log("\n  阶段 2: 长推理时间 (100ms)")
            for i in range(10):
                inference_time = np.random.normal(0.10, 0.01)  # 100±10ms
                inference_time = max(0.08, inference_time)
                
                action_seq = np.random.randn(self.prediction_horizon, self.action_dim) * 0.1
                t = time.time()
                chunk = manager.submit_new_chunk(
                    action_seq=action_seq,
                    obs_time=t - inference_time,
                    inference_time=inference_time,
                    current_time=t,
                )
                d_values.append(chunk.inference_delay)
            
            d_phase2 = d_values[-1]
            self.log(f"    最终 d 值: {d_phase2}")
            
            # 验证自适应行为
            self.log(f"\n  d 值变化: {d_values}")
            
            # 短推理时间应该有较小的 d
            # 长推理时间应该有较大的 d
            assert d_phase2 >= d_phase1, "长推理时间的 d 值应该 >= 短推理时间"
            
            self.log("\n✓ 推理延迟自适应正常")
            return True
            
        except Exception as e:
            self.log(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========= RTC vs 传统模式对比 =========
    
    def simulate_traditional_mode(
        self,
        trajectory: np.ndarray,
        inference_times: List[float],
        eval_dt: float = 0.1,
    ) -> TestMetrics:
        """
        模拟传统模式
        
        Args:
            trajectory: (N, action_dim) 目标轨迹
            inference_times: 模拟的推理时间列表
            eval_dt: 评估循环周期 (默认 100ms = 10Hz)
        """
        metrics = TestMetrics()
        metrics.inference_times = inference_times.copy()
        
        chunks: List[ChunkRecord] = []
        current_time = 0.0
        traj_idx = 0
        
        for i, inference_time in enumerate(inference_times):
            # 模拟观测获取
            obs_time = current_time
            
            # 模拟推理完成
            infer_end_time = obs_time + inference_time
            
            # 传统模式: 使用自适应延迟
            if len(metrics.inference_times[:i+1]) >= 3:
                adaptive_delay = np.percentile(metrics.inference_times[:i+1], 95) + 0.01
            else:
                adaptive_delay = 0.08
            
            start_time = infer_end_time + adaptive_delay
            end_time = start_time + self.action_horizon * self.action_dt
            
            # 提取动作序列
            action_start = min(traj_idx, len(trajectory) - self.prediction_horizon)
            action_seq = trajectory[action_start:action_start + self.prediction_horizon].copy()
            if len(action_seq) < self.prediction_horizon:
                padding = np.tile(action_seq[-1:], (self.prediction_horizon - len(action_seq), 1))
                action_seq = np.vstack([action_seq, padding])
            
            chunk = ChunkRecord(
                chunk_id=i + 1,
                creation_time=infer_end_time,
                scheduled_start=start_time,
                scheduled_end=end_time,
                inference_delay=0,  # 传统模式无 d 值概念
                execute_horizon=self.action_horizon,
                actions=action_seq[:self.action_horizon],
                obs_time=obs_time,
                inference_time=inference_time,
            )
            chunks.append(chunk)
            
            # 计算重叠
            if len(chunks) > 1:
                prev_chunk = chunks[-2]
                overlap = (prev_chunk.scheduled_end - chunk.scheduled_start) * 1000  # ms
                metrics.overlap_times.append(overlap)
                
                # 计算动作不连续性
                # 比较前一个 chunk 最后执行的动作 vs 当前 chunk 第一个动作
                prev_last_action = prev_chunk.actions[-1]
                curr_first_action = chunk.actions[0]
                discontinuity = np.linalg.norm(curr_first_action - prev_last_action)
                metrics.action_discontinuities.append(discontinuity)
            
            # 更新时间
            current_time = obs_time + eval_dt
            traj_idx += int(eval_dt / self.action_dt)
        
        # 计算统计
        metrics.total_chunks = len(chunks)
        metrics.inference_time_mean = np.mean(metrics.inference_times)
        metrics.inference_time_p95 = np.percentile(metrics.inference_times, 95)
        
        if metrics.overlap_times:
            metrics.overlap_mean = np.mean(metrics.overlap_times)
            metrics.overlap_max = np.max(metrics.overlap_times)
            metrics.positive_overlap_count = sum(1 for o in metrics.overlap_times if o > 0)
        
        if metrics.action_discontinuities:
            metrics.discontinuity_mean = np.mean(metrics.action_discontinuities)
            metrics.discontinuity_max = np.max(metrics.action_discontinuities)
        
        return metrics
    
    def simulate_rtc_mode(
        self,
        trajectory: np.ndarray,
        inference_times: List[float],
        eval_dt: float = 0.1,
        soft_mask_schedule: str = "exp",
    ) -> TestMetrics:
        """
        模拟 RTC 模式
        
        Args:
            trajectory: (N, action_dim) 目标轨迹
            inference_times: 模拟的推理时间列表
            eval_dt: 评估循环周期
            soft_mask_schedule: 软掩码类型
        """
        config = RTCConfig(
            prediction_horizon=self.prediction_horizon,
            execute_horizon=self.execute_horizon,
            action_dim=self.action_dim,
            action_dt=self.action_dt,
            min_inference_delay_steps=2,
            max_inference_delay_steps=5,
            soft_mask_schedule=soft_mask_schedule,
            soft_mask_decay_rate=2.0,
            enable_soft_masking=True,
        )
        manager = ActionChunkManager(config, verbose=False)
        
        metrics = TestMetrics()
        metrics.inference_times = inference_times.copy()
        
        chunks: List[ChunkRecord] = []
        current_time = 0.0
        traj_idx = 0
        
        for i, inference_time in enumerate(inference_times):
            # 模拟观测获取
            obs_time = current_time
            
            # 模拟推理完成
            infer_end_time = obs_time + inference_time
            
            # 提取动作序列
            action_start = min(traj_idx, len(trajectory) - self.prediction_horizon)
            action_seq = trajectory[action_start:action_start + self.prediction_horizon].copy()
            if len(action_seq) < self.prediction_horizon:
                padding = np.tile(action_seq[-1:], (self.prediction_horizon - len(action_seq), 1))
                action_seq = np.vstack([action_seq, padding])
            
            # RTC 提交 chunk
            chunk_obj = manager.submit_new_chunk(
                action_seq=action_seq,
                obs_time=obs_time,
                inference_time=inference_time,
                current_time=infer_end_time,
            )
            
            # 获取调度信息
            actions, timestamps, start_time, end_time = manager.get_scheduled_actions(
                start_time=infer_end_time
            )
            
            chunk = ChunkRecord(
                chunk_id=chunk_obj.chunk_id,
                creation_time=infer_end_time,
                scheduled_start=start_time,
                scheduled_end=end_time,
                inference_delay=chunk_obj.inference_delay,
                execute_horizon=chunk_obj.execute_horizon,
                actions=actions,
                obs_time=obs_time,
                inference_time=inference_time,
            )
            chunks.append(chunk)
            metrics.d_values.append(chunk_obj.inference_delay)
            
            # 计算重叠
            if len(chunks) > 1:
                prev_chunk = chunks[-2]
                overlap = (prev_chunk.scheduled_end - chunk.scheduled_start) * 1000  # ms
                metrics.overlap_times.append(overlap)
                
                # 计算动作不连续性 (RTC 的 soft masking 应该减少不连续性)
                prev_last_action = prev_chunk.actions[-1]
                curr_first_action = chunk.actions[0]
                discontinuity = np.linalg.norm(curr_first_action - prev_last_action)
                metrics.action_discontinuities.append(discontinuity)
            
            # 更新时间
            current_time = obs_time + eval_dt
            traj_idx += int(eval_dt / self.action_dt)
        
        # 计算统计
        metrics.total_chunks = len(chunks)
        metrics.splice_count = manager.get_statistics()['splice_count']
        metrics.inference_time_mean = np.mean(metrics.inference_times)
        metrics.inference_time_p95 = np.percentile(metrics.inference_times, 95)
        
        if metrics.d_values:
            metrics.d_mean = np.mean(metrics.d_values)
            metrics.d_std = np.std(metrics.d_values)
        
        if metrics.overlap_times:
            metrics.overlap_mean = np.mean(metrics.overlap_times)
            metrics.overlap_max = np.max(metrics.overlap_times)
            metrics.positive_overlap_count = sum(1 for o in metrics.overlap_times if o > 0)
        
        if metrics.action_discontinuities:
            metrics.discontinuity_mean = np.mean(metrics.action_discontinuities)
            metrics.discontinuity_max = np.max(metrics.action_discontinuities)
        
        return metrics
    
    def test_mode_comparison(
        self,
        num_steps: int = 50,
        inference_mean: float = 0.08,
        inference_std: float = 0.02,
    ) -> bool:
        """对比 RTC 和传统模式"""
        self.log("\n" + "=" * 60)
        self.log("测试 5: RTC vs 传统模式对比")
        self.log("=" * 60)
        
        try:
            # 生成测试轨迹 (平滑正弦波)
            total_frames = num_steps * 3 + self.prediction_horizon
            t = np.linspace(0, 4 * np.pi, total_frames)
            trajectory = np.zeros((total_frames, self.action_dim))
            for dim in range(self.action_dim):
                freq = np.random.uniform(0.5, 2.0)
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.1, 0.3)
                trajectory[:, dim] = amplitude * np.sin(freq * t + phase)
            
            # 生成推理时间
            inference_times = [
                max(0.05, np.random.normal(inference_mean, inference_std))
                for _ in range(num_steps)
            ]
            
            self.log(f"\n  配置:")
            self.log(f"    步数: {num_steps}")
            self.log(f"    推理时间: {inference_mean*1000:.0f} ± {inference_std*1000:.0f} ms")
            self.log(f"    prediction_horizon (H): {self.prediction_horizon}")
            self.log(f"    execute_horizon (s): {self.execute_horizon}")
            self.log(f"    action_horizon: {self.action_horizon}")
            
            # 运行传统模式
            self.log("\n  运行传统模式...")
            trad_metrics = self.simulate_traditional_mode(
                trajectory=trajectory,
                inference_times=inference_times,
            )
            
            # 运行 RTC 模式
            self.log("  运行 RTC 模式 (exp)...")
            rtc_exp_metrics = self.simulate_rtc_mode(
                trajectory=trajectory,
                inference_times=inference_times,
                soft_mask_schedule="exp",
            )
            
            self.log("  运行 RTC 模式 (linear)...")
            rtc_linear_metrics = self.simulate_rtc_mode(
                trajectory=trajectory,
                inference_times=inference_times,
                soft_mask_schedule="linear",
            )
            
            # 打印对比结果
            self.log("\n" + "-" * 60)
            self.log("对比结果:")
            self.log("-" * 60)
            
            self.log(f"\n  {'指标':<25} {'Traditional':>12} {'RTC (exp)':>12} {'RTC (linear)':>12}")
            self.log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
            
            self.log(f"  {'Chunk 重叠均值 (ms)':<25} {trad_metrics.overlap_mean:>12.1f} {rtc_exp_metrics.overlap_mean:>12.1f} {rtc_linear_metrics.overlap_mean:>12.1f}")
            self.log(f"  {'Chunk 重叠最大 (ms)':<25} {trad_metrics.overlap_max:>12.1f} {rtc_exp_metrics.overlap_max:>12.1f} {rtc_linear_metrics.overlap_max:>12.1f}")
            self.log(f"  {'重叠次数':<25} {trad_metrics.positive_overlap_count:>12} {rtc_exp_metrics.positive_overlap_count:>12} {rtc_linear_metrics.positive_overlap_count:>12}")
            self.log(f"  {'动作不连续均值':<25} {trad_metrics.discontinuity_mean:>12.4f} {rtc_exp_metrics.discontinuity_mean:>12.4f} {rtc_linear_metrics.discontinuity_mean:>12.4f}")
            self.log(f"  {'动作不连续最大':<25} {trad_metrics.discontinuity_max:>12.4f} {rtc_exp_metrics.discontinuity_max:>12.4f} {rtc_linear_metrics.discontinuity_max:>12.4f}")
            
            self.log(f"\n  RTC 特有指标:")
            self.log(f"    d 值均值: exp={rtc_exp_metrics.d_mean:.2f}, linear={rtc_linear_metrics.d_mean:.2f}")
            self.log(f"    拼接次数: exp={rtc_exp_metrics.splice_count}, linear={rtc_linear_metrics.splice_count}")
            
            # 计算改进
            overlap_improvement = trad_metrics.overlap_mean - rtc_exp_metrics.overlap_mean
            discontinuity_improvement = trad_metrics.discontinuity_mean - rtc_exp_metrics.discontinuity_mean
            
            self.log(f"\n  RTC (exp) 改进:")
            self.log(f"    重叠减少: {overlap_improvement:.1f} ms")
            self.log(f"    不连续性减少: {discontinuity_improvement:.4f}")
            
            self.log("\n✓ 模式对比测试完成")
            return True
            
        except Exception as e:
            self.log(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========= 运行所有测试 =========
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        self.log("\n" + "=" * 60)
        self.log("RTC 控制器测试套件")
        self.log("=" * 60)
        
        # 测试 1: Soft Masking 权重
        self.test_results['soft_mask_weights'] = self.test_soft_mask_weights()
        
        # 测试 2: Soft Masking 混合
        self.test_results['soft_mask_blending'] = self.test_soft_mask_blending()
        
        # 测试 3: ActionChunkManager 基本功能
        self.test_results['chunk_manager_basic'] = self.test_chunk_manager_basic()
        
        # 测试 4: 推理延迟自适应
        self.test_results['inference_delay_adaptation'] = self.test_inference_delay_adaptation()
        
        # 测试 5: 模式对比
        self.test_results['mode_comparison'] = self.test_mode_comparison()
        
        # 总结
        self.log("\n" + "=" * 60)
        self.log("测试总结")
        self.log("=" * 60)
        
        all_passed = True
        for name, passed in self.test_results.items():
            status = "✓ 通过" if passed else "✗ 失败"
            self.log(f"  {name}: {status}")
            if not passed:
                all_passed = False
        
        self.log("\n" + "=" * 60)
        if all_passed:
            self.log("✓ 所有测试通过")
        else:
            self.log("✗ 部分测试失败")
        self.log("=" * 60)
        
        return all_passed


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(description="RTC 控制器对比测试")
    parser.add_argument("--prediction-horizon", type=int, default=16, help="H 值")
    parser.add_argument("--execute-horizon", type=int, default=8, help="s 值 (RTC)")
    parser.add_argument("--action-horizon", type=int, default=8, help="传统模式 action_horizon")
    parser.add_argument("--num-steps", type=int, default=50, help="对比测试步数")
    parser.add_argument("--inference-mean", type=float, default=0.08, help="推理时间均值 (s)")
    parser.add_argument("--inference-std", type=float, default=0.02, help="推理时间标准差 (s)")
    parser.add_argument("--mask-schedule", type=str, default="exp", choices=["exp", "linear"],
                        help="软掩码衰减类型: exp (指数) 或 linear (线性)")
    parser.add_argument("--real-robot", action="store_true", help="真机测试模式 (当前未实现，仅占位)")
    parser.add_argument("-v", "--verbose", action="store_true", default=True, help="详细输出")
    
    args = parser.parse_args()
    
    if args.real_robot:
        print("\n⚠ 真机测试模式尚未实现，请使用 test_rtc_replay.py 进行真机测试")
        print("  示例: python -m consistency_policy.tests.test_rtc_replay --demo /path/to/demo.h5")
        return 1
    
    print("=" * 60)
    print("RTC 控制器对比测试")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  H (prediction_horizon): {args.prediction_horizon}")
    print(f"  s (execute_horizon): {args.execute_horizon}")
    print(f"  action_horizon: {args.action_horizon}")
    print(f"  mask_schedule: {args.mask_schedule}")
    
    tester = RTCControllerTester(
        prediction_horizon=args.prediction_horizon,
        execute_horizon=args.execute_horizon,
        action_horizon=args.action_horizon,
        verbose=args.verbose,
    )
    
    # 如果指定了 mask_schedule，在对比测试中使用
    tester.default_mask_schedule = args.mask_schedule
    
    success = tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
