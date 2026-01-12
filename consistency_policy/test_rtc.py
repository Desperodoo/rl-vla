#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTC (Real-Time Chunking) 效果测试

模拟策略推理和动作调度流程，对比 RTC 模式和传统模式的差异。

用法:
    python -m consistency_policy.test_rtc
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import List

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not available, skipping visualization")

from consistency_policy.rtc_manager import (
    RTCConfig,
    ActionChunkManager,
    compute_soft_mask_weights,
)


@dataclass
class SimulationConfig:
    """模拟配置"""
    prediction_horizon: int = 16       # H
    execute_horizon: int = 8           # s
    action_dim: int = 7
    action_dt: float = 1.0 / 30.0      # 33.3ms
    
    inference_time_mean: float = 0.08  # 80ms
    inference_time_std: float = 0.02   # ±20ms
    
    eval_frequency: float = 10.0       # 10Hz 推理循环
    
    num_steps: int = 50                # 模拟步数


def generate_smooth_trajectory(num_points: int, action_dim: int, seed: int = 42) -> np.ndarray:
    """
    生成平滑的测试轨迹
    
    使用正弦波叠加生成平滑的多维轨迹
    """
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, num_points)
    
    trajectory = np.zeros((num_points, action_dim))
    for dim in range(action_dim):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.1, 0.3)
        trajectory[:, dim] = amp * np.sin(freq * t + phase)
    
    return trajectory


def simulate_policy_output(
    ground_truth: np.ndarray,
    current_idx: int,
    pred_horizon: int,
    noise_std: float = 0.01,
) -> np.ndarray:
    """
    模拟策略输出
    
    基于 ground truth 添加噪声
    """
    start_idx = max(0, current_idx)
    end_idx = min(len(ground_truth), current_idx + pred_horizon)
    
    # 提取片段
    action_seq = np.zeros((pred_horizon, ground_truth.shape[1]))
    valid_len = end_idx - start_idx
    action_seq[:valid_len] = ground_truth[start_idx:end_idx]
    
    # 如果超出范围，用最后一个值填充
    if valid_len < pred_horizon:
        action_seq[valid_len:] = ground_truth[-1]
    
    # 添加噪声
    action_seq += np.random.randn(*action_seq.shape) * noise_std
    
    return action_seq


def simulate_traditional_mode(config: SimulationConfig, ground_truth: np.ndarray):
    """
    模拟传统模式 (无 RTC)
    
    每次推理完成后立即调度新的 action chunk
    """
    eval_dt = 1.0 / config.eval_frequency
    action_dt = config.action_dt
    H = config.prediction_horizon
    s = config.execute_horizon
    
    # 记录
    scheduled_actions = []  # [(start_time, actions), ...]
    executed_actions = []   # [(time, action), ...]
    chunk_overlaps = []
    
    current_time = 0.0
    trajectory_idx = 0
    
    for step in range(config.num_steps):
        # 获取观测时间
        obs_time = current_time
        
        # 模拟推理
        inference_time = np.random.normal(config.inference_time_mean, config.inference_time_std)
        inference_time = max(0.05, inference_time)  # 最小 50ms
        
        # 模拟策略输出
        action_seq = simulate_policy_output(
            ground_truth, trajectory_idx, H
        )
        
        # 推理完成时间
        current_time = obs_time + inference_time
        
        # 调度动作 (传统模式: 推理完成后立即调度)
        # 使用自适应延迟
        adaptive_delay = inference_time + 0.01
        start_time = current_time + adaptive_delay
        
        # 记录调度
        scheduled_actions.append({
            'step': step,
            'start_time': start_time,
            'end_time': start_time + s * action_dt,
            'actions': action_seq[:s].copy(),
        })
        
        # 检查与前一个 chunk 的重叠
        if len(scheduled_actions) > 1:
            prev = scheduled_actions[-2]
            curr = scheduled_actions[-1]
            overlap = prev['end_time'] - curr['start_time']
            chunk_overlaps.append(overlap)
        
        # 模拟等待到下一个循环
        wait_time = eval_dt - inference_time
        if wait_time > 0:
            current_time += wait_time
        
        trajectory_idx += int(eval_dt / action_dt)
    
    return {
        'scheduled_actions': scheduled_actions,
        'chunk_overlaps': chunk_overlaps,
        'mode': 'traditional',
    }


def simulate_rtc_mode(config: SimulationConfig, ground_truth: np.ndarray):
    """
    模拟 RTC 模式
    
    使用 ActionChunkManager 进行 chunk 管理和 soft masking
    """
    eval_dt = 1.0 / config.eval_frequency
    action_dt = config.action_dt
    H = config.prediction_horizon
    s = config.execute_horizon
    
    # 创建 RTC 管理器
    rtc_config = RTCConfig(
        prediction_horizon=H,
        execute_horizon=s,
        action_dim=config.action_dim,
        action_dt=action_dt,
        min_inference_delay_steps=2,
        max_inference_delay_steps=5,
        soft_mask_schedule='exp',
        soft_mask_decay_rate=2.0,
    )
    rtc_manager = ActionChunkManager(rtc_config, verbose=False)
    
    # 记录
    scheduled_actions = []
    chunk_overlaps = []
    d_values = []
    
    current_time = 0.0
    trajectory_idx = 0
    
    for step in range(config.num_steps):
        # 获取观测时间
        obs_time = current_time
        
        # 模拟推理
        inference_time = np.random.normal(config.inference_time_mean, config.inference_time_std)
        inference_time = max(0.05, inference_time)
        
        # 模拟策略输出
        action_seq = simulate_policy_output(
            ground_truth, trajectory_idx, H
        )
        
        # 推理完成时间
        infer_end_time = obs_time + inference_time
        
        # 使用 RTC 提交新 chunk
        chunk = rtc_manager.submit_new_chunk(
            action_seq=action_seq,
            obs_time=obs_time,
            inference_time=inference_time,
            current_time=infer_end_time,
        )
        
        # 获取调度用的动作
        actions, timestamps, chunk_start, chunk_end = rtc_manager.get_scheduled_actions(
            start_time=infer_end_time
        )
        
        # 记录
        scheduled_actions.append({
            'step': step,
            'start_time': chunk_start,
            'end_time': chunk_end,
            'actions': actions.copy(),
            'd': chunk.inference_delay,
        })
        d_values.append(chunk.inference_delay)
        
        # 检查与前一个 chunk 的重叠
        if len(scheduled_actions) > 1:
            prev = scheduled_actions[-2]
            curr = scheduled_actions[-1]
            overlap = prev['end_time'] - curr['start_time']
            chunk_overlaps.append(overlap)
        
        # 模拟等待到下一个循环
        current_time = infer_end_time
        wait_time = eval_dt - inference_time
        if wait_time > 0:
            current_time += wait_time
        
        trajectory_idx += int(eval_dt / action_dt)
    
    return {
        'scheduled_actions': scheduled_actions,
        'chunk_overlaps': chunk_overlaps,
        'd_values': d_values,
        'mode': 'rtc',
        'rtc_stats': rtc_manager.get_statistics(),
    }


def visualize_results(traditional_result, rtc_result, save_path: str = None):
    """可视化对比结果"""
    if not HAS_MATPLOTLIB:
        print("Skipping visualization (matplotlib not available)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Chunk 重叠时间对比
    ax1 = axes[0, 0]
    trad_overlaps = np.array(traditional_result['chunk_overlaps']) * 1000  # 转为 ms
    rtc_overlaps = np.array(rtc_result['chunk_overlaps']) * 1000
    
    x = np.arange(len(trad_overlaps))
    ax1.bar(x - 0.2, trad_overlaps, 0.4, label='Traditional', alpha=0.7)
    ax1.bar(x + 0.2, rtc_overlaps, 0.4, label='RTC', alpha=0.7)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Chunk Transition')
    ax1.set_ylabel('Overlap Time (ms)')
    ax1.set_title('Chunk Overlap Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 重叠时间分布
    ax2 = axes[0, 1]
    ax2.hist(trad_overlaps, bins=20, alpha=0.7, label='Traditional')
    ax2.hist(rtc_overlaps, bins=20, alpha=0.7, label='RTC')
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Overlap Time (ms)')
    ax2.set_ylabel('Count')
    ax2.set_title('Overlap Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Chunk 执行时间线
    ax3 = axes[1, 0]
    
    # Traditional mode
    for i, chunk in enumerate(traditional_result['scheduled_actions'][:20]):
        y = 0
        ax3.barh(y, chunk['end_time'] - chunk['start_time'], 
                 left=chunk['start_time'], height=0.3, alpha=0.7,
                 color=f'C{i % 10}')
    
    # RTC mode
    for i, chunk in enumerate(rtc_result['scheduled_actions'][:20]):
        y = 1
        ax3.barh(y, chunk['end_time'] - chunk['start_time'], 
                 left=chunk['start_time'], height=0.3, alpha=0.7,
                 color=f'C{i % 10}')
    
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Traditional', 'RTC'])
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Chunk Execution Timeline (first 20 chunks)')
    ax3.grid(True, alpha=0.3)
    
    # 4. RTC d 值变化
    ax4 = axes[1, 1]
    d_values = rtc_result['d_values']
    ax4.plot(d_values, marker='o', markersize=3, linewidth=1)
    ax4.axhline(y=np.mean(d_values), color='r', linestyle='--', label=f'Mean: {np.mean(d_values):.2f}')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('d (inference delay steps)')
    ax4.set_title('RTC Inference Delay Estimation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图表已保存: {save_path}")
    
    plt.show()


def print_comparison(traditional_result, rtc_result):
    """打印对比统计"""
    print("\n" + "=" * 60)
    print("RTC vs Traditional Mode Comparison")
    print("=" * 60)
    
    trad_overlaps = np.array(traditional_result['chunk_overlaps']) * 1000
    rtc_overlaps = np.array(rtc_result['chunk_overlaps']) * 1000
    
    print("\n[Chunk Overlap Statistics (ms)]")
    print(f"{'Metric':<20} {'Traditional':>15} {'RTC':>15}")
    print("-" * 50)
    print(f"{'Mean':<20} {np.mean(trad_overlaps):>15.2f} {np.mean(rtc_overlaps):>15.2f}")
    print(f"{'Std':<20} {np.std(trad_overlaps):>15.2f} {np.std(rtc_overlaps):>15.2f}")
    print(f"{'Max':<20} {np.max(trad_overlaps):>15.2f} {np.max(rtc_overlaps):>15.2f}")
    print(f"{'Min':<20} {np.min(trad_overlaps):>15.2f} {np.min(rtc_overlaps):>15.2f}")
    print(f"{'Positive (overlap)':<20} {np.sum(trad_overlaps > 0):>15} {np.sum(rtc_overlaps > 0):>15}")
    print(f"{'Negative (gap)':<20} {np.sum(trad_overlaps < 0):>15} {np.sum(rtc_overlaps < 0):>15}")
    
    print("\n[RTC Statistics]")
    stats = rtc_result['rtc_stats']
    print(f"  Total chunks: {stats['chunk_count']}")
    print(f"  Final d value: {stats['current_d']}")
    print(f"  Inference time P95: {stats['inference_time_p95']*1000:.2f} ms")
    print(f"  Splice count: {stats['splice_count']}")
    
    # 计算有效执行率
    # Traditional: 由于大量重叠，实际执行的动作比例较低
    # RTC: 通过 soft masking 确保连续性
    trad_positive_overlaps = trad_overlaps[trad_overlaps > 0]
    rtc_positive_overlaps = rtc_overlaps[rtc_overlaps > 0]
    
    action_dt = 33.3  # ms
    s = 8  # execute_horizon
    chunk_duration = s * action_dt
    
    if len(trad_positive_overlaps) > 0:
        trad_effective_ratio = 1 - np.mean(trad_positive_overlaps) / chunk_duration
        print(f"\n[Effective Execution Ratio]")
        print(f"  Traditional: {trad_effective_ratio*100:.1f}%")
    
    if len(rtc_positive_overlaps) > 0:
        rtc_effective_ratio = 1 - np.mean(rtc_positive_overlaps) / chunk_duration
        print(f"  RTC: {rtc_effective_ratio*100:.1f}%")
    
    print("\n[Conclusion]")
    improvement = np.mean(trad_overlaps) - np.mean(rtc_overlaps)
    if improvement > 0:
        print(f"  ✓ RTC 减少了平均 {improvement:.2f}ms 的 chunk 重叠")
    else:
        print(f"  RTC 模式的重叠增加了 {-improvement:.2f}ms (检查参数配置)")


def main():
    print("=" * 60)
    print("RTC (Real-Time Chunking) 效果测试")
    print("=" * 60)
    
    # 配置
    config = SimulationConfig(
        prediction_horizon=16,
        execute_horizon=8,
        action_dim=7,
        action_dt=1.0 / 30.0,
        inference_time_mean=0.08,   # 80ms
        inference_time_std=0.02,     # ±20ms
        eval_frequency=10.0,         # 10Hz
        num_steps=50,
    )
    
    print(f"\n配置:")
    print(f"  H (prediction_horizon): {config.prediction_horizon}")
    print(f"  s (execute_horizon): {config.execute_horizon}")
    print(f"  action_dt: {config.action_dt*1000:.1f}ms")
    print(f"  inference_time: {config.inference_time_mean*1000:.0f} ± {config.inference_time_std*1000:.0f}ms")
    print(f"  eval_frequency: {config.eval_frequency}Hz")
    print(f"  chunk_duration: {config.execute_horizon * config.action_dt * 1000:.1f}ms")
    print(f"  eval_period: {1000/config.eval_frequency:.1f}ms")
    
    # 生成测试轨迹
    total_points = int(config.num_steps * (1.0 / config.eval_frequency) / config.action_dt) + 100
    ground_truth = generate_smooth_trajectory(total_points, config.action_dim)
    print(f"\n生成测试轨迹: {ground_truth.shape}")
    
    # 运行模拟
    print("\n运行传统模式模拟...")
    traditional_result = simulate_traditional_mode(config, ground_truth)
    
    print("运行 RTC 模式模拟...")
    rtc_result = simulate_rtc_mode(config, ground_truth)
    
    # 打印对比
    print_comparison(traditional_result, rtc_result)
    
    # 可视化
    print("\n生成可视化图表...")
    try:
        visualize_results(traditional_result, rtc_result, 
                         save_path='/home/lizh/rl-vla/consistency_policy/rtc_comparison.png')
    except Exception as e:
        print(f"可视化失败 (可能没有显示设备): {e}")
        # 保存数据
        np.savez('/home/lizh/rl-vla/consistency_policy/rtc_test_results.npz',
                 trad_overlaps=traditional_result['chunk_overlaps'],
                 rtc_overlaps=rtc_result['chunk_overlaps'],
                 rtc_d_values=rtc_result['d_values'])
        print("数据已保存到 rtc_test_results.npz")


if __name__ == "__main__":
    main()
