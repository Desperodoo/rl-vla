#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action Chunking 时序分析脚本

分析 eval_real_mp.py 输出的时序日志，验证 action chunking 的时序问题：
1. Chunk 之间是否有重叠/间隙
2. 推理延迟是否导致动作执行滞后
3. 自适应延迟补偿是否有效

用法:
    python -m consistency_policy.analyze_timing timing_log_xxx.npz
    
输出:
    - 时序统计报告
    - chunk 时间线图 (可选)
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

# 尝试导入 matplotlib (可选)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found, visualization disabled")


def load_timing_log(filepath: str) -> Dict[str, np.ndarray]:
    """加载时序日志"""
    data = dict(np.load(filepath, allow_pickle=True))
    print(f"[分析] 加载日志: {filepath}")
    print(f"  Keys: {list(data.keys())}")
    return data


def analyze_inference_timing(data: Dict[str, np.ndarray]) -> Dict:
    """分析推理时序"""
    if 't_infer_start' not in data:
        return {}
    
    infer_start = data['t_infer_start']
    infer_end = data['t_infer_end']
    infer_duration = data['infer_duration']
    adaptive_delay = data['adaptive_delay']
    chunk_start = data['chunk_start_time']
    chunk_end = data['chunk_end_time']
    
    n_steps = len(infer_start)
    
    # 基本统计
    stats = {
        'n_steps': n_steps,
        'infer_duration_mean': np.mean(infer_duration) * 1000,
        'infer_duration_std': np.std(infer_duration) * 1000,
        'infer_duration_p95': np.percentile(infer_duration, 95) * 1000,
        'adaptive_delay_mean': np.mean(adaptive_delay) * 1000,
    }
    
    # Chunk 时间范围统计
    chunk_duration = chunk_end - chunk_start
    stats['chunk_duration_mean'] = np.mean(chunk_duration) * 1000
    stats['chunk_duration_std'] = np.std(chunk_duration) * 1000
    
    # 推理周期 (相邻两次推理的时间间隔)
    if n_steps > 1:
        infer_intervals = np.diff(infer_start)
        stats['infer_interval_mean'] = np.mean(infer_intervals) * 1000
        stats['infer_interval_std'] = np.std(infer_intervals) * 1000
    
    return stats


def analyze_chunk_overlap(data: Dict[str, np.ndarray]) -> Dict:
    """
    分析 chunk 之间的重叠/间隙
    
    关键问题: 新 chunk 的开始时间是否在上一个 chunk 结束之前？
    """
    if 'chunk_start_time' not in data:
        return {}
    
    chunk_start = data['chunk_start_time']
    chunk_end = data['chunk_end_time']
    t_schedule = data['t_schedule']
    
    n_steps = len(chunk_start)
    
    if n_steps < 2:
        return {'n_steps': n_steps, 'warning': '样本太少'}
    
    # 计算 chunk 重叠/间隙
    overlaps = []
    gaps = []
    
    for i in range(1, n_steps):
        prev_end = chunk_end[i-1]
        curr_start = chunk_start[i]
        
        delta = curr_start - prev_end
        if delta < 0:
            overlaps.append(-delta)  # 重叠时间 (正数)
        else:
            gaps.append(delta)  # 间隙时间 (正数)
    
    # 计算实际执行时间 vs 预期时间
    # 问题: 当新 chunk 来的时候，旧 chunk 执行了多少？
    executed_before_override = []
    for i in range(1, n_steps):
        # 从上一个 chunk 开始到新 chunk 发送的时间
        actual_exec_time = t_schedule[i] - chunk_start[i-1]
        expected_exec_time = chunk_end[i-1] - chunk_start[i-1]
        executed_ratio = actual_exec_time / expected_exec_time if expected_exec_time > 0 else 0
        executed_before_override.append(executed_ratio)
    
    stats = {
        'n_overlaps': len(overlaps),
        'n_gaps': len(gaps),
        'overlap_mean_ms': np.mean(overlaps) * 1000 if overlaps else 0,
        'overlap_max_ms': np.max(overlaps) * 1000 if overlaps else 0,
        'gap_mean_ms': np.mean(gaps) * 1000 if gaps else 0,
        'gap_max_ms': np.max(gaps) * 1000 if gaps else 0,
        'executed_ratio_mean': np.mean(executed_before_override),
        'executed_ratio_min': np.min(executed_before_override),
        'executed_ratio_max': np.max(executed_before_override),
    }
    
    return stats


def analyze_timing_alignment(data: Dict[str, np.ndarray]) -> Dict:
    """
    分析时间对齐问题
    
    检查: chunk_start_time 是否总是在 t_schedule 之后 (即有正确的延迟补偿)
    """
    if 't_schedule' not in data or 'chunk_start_time' not in data:
        return {}
    
    t_schedule = data['t_schedule']
    chunk_start = data['chunk_start_time']
    adaptive_delay = data['adaptive_delay']
    
    # 计算实际延迟 (chunk 开始时间 - 调度时间)
    actual_delay = chunk_start - t_schedule
    
    # 检查是否有负延迟 (chunk 开始时间在调度之前，说明延迟补偿过大)
    n_negative = np.sum(actual_delay < 0)
    
    stats = {
        'actual_delay_mean_ms': np.mean(actual_delay) * 1000,
        'actual_delay_min_ms': np.min(actual_delay) * 1000,
        'actual_delay_max_ms': np.max(actual_delay) * 1000,
        'n_negative_delay': n_negative,
        'adaptive_delay_mean_ms': np.mean(adaptive_delay) * 1000,
    }
    
    return stats


def print_report(stats: Dict, title: str):
    """打印统计报告"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def visualize_timeline(data: Dict[str, np.ndarray], output_path: str = None):
    """
    可视化时间线
    
    显示:
    - 每个 chunk 的时间范围 (矩形)
    - 推理时间点 (垂直线)
    - 重叠区域 (红色高亮)
    """
    if not HAS_MATPLOTLIB:
        print("[Warning] matplotlib not available, skipping visualization")
        return
    
    chunk_start = data['chunk_start_time']
    chunk_end = data['chunk_end_time']
    t_infer_start = data['t_infer_start']
    t_infer_end = data['t_infer_end']
    t_schedule = data['t_schedule']
    
    n_steps = len(chunk_start)
    
    # 归一化时间 (相对于第一个推理开始时间)
    t0 = t_infer_start[0]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # === 图1: Chunk 时间线 ===
    ax1 = axes[0]
    colors = plt.cm.tab10(np.arange(n_steps) % 10)
    
    for i in range(n_steps):
        # Chunk 矩形
        start = chunk_start[i] - t0
        end = chunk_end[i] - t0
        ax1.barh(0, end - start, left=start, height=0.3, 
                 color=colors[i], alpha=0.6, edgecolor='black')
        
        # 推理时间点
        infer_s = t_infer_start[i] - t0
        infer_e = t_infer_end[i] - t0
        ax1.axvline(infer_s, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axvline(infer_e, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.set_ylabel('Chunks')
    ax1.set_title('Action Chunk Timeline (Blue=Chunk, Green=Infer Start, Red=Infer End)')
    ax1.set_ylim(-0.5, 0.5)
    
    # === 图2: Chunk 重叠分析 ===
    ax2 = axes[1]
    
    for i in range(n_steps):
        start = chunk_start[i] - t0
        end = chunk_end[i] - t0
        ax2.plot([start, end], [i, i], 'b-', linewidth=4, alpha=0.7)
        ax2.scatter([start], [i], color='green', s=50, zorder=5)
        ax2.scatter([end], [i], color='red', s=50, zorder=5)
    
    # 标记重叠区域
    for i in range(1, n_steps):
        prev_end = chunk_end[i-1] - t0
        curr_start = chunk_start[i] - t0
        if curr_start < prev_end:
            ax2.axvspan(curr_start, prev_end, alpha=0.3, color='red', 
                        label='Overlap' if i == 1 else '')
    
    ax2.set_ylabel('Step')
    ax2.set_title('Chunk Overlap Analysis (Red shading = overlap)')
    ax2.legend(loc='upper right')
    
    # === 图3: 时间延迟分析 ===
    ax3 = axes[2]
    
    # 计算各种延迟
    infer_duration = (t_infer_end - t_infer_start) * 1000
    schedule_delay = (t_schedule - t_infer_end) * 1000
    total_delay = (chunk_start - t_infer_start) * 1000
    
    steps = np.arange(n_steps)
    ax3.plot(steps, infer_duration, 'b-o', label='Inference Duration', markersize=4)
    ax3.plot(steps, total_delay, 'r-s', label='Total Delay (infer_start → chunk_start)', markersize=4)
    ax3.plot(steps, data['adaptive_delay'] * 1000, 'g--', label='Adaptive Delay', linewidth=2)
    
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Timing Analysis')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"[分析] 图表已保存: {output_path}")
    else:
        plt.show()


def diagnose_issues(data: Dict[str, np.ndarray]) -> List[str]:
    """诊断潜在问题"""
    issues = []
    
    overlap_stats = analyze_chunk_overlap(data)
    timing_stats = analyze_timing_alignment(data)
    infer_stats = analyze_inference_timing(data)
    
    # 检查 chunk 覆盖问题
    if overlap_stats.get('n_overlaps', 0) > 0:
        overlap_ratio = overlap_stats['n_overlaps'] / (overlap_stats['n_overlaps'] + overlap_stats['n_gaps'])
        if overlap_ratio > 0.1:
            issues.append(f"⚠️ 严重: {overlap_stats['n_overlaps']} 次 chunk 重叠 "
                         f"(平均重叠 {overlap_stats['overlap_mean_ms']:.1f}ms)")
    
    # 检查 chunk 执行比例
    exec_ratio = overlap_stats.get('executed_ratio_mean', 1.0)
    if exec_ratio < 0.5:
        issues.append(f"⚠️ 严重: 平均只执行了 {exec_ratio*100:.1f}% 的 chunk 就被新 chunk 覆盖")
    
    # 检查推理延迟
    if infer_stats.get('infer_duration_p95', 0) > 100:
        issues.append(f"⚠️ 推理延迟较高: P95 = {infer_stats['infer_duration_p95']:.1f}ms")
    
    # 检查 chunk 持续时间 vs 推理周期
    chunk_duration = infer_stats.get('chunk_duration_mean', 0)
    infer_interval = infer_stats.get('infer_interval_mean', 0)
    if chunk_duration > 0 and infer_interval > 0:
        if chunk_duration < infer_interval:
            issues.append(f"⚠️ Chunk 持续时间 ({chunk_duration:.1f}ms) < 推理周期 ({infer_interval:.1f}ms)，"
                         "会导致动作间隙")
        elif chunk_duration > infer_interval * 3:
            issues.append(f"⚠️ Chunk 持续时间 ({chunk_duration:.1f}ms) >> 推理周期 ({infer_interval:.1f}ms)，"
                         "会导致大量覆盖浪费")
    
    # 检查负延迟
    if timing_stats.get('n_negative_delay', 0) > 0:
        issues.append(f"⚠️ {timing_stats['n_negative_delay']} 次负延迟，自适应延迟补偿可能过大")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="Action Chunking 时序分析")
    parser.add_argument("logfile", help="时序日志文件 (.npz)")
    parser.add_argument("-o", "--output", help="输出图表路径 (可选)")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    
    args = parser.parse_args()
    
    # 加载日志
    data = load_timing_log(args.logfile)
    
    # 打印配置信息
    print(f"\n[配置信息]")
    for key in data.keys():
        if key.startswith('config_'):
            print(f"  {key}: {data[key]}")
    
    # 分析推理时序
    infer_stats = analyze_inference_timing(data)
    if infer_stats:
        print_report(infer_stats, "推理时序统计")
    
    # 分析 chunk 重叠
    overlap_stats = analyze_chunk_overlap(data)
    if overlap_stats:
        print_report(overlap_stats, "Chunk 重叠/间隙分析")
    
    # 分析时间对齐
    timing_stats = analyze_timing_alignment(data)
    if timing_stats:
        print_report(timing_stats, "时间对齐分析")
    
    # 诊断问题
    issues = diagnose_issues(data)
    print(f"\n{'='*60}")
    print("问题诊断")
    print('='*60)
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✓ 未发现明显问题")
    
    # 可视化
    if not args.no_plot and HAS_MATPLOTLIB:
        output_path = args.output
        if output_path is None:
            output_path = args.logfile.replace('.npz', '_analysis.png')
        visualize_timeline(data, output_path)
    
    print(f"\n[分析完成]")


if __name__ == "__main__":
    main()
