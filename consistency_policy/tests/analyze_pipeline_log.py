#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流水线模式日志分析脚本

分析 eval_real_mp.py --pipeline 产生的时序日志，
计算与 test_rtc_replay.py 类似的指标进行对比。

用法:
    python -m consistency_policy.tests.analyze_pipeline_log \
        --log ./eval_output/timing_log_20260108_123456.npz
    
    # 对比两个日志
    python -m consistency_policy.tests.analyze_pipeline_log \
        --log ./log1.npz --log2 ./log2.npz
    
    # 绘制图表
    python -m consistency_policy.tests.analyze_pipeline_log \
        --log ./log.npz --plot

环境:
    conda activate arx-py310
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# 添加项目路径
CONSISTENCY_POLICY_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_VLA_PATH = os.path.dirname(CONSISTENCY_POLICY_PATH)
if RL_VLA_PATH not in sys.path:
    sys.path.insert(0, RL_VLA_PATH)


@dataclass
class PipelineMetrics:
    """流水线模式指标 (兼容 test_rtc_replay.py 的 ReplayMetrics)"""
    mode: str = "pipeline"
    
    # 基础统计
    total_chunks: int = 0
    total_duration: float = 0.0
    
    # 推理时间
    inference_times: List[float] = field(default_factory=list)
    inference_time_mean: float = 0.0
    inference_time_std: float = 0.0
    inference_time_p95: float = 0.0
    inference_time_max: float = 0.0
    
    # Chunk 重叠 (ms) - 正值表示重叠，负值表示间隙
    overlap_times: List[float] = field(default_factory=list)
    overlap_mean: float = 0.0
    overlap_std: float = 0.0
    overlap_min: float = 0.0
    overlap_max: float = 0.0
    positive_overlap_count: int = 0  # 正重叠次数
    negative_overlap_count: int = 0  # 负重叠 (间隙) 次数
    
    # 动作连续性 (相邻 chunk 的第一个动作与上一个 chunk 最后一个动作的差)
    action_discontinuities: List[float] = field(default_factory=list)
    discontinuity_mean: float = 0.0
    discontinuity_max: float = 0.0
    
    # 跟踪误差 (如果有实际关节位置)
    joint_errors: List[float] = field(default_factory=list)
    joint_error_mean: float = 0.0
    joint_error_max: float = 0.0
    joint_error_std: float = 0.0
    
    # RTC 相关
    d_values: List[int] = field(default_factory=list)
    d_mean: float = 0.0
    d_std: float = 0.0
    
    # 调度延迟 (obs_time 到 chunk_start 的时间)
    schedule_delays: List[float] = field(default_factory=list)
    schedule_delay_mean: float = 0.0
    schedule_delay_std: float = 0.0


def load_timing_log(log_path: str) -> Dict[str, np.ndarray]:
    """加载时序日志"""
    data = np.load(log_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def analyze_log(log_data: Dict[str, np.ndarray], verbose: bool = True) -> PipelineMetrics:
    """分析日志数据，计算指标"""
    metrics = PipelineMetrics()
    
    # 检查必需字段
    required_fields = ['chunk_start_time', 'chunk_end_time', 'infer_duration', 'action_seqs']
    for field in required_fields:
        if field not in log_data:
            print(f"警告: 缺少字段 {field}")
            return metrics
    
    chunk_starts = log_data['chunk_start_time']
    chunk_ends = log_data['chunk_end_time']
    infer_durations = log_data['infer_duration']
    action_seqs = log_data['action_seqs']
    
    n_chunks = len(chunk_starts)
    metrics.total_chunks = n_chunks
    
    if n_chunks < 2:
        print("警告: chunk 数量不足")
        return metrics
    
    # 1. 推理时间统计
    metrics.inference_times = infer_durations.tolist()
    metrics.inference_time_mean = np.mean(infer_durations)
    metrics.inference_time_std = np.std(infer_durations)
    metrics.inference_time_p95 = np.percentile(infer_durations, 95)
    metrics.inference_time_max = np.max(infer_durations)
    
    # 2. Chunk 重叠计算
    # overlap = prev_chunk_end - current_chunk_start
    # 正值表示重叠，负值表示间隙
    for i in range(1, n_chunks):
        overlap_sec = chunk_ends[i-1] - chunk_starts[i]
        overlap_ms = overlap_sec * 1000
        metrics.overlap_times.append(overlap_ms)
        if overlap_ms > 0:
            metrics.positive_overlap_count += 1
        else:
            metrics.negative_overlap_count += 1
    
    if metrics.overlap_times:
        metrics.overlap_mean = np.mean(metrics.overlap_times)
        metrics.overlap_std = np.std(metrics.overlap_times)
        metrics.overlap_min = np.min(metrics.overlap_times)
        metrics.overlap_max = np.max(metrics.overlap_times)
    
    # 3. 动作不连续性
    # 比较相邻 chunk: 当前 chunk 的第一个动作 vs 上一个 chunk 实际执行的最后一个动作
    # 注意: RTC 模式下只执行 execute_horizon 个动作，不是整个 pred_horizon
    execute_horizon = log_data.get('config_action_horizon', 8)  # 默认 8
    # 如果日志有 execute_horizon 信息，优先使用
    if 'config_execute_horizon' in log_data:
        execute_horizon = int(log_data['config_execute_horizon'])
    
    for i in range(1, n_chunks):
        # action_seqs shape: (n_chunks, pred_horizon, action_dim)
        # 上一个 chunk 实际执行的最后一个动作 (index = execute_horizon - 1)
        prev_executed_last = action_seqs[i-1, min(execute_horizon-1, action_seqs.shape[1]-1), :6]
        curr_first_action = action_seqs[i, 0, :6]    # 当前 chunk 第一个动作的关节部分
        discontinuity = np.linalg.norm(curr_first_action - prev_executed_last)
        metrics.action_discontinuities.append(discontinuity)
    
    if metrics.action_discontinuities:
        metrics.discontinuity_mean = np.mean(metrics.action_discontinuities)
        metrics.discontinuity_max = np.max(metrics.action_discontinuities)
    
    # 4. 跟踪误差 (如果有实际关节位置)
    if 'actual_joint_pos' in log_data:
        actual_pos = log_data['actual_joint_pos']
        for i in range(n_chunks):
            # 目标位置: 当前 chunk 第一个动作
            target_joints = action_seqs[i, 0, :6]
            actual_joints = actual_pos[i]
            error = np.linalg.norm(target_joints - actual_joints)
            metrics.joint_errors.append(error)
        
        if metrics.joint_errors:
            metrics.joint_error_mean = np.mean(metrics.joint_errors)
            metrics.joint_error_max = np.max(metrics.joint_errors)
            metrics.joint_error_std = np.std(metrics.joint_errors)
    
    # 5. d 值统计
    if 'd_values' in log_data:
        d_vals = log_data['d_values']
        metrics.d_values = d_vals.tolist()
        metrics.d_mean = np.mean(d_vals)
        metrics.d_std = np.std(d_vals)
    
    # 6. 调度延迟
    if 't_obs_get' in log_data:
        obs_times = log_data['t_obs_get']
        for i in range(n_chunks):
            delay = chunk_starts[i] - obs_times[i]
            metrics.schedule_delays.append(delay)
        metrics.schedule_delay_mean = np.mean(metrics.schedule_delays)
        metrics.schedule_delay_std = np.std(metrics.schedule_delays)
    
    # 7. 总时长
    metrics.total_duration = chunk_ends[-1] - chunk_starts[0]
    
    return metrics


def print_metrics(metrics: PipelineMetrics, title: str = "流水线模式指标"):
    """打印指标"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    
    print(f"\n基础统计:")
    print(f"  Chunk 数量: {metrics.total_chunks}")
    print(f"  总时长: {metrics.total_duration:.2f}s")
    
    print(f"\n推理时间:")
    print(f"  均值: {metrics.inference_time_mean*1000:.1f}ms")
    print(f"  标准差: {metrics.inference_time_std*1000:.1f}ms")
    print(f"  P95: {metrics.inference_time_p95*1000:.1f}ms")
    print(f"  最大: {metrics.inference_time_max*1000:.1f}ms")
    
    print(f"\nChunk 重叠 (正=重叠, 负=间隙):")
    print(f"  均值: {metrics.overlap_mean:.1f}ms")
    print(f"  标准差: {metrics.overlap_std:.1f}ms")
    print(f"  范围: [{metrics.overlap_min:.1f}, {metrics.overlap_max:.1f}]ms")
    print(f"  正重叠次数: {metrics.positive_overlap_count} / {len(metrics.overlap_times)}")
    print(f"  间隙次数: {metrics.negative_overlap_count} / {len(metrics.overlap_times)}")
    
    print(f"\n动作连续性:")
    print(f"  不连续均值: {metrics.discontinuity_mean:.4f}")
    print(f"  不连续最大: {metrics.discontinuity_max:.4f}")
    
    if metrics.joint_errors:
        print(f"\n跟踪误差:")
        print(f"  均值: {metrics.joint_error_mean:.4f} rad")
        print(f"  最大: {metrics.joint_error_max:.4f} rad")
        print(f"  标准差: {metrics.joint_error_std:.4f} rad")
    
    if metrics.d_values:
        print(f"\nRTC d 值:")
        print(f"  均值: {metrics.d_mean:.2f}")
        print(f"  标准差: {metrics.d_std:.2f}")
        print(f"  分布: {metrics.d_values[:10]}..." if len(metrics.d_values) > 10 else f"  分布: {metrics.d_values}")
    
    if metrics.schedule_delays:
        print(f"\n调度延迟 (obs → chunk_start):")
        print(f"  均值: {metrics.schedule_delay_mean*1000:.1f}ms")
        print(f"  标准差: {metrics.schedule_delay_std*1000:.1f}ms")
    
    print("=" * 70)


def compare_metrics(m1: PipelineMetrics, m2: PipelineMetrics, 
                    label1: str = "Log1", label2: str = "Log2"):
    """对比两个日志的指标"""
    print("\n" + "=" * 70)
    print(f"对比分析: {label1} vs {label2}")
    print("=" * 70)
    
    print(f"\n{'指标':<35} {label1:>15} {label2:>15} {'差异':>15}")
    print("-" * 80)
    
    # 推理时间
    print(f"{'推理时间均值 (ms)':<35} {m1.inference_time_mean*1000:>15.1f} {m2.inference_time_mean*1000:>15.1f} {(m2.inference_time_mean-m1.inference_time_mean)*1000:>+15.1f}")
    print(f"{'推理时间 P95 (ms)':<35} {m1.inference_time_p95*1000:>15.1f} {m2.inference_time_p95*1000:>15.1f} {(m2.inference_time_p95-m1.inference_time_p95)*1000:>+15.1f}")
    
    # Chunk 重叠
    print(f"{'Chunk 重叠均值 (ms)':<35} {m1.overlap_mean:>15.1f} {m2.overlap_mean:>15.1f} {m2.overlap_mean-m1.overlap_mean:>+15.1f}")
    print(f"{'正重叠比例':<35} {m1.positive_overlap_count/max(len(m1.overlap_times),1)*100:>14.1f}% {m2.positive_overlap_count/max(len(m2.overlap_times),1)*100:>14.1f}% {'-':>15}")
    
    # 动作连续性
    print(f"{'动作不连续均值':<35} {m1.discontinuity_mean:>15.4f} {m2.discontinuity_mean:>15.4f} {m2.discontinuity_mean-m1.discontinuity_mean:>+15.4f}")
    print(f"{'动作不连续最大':<35} {m1.discontinuity_max:>15.4f} {m2.discontinuity_max:>15.4f} {m2.discontinuity_max-m1.discontinuity_max:>+15.4f}")
    
    # 跟踪误差
    if m1.joint_errors and m2.joint_errors:
        print(f"{'关节误差均值 (rad)':<35} {m1.joint_error_mean:>15.4f} {m2.joint_error_mean:>15.4f} {m2.joint_error_mean-m1.joint_error_mean:>+15.4f}")
    
    # d 值
    if m1.d_values and m2.d_values:
        print(f"{'RTC d 值均值':<35} {m1.d_mean:>15.2f} {m2.d_mean:>15.2f} {m2.d_mean-m1.d_mean:>+15.2f}")
    
    # 总时长
    print(f"{'总时长 (s)':<35} {m1.total_duration:>15.2f} {m2.total_duration:>15.2f} {m2.total_duration-m1.total_duration:>+15.2f}")
    
    print("=" * 70)


def plot_metrics(metrics: PipelineMetrics, output_path: Optional[str] = None):
    """绘制指标图表"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("警告: matplotlib 未安装，跳过绘图")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 推理时间直方图
    ax = axes[0, 0]
    ax.hist(np.array(metrics.inference_times) * 1000, bins=30, edgecolor='black')
    ax.axvline(metrics.inference_time_mean * 1000, color='r', linestyle='--', label=f'Mean: {metrics.inference_time_mean*1000:.1f}ms')
    ax.axvline(metrics.inference_time_p95 * 1000, color='orange', linestyle='--', label=f'P95: {metrics.inference_time_p95*1000:.1f}ms')
    ax.set_xlabel('推理时间 (ms)')
    ax.set_ylabel('频数')
    ax.set_title('推理时间分布')
    ax.legend()
    
    # 2. Chunk 重叠直方图
    ax = axes[0, 1]
    ax.hist(metrics.overlap_times, bins=30, edgecolor='black')
    ax.axvline(0, color='r', linestyle='-', linewidth=2, label='零点 (无重叠/间隙)')
    ax.axvline(metrics.overlap_mean, color='g', linestyle='--', label=f'Mean: {metrics.overlap_mean:.1f}ms')
    ax.set_xlabel('Chunk 重叠 (ms)')
    ax.set_ylabel('频数')
    ax.set_title('Chunk 重叠分布 (正=重叠, 负=间隙)')
    ax.legend()
    
    # 3. 动作不连续性
    ax = axes[0, 2]
    ax.plot(metrics.action_discontinuities, 'b-', alpha=0.7)
    ax.axhline(metrics.discontinuity_mean, color='r', linestyle='--', label=f'Mean: {metrics.discontinuity_mean:.4f}')
    ax.set_xlabel('Chunk 索引')
    ax.set_ylabel('动作不连续性')
    ax.set_title('相邻 Chunk 动作不连续性')
    ax.legend()
    
    # 4. 跟踪误差 (如果有)
    ax = axes[1, 0]
    if metrics.joint_errors:
        ax.plot(metrics.joint_errors, 'g-', alpha=0.7)
        ax.axhline(metrics.joint_error_mean, color='r', linestyle='--', label=f'Mean: {metrics.joint_error_mean:.4f}')
        ax.set_xlabel('Chunk 索引')
        ax.set_ylabel('关节误差 (rad)')
        ax.set_title('跟踪误差')
        ax.legend()
    else:
        ax.text(0.5, 0.5, '无跟踪误差数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('跟踪误差')
    
    # 5. d 值分布
    ax = axes[1, 1]
    if metrics.d_values:
        unique_d, counts = np.unique(metrics.d_values, return_counts=True)
        ax.bar(unique_d, counts, edgecolor='black')
        ax.set_xlabel('d 值')
        ax.set_ylabel('频数')
        ax.set_title(f'd 值分布 (Mean: {metrics.d_mean:.2f})')
    else:
        ax.text(0.5, 0.5, '无 d 值数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('d 值分布')
    
    # 6. 调度延迟
    ax = axes[1, 2]
    if metrics.schedule_delays:
        ax.hist(np.array(metrics.schedule_delays) * 1000, bins=30, edgecolor='black')
        ax.axvline(metrics.schedule_delay_mean * 1000, color='r', linestyle='--', 
                   label=f'Mean: {metrics.schedule_delay_mean*1000:.1f}ms')
        ax.set_xlabel('调度延迟 (ms)')
        ax.set_ylabel('频数')
        ax.set_title('调度延迟分布 (obs → chunk_start)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, '无调度延迟数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('调度延迟分布')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"图表已保存: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="流水线模式日志分析")
    parser.add_argument("--log", "-l", required=True, help="时序日志文件路径 (.npz)")
    parser.add_argument("--log2", help="第二个日志文件 (用于对比)")
    parser.add_argument("--plot", "-p", action="store_true", help="绘制图表")
    parser.add_argument("--output", "-o", help="图表输出路径 (默认显示)")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="详细输出")
    
    args = parser.parse_args()
    
    # 加载并分析第一个日志
    print(f"加载日志: {args.log}")
    log_data = load_timing_log(args.log)
    
    print(f"\n日志字段: {list(log_data.keys())}")
    
    metrics = analyze_log(log_data, verbose=args.verbose)
    print_metrics(metrics, f"分析结果: {os.path.basename(args.log)}")
    
    # 如果有第二个日志，进行对比
    if args.log2:
        print(f"\n加载日志: {args.log2}")
        log_data2 = load_timing_log(args.log2)
        metrics2 = analyze_log(log_data2, verbose=args.verbose)
        print_metrics(metrics2, f"分析结果: {os.path.basename(args.log2)}")
        
        compare_metrics(metrics, metrics2, 
                        os.path.basename(args.log), 
                        os.path.basename(args.log2))
    
    # 绘制图表
    if args.plot:
        plot_metrics(metrics, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
