#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹对比分析脚本

用于对比原版单进程和多进程版本的推理/控制行为差异，
找出导致多进程版本抖动的根本原因。

用法:
    # 对比两个版本的轨迹
    python -m inference.compare_trajectories \
        --original trajectory_original.npz \
        --multi-inference trajectory_multi_process_inference.npz \
        --multi-control trajectory_multi_process_control.npz
    
    # 仅分析单个轨迹
    python -m inference.compare_trajectories --single trajectory.npz
    
    # 输出报告到文件
    python -m inference.compare_trajectories --original a.npz --multi-control b.npz --output report.txt
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[警告] matplotlib 未安装，将禁用绘图功能")


@dataclass
class TrajectoryStats:
    """轨迹统计信息"""
    name: str
    total_steps: int
    total_duration: float  # 秒
    
    # 推理统计
    inference_count: int = 0
    avg_inference_time: float = 0.0
    max_inference_time: float = 0.0
    
    # 动作统计
    mean_action_delta: np.ndarray = None  # 平均帧间变化
    max_action_delta: np.ndarray = None   # 最大帧间变化
    action_delta_std: np.ndarray = None   # 帧间变化标准差
    
    # 跳变统计 (chunk 边界)
    jump_count: int = 0          # 大跳变次数
    max_jump: float = 0.0        # 最大跳变量
    avg_jump: float = 0.0        # 平均跳变量
    
    # 滤波统计
    filter_effect: float = 0.0   # 滤波强度 (raw vs filtered 的差异)


def load_trajectory(path: str) -> Dict[str, Any]:
    """加载轨迹文件"""
    data = np.load(path, allow_pickle=True)
    result = {}
    for key in data.files:
        result[key] = data[key]
    return result


def analyze_single_trajectory(data: Dict[str, Any], name: str = "trajectory") -> TrajectoryStats:
    """分析单个轨迹"""
    
    # 检测数据类型 (推理 vs 控制)
    has_inference = 'inference_timestamps' in data or any('inference' in k.lower() for k in data.keys())
    has_control = 'control_timestamps' in data or 'executed_actions' in data
    
    stats = TrajectoryStats(
        name=name,
        total_steps=0,
        total_duration=0.0,
    )
    
    # 推理数据分析
    if 'inference_timestamps' in data:
        inf_ts = data['inference_timestamps']
        stats.inference_count = len(inf_ts)
        
        if 'inference_times' in data:
            inf_times = data['inference_times']
            stats.avg_inference_time = float(np.mean(inf_times))
            stats.max_inference_time = float(np.max(inf_times))
        
        if len(inf_ts) > 1:
            stats.total_duration = float(inf_ts[-1] - inf_ts[0])
    
    # 控制数据分析
    if 'control_timestamps' in data:
        ctrl_ts = data['control_timestamps']
        stats.total_steps = len(ctrl_ts)
        
        if len(ctrl_ts) > 1:
            duration = float(ctrl_ts[-1] - ctrl_ts[0])
            if duration > stats.total_duration:
                stats.total_duration = duration
    
    # 动作分析
    action_key = None
    for key in ['executed_actions', 'raw_actions', 'actions']:
        if key in data:
            action_key = key
            break
    
    if action_key and len(data[action_key]) > 1:
        actions = data[action_key]
        
        # 帧间变化
        deltas = np.diff(actions, axis=0)
        stats.mean_action_delta = np.mean(np.abs(deltas), axis=0)
        stats.max_action_delta = np.max(np.abs(deltas), axis=0)
        stats.action_delta_std = np.std(deltas, axis=0)
        
        # 检测跳变
        joint_deltas = np.abs(deltas[:, :6]).max(axis=1)
        gripper_deltas = np.abs(deltas[:, 6])
        
        # 跳变阈值
        JOINT_JUMP_THRESHOLD = 0.02  # rad
        GRIPPER_JUMP_THRESHOLD = 0.005  # m
        
        jumps = (joint_deltas > JOINT_JUMP_THRESHOLD) | (gripper_deltas > GRIPPER_JUMP_THRESHOLD)
        stats.jump_count = int(np.sum(jumps))
        
        if stats.jump_count > 0:
            jump_magnitudes = joint_deltas[jumps]
            stats.max_jump = float(np.max(jump_magnitudes))
            stats.avg_jump = float(np.mean(jump_magnitudes))
    
    # 滤波效果分析
    if 'raw_actions' in data and 'filtered_actions' in data:
        raw = data['raw_actions']
        filtered = data['filtered_actions']
        diff = np.abs(raw - filtered)
        stats.filter_effect = float(np.mean(diff))
    
    return stats


def print_stats(stats: TrajectoryStats, indent: str = "  "):
    """打印统计信息"""
    print(f"\n{indent}=== {stats.name} ===")
    print(f"{indent}总步数: {stats.total_steps}")
    print(f"{indent}总时长: {stats.total_duration:.2f}s")
    
    if stats.inference_count > 0:
        print(f"{indent}推理次数: {stats.inference_count}")
        print(f"{indent}平均推理时间: {stats.avg_inference_time*1000:.1f}ms")
        print(f"{indent}最大推理时间: {stats.max_inference_time*1000:.1f}ms")
    
    if stats.mean_action_delta is not None:
        print(f"{indent}平均帧间变化:")
        print(f"{indent}  关节: {stats.mean_action_delta[:6]}")
        print(f"{indent}  夹爪: {stats.mean_action_delta[6]:.6f}")
        print(f"{indent}最大帧间变化:")
        print(f"{indent}  关节: {stats.max_action_delta[:6]}")
        print(f"{indent}  夹爪: {stats.max_action_delta[6]:.6f}")
    
    print(f"{indent}跳变次数: {stats.jump_count}")
    if stats.jump_count > 0:
        print(f"{indent}  最大跳变: {stats.max_jump:.4f} rad")
        print(f"{indent}  平均跳变: {stats.avg_jump:.4f} rad")
    
    if stats.filter_effect > 0:
        print(f"{indent}滤波效果: {stats.filter_effect:.6f}")


def compare_trajectories(
    original_data: Dict[str, Any],
    multi_inference_data: Optional[Dict[str, Any]] = None,
    multi_control_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """对比分析两个版本的轨迹"""
    
    print("\n" + "=" * 70)
    print("轨迹对比分析")
    print("=" * 70)
    
    results = {}
    
    # 分析原版
    original_stats = analyze_single_trajectory(original_data, "原版 (单进程)")
    print_stats(original_stats)
    results['original'] = original_stats
    
    # 分析多进程推理
    if multi_inference_data:
        multi_inf_stats = analyze_single_trajectory(multi_inference_data, "多进程 (推理节点)")
        print_stats(multi_inf_stats)
        results['multi_inference'] = multi_inf_stats
    
    # 分析多进程控制
    if multi_control_data:
        multi_ctrl_stats = analyze_single_trajectory(multi_control_data, "多进程 (控制节点)")
        print_stats(multi_ctrl_stats)
        results['multi_control'] = multi_ctrl_stats
    
    # 对比分析
    print("\n" + "=" * 70)
    print("关键差异分析")
    print("=" * 70)
    
    # 1. 推理时间对比
    if multi_inference_data and 'multi_inference' in results:
        orig_inf_time = original_stats.avg_inference_time
        multi_inf_time = results['multi_inference'].avg_inference_time
        if orig_inf_time > 0 and multi_inf_time > 0:
            diff_pct = (multi_inf_time - orig_inf_time) / orig_inf_time * 100
            print(f"\n推理时间对比:")
            print(f"  原版: {orig_inf_time*1000:.1f}ms")
            print(f"  多进程: {multi_inf_time*1000:.1f}ms")
            print(f"  差异: {diff_pct:+.1f}%")
    
    # 2. 跳变对比
    multi_jumps = 0
    if 'multi_control' in results:
        multi_jumps = results['multi_control'].jump_count
    elif 'multi_inference' in results:
        multi_jumps = results['multi_inference'].jump_count
    
    print(f"\n跳变次数对比:")
    print(f"  原版: {original_stats.jump_count}")
    print(f"  多进程: {multi_jumps}")
    
    if multi_jumps > original_stats.jump_count:
        print(f"  ⚠ 多进程版本跳变更多! 这可能是抖动的原因")
        results['warning'] = "多进程版本跳变次数更多"
    
    # 3. 动作平滑度对比
    if original_stats.action_delta_std is not None:
        orig_smoothness = np.mean(original_stats.action_delta_std[:6])
        print(f"\n动作平滑度 (帧间变化标准差):")
        print(f"  原版: {orig_smoothness:.6f} rad")
        
        if 'multi_control' in results and results['multi_control'].action_delta_std is not None:
            multi_smoothness = np.mean(results['multi_control'].action_delta_std[:6])
            print(f"  多进程 (控制): {multi_smoothness:.6f} rad")
            
            if multi_smoothness > orig_smoothness * 1.2:
                print(f"  ⚠ 多进程版本动作更不平滑!")
                results['warning'] = "多进程版本动作不够平滑"
    
    # 4. 建议
    print("\n" + "=" * 70)
    print("诊断建议")
    print("=" * 70)
    
    if multi_jumps > original_stats.jump_count:
        print("""
可能的原因:
1. Chunk 切换时机不同
   - 原版: 固定步数后切换 (idx >= steps_per_chunk)
   - 多进程: 缓冲区空后切换 (is_action_buffer_empty)
   
2. 状态同步延迟
   - 多进程通过共享内存同步状态
   - 推理时使用的 current_position 可能滞后于实际执行位置
   
3. EMA 滤波时机
   - 确保两版本都在 500Hz 控制循环中应用滤波

建议调试步骤:
1. 检查 chunk 边界时的动作跳变
2. 对比推理触发的时机
3. 验证状态同步的延迟
""")
    else:
        print("两版本行为相似，抖动可能来自其他因素。")
    
    return results


def plot_comparison(
    original_data: Dict[str, Any],
    multi_control_data: Optional[Dict[str, Any]] = None,
    output_path: str = "trajectory_comparison.png",
):
    """绘制对比图"""
    if not HAS_MATPLOTLIB:
        print("[警告] matplotlib 未安装，跳过绘图")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # 1. 关节位置对比 (前 3 个关节)
    for i in range(3):
        ax = axes[0, 0] if i == 0 else (axes[0, 1] if i == 1 else axes[1, 0])
        
        if 'executed_actions' in original_data:
            actions = original_data['executed_actions']
            ax.plot(actions[:, i], label='原版', alpha=0.7)
        
        if multi_control_data and 'executed_actions' in multi_control_data:
            actions = multi_control_data['executed_actions']
            ax.plot(actions[:, i], label='多进程', alpha=0.7)
        
        ax.set_title(f'关节 {i+1}')
        ax.set_xlabel('步数')
        ax.set_ylabel('位置 (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. 夹爪位置对比
    ax = axes[1, 1]
    if 'executed_actions' in original_data:
        actions = original_data['executed_actions']
        ax.plot(actions[:, 6], label='原版', alpha=0.7)
    
    if multi_control_data and 'executed_actions' in multi_control_data:
        actions = multi_control_data['executed_actions']
        ax.plot(actions[:, 6], label='多进程', alpha=0.7)
    
    ax.set_title('夹爪位置')
    ax.set_xlabel('步数')
    ax.set_ylabel('位置 (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 帧间变化对比 (关节)
    ax = axes[2, 0]
    
    if 'executed_actions' in original_data:
        actions = original_data['executed_actions']
        deltas = np.abs(np.diff(actions[:, :6], axis=0)).max(axis=1)
        ax.plot(deltas, label='原版', alpha=0.7)
    
    if multi_control_data and 'executed_actions' in multi_control_data:
        actions = multi_control_data['executed_actions']
        deltas = np.abs(np.diff(actions[:, :6], axis=0)).max(axis=1)
        ax.plot(deltas, label='多进程', alpha=0.7)
    
    ax.axhline(y=0.02, color='r', linestyle='--', label='跳变阈值', alpha=0.5)
    ax.set_title('最大关节帧间变化')
    ax.set_xlabel('步数')
    ax.set_ylabel('变化量 (rad)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 推理时间对比
    ax = axes[2, 1]
    
    if 'inference_times' in original_data:
        times = original_data['inference_times'] * 1000  # ms
        ax.plot(times, label='原版', alpha=0.7)
    
    if multi_control_data and 'inference_times' in multi_control_data:
        times = multi_control_data['inference_times'] * 1000
        ax.plot(times, label='多进程', alpha=0.7)
    
    ax.set_title('推理时间')
    ax.set_xlabel('推理次数')
    ax.set_ylabel('时间 (ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n对比图已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="轨迹对比分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--original", "-o", type=str,
                       help="原版轨迹文件 (.npz)")
    parser.add_argument("--multi-inference", "-i", type=str,
                       help="多进程推理节点轨迹文件 (.npz)")
    parser.add_argument("--multi-control", "-c", type=str,
                       help="多进程控制节点轨迹文件 (.npz)")
    parser.add_argument("--single", "-s", type=str,
                       help="单独分析一个轨迹文件")
    parser.add_argument("--output", type=str,
                       help="输出报告文件")
    parser.add_argument("--plot", action="store_true",
                       help="生成对比图")
    parser.add_argument("--plot-output", type=str, default="trajectory_comparison.png",
                       help="对比图输出路径")
    
    args = parser.parse_args()
    
    if args.single:
        # 单独分析
        print(f"\n加载轨迹: {args.single}")
        data = load_trajectory(args.single)
        stats = analyze_single_trajectory(data, os.path.basename(args.single))
        print_stats(stats)
        
    elif args.original:
        # 对比分析
        print(f"\n加载原版轨迹: {args.original}")
        original_data = load_trajectory(args.original)
        
        multi_inference_data = None
        multi_control_data = None
        
        if args.multi_inference:
            print(f"加载多进程推理轨迹: {args.multi_inference}")
            multi_inference_data = load_trajectory(args.multi_inference)
        
        if args.multi_control:
            print(f"加载多进程控制轨迹: {args.multi_control}")
            multi_control_data = load_trajectory(args.multi_control)
        
        # 对比分析
        results = compare_trajectories(
            original_data,
            multi_inference_data,
            multi_control_data,
        )
        
        # 绘图
        if args.plot:
            control_data = multi_control_data or multi_inference_data
            if control_data:
                plot_comparison(original_data, control_data, args.plot_output)
        
        # 输出报告
        if args.output:
            import json
            # 转换为可序列化的格式
            report = {}
            for key, stats in results.items():
                if isinstance(stats, TrajectoryStats):
                    report[key] = {
                        'name': stats.name,
                        'total_steps': stats.total_steps,
                        'total_duration': stats.total_duration,
                        'inference_count': stats.inference_count,
                        'avg_inference_time': stats.avg_inference_time,
                        'jump_count': stats.jump_count,
                        'max_jump': stats.max_jump,
                    }
                else:
                    report[key] = str(stats)
            
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n报告已保存: {args.output}")
    else:
        parser.print_help()
        print("\n错误: 请指定 --original 或 --single 参数")
        sys.exit(1)


if __name__ == "__main__":
    main()
