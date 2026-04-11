#!/usr/bin/env python3
"""
推理数据分析脚本 - 分析 inference_episode_*.hdf5 数据。

当前语义:
1. `action_model` 表示模型输出 chunk
2. `action_executed` 表示真实执行 chunk
3. `action` 表示 `action_executed[:, 0, :]` 的兼容单步视图

使用方法:
    python analyze_inference_data.py --data_dir /path/to/inference_logs
    python analyze_inference_data.py --files inference_episode_0001.hdf5
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any

import h5py
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found, visualization disabled")


def load_hdf5_data(filepath: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    with h5py.File(filepath, 'r') as f:
        data['action'] = f['action'][:]
        data['action_model'] = f['action_model'][:]
        data['action_executed'] = f['action_executed'][:]
        data['images'] = f['observations/images'][:]
        data['gripper'] = f['observations/gripper'][:]
        data['qpos'] = f['observations/qpos'][:]
        data['qpos_end'] = f['observations/qpos_end'][:]
        data['qpos_joint'] = f['observations/qpos_joint'][:]
        data['timestamps'] = f['observations/timestamps'][:]
        data['attrs'] = dict(f.attrs)
    return data


def analyze_basic_stats(data: dict[str, Any], filename: str) -> dict[str, Any]:
    timestamps = data['timestamps']
    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0
    avg_dt = float(np.mean(np.diff(timestamps))) if len(timestamps) >= 2 else 0.0
    freq = 1.0 / avg_dt if avg_dt > 0 else 0.0
    return {
        'filename': os.path.basename(filename),
        'num_steps': int(data['attrs'].get('num_steps', len(timestamps))),
        'duration_sec': duration,
        'avg_freq_hz': freq,
        'pred_horizon': int(data['attrs'].get('pred_horizon', data['action_model'].shape[1])),
        'action_dim': int(data['attrs'].get('action_dim', data['action_model'].shape[2])),
        'success': data['attrs'].get('success'),
        'outcome_label': data['attrs'].get('outcome_label'),
        'safety_clip_rate': data['attrs'].get('safety_clip_rate'),
    }


def analyze_execution_difference(data: dict[str, Any]) -> dict[str, Any]:
    action_model_first = data['action_model'][:, 0, :]
    action_executed_first = data['action_executed'][:, 0, :]
    diff = action_executed_first - action_model_first
    diff_abs = np.abs(diff)

    changed_steps = np.any(diff_abs > 1e-9, axis=1)
    num_changed_steps = int(changed_steps.sum())
    action_dim = diff.shape[1]

    xyz_cols = [0, 1, 2] if action_dim >= 3 else []
    quat_cols = [3, 4, 5, 6] if action_dim >= 7 else []
    gripper_col = action_dim - 1 if action_dim >= 1 else None

    dim_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
    dim_stats = {}
    for idx in range(action_dim):
        name = dim_names[idx] if idx < len(dim_names) else f'dim_{idx}'
        dim_stats[name] = {
            'mean_abs': float(np.mean(diff_abs[:, idx])),
            'max_abs': float(np.max(diff_abs[:, idx])),
            'p95_abs': float(np.percentile(diff_abs[:, idx], 95)),
        }

    xyz_norm = np.linalg.norm(diff[:, xyz_cols], axis=1) if xyz_cols else np.zeros(len(diff))
    quat_norm = np.linalg.norm(diff[:, quat_cols], axis=1) if quat_cols else np.zeros(len(diff))
    gripper_abs = diff_abs[:, gripper_col] if gripper_col is not None else np.zeros(len(diff))

    return {
        'diff': diff,
        'changed_steps_mask': changed_steps,
        'num_changed_steps': num_changed_steps,
        'changed_step_ratio': float(num_changed_steps / len(diff)) if len(diff) > 0 else 0.0,
        'overall_mse': float(np.mean(diff ** 2)),
        'overall_mae': float(np.mean(diff_abs)),
        'xyz_diff_norm_mean': float(np.mean(xyz_norm)),
        'xyz_diff_norm_p95': float(np.percentile(xyz_norm, 95)) if len(xyz_norm) > 0 else 0.0,
        'quat_diff_norm_mean': float(np.mean(quat_norm)),
        'quat_diff_norm_p95': float(np.percentile(quat_norm, 95)) if len(quat_norm) > 0 else 0.0,
        'gripper_diff_mean': float(np.mean(gripper_abs)),
        'gripper_diff_p95': float(np.percentile(gripper_abs, 95)) if len(gripper_abs) > 0 else 0.0,
        'dim_stats': dim_stats,
    }


def analyze_trajectory(data: dict[str, Any]) -> dict[str, Any]:
    qpos_end = data['qpos_end']
    qpos_joint = data['qpos_joint']
    timestamps = data['timestamps']
    xyz = qpos_end[:, :3]
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    xyz_range = xyz_max - xyz_min
    xyz_diff = np.diff(xyz, axis=0)
    step_distances = np.linalg.norm(xyz_diff, axis=1)
    total_distance = float(step_distances.sum())
    dt = np.diff(timestamps)
    velocities = step_distances / np.clip(dt, 1e-6, None) if len(dt) > 0 else np.array([], dtype=float)
    joints = qpos_joint[:, :6]
    joint_ranges = joints.max(axis=0) - joints.min(axis=0)
    gripper = data['gripper']
    gripper_changes = int(np.sum(np.abs(np.diff(gripper)) > 0.01))
    return {
        'xyz': xyz,
        'xyz_min': xyz_min,
        'xyz_max': xyz_max,
        'xyz_range': xyz_range,
        'total_distance': total_distance,
        'avg_velocity': float(np.mean(velocities)) if len(velocities) > 0 else 0.0,
        'max_velocity': float(np.max(velocities)) if len(velocities) > 0 else 0.0,
        'joints': joints,
        'joint_ranges': joint_ranges,
        'gripper': gripper,
        'gripper_changes': gripper_changes,
        'timestamps': timestamps,
    }


def print_analysis_report(
    basic_stats: dict[str, Any],
    execution_diff: dict[str, Any],
    trajectory: dict[str, Any],
) -> None:
    print("\n" + "=" * 70)
    print(f"数据分析报告: {basic_stats['filename']}")
    print("=" * 70)
    print("\n【基本统计】")
    print(f"  总步数: {basic_stats['num_steps']}")
    print(f"  时长: {basic_stats['duration_sec']:.2f} 秒")
    print(f"  平均频率: {basic_stats['avg_freq_hz']:.1f} Hz")
    print(f"  预测 horizon: {basic_stats['pred_horizon']}")
    print(f"  动作维度: {basic_stats['action_dim']}")
    print(f"  outcome: {basic_stats['outcome_label']} (success={basic_stats['success']})")
    if basic_stats['safety_clip_rate'] is not None:
        print(f"  safety_clip_rate: {float(basic_stats['safety_clip_rate']):.2%}")

    print("\n【执行差异 (model vs executed)】")
    print(f"  执行改写步数: {execution_diff['num_changed_steps']} / {basic_stats['num_steps']} ({execution_diff['changed_step_ratio']:.1%})")
    print(f"  整体 MSE: {execution_diff['overall_mse']:.6f}")
    print(f"  整体 MAE: {execution_diff['overall_mae']:.6f}")
    print(f"  XYZ diff norm: mean={execution_diff['xyz_diff_norm_mean']:.6f}, p95={execution_diff['xyz_diff_norm_p95']:.6f}")
    print(f"  Quaternion diff norm: mean={execution_diff['quat_diff_norm_mean']:.6f}, p95={execution_diff['quat_diff_norm_p95']:.6f}")
    print(f"  Gripper diff abs: mean={execution_diff['gripper_diff_mean']:.6f}, p95={execution_diff['gripper_diff_p95']:.6f}")

    print("\n【轨迹分析】")
    print(f"  XYZ 范围:")
    print(f"    X: [{trajectory['xyz_min'][0]:.4f}, {trajectory['xyz_max'][0]:.4f}] (范围: {trajectory['xyz_range'][0]:.4f})")
    print(f"    Y: [{trajectory['xyz_min'][1]:.4f}, {trajectory['xyz_max'][1]:.4f}] (范围: {trajectory['xyz_range'][1]:.4f})")
    print(f"    Z: [{trajectory['xyz_min'][2]:.4f}, {trajectory['xyz_max'][2]:.4f}] (范围: {trajectory['xyz_range'][2]:.4f})")
    print(f"  总运动距离: {trajectory['total_distance']:.4f} m")
    print(f"  平均速度: {trajectory['avg_velocity'] * 100:.2f} cm/s")
    print(f"  最大速度: {trajectory['max_velocity'] * 100:.2f} cm/s")
    print(f"  Gripper 切换次数: {trajectory['gripper_changes']}")
    print("\n" + "=" * 70)


def plot_trajectory_3d(
    trajectory: dict[str, Any],
    execution_diff: dict[str, Any],
    save_path: str | None = None,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    fig = plt.figure(figsize=(12, 5))
    xyz = trajectory['xyz']
    timestamps = trajectory['timestamps']
    changed_mask = execution_diff['changed_steps_mask']

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', alpha=0.3, linewidth=0.5)
    ax1.scatter(xyz[changed_mask, 0], xyz[changed_mask, 1], xyz[changed_mask, 2], c='red', s=10, alpha=0.5, label='Executed != model')
    ax1.scatter(xyz[~changed_mask, 0], xyz[~changed_mask, 1], xyz[~changed_mask, 2], c='blue', s=2, alpha=0.2, label='Executed == model')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('End-effector Trajectory (3D)')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    t = timestamps - timestamps[0]
    ax2.plot(t, xyz[:, 0], 'r-', label='X', alpha=0.7)
    ax2.plot(t, xyz[:, 1], 'g-', label='Y', alpha=0.7)
    ax2.plot(t, xyz[:, 2], 'b-', label='Z', alpha=0.7)
    if np.any(changed_mask):
        ax2.scatter(t[changed_mask], xyz[changed_mask, 0], c='red', s=8, alpha=0.4, label='Changed step')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('XYZ vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  轨迹图已保存: {save_path}")
    plt.close()


def plot_execution_analysis(
    execution_diff: dict[str, Any],
    trajectory: dict[str, Any],
    save_path: str | None = None,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    diff = execution_diff['diff']
    changed_mask = execution_diff['changed_steps_mask']
    timestamps = trajectory['timestamps']
    t = timestamps - timestamps[0]

    ax1 = axes[0, 0]
    change_signal = changed_mask.astype(float)
    window = min(30, max(len(change_signal), 1))
    if len(change_signal) > 0:
        density = np.convolve(change_signal, np.ones(window) / window, mode='same')
        ax1.fill_between(np.arange(len(density)), density, alpha=0.5, color='orange')
        ax1.plot(np.arange(len(density)), density, color='orange', linewidth=1)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Execution change density')
    ax1.set_title('Executed != Model Density')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    dim_max = np.max(np.abs(diff), axis=0) if len(diff) > 0 else np.array([])
    dim_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
    x = np.arange(len(dim_max))
    labels = [dim_names[i] if i < len(dim_names) else f'd{i}' for i in x]
    ax2.bar(x, dim_max, color='steelblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylabel('Max abs diff')
    ax2.set_title('Per-dimension max |executed - model|')
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = axes[1, 0]
    if diff.shape[1] >= 3:
        ax3.hist(diff[:, 0], bins=50, alpha=0.5, label='X', color='red')
        ax3.hist(diff[:, 1], bins=50, alpha=0.5, label='Y', color='green')
        ax3.hist(diff[:, 2], bins=50, alpha=0.5, label='Z', color='blue')
    ax3.set_xlabel('Action Difference (executed - model)')
    ax3.set_ylabel('Count')
    ax3.set_title('Position Difference Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    gripper = trajectory['gripper']
    ax4.plot(t, gripper, 'b-', linewidth=1, label='Gripper state')
    if np.any(changed_mask):
        ax4.scatter(t[changed_mask], gripper[changed_mask], c='red', s=15, alpha=0.5, label='Changed step', zorder=5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Gripper Position')
    ax4.set_title('Gripper with execution-change markers')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  执行分析图已保存: {save_path}")
    plt.close()


def plot_action_comparison(
    data: dict[str, Any],
    execution_diff: dict[str, Any],
    save_path: str | None = None,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    action_model = data['action_model'][:, 0, :]
    action_executed = data['action_executed'][:, 0, :]
    timestamps = data['timestamps']
    t = timestamps - timestamps[0]
    changed_mask = execution_diff['changed_steps_mask']
    dim_idx = [0, 1, 2]
    dim_names = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']

    for i, (idx, name, color) in enumerate(zip(dim_idx, dim_names, colors, strict=False)):
        ax = axes[i]
        if idx < action_model.shape[1]:
            ax.plot(t, action_model[:, idx], color=color, alpha=0.5, linewidth=1, label=f'Model {name}')
            ax.plot(t, action_executed[:, idx], color=color, linestyle='--', linewidth=1, label=f'Executed {name}')
        if np.any(changed_mask):
            ax.scatter(t[changed_mask], action_executed[changed_mask, idx], c='black', s=6, alpha=0.4)
        ax.set_ylabel(f'{name} Action')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel('Time (s)')
    axes[0].set_title('Model Output vs Executed Output')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  动作对比图已保存: {save_path}")
    plt.close()


def analyze_single_file(filepath: str, save_dir: str | None = None) -> dict[str, Any]:
    print(f"\n正在分析: {filepath}")
    data = load_hdf5_data(filepath)
    basic_stats = analyze_basic_stats(data, filepath)
    execution_diff = analyze_execution_difference(data)
    trajectory = analyze_trajectory(data)
    print_analysis_report(basic_stats, execution_diff, trajectory)

    if save_dir and HAS_MATPLOTLIB:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        print("\n生成可视化图表...")
        plot_trajectory_3d(trajectory, execution_diff, os.path.join(save_dir, f'{base_name}_trajectory.png'))
        plot_execution_analysis(execution_diff, trajectory, os.path.join(save_dir, f'{base_name}_execution.png'))
        plot_action_comparison(data, execution_diff, os.path.join(save_dir, f'{base_name}_action_compare.png'))

    return {
        'basic_stats': basic_stats,
        'execution_diff': execution_diff,
        'trajectory': trajectory,
    }


def compare_episodes(results: list[dict[str, Any]], save_dir: str | None = None) -> None:
    if len(results) < 2:
        return
    print("\n" + "=" * 70)
    print("多 Episode 对比")
    print("=" * 70)
    print("\n{:<40} {:>10} {:>10} {:>14} {:>10}".format(
        'Episode', 'Steps', 'Duration', 'Exec Diff', 'Distance'
    ))
    print("-" * 90)
    for r in results:
        bs = r['basic_stats']
        ed = r['execution_diff']
        tr = r['trajectory']
        print("{:<40} {:>10} {:>10.1f}s {:>13.1%} {:>10.4f}m".format(
            bs['filename'][:40],
            bs['num_steps'],
            bs['duration_sec'],
            ed['changed_step_ratio'],
            tr['total_distance'],
        ))

    if save_dir and HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        names = [r['basic_stats']['filename'][:20] for r in results]

        steps = [r['basic_stats']['num_steps'] for r in results]
        axes[0].bar(names, steps, color='steelblue')
        axes[0].set_ylabel('Steps')
        axes[0].set_title('Episode Length')
        axes[0].tick_params(axis='x', rotation=45)

        diff_ratios = [r['execution_diff']['changed_step_ratio'] * 100 for r in results]
        axes[1].bar(names, diff_ratios, color='coral')
        axes[1].set_ylabel('Executed!=Model Ratio (%)')
        axes[1].set_title('Execution Override Ratio')
        axes[1].tick_params(axis='x', rotation=45)

        distances = [r['trajectory']['total_distance'] * 100 for r in results]
        axes[2].bar(names, distances, color='seagreen')
        axes[2].set_ylabel('Distance (cm)')
        axes[2].set_title('Total Movement Distance')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        path = os.path.join(save_dir, 'episode_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\n对比图已保存: {path}")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='分析 inference 采集数据')
    parser.add_argument('--data_dir', type=str, default='/home/lizh/rl-vla/inference_logs', help='数据目录')
    parser.add_argument('--files', type=str, nargs='+', default=None, help='指定要分析的文件')
    parser.add_argument('--save_dir', type=str, default=None, help='保存可视化图表的目录 (默认: data_dir/analysis)')
    parser.add_argument('--pattern', type=str, default='inference_episode_*.hdf5', help='文件匹配模式')
    parser.add_argument('--no_viz', action='store_true', help='禁用可视化')
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))

    if not files:
        print(f"[ERROR] 未找到匹配的文件: {args.pattern}")
        print(f"  目录: {args.data_dir}")
        return

    print(f"找到 {len(files)} 个文件待分析:")
    for file_path in files:
        print(f"  - {os.path.basename(file_path)}")

    save_dir = None
    if not args.no_viz:
        save_dir = args.save_dir or os.path.join(args.data_dir, 'analysis')

    results = []
    for filepath in files:
        try:
            results.append(analyze_single_file(filepath, save_dir))
        except Exception as exc:
            print(f"[ERROR] 分析 {filepath} 失败: {exc}")
            import traceback
            traceback.print_exc()

    if len(results) > 1:
        compare_episodes(results, save_dir)

    print("\n分析完成!")
    if save_dir:
        print(f"可视化结果保存在: {save_dir}")


if __name__ == '__main__':
    main()
