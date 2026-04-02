#!/usr/bin/env python3
"""
推理数据分析脚本 - 分析 recorder 产出的 inference_episode_*.hdf5 数据

功能:
1. 基本统计: 步数、干预率、时间分布
2. 干预分析: 干预频率、持续时间、维度分布
3. 动作分析: 模型输出 vs 干预后输出的差异
4. 轨迹可视化: xyz 轨迹、gripper 状态、关节角度
5. 干预时刻可视化: 标注干预发生的时间点

使用方法:
    python analyze_inference_data.py --data_dir /path/to/inference_logs
    python analyze_inference_data.py --files inference_episode_0001.hdf5 inference_episode_0002.hdf5
    python analyze_inference_data.py --data_dir /path/to/logs --save_dir /path/to/output
"""

import argparse
import os
import sys
import glob
import numpy as np
import h5py
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# 尝试导入可视化库
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found, visualization disabled")


def load_hdf5_data(filepath: str) -> Dict[str, Any]:
    """加载 recorder HDF5 数据。"""
    data = {}
    with h5py.File(filepath, 'r') as f:
        data['action'] = f['action'][:]
        data['action_model'] = f['action_model'][:]
        data['action_intervened'] = f['action_intervened'][:]
        data['intervention_mask'] = f['intervention_mask'][:]

        data['images'] = f['observations/images'][:]
        data['gripper'] = f['observations/gripper'][:]
        data['qpos'] = f['observations/qpos'][:]
        data['qpos_end'] = f['observations/qpos_end'][:]
        data['qpos_joint'] = f['observations/qpos_joint'][:]
        data['timestamps'] = f['observations/timestamps'][:]
        data['attrs'] = dict(f.attrs)

    return data


def analyze_basic_stats(data: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """基本统计分析"""
    stats = {}
    
    num_steps = data['attrs'].get('num_steps', len(data['timestamps']))
    intervention_ratio = data['attrs'].get('intervention_ratio', 0)
    
    # 时间分析
    timestamps = data['timestamps']
    duration = timestamps[-1] - timestamps[0]
    avg_dt = np.mean(np.diff(timestamps))
    freq = 1.0 / avg_dt if avg_dt > 0 else 0
    
    stats['filename'] = os.path.basename(filename)
    stats['num_steps'] = num_steps
    stats['duration_sec'] = duration
    stats['avg_freq_hz'] = freq
    stats['intervention_ratio'] = intervention_ratio
    stats['pred_horizon'] = data['attrs'].get('pred_horizon', 16)
    stats['action_dim'] = data['attrs'].get('action_dim', 15)
    
    return stats


def analyze_intervention(data: Dict[str, Any]) -> Dict[str, Any]:
    """干预分析"""
    mask = data['intervention_mask']  # [T, pred_horizon, action_dim]
    T, pred_horizon, action_dim = mask.shape
    
    # 每步是否有干预 (任意维度被干预)
    step_has_intervention = mask.any(axis=(1, 2))  # [T]
    num_intervention_steps = step_has_intervention.sum()
    
    # 干预维度分布 (哪些维度被干预最多)
    dim_intervention_count = mask.sum(axis=(0, 1))  # [action_dim]
    
    # ===== 更智能的干预检测 =====
    # 区分"真正的新干预"和"状态保持"
    # 通过检测 action 差异变化来识别
    action_model = data['action_model'][:, 0, :]  # [T, action_dim]
    action_intervened = data['action_intervened'][:, 0, :]
    diff = action_intervened - action_model  # [T, action_dim]
    
    # XYZ 干预检测 (维度 7,8,9 对于 15D action)
    xyz_cols = [7, 8, 9] if action_dim >= 10 else [0, 1, 2]
    xyz_diff = diff[:, xyz_cols]  # [T, 3]
    xyz_diff_mag = np.linalg.norm(xyz_diff, axis=1)  # [T]
    
    # 检测 XYZ 干预：差异变化超过阈值
    xyz_threshold = 0.001  # 1mm
    xyz_intervened = xyz_diff_mag > xyz_threshold
    
    # Gripper 干预检测：检测变化点
    gripper_col = 14 if action_dim == 15 else (7 if action_dim == 8 else 6)
    gripper_diff = diff[:, gripper_col]
    
    # 检测 gripper 状态变化点（真正的干预时刻）
    gripper_diff_change = np.abs(np.diff(gripper_diff, prepend=gripper_diff[0]))
    gripper_threshold = 0.1
    gripper_change_points = gripper_diff_change > gripper_threshold
    
    # 合并：真正有干预的步骤
    actual_intervention = xyz_intervened | gripper_change_points
    num_actual_intervention = actual_intervention.sum()
    
    # 干预段落分析 (连续干预的段落) - 使用更智能的检测
    intervention_segments = []
    in_segment = False
    segment_start = 0
    
    for i, has_int in enumerate(actual_intervention):
        if has_int and not in_segment:
            in_segment = True
            segment_start = i
        elif not has_int and in_segment:
            in_segment = False
            intervention_segments.append((segment_start, i))
    if in_segment:
        intervention_segments.append((segment_start, T))
    
    segment_lengths = [end - start for start, end in intervention_segments]
    
    # 干预密度 (滑动窗口) - 使用更智能的检测
    window_size = 30
    intervention_density = np.convolve(
        actual_intervention.astype(float), 
        np.ones(window_size) / window_size, 
        mode='valid'
    )
    
    return {
        'num_intervention_steps': int(num_intervention_steps),
        'intervention_ratio': num_intervention_steps / T,
        # 更准确的干预统计
        'num_actual_intervention': int(num_actual_intervention),
        'actual_intervention_ratio': num_actual_intervention / T,
        'xyz_intervention_count': int(xyz_intervened.sum()),
        'gripper_change_count': int(gripper_change_points.sum()),
        # 原始统计
        'dim_intervention_count': dim_intervention_count,
        'num_segments': len(intervention_segments),
        'segments': intervention_segments,
        'segment_lengths': segment_lengths,
        'avg_segment_length': np.mean(segment_lengths) if segment_lengths else 0,
        'max_segment_length': max(segment_lengths) if segment_lengths else 0,
        'intervention_density': intervention_density,
        'step_has_intervention': step_has_intervention,
        # 新增：更智能的干预标记
        'actual_intervention': actual_intervention,
        'xyz_intervened': xyz_intervened,
        'gripper_change_points': gripper_change_points,
    }


def analyze_action_difference(data: Dict[str, Any]) -> Dict[str, Any]:
    """分析模型输出和干预后输出的差异"""
    action_model = data['action_model']  # [T, pred_horizon, action_dim]
    action_intervened = data['action_intervened']  # [T, pred_horizon, action_dim]
    mask = data['intervention_mask']  # [T, pred_horizon, action_dim]
    
    # 只分析第一步动作 (实际执行的)
    action_model_first = action_model[:, 0, :]  # [T, action_dim]
    action_intervened_first = action_intervened[:, 0, :]  # [T, action_dim]
    
    # 差异
    diff = action_intervened_first - action_model_first  # [T, action_dim]
    
    # 被干预步骤的差异
    step_has_intervention = mask.any(axis=(1, 2))
    diff_when_intervened = diff[step_has_intervention]
    
    # 各维度差异统计
    dim_names = [
        'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'gripper_j',
        'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper_e'
    ]
    
    dim_stats = {}
    for i in range(min(len(dim_names), diff.shape[1])):
        d = diff[:, i]
        d_int = diff_when_intervened[:, i] if len(diff_when_intervened) > 0 else np.array([0])
        dim_stats[dim_names[i]] = {
            'mean': float(np.mean(d)),
            'std': float(np.std(d)),
            'max_abs': float(np.max(np.abs(d))),
            'mean_when_intervened': float(np.mean(d_int)),
            'std_when_intervened': float(np.std(d_int)),
        }
    
    return {
        'diff': diff,
        'diff_when_intervened': diff_when_intervened,
        'dim_stats': dim_stats,
        'overall_mse': float(np.mean(diff ** 2)),
        'overall_mae': float(np.mean(np.abs(diff))),
    }


def analyze_trajectory(data: Dict[str, Any]) -> Dict[str, Any]:
    """轨迹分析"""
    qpos_end = data['qpos_end']  # [T, 8]: x, y, z, qx, qy, qz, qw, gripper
    qpos_joint = data['qpos_joint']  # [T, 7]: j1-j6, gripper
    timestamps = data['timestamps']
    
    # 末端位置
    xyz = qpos_end[:, :3]  # [T, 3]
    
    # 运动范围
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    xyz_range = xyz_max - xyz_min
    
    # 运动距离
    xyz_diff = np.diff(xyz, axis=0)
    step_distances = np.linalg.norm(xyz_diff, axis=1)
    total_distance = step_distances.sum()
    
    # 速度分析
    dt = np.diff(timestamps)
    velocities = step_distances / np.clip(dt, 1e-6, None)
    
    # 关节分析
    joints = qpos_joint[:, :6]  # [T, 6]
    joint_ranges = joints.max(axis=0) - joints.min(axis=0)
    
    # Gripper 状态
    gripper = data['gripper']
    gripper_changes = np.sum(np.abs(np.diff(gripper)) > 0.01)
    
    return {
        'xyz': xyz,
        'xyz_min': xyz_min,
        'xyz_max': xyz_max,
        'xyz_range': xyz_range,
        'total_distance': total_distance,
        'avg_velocity': float(np.mean(velocities)),
        'max_velocity': float(np.max(velocities)),
        'joints': joints,
        'joint_ranges': joint_ranges,
        'gripper': gripper,
        'gripper_changes': gripper_changes,
        'timestamps': timestamps,
    }


def print_analysis_report(
    basic_stats: Dict[str, Any],
    intervention_analysis: Dict[str, Any],
    action_diff: Dict[str, Any],
    trajectory: Dict[str, Any],
):
    """打印分析报告"""
    print("\n" + "=" * 70)
    print(f"数据分析报告: {basic_stats['filename']}")
    print("=" * 70)
    
    # 基本统计
    print("\n【基本统计】")
    print(f"  总步数: {basic_stats['num_steps']}")
    print(f"  时长: {basic_stats['duration_sec']:.2f} 秒")
    print(f"  平均频率: {basic_stats['avg_freq_hz']:.1f} Hz")
    print(f"  预测 horizon: {basic_stats['pred_horizon']}")
    print(f"  动作维度: {basic_stats['action_dim']}")
    
    # 干预分析
    print("\n【干预分析】")
    print(f"  (原始) 干预标记步数: {intervention_analysis['num_intervention_steps']} / {basic_stats['num_steps']} ({intervention_analysis['intervention_ratio']:.1%})")
    print(f"  (智能检测) 实际干预步数: {intervention_analysis['num_actual_intervention']} / {basic_stats['num_steps']} ({intervention_analysis['actual_intervention_ratio']:.1%})")
    print(f"    - XYZ 位移干预: {intervention_analysis['xyz_intervention_count']} 步")
    print(f"    - Gripper 变化点: {intervention_analysis['gripper_change_count']} 次")
    print(f"  干预段落数: {intervention_analysis['num_segments']}")
    print(f"  平均段落长度: {intervention_analysis['avg_segment_length']:.1f} 步")
    print(f"  最长段落: {intervention_analysis['max_segment_length']} 步")
    
    # 干预维度分布
    dim_counts = intervention_analysis['dim_intervention_count']
    dim_names = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'grip_j', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'grip_e']
    print("\n  干预维度分布 (前10):")
    sorted_dims = sorted(enumerate(dim_counts), key=lambda x: x[1], reverse=True)
    for i, count in sorted_dims[:10]:
        name = dim_names[i] if i < len(dim_names) else f'd{i}'
        print(f"    {name}: {int(count)}")
    
    # 动作差异
    print("\n【动作差异 (模型 vs 干预后)】")
    print(f"  整体 MSE: {action_diff['overall_mse']:.6f}")
    print(f"  整体 MAE: {action_diff['overall_mae']:.6f}")
    print("\n  各维度差异 (干预时):")
    for dim, stats in list(action_diff['dim_stats'].items())[:8]:
        print(f"    {dim}: mean={stats['mean_when_intervened']:.4f}, std={stats['std_when_intervened']:.4f}")
    
    # 轨迹分析
    print("\n【轨迹分析】")
    print(f"  XYZ 范围:")
    print(f"    X: [{trajectory['xyz_min'][0]:.4f}, {trajectory['xyz_max'][0]:.4f}] (范围: {trajectory['xyz_range'][0]:.4f})")
    print(f"    Y: [{trajectory['xyz_min'][1]:.4f}, {trajectory['xyz_max'][1]:.4f}] (范围: {trajectory['xyz_range'][1]:.4f})")
    print(f"    Z: [{trajectory['xyz_min'][2]:.4f}, {trajectory['xyz_max'][2]:.4f}] (范围: {trajectory['xyz_range'][2]:.4f})")
    print(f"  总运动距离: {trajectory['total_distance']:.4f} m")
    print(f"  平均速度: {trajectory['avg_velocity']*100:.2f} cm/s")
    print(f"  最大速度: {trajectory['max_velocity']*100:.2f} cm/s")
    print(f"  Gripper 切换次数: {trajectory['gripper_changes']}")
    
    print("\n" + "=" * 70)


def plot_trajectory_3d(
    trajectory: Dict[str, Any],
    intervention_analysis: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """绘制 3D 轨迹"""
    if not HAS_MATPLOTLIB:
        return
    
    fig = plt.figure(figsize=(12, 5))
    
    xyz = trajectory['xyz']
    timestamps = trajectory['timestamps']
    # 使用更智能的干预检测
    step_has_int = intervention_analysis.get('actual_intervention', 
                                              intervention_analysis['step_has_intervention'])
    
    # 3D 轨迹
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 分段绘制: 干预 vs 非干预
    colors = ['blue' if not h else 'red' for h in step_has_int]
    
    # 绘制轨迹线
    ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', alpha=0.3, linewidth=0.5)
    
    # 标记干预点
    int_mask = step_has_int
    ax1.scatter(xyz[int_mask, 0], xyz[int_mask, 1], xyz[int_mask, 2], 
                c='red', s=10, alpha=0.5, label='Intervention')
    ax1.scatter(xyz[~int_mask, 0], xyz[~int_mask, 1], xyz[~int_mask, 2], 
                c='blue', s=2, alpha=0.2, label='Model only')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('End-effector Trajectory (3D)')
    ax1.legend()
    
    # XYZ 随时间变化
    ax2 = fig.add_subplot(122)
    t = timestamps - timestamps[0]
    
    ax2.plot(t, xyz[:, 0], 'r-', label='X', alpha=0.7)
    ax2.plot(t, xyz[:, 1], 'g-', label='Y', alpha=0.7)
    ax2.plot(t, xyz[:, 2], 'b-', label='Z', alpha=0.7)
    
    # 标记干预时段
    for start, end in intervention_analysis['segments']:
        ax2.axvspan(t[start], t[min(end, len(t)-1)], alpha=0.2, color='yellow')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('XYZ vs Time (yellow = intervention)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  轨迹图已保存: {save_path}")
    plt.close()


def plot_intervention_analysis(
    intervention_analysis: Dict[str, Any],
    action_diff: Dict[str, Any],
    trajectory: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """绘制干预分析图"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 干预密度随时间变化
    ax1 = axes[0, 0]
    density = intervention_analysis['intervention_density']
    t = np.arange(len(density))
    ax1.fill_between(t, density, alpha=0.5, color='orange')
    ax1.plot(t, density, 'orange', linewidth=1)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Intervention Density')
    ax1.set_title('Intervention Density (30-step window)')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # 2. 干预维度分布
    ax2 = axes[0, 1]
    dim_counts = intervention_analysis['dim_intervention_count']
    dim_names = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'g_j', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'g_e']
    x = np.arange(len(dim_counts))
    colors = ['steelblue'] * 7 + ['coral'] * 8  # 关节蓝色，末端橙色
    ax2.bar(x, dim_counts, color=colors[:len(dim_counts)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(dim_names[:len(dim_counts)], rotation=45)
    ax2.set_xlabel('Action Dimension')
    ax2.set_ylabel('Intervention Count')
    ax2.set_title('Intervention by Dimension')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 动作差异分布 (XYZ)
    ax3 = axes[1, 0]
    diff = action_diff['diff']
    if diff.shape[1] >= 10:
        # XYZ 在维度 7, 8, 9
        ax3.hist(diff[:, 7], bins=50, alpha=0.5, label='X', color='red')
        ax3.hist(diff[:, 8], bins=50, alpha=0.5, label='Y', color='green')
        ax3.hist(diff[:, 9], bins=50, alpha=0.5, label='Z', color='blue')
    ax3.set_xlabel('Action Difference (intervened - model)')
    ax3.set_ylabel('Count')
    ax3.set_title('Action Difference Distribution (XYZ)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Gripper 状态和干预
    ax4 = axes[1, 1]
    gripper = trajectory['gripper']
    timestamps = trajectory['timestamps']
    t = timestamps - timestamps[0]
    # 使用更智能的干预检测
    step_has_int = intervention_analysis.get('actual_intervention', 
                                              intervention_analysis['step_has_intervention'])
    
    ax4.plot(t, gripper, 'b-', linewidth=1, label='Gripper state')
    
    # 标记干预点
    int_times = t[step_has_int]
    int_gripper = gripper[step_has_int]
    ax4.scatter(int_times, int_gripper, c='red', s=15, alpha=0.5, label='Intervention', zorder=5)
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Gripper Position')
    ax4.set_title('Gripper State with Intervention Markers')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  干预分析图已保存: {save_path}")
    plt.close()


def plot_action_comparison(
    data: Dict[str, Any],
    intervention_analysis: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """绘制模型输出和干预后输出的对比"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    action_model = data['action_model'][:, 0, :]  # [T, action_dim]
    action_intervened = data['action_intervened'][:, 0, :]
    timestamps = data['timestamps']
    t = timestamps - timestamps[0]
    step_has_int = intervention_analysis['step_has_intervention']
    
    # XYZ 对比 (维度 7, 8, 9)
    dim_idx = [7, 8, 9]
    dim_names = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    
    for i, (idx, name, color) in enumerate(zip(dim_idx, dim_names, colors)):
        ax = axes[i]
        
        if idx < action_model.shape[1]:
            ax.plot(t, action_model[:, idx], color=color, alpha=0.5, linewidth=1, label=f'Model {name}')
            ax.plot(t, action_intervened[:, idx], color=color, linestyle='--', linewidth=1, label=f'Intervened {name}')
        
        # 标记干预时段
        for start, end in intervention_analysis['segments']:
            ax.axvspan(t[start], t[min(end, len(t)-1)], alpha=0.15, color='yellow')
        
        ax.set_ylabel(f'{name} Action')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time (s)')
    axes[0].set_title('Model Output vs Intervened Output (yellow = intervention period)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  动作对比图已保存: {save_path}")
    plt.close()


def plot_joint_analysis(
    trajectory: Dict[str, Any],
    intervention_analysis: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """绘制关节角度分析"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    joints = trajectory['joints']  # [T, 6]
    timestamps = trajectory['timestamps']
    t = timestamps - timestamps[0]
    # 使用更智能的干预检测
    step_has_int = intervention_analysis.get('actual_intervention', 
                                              intervention_analysis['step_has_intervention'])
    
    for i in range(6):
        ax = axes[i]
        ax.plot(t, joints[:, i], 'b-', linewidth=1)
        
        # 标记干预点
        int_times = t[step_has_int]
        int_joints = joints[step_has_int, i]
        ax.scatter(int_times, int_joints, c='red', s=10, alpha=0.3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rad)')
        ax.set_title(f'Joint {i}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Joint Angles (red = intervention)', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  关节分析图已保存: {save_path}")
    plt.close()


def plot_sample_images(
    data: Dict[str, Any],
    intervention_analysis: Dict[str, Any],
    num_samples: int = 8,
    save_path: Optional[str] = None,
):
    """绘制样本图像"""
    if not HAS_MATPLOTLIB:
        return
    
    images = data['images']
    # 使用更智能的干预检测
    step_has_int = intervention_analysis.get('actual_intervention', 
                                              intervention_analysis['step_has_intervention'])
    T = len(images)
    
    # 选择均匀分布的样本
    indices = np.linspace(0, T - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(images[idx])
        
        title = f'Step {idx}'
        if step_has_int[idx]:
            title += ' [INT]'
            ax.patch.set_edgecolor('red')
            ax.patch.set_linewidth(3)
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Sample Images ([INT] = during intervention)', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  样本图像已保存: {save_path}")
    plt.close()


def analyze_single_file(filepath: str, save_dir: Optional[str] = None) -> Dict[str, Any]:
    """分析单个文件"""
    print(f"\n正在分析: {filepath}")
    
    # 加载数据
    data = load_hdf5_data(filepath)
    
    # 分析
    basic_stats = analyze_basic_stats(data, filepath)
    intervention_analysis = analyze_intervention(data)
    action_diff = analyze_action_difference(data)
    trajectory = analyze_trajectory(data)
    
    # 打印报告
    print_analysis_report(basic_stats, intervention_analysis, action_diff, trajectory)
    
    # 可视化
    if save_dir and HAS_MATPLOTLIB:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        
        print("\n生成可视化图表...")
        plot_trajectory_3d(trajectory, intervention_analysis, 
                          os.path.join(save_dir, f'{base_name}_trajectory.png'))
        plot_intervention_analysis(intervention_analysis, action_diff, trajectory,
                                   os.path.join(save_dir, f'{base_name}_intervention.png'))
        plot_action_comparison(data, intervention_analysis,
                              os.path.join(save_dir, f'{base_name}_action_compare.png'))
        plot_joint_analysis(trajectory, intervention_analysis,
                           os.path.join(save_dir, f'{base_name}_joints.png'))
        plot_sample_images(data, intervention_analysis,
                          save_path=os.path.join(save_dir, f'{base_name}_samples.png'))
    
    return {
        'basic_stats': basic_stats,
        'intervention_analysis': intervention_analysis,
        'action_diff': action_diff,
        'trajectory': trajectory,
    }


def compare_episodes(results: List[Dict[str, Any]], save_dir: Optional[str] = None):
    """比较多个 episode"""
    if len(results) < 2:
        return
    
    print("\n" + "=" * 70)
    print("多 Episode 对比")
    print("=" * 70)
    
    print("\n{:<40} {:>10} {:>10} {:>12} {:>10}".format(
        'Episode', 'Steps', 'Duration', 'Int. Ratio', 'Distance'
    ))
    print("-" * 82)
    
    for r in results:
        bs = r['basic_stats']
        ia = r['intervention_analysis']
        tr = r['trajectory']
        # 使用更准确的干预率
        int_ratio = ia.get('actual_intervention_ratio', ia['intervention_ratio'])
        print("{:<40} {:>10} {:>10.1f}s {:>11.1%} {:>10.4f}m".format(
            bs['filename'][:40],
            bs['num_steps'],
            bs['duration_sec'],
            int_ratio,
            tr['total_distance'],
        ))
    
    if save_dir and HAS_MATPLOTLIB:
        # 对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        names = [r['basic_stats']['filename'][:20] for r in results]
        
        # 步数对比
        ax1 = axes[0]
        steps = [r['basic_stats']['num_steps'] for r in results]
        ax1.bar(names, steps, color='steelblue')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode Length')
        ax1.tick_params(axis='x', rotation=45)
        
        # 干预率对比 (使用更准确的)
        ax2 = axes[1]
        int_ratios = [r['intervention_analysis'].get('actual_intervention_ratio', 
                      r['intervention_analysis']['intervention_ratio']) * 100 for r in results]
        ax2.bar(names, int_ratios, color='coral')
        ax2.set_ylabel('Intervention Ratio (%)')
        ax2.set_title('Actual Intervention Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # 运动距离对比
        ax3 = axes[2]
        distances = [r['trajectory']['total_distance'] * 100 for r in results]
        ax3.bar(names, distances, color='seagreen')
        ax3.set_ylabel('Distance (cm)')
        ax3.set_title('Total Movement Distance')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'episode_comparison.png'), dpi=150, bbox_inches='tight')
        print(f"\n对比图已保存: {os.path.join(save_dir, 'episode_comparison.png')}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='分析 inference 采集数据')
    parser.add_argument('--data_dir', type=str, default='/home/lizh/rl-vla/inference_logs',
                        help='数据目录')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                        help='指定要分析的文件')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='保存可视化图表的目录 (默认: data_dir/analysis)')
    parser.add_argument('--pattern', type=str, default='inference_episode_*.hdf5',
                        help='文件匹配模式')
    parser.add_argument('--no_viz', action='store_true',
                        help='禁用可视化')
    
    args = parser.parse_args()
    
    # 确定要分析的文件
    if args.files:
        files = args.files
    else:
        files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    
    if not files:
        print(f"[ERROR] 未找到匹配的文件: {args.pattern}")
        print(f"  目录: {args.data_dir}")
        return
    
    print(f"找到 {len(files)} 个文件待分析:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # 确定保存目录
    save_dir = None
    if not args.no_viz:
        if args.save_dir:
            save_dir = args.save_dir
        else:
            save_dir = os.path.join(args.data_dir, 'analysis')
    
    # 分析每个文件
    results = []
    for filepath in files:
        try:
            result = analyze_single_file(filepath, save_dir)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] 分析 {filepath} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 多文件对比
    if len(results) > 1:
        compare_episodes(results, save_dir)
    
    print("\n分析完成!")
    if save_dir:
        print(f"可视化结果保存在: {save_dir}")


if __name__ == '__main__':
    main()
