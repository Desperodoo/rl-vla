#!/usr/bin/env python3
"""
对比分析 Action 抖动情况

比较:
1. Policy 真机部署时输出的 action (relative_pose)
2. 遥操作记录数据转换成的 relative action

这样可以判断抖动是否来自训练数据本身还是模型问题

Action 格式: 15D = [joint(6), gripper(1), relative_pose(7), gripper(1)]
- relative_pose[7] = [dx, dy, dz, qx, qy, qz, qw] 相对于当前位姿的变换

用法:
    python compare_action_jitter.py --teleop_dir /path/to/recorded_data --inference_dir /path/to/inference_logs
"""

import os
import sys
import argparse
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy import signal
from typing import Dict, List, Tuple, Optional


# ============================================================================
# Pose Transformation Utilities (from carm_utils.py)
# ============================================================================

def pose_to_transform_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Convert pose (xyz + quaternion) to 4x4 transformation matrix."""
    rotation = R.from_quat(quaternion).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def compute_relative_pose_transform(pose_current: np.ndarray, pose_target: np.ndarray) -> np.ndarray:
    """
    Compute relative pose transformation from current to target.
    relative_transform = current_pose^{-1} @ target_pose
    """
    T_current = pose_to_transform_matrix(pose_current[:3], pose_current[3:7])
    T_target = pose_to_transform_matrix(pose_target[:3], pose_target[3:7])
    T_relative = np.linalg.inv(T_current) @ T_target
    
    position = T_relative[:3, 3]
    quaternion = R.from_matrix(T_relative[:3, :3]).as_quat()
    return np.concatenate([position, quaternion])


# ============================================================================
# Data Loading
# ============================================================================

def load_teleop_episode(filepath: str) -> Dict[str, np.ndarray]:
    """加载遥操作数据文件"""
    data = {}
    with h5py.File(filepath, 'r') as f:
        obs = f['observations']
        data['qpos_end'] = np.array(obs['qpos_end'])       # [T, 8] (7 pose + gripper)
        data['qpos_joint'] = np.array(obs['qpos_joint'])   # [T, 7]
        data['gripper'] = np.array(obs['gripper'])         # [T]
        data['timestamps'] = np.array(obs['timestamps'])   # [T]
        
        # Raw action (absolute poses)
        if 'action' in f:
            data['action'] = np.array(f['action'])         # [T, 15]
    return data


def convert_teleop_to_relative_actions(data: Dict[str, np.ndarray], horizon: int = 1) -> np.ndarray:
    """
    将遥操作数据转换为相对动作 (模拟训练时的处理方式)
    
    对于每个时间步 t，计算从 t 到 t+horizon 的相对位姿变换
    
    Args:
        data: 遥操作数据字典
        horizon: action horizon (default=1 表示下一帧)
    
    Returns:
        relative_actions: [T-horizon, 7] 相对位姿 (dx, dy, dz, qx, qy, qz, qw)
    """
    qpos_end = data['qpos_end']  # [T, 8]
    T = len(qpos_end)
    
    relative_actions = []
    for t in range(T - horizon):
        current_pose = qpos_end[t, :7]   # [x, y, z, qx, qy, qz, qw]
        target_pose = qpos_end[t + horizon, :7]
        
        # 计算相对变换 (和 train_carm.py 中一样)
        relative_pose = compute_relative_pose_transform(current_pose, target_pose)
        relative_actions.append(relative_pose)
    
    return np.array(relative_actions)


def load_inference_episode(filepath: str) -> Dict[str, np.ndarray]:
    """加载推理数据文件"""
    data = {}
    with h5py.File(filepath, 'r') as f:
        obs = f['observations']
        data['qpos_end'] = np.array(obs['qpos_end'])
        data['timestamps'] = np.array(obs['timestamps'])
        
        # action_model: [T, pred_horizon, 15]
        data['action_model'] = np.array(f['action_model'])
        data['action_intervened'] = np.array(f['action_intervened'])
        data['intervention_mask'] = np.array(f['intervention_mask'])
    return data


def extract_inference_relative_actions(data: Dict[str, np.ndarray], step_idx: int = 0) -> np.ndarray:
    """
    从推理数据中提取相对动作
    
    Args:
        data: 推理数据字典
        step_idx: 使用 pred_horizon 中的第几步 (default=0, 即第一步)
    
    Returns:
        relative_actions: [T, 7] 相对位姿
    """
    action_model = data['action_model']  # [T, pred_horizon, 15]
    # 提取 relative_pose 部分: index 7:14
    return action_model[:, step_idx, 7:14]


# ============================================================================
# Jitter Analysis
# ============================================================================

def compute_jitter_metrics(actions: np.ndarray, timestamps: np.ndarray = None) -> Dict[str, float]:
    """
    计算动作抖动指标
    
    Args:
        actions: [T, D] 动作序列
        timestamps: [T] 时间戳 (可选)
    
    Returns:
        metrics: 各种抖动指标
    """
    T, D = actions.shape
    
    metrics = {}
    
    # 1. 一阶差分 (速度变化)
    diff1 = np.diff(actions, axis=0)  # [T-1, D]
    
    # 2. 二阶差分 (加速度变化 / 抖动)
    diff2 = np.diff(diff1, axis=0)    # [T-2, D]
    
    # 3. 各维度的统计量
    for i, name in enumerate(['dx', 'dy', 'dz', 'qx', 'qy', 'qz', 'qw'][:D]):
        col = actions[:, i]
        d1 = diff1[:, i]
        d2 = diff2[:, i] if D <= actions.shape[1] else diff2[:, i]
        
        metrics[f'{name}_mean'] = np.mean(col)
        metrics[f'{name}_std'] = np.std(col)
        metrics[f'{name}_range'] = np.ptp(col)
        metrics[f'{name}_diff1_std'] = np.std(d1)  # 速度变化
        metrics[f'{name}_diff2_std'] = np.std(d2) if len(d2) > 0 else 0  # 加速度变化
    
    # 4. XYZ 整体抖动
    xyz = actions[:, :3]
    xyz_diff1 = diff1[:, :3]
    xyz_diff2 = diff2[:, :3]
    
    # 位移幅度
    xyz_mag = np.linalg.norm(xyz, axis=1)
    metrics['xyz_mag_mean'] = np.mean(xyz_mag)
    metrics['xyz_mag_std'] = np.std(xyz_mag)
    
    # 速度 (一阶差分幅度)
    velocity = np.linalg.norm(xyz_diff1, axis=1)
    metrics['velocity_mean'] = np.mean(velocity)
    metrics['velocity_std'] = np.std(velocity)
    metrics['velocity_max'] = np.max(velocity)
    
    # 加速度 (二阶差分幅度) - 抖动的核心指标
    acceleration = np.linalg.norm(xyz_diff2, axis=1) if len(xyz_diff2) > 0 else np.array([0])
    metrics['acceleration_mean'] = np.mean(acceleration)
    metrics['acceleration_std'] = np.std(acceleration)
    metrics['acceleration_max'] = np.max(acceleration)
    
    # 5. 方向变化 (符号翻转次数)
    for i, name in enumerate(['dx', 'dy', 'dz']):
        d1 = diff1[:, i]
        sign_changes = np.sum(np.abs(np.diff(np.sign(d1))) > 0)
        metrics[f'{name}_sign_changes'] = sign_changes
        metrics[f'{name}_sign_change_rate'] = sign_changes / max(1, len(d1) - 1)
    
    # 6. 频域分析 - 高频能量占比
    for i, name in enumerate(['dx', 'dy', 'dz']):
        col = actions[:, i]
        if len(col) > 10:
            # FFT
            fft = np.fft.rfft(col - np.mean(col))
            freqs = np.fft.rfftfreq(len(col))
            power = np.abs(fft) ** 2
            
            # 高频能量占比 (频率 > 0.2)
            high_freq_mask = freqs > 0.2
            if np.sum(power) > 0:
                high_freq_ratio = np.sum(power[high_freq_mask]) / np.sum(power)
            else:
                high_freq_ratio = 0
            metrics[f'{name}_high_freq_ratio'] = high_freq_ratio
    
    return metrics


def compute_smoothness_metrics(actions: np.ndarray) -> Dict[str, float]:
    """
    计算平滑度指标
    """
    metrics = {}
    
    # Savitzky-Golay 滤波后的残差
    xyz = actions[:, :3]
    if len(xyz) > 11:
        for i, name in enumerate(['dx', 'dy', 'dz']):
            col = xyz[:, i]
            # 平滑滤波
            smoothed = signal.savgol_filter(col, window_length=11, polyorder=3)
            residual = col - smoothed
            metrics[f'{name}_smooth_residual_std'] = np.std(residual)
            metrics[f'{name}_smooth_residual_max'] = np.max(np.abs(residual))
    
    return metrics


# ============================================================================
# Visualization
# ============================================================================

def plot_action_comparison(teleop_actions: np.ndarray, inference_actions: np.ndarray,
                           teleop_timestamps: np.ndarray, inference_timestamps: np.ndarray,
                           save_path: str, title_suffix: str = ""):
    """绘制动作对比图"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(f'Action Comparison: Teleop vs Policy Output {title_suffix}', fontsize=14)
    
    # 归一化时间轴到 [0, 1]
    teleop_t = (teleop_timestamps - teleop_timestamps[0]) / (teleop_timestamps[-1] - teleop_timestamps[0] + 1e-6)
    inference_t = (inference_timestamps - inference_timestamps[0]) / (inference_timestamps[-1] - inference_timestamps[0] + 1e-6)
    
    components = ['dx', 'dy', 'dz', 'qx', 'qy', 'qz', 'qw']
    colors = ['red', 'green', 'blue', 'orange']
    
    for i, name in enumerate(components[:7]):
        ax = axes[i // 2, i % 2]
        
        teleop_col = teleop_actions[:, i] if i < teleop_actions.shape[1] else np.zeros(len(teleop_t))
        inference_col = inference_actions[:, i] if i < inference_actions.shape[1] else np.zeros(len(inference_t))
        
        ax.plot(teleop_t[:len(teleop_col)], teleop_col, 'b-', alpha=0.7, label='Teleop', linewidth=1.0)
        ax.plot(inference_t[:len(inference_col)], inference_col, 'r-', alpha=0.7, label='Policy', linewidth=1.0)
        
        ax.set_ylabel(name)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 显示统计量
        teleop_std = np.std(teleop_col)
        inference_std = np.std(inference_col)
        ax.set_title(f'{name}: Teleop std={teleop_std:.5f}, Policy std={inference_std:.5f}')
    
    # 最后一个子图: XYZ 幅度
    ax = axes[3, 1]
    teleop_xyz_mag = np.linalg.norm(teleop_actions[:, :3], axis=1)
    inference_xyz_mag = np.linalg.norm(inference_actions[:, :3], axis=1)
    
    ax.plot(teleop_t[:len(teleop_xyz_mag)], teleop_xyz_mag, 'b-', alpha=0.7, label='Teleop', linewidth=1.0)
    ax.plot(inference_t[:len(inference_xyz_mag)], inference_xyz_mag, 'r-', alpha=0.7, label='Policy', linewidth=1.0)
    ax.set_ylabel('|dXYZ|')
    ax.set_xlabel('Normalized Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'XYZ Magnitude: Teleop std={np.std(teleop_xyz_mag):.5f}, Policy std={np.std(inference_xyz_mag):.5f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_jitter_analysis(teleop_actions: np.ndarray, inference_actions: np.ndarray,
                         save_path: str, title_suffix: str = ""):
    """绘制抖动分析图"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Jitter Analysis: Teleop vs Policy Output {title_suffix}', fontsize=14)
    
    # 计算差分
    teleop_diff1 = np.diff(teleop_actions[:, :3], axis=0)
    inference_diff1 = np.diff(inference_actions[:, :3], axis=0)
    
    teleop_diff2 = np.diff(teleop_diff1, axis=0)
    inference_diff2 = np.diff(inference_diff1, axis=0)
    
    # Row 1: 一阶差分 (速度)
    for i, name in enumerate(['dx', 'dy', 'dz']):
        ax = axes[0, i]
        ax.hist(teleop_diff1[:, i], bins=50, alpha=0.6, label='Teleop', color='blue', density=True)
        ax.hist(inference_diff1[:, i], bins=50, alpha=0.6, label='Policy', color='red', density=True)
        ax.set_xlabel(f'd{name}/dt')
        ax.set_ylabel('Density')
        ax.set_title(f'Velocity ({name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 2: 二阶差分 (加速度 / 抖动)
    for i, name in enumerate(['dx', 'dy', 'dz']):
        ax = axes[1, i]
        ax.hist(teleop_diff2[:, i], bins=50, alpha=0.6, label='Teleop', color='blue', density=True)
        ax.hist(inference_diff2[:, i], bins=50, alpha=0.6, label='Policy', color='red', density=True)
        ax.set_xlabel(f'd²{name}/dt²')
        ax.set_ylabel('Density')
        ax.set_title(f'Acceleration ({name}) - Jitter Indicator')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 3: 频谱分析
    for i, name in enumerate(['dx', 'dy', 'dz']):
        ax = axes[2, i]
        
        # Teleop FFT
        teleop_col = teleop_actions[:, i]
        if len(teleop_col) > 10:
            teleop_fft = np.abs(np.fft.rfft(teleop_col - np.mean(teleop_col)))
            teleop_freqs = np.fft.rfftfreq(len(teleop_col))
            ax.plot(teleop_freqs[1:], teleop_fft[1:], 'b-', alpha=0.7, label='Teleop')
        
        # Inference FFT
        inference_col = inference_actions[:, i]
        if len(inference_col) > 10:
            inference_fft = np.abs(np.fft.rfft(inference_col - np.mean(inference_col)))
            inference_freqs = np.fft.rfftfreq(len(inference_col))
            ax.plot(inference_freqs[1:], inference_fft[1:], 'r-', alpha=0.7, label='Policy')
        
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'FFT ({name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 0.5])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_comparison(teleop_metrics_list: List[Dict], inference_metrics_list: List[Dict],
                            save_path: str):
    """绘制汇总对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Summary: Teleop vs Policy Jitter Metrics', fontsize=14)
    
    # 提取关键指标
    key_metrics = [
        ('dx_diff1_std', 'X Velocity Std'),
        ('dy_diff1_std', 'Y Velocity Std'),
        ('dz_diff1_std', 'Z Velocity Std'),
        ('acceleration_mean', 'Acceleration Mean'),
        ('acceleration_std', 'Acceleration Std'),
        ('dx_sign_change_rate', 'X Direction Change Rate'),
    ]
    
    for idx, (metric_key, metric_name) in enumerate(key_metrics):
        ax = axes[idx // 3, idx % 3]
        
        teleop_vals = [m.get(metric_key, 0) for m in teleop_metrics_list]
        inference_vals = [m.get(metric_key, 0) for m in inference_metrics_list]
        
        x = np.arange(2)
        width = 0.35
        
        teleop_mean = np.mean(teleop_vals) if teleop_vals else 0
        teleop_std = np.std(teleop_vals) if teleop_vals else 0
        inference_mean = np.mean(inference_vals) if inference_vals else 0
        inference_std = np.std(inference_vals) if inference_vals else 0
        
        bars = ax.bar(x, [teleop_mean, inference_mean], width, 
                     yerr=[teleop_std, inference_std],
                     capsize=5, color=['blue', 'red'], alpha=0.7)
        
        ax.set_ylabel(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(['Teleop', 'Policy'])
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 显示数值
        for bar, val in zip(bars, [teleop_mean, inference_mean]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.5f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_individual_trajectories(teleop_actions_list: List[np.ndarray], 
                                 inference_actions_list: List[np.ndarray],
                                 save_path: str):
    """绘制各个 episode 的轨迹"""
    n_teleop = len(teleop_actions_list)
    n_inference = len(inference_actions_list)
    
    fig = plt.figure(figsize=(14, 10))
    
    # Teleop trajectories
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    for i, actions in enumerate(teleop_actions_list[:10]):  # 最多显示10个
        xyz = np.cumsum(actions[:, :3], axis=0)  # 累积得到轨迹
        ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], alpha=0.7, linewidth=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Teleop Trajectories (n={min(n_teleop, 10)})')
    
    # Inference trajectories
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    for i, actions in enumerate(inference_actions_list[:10]):
        xyz = np.cumsum(actions[:, :3], axis=0)
        ax2.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], alpha=0.7, linewidth=0.8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Policy Trajectories (n={min(n_inference, 10)})')
    
    # XYZ time series comparison (first episode each)
    ax3 = fig.add_subplot(2, 2, 3)
    if teleop_actions_list:
        actions = teleop_actions_list[0]
        t = np.arange(len(actions))
        for i, (name, color) in enumerate(zip(['dx', 'dy', 'dz'], ['r', 'g', 'b'])):
            ax3.plot(t, actions[:, i], color=color, alpha=0.7, label=name)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Relative Position')
    ax3.set_title('Teleop Episode 1: Relative XYZ')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(2, 2, 4)
    if inference_actions_list:
        actions = inference_actions_list[0]
        t = np.arange(len(actions))
        for i, (name, color) in enumerate(zip(['dx', 'dy', 'dz'], ['r', 'g', 'b'])):
            ax4.plot(t, actions[:, i], color=color, alpha=0.7, label=name)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Relative Position')
    ax4.set_title('Policy Episode 1: Relative XYZ')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_teleop_data(data_dir: str, max_episodes: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    """分析遥操作数据"""
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    if max_episodes:
        files = files[:max_episodes]
    
    print(f"\n=== Analyzing Teleop Data ===")
    print(f"Found {len(files)} episodes in {data_dir}")
    
    all_actions = []
    all_timestamps = []
    all_metrics = []
    
    for filepath in files:
        try:
            data = load_teleop_episode(filepath)
            
            # 转换为相对动作
            relative_actions = convert_teleop_to_relative_actions(data, horizon=1)
            timestamps = data['timestamps'][:-1]  # 长度匹配
            
            # 计算指标
            metrics = compute_jitter_metrics(relative_actions, timestamps)
            metrics.update(compute_smoothness_metrics(relative_actions))
            metrics['episode'] = os.path.basename(filepath)
            metrics['num_steps'] = len(relative_actions)
            
            all_actions.append(relative_actions)
            all_timestamps.append(timestamps)
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
    
    print(f"Successfully loaded {len(all_actions)} episodes")
    return all_actions, all_timestamps, all_metrics


def analyze_inference_data(data_dir: str, max_episodes: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    """分析推理数据"""
    files = sorted(glob.glob(os.path.join(data_dir, "inference_episode_*.hdf5")))
    if max_episodes:
        files = files[:max_episodes]
    
    print(f"\n=== Analyzing Inference Data ===")
    print(f"Found {len(files)} episodes in {data_dir}")
    
    all_actions = []
    all_timestamps = []
    all_metrics = []
    
    for filepath in files:
        try:
            data = load_inference_episode(filepath)
            
            # 提取模型输出的相对动作 (使用 step_idx=0)
            relative_actions = extract_inference_relative_actions(data, step_idx=0)
            timestamps = data['timestamps']
            
            # 计算指标
            metrics = compute_jitter_metrics(relative_actions, timestamps)
            metrics.update(compute_smoothness_metrics(relative_actions))
            metrics['episode'] = os.path.basename(filepath)
            metrics['num_steps'] = len(relative_actions)
            
            all_actions.append(relative_actions)
            all_timestamps.append(timestamps)
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
    
    print(f"Successfully loaded {len(all_actions)} episodes")
    return all_actions, all_timestamps, all_metrics


def print_comparison_report(teleop_metrics: List[Dict], inference_metrics: List[Dict]):
    """打印对比报告"""
    print("\n" + "=" * 80)
    print("JITTER COMPARISON REPORT")
    print("=" * 80)
    
    def aggregate_metrics(metrics_list, key):
        vals = [m.get(key, 0) for m in metrics_list if key in m]
        if vals:
            return np.mean(vals), np.std(vals)
        return 0, 0
    
    key_metrics = [
        ('xyz_mag_mean', 'XYZ Magnitude Mean', 'm'),
        ('xyz_mag_std', 'XYZ Magnitude Std', 'm'),
        ('velocity_mean', 'Velocity Mean', 'm/step'),
        ('velocity_std', 'Velocity Std', 'm/step'),
        ('velocity_max', 'Velocity Max', 'm/step'),
        ('acceleration_mean', 'Acceleration Mean (Jitter)', 'm/step²'),
        ('acceleration_std', 'Acceleration Std', 'm/step²'),
        ('acceleration_max', 'Acceleration Max', 'm/step²'),
        ('dx_sign_change_rate', 'X Direction Change Rate', '%'),
        ('dy_sign_change_rate', 'Y Direction Change Rate', '%'),
        ('dz_sign_change_rate', 'Z Direction Change Rate', '%'),
        ('dx_high_freq_ratio', 'X High Freq Energy Ratio', '%'),
        ('dy_high_freq_ratio', 'Y High Freq Energy Ratio', '%'),
        ('dz_high_freq_ratio', 'Z High Freq Energy Ratio', '%'),
    ]
    
    print(f"\n{'Metric':<40} {'Teleop':<25} {'Policy':<25} {'Ratio':<10}")
    print("-" * 100)
    
    for key, name, unit in key_metrics:
        t_mean, t_std = aggregate_metrics(teleop_metrics, key)
        i_mean, i_std = aggregate_metrics(inference_metrics, key)
        
        ratio = i_mean / t_mean if t_mean > 1e-10 else float('inf')
        
        if '%' in unit:
            t_str = f"{t_mean*100:.2f} ± {t_std*100:.2f} {unit}"
            i_str = f"{i_mean*100:.2f} ± {i_std*100:.2f} {unit}"
        else:
            t_str = f"{t_mean:.6f} ± {t_std:.6f}"
            i_str = f"{i_mean:.6f} ± {i_std:.6f}"
        
        ratio_str = f"{ratio:.2f}x" if ratio < 100 else ">>>"
        print(f"{name:<40} {t_str:<25} {i_str:<25} {ratio_str:<10}")
    
    print("\n" + "-" * 100)
    print("INTERPRETATION:")
    print("-" * 100)
    
    # 计算关键抖动指标
    t_acc_mean, _ = aggregate_metrics(teleop_metrics, 'acceleration_mean')
    i_acc_mean, _ = aggregate_metrics(inference_metrics, 'acceleration_mean')
    
    t_vel_std, _ = aggregate_metrics(teleop_metrics, 'velocity_std')
    i_vel_std, _ = aggregate_metrics(inference_metrics, 'velocity_std')
    
    t_sign_rate, _ = aggregate_metrics(teleop_metrics, 'dx_sign_change_rate')
    i_sign_rate, _ = aggregate_metrics(inference_metrics, 'dx_sign_change_rate')
    
    if i_acc_mean > t_acc_mean * 1.5:
        print(f"⚠️  Policy acceleration (jitter) is {i_acc_mean/t_acc_mean:.1f}x higher than teleop data.")
        print("   This suggests the model is adding noise/jitter beyond the training data.")
    elif i_acc_mean > t_acc_mean * 1.1:
        print(f"ℹ️  Policy acceleration is slightly higher ({i_acc_mean/t_acc_mean:.1f}x) than teleop data.")
        print("   Some jitter exists but is within acceptable range.")
    else:
        print(f"✓  Policy acceleration is similar to ({i_acc_mean/t_acc_mean:.1f}x) or lower than teleop data.")
        print("   Jitter is likely inherited from training data, not model-induced.")
    
    if i_sign_rate > t_sign_rate * 1.5:
        print(f"\n⚠️  Policy has {i_sign_rate/t_sign_rate:.1f}x more direction changes than teleop.")
        print("   This indicates high-frequency oscillation in policy output.")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Compare action jitter between teleop and inference data')
    parser.add_argument('--teleop_dir', type=str, default='/home/lizh/rl-vla/recorded_data/mix',
                       help='Directory containing teleop demonstration data')
    parser.add_argument('--inference_dir', type=str, default='/home/lizh/rl-vla/inference_logs',
                       help='Directory containing inference log data')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save analysis results (default: inference_dir/jitter_analysis)')
    parser.add_argument('--max_teleop', type=int, default=20,
                       help='Maximum number of teleop episodes to analyze')
    parser.add_argument('--max_inference', type=int, default=None,
                       help='Maximum number of inference episodes to analyze')
    parser.add_argument('--no_viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # 设置保存目录
    save_dir = args.save_dir or os.path.join(args.inference_dir, 'jitter_analysis')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Teleop data: {args.teleop_dir}")
    print(f"Inference data: {args.inference_dir}")
    print(f"Save directory: {save_dir}")
    
    # 分析遥操作数据
    teleop_actions, teleop_timestamps, teleop_metrics = analyze_teleop_data(
        args.teleop_dir, max_episodes=args.max_teleop
    )
    
    # 分析推理数据
    inference_actions, inference_timestamps, inference_metrics = analyze_inference_data(
        args.inference_dir, max_episodes=args.max_inference
    )
    
    if not teleop_actions:
        print("Error: No teleop data found!")
        return
    
    if not inference_actions:
        print("Error: No inference data found!")
        return
    
    # 打印对比报告
    print_comparison_report(teleop_metrics, inference_metrics)
    
    # 生成可视化
    if not args.no_viz:
        print("\n=== Generating Visualizations ===")
        
        # 1. 单个 episode 对比 (取第一个)
        plot_action_comparison(
            teleop_actions[0], inference_actions[0],
            teleop_timestamps[0], inference_timestamps[0],
            os.path.join(save_dir, 'action_comparison_ep1.png'),
            title_suffix="(Episode 1)"
        )
        
        # 2. 抖动分析对比 (取第一个)
        plot_jitter_analysis(
            teleop_actions[0], inference_actions[0],
            os.path.join(save_dir, 'jitter_analysis_ep1.png'),
            title_suffix="(Episode 1)"
        )
        
        # 3. 汇总统计对比
        plot_summary_comparison(
            teleop_metrics, inference_metrics,
            os.path.join(save_dir, 'summary_comparison.png')
        )
        
        # 4. 轨迹可视化
        plot_individual_trajectories(
            teleop_actions, inference_actions,
            os.path.join(save_dir, 'trajectories.png')
        )
        
        # 5. 如果有多个 inference episode，也分析一下
        if len(inference_actions) > 1:
            plot_action_comparison(
                teleop_actions[0], inference_actions[-1],
                teleop_timestamps[0], inference_timestamps[-1],
                os.path.join(save_dir, 'action_comparison_latest.png'),
                title_suffix="(Latest Inference vs Teleop)"
            )
    
    # 保存详细指标到文件
    import json
    
    def convert_metrics(metrics_list):
        """转换指标为可序列化格式"""
        result = []
        for m in metrics_list:
            converted = {}
            for k, v in m.items():
                if isinstance(v, (np.floating, np.integer)):
                    converted[k] = float(v)
                else:
                    converted[k] = v
            result.append(converted)
        return result
    
    report = {
        'teleop': {
            'num_episodes': len(teleop_metrics),
            'metrics': convert_metrics(teleop_metrics),
        },
        'inference': {
            'num_episodes': len(inference_metrics),
            'metrics': convert_metrics(inference_metrics),
        },
    }
    
    with open(os.path.join(save_dir, 'jitter_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved detailed report to: {os.path.join(save_dir, 'jitter_report.json')}")
    
    print(f"\n✓ Analysis complete! Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
