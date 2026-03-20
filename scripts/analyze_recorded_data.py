#!/usr/bin/env python3
"""
分析 record_data_ros.py 采集的 v2 HDF5 数据
目标：
1) 数据完整性与格式检查
2) 参数设置诊断
3) 时间线分析（帧率、延迟、抖动）
4) 轨迹可视化（采样图片 + 运动轨迹）
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

# 可选: matplotlib（生成图片）
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available, skipping plots")


def analyze_single_episode(filepath: str, verbose: bool = True) -> dict:
    """分析单个 episode HDF5 文件"""
    result = {
        'file': os.path.basename(filepath),
        'filepath': filepath,
        'errors': [],
        'warnings': [],
        'info': {},
    }

    with h5py.File(filepath, 'r') as f:
        # ---- 1. 元数据检查 ----
        attrs = dict(f.attrs)
        result['info']['attrs'] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in attrs.items()}

        data_version = attrs.get('data_version', 'unknown')
        if data_version != 'v2':
            result['errors'].append(f"data_version={data_version}, expected 'v2'")

        num_steps = int(attrs.get('num_steps', 0))
        record_freq = int(attrs.get('record_freq', 30))
        image_w = int(attrs.get('image_width', 0))
        image_h = int(attrs.get('image_height', 0))
        result['info']['num_steps'] = num_steps
        result['info']['record_freq'] = record_freq
        result['info']['image_size'] = (image_h, image_w)

        # ---- 2. 数据集存在性和 shape 检查 ----
        expected_datasets = {
            'observations/images': f'[T, {image_h}, {image_w}, 3]',
            'observations/qpos_joint': '[T, 7]',
            'observations/qpos_end': '[T, 8]',
            'observations/qpos': '[T, 15]',
            'observations/gripper': '[T]',
            'observations/timestamps': '[T]',
            'action': '[T, 8]',
            'teleop_scale': '[T]',
        }

        actual_shapes = {}
        for ds_name, expected_desc in expected_datasets.items():
            if ds_name not in f:
                result['errors'].append(f"Missing dataset: {ds_name}")
            else:
                shape = f[ds_name].shape
                dtype = f[ds_name].dtype
                actual_shapes[ds_name] = {'shape': shape, 'dtype': str(dtype)}
                if verbose:
                    print(f"  {ds_name}: shape={shape}, dtype={dtype}")

        result['info']['shapes'] = {k: str(v['shape']) for k, v in actual_shapes.items()}

        # 检查 T 维度一致性
        t_values = {}
        for ds_name in expected_datasets:
            if ds_name in f:
                t_values[ds_name] = f[ds_name].shape[0]

        unique_ts = set(t_values.values())
        if len(unique_ts) > 1:
            result['errors'].append(f"Inconsistent T dimension across datasets: {t_values}")
        elif len(unique_ts) == 1:
            T = unique_ts.pop()
            if T != num_steps:
                result['warnings'].append(f"attrs.num_steps={num_steps} != actual T={T}")

        # ---- 3. 图像检查 ----
        if 'observations/images' in f:
            images = f['observations/images']
            img_shape = images.shape
            if len(img_shape) != 4:
                result['errors'].append(f"images shape should be 4D, got {img_shape}")
            else:
                T_img, H, W, C = img_shape
                if C != 3:
                    result['errors'].append(f"images channels={C}, expected 3 (RGB)")
                if H != image_h or W != image_w:
                    result['warnings'].append(
                        f"images HxW=({H},{W}) != attrs ({image_h},{image_w})")

                # 抽样检查像素值范围
                sample_indices = [0, T_img // 2, T_img - 1]
                for idx in sample_indices:
                    img = images[idx]
                    if img.max() == 0:
                        result['warnings'].append(f"Image at step {idx} is all black")
                    if img.min() == img.max():
                        result['warnings'].append(f"Image at step {idx} is constant value={img.min()}")

                result['info']['image_stats'] = {
                    'first_frame_mean': float(images[0].mean()),
                    'last_frame_mean': float(images[-1].mean()),
                    'dtype': str(images.dtype),
                }
                # 保存采样图片数据
                result['_sample_images'] = [images[i] for i in sample_indices]
                result['_sample_indices'] = sample_indices

        # ---- 4. 状态数据检查 ----
        if 'observations/qpos_end' in f:
            qpos_end = f['observations/qpos_end'][:]  # [T, 8]
            ee_xyz = qpos_end[:, :3]
            ee_quat = qpos_end[:, 3:7]
            ee_gripper = qpos_end[:, 7]

            result['info']['ee_xyz_range'] = {
                'min': ee_xyz.min(axis=0).tolist(),
                'max': ee_xyz.max(axis=0).tolist(),
                'mean': ee_xyz.mean(axis=0).tolist(),
            }
            result['info']['ee_xyz_delta'] = {
                'total_distance': float(np.sum(np.linalg.norm(np.diff(ee_xyz, axis=0), axis=1))),
                'mean_step_dist': float(np.mean(np.linalg.norm(np.diff(ee_xyz, axis=0), axis=1))),
                'max_step_dist': float(np.max(np.linalg.norm(np.diff(ee_xyz, axis=0), axis=1))),
            }

            # 四元数范数检查（应≈1）
            quat_norms = np.linalg.norm(ee_quat, axis=1)
            if np.any(np.abs(quat_norms - 1.0) > 0.01):
                result['warnings'].append(
                    f"Quaternion norms deviate from 1: min={quat_norms.min():.4f}, max={quat_norms.max():.4f}")

            result['info']['gripper_range'] = {
                'min': float(ee_gripper.min()),
                'max': float(ee_gripper.max()),
                'num_changes': int(np.sum(np.abs(np.diff(ee_gripper)) > 1e-4)),
            }

            result['_ee_xyz'] = ee_xyz
            result['_ee_gripper'] = ee_gripper

        if 'observations/qpos_joint' in f:
            qpos_joint = f['observations/qpos_joint'][:]  # [T, 7]
            result['info']['joint_range'] = {
                'min': qpos_joint[:, :6].min(axis=0).tolist(),
                'max': qpos_joint[:, :6].max(axis=0).tolist(),
            }
            result['_qpos_joint'] = qpos_joint

        # ---- 5. Action 检查 ----
        if 'action' in f:
            action = f['action'][:]  # [T, 8]
            act_xyz = action[:, :3]
            act_quat = action[:, 3:7]
            act_gripper = action[:, 7]

            result['info']['action_xyz_range'] = {
                'min': act_xyz.min(axis=0).tolist(),
                'max': act_xyz.max(axis=0).tolist(),
            }

            # Action vs qpos_end 差异 (action 是 target, qpos_end 是 actual)
            if 'observations/qpos_end' in f:
                qpos_end = f['observations/qpos_end'][:]
                obs_xyz = qpos_end[:, :3]
                obs_quat = qpos_end[:, 3:7]

                xyz_diff = np.linalg.norm(act_xyz - obs_xyz, axis=1)
                result['info']['action_vs_obs'] = {
                    'xyz_diff_mean_mm': float(xyz_diff.mean() * 1000),
                    'xyz_diff_max_mm': float(xyz_diff.max() * 1000),
                    'xyz_diff_p95_mm': float(np.percentile(xyz_diff, 95) * 1000),
                    'xyz_diff_std_mm': float(xyz_diff.std() * 1000),
                }

                # 四元数差异（角度距离）
                # dot product between quaternions
                dots = np.abs(np.sum(act_quat * obs_quat, axis=1))
                dots = np.clip(dots, 0, 1)
                angle_diffs_deg = np.degrees(2 * np.arccos(dots))
                result['info']['action_vs_obs']['quat_diff_mean_deg'] = float(angle_diffs_deg.mean())
                result['info']['action_vs_obs']['quat_diff_max_deg'] = float(angle_diffs_deg.max())

                result['_xyz_diff'] = xyz_diff

            result['_action'] = action

        # ---- 6. Teleop scale 检查 ----
        if 'teleop_scale' in f:
            teleop_scale = f['teleop_scale'][:]
            n_active = int(np.sum(teleop_scale > 0))
            n_inactive = int(np.sum(teleop_scale == 0))
            unique_scales = np.unique(teleop_scale)

            result['info']['teleop_scale'] = {
                'n_active': n_active,
                'n_inactive': n_inactive,
                'active_ratio': float(n_active / max(len(teleop_scale), 1)),
                'unique_values': unique_scales.tolist(),
            }

            if n_inactive > 0:
                result['warnings'].append(
                    f"{n_inactive}/{len(teleop_scale)} steps have teleop_scale=0 (inactive)")

            result['_teleop_scale'] = teleop_scale

        # ---- 7. 时间戳分析 ----
        if 'observations/timestamps' in f:
            timestamps = f['observations/timestamps'][:]
            dt = np.diff(timestamps)

            result['info']['timing'] = {
                'duration_sec': float(timestamps[-1] - timestamps[0]),
                'num_steps': len(timestamps),
                'dt_mean_ms': float(dt.mean() * 1000),
                'dt_std_ms': float(dt.std() * 1000),
                'dt_min_ms': float(dt.min() * 1000),
                'dt_max_ms': float(dt.max() * 1000),
                'dt_p5_ms': float(np.percentile(dt, 5) * 1000),
                'dt_p95_ms': float(np.percentile(dt, 95) * 1000),
                'actual_freq_hz': float(1.0 / dt.mean()) if dt.mean() > 0 else 0,
            }

            # 看有没有异常大的时间间隔（>3x mean）
            outlier_threshold = dt.mean() * 3
            n_outliers = int(np.sum(dt > outlier_threshold))
            if n_outliers > 0:
                outlier_indices = np.where(dt > outlier_threshold)[0]
                result['warnings'].append(
                    f"{n_outliers} timestamp outliers (>{outlier_threshold*1000:.1f}ms): "
                    f"steps {outlier_indices.tolist()[:10]}")
                result['info']['timing']['n_outliers'] = n_outliers

            result['_timestamps'] = timestamps
            result['_dt'] = dt

        # ---- 8. qpos 冗余性检查 ----
        if 'observations/qpos' in f and 'observations/qpos_joint' in f and 'observations/qpos_end' in f:
            qpos = f['observations/qpos'][:]  # [T, 15]
            qpos_joint = f['observations/qpos_joint'][:]  # [T, 7]
            qpos_end = f['observations/qpos_end'][:]  # [T, 8]
            reconstructed = np.concatenate([qpos_joint, qpos_end], axis=1)
            if not np.allclose(qpos, reconstructed):
                result['warnings'].append(
                    "qpos != concat(qpos_joint, qpos_end) — redundancy inconsistency")
            else:
                result['info']['qpos_redundancy'] = 'consistent (qpos == concat(qpos_joint, qpos_end))'

    return result


def analyze_timeline_log(jsonl_path: str) -> dict:
    """分析时间线 JSONL 日志"""
    events = []
    with open(jsonl_path, 'r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_no}: JSON parse error: {e}")

    if not events:
        return {'error': 'No events found'}

    result = {
        'total_events': len(events),
        'event_types': {},
        'episodes': {},
    }

    # 分类事件
    for ev in events:
        etype = ev.get('event', 'unknown')
        if etype not in result['event_types']:
            result['event_types'][etype] = 0
        result['event_types'][etype] += 1

    # 分析 record_step 事件
    record_steps = [e for e in events if e.get('event') == 'record_step']
    if record_steps:
        # 按 episode 分组
        by_episode = {}
        for ev in record_steps:
            ep = ev.get('episode', 0)
            if ep not in by_episode:
                by_episode[ep] = []
            by_episode[ep].append(ev)

        for ep_id, ep_events in sorted(by_episode.items()):
            ep_info = {'num_steps': len(ep_events)}

            # 系统时间间隔
            sys_times = [e['t_sys'] for e in ep_events]
            if len(sys_times) > 1:
                dt_sys = np.diff(sys_times)
                ep_info['sys_dt_mean_ms'] = float(np.mean(dt_sys) * 1000)
                ep_info['sys_dt_std_ms'] = float(np.std(dt_sys) * 1000)
                ep_info['sys_dt_min_ms'] = float(np.min(dt_sys) * 1000)
                ep_info['sys_dt_max_ms'] = float(np.max(dt_sys) * 1000)
                ep_info['sys_actual_freq_hz'] = float(1.0 / np.mean(dt_sys))

            # delta_obs: obs ROS stamp → obs ready 的延迟
            delta_obs_vals = [e.get('delta_obs') for e in ep_events if e.get('delta_obs') is not None]
            if delta_obs_vals:
                delta_obs_arr = np.array(delta_obs_vals)
                ep_info['delta_obs_mean_ms'] = float(delta_obs_arr.mean() * 1000)
                ep_info['delta_obs_std_ms'] = float(delta_obs_arr.std() * 1000)
                ep_info['delta_obs_min_ms'] = float(delta_obs_arr.min() * 1000)
                ep_info['delta_obs_max_ms'] = float(delta_obs_arr.max() * 1000)

            # delta_action_obs: obs ROS stamp → action query 的延迟
            delta_action_vals = [e.get('delta_action_obs') for e in ep_events
                                 if e.get('delta_action_obs') is not None]
            if delta_action_vals:
                delta_action_arr = np.array(delta_action_vals)
                ep_info['delta_action_obs_mean_ms'] = float(delta_action_arr.mean() * 1000)
                ep_info['delta_action_obs_std_ms'] = float(delta_action_arr.std() * 1000)

            # ROS stamp 间隔
            ros_stamps = [e.get('obs_stamp_ros') for e in ep_events
                          if e.get('obs_stamp_ros') is not None]
            if len(ros_stamps) > 1:
                dt_ros = np.diff(ros_stamps)
                ep_info['ros_dt_mean_ms'] = float(np.mean(dt_ros) * 1000)
                ep_info['ros_dt_std_ms'] = float(np.std(dt_ros) * 1000)
                ep_info['ros_dt_min_ms'] = float(np.min(dt_ros) * 1000)
                ep_info['ros_dt_max_ms'] = float(np.max(dt_ros) * 1000)
                ep_info['ros_actual_freq_hz'] = float(1.0 / np.mean(dt_ros))

                # 检查 ROS stamp 是否单调递增
                non_monotonic = np.sum(dt_ros <= 0)
                if non_monotonic > 0:
                    ep_info['ros_non_monotonic_count'] = int(non_monotonic)

            result['episodes'][f'episode_{ep_id}'] = ep_info

    return result


def plot_episode(result: dict, output_dir: str, episode_idx: int):
    """为单个 episode 生成可视化"""
    if not HAS_MPL:
        return

    prefix = f"ep{episode_idx}"

    # --- 1. 采样图像 ---
    if '_sample_images' in result:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (img, idx) in enumerate(zip(result['_sample_images'], result['_sample_indices'])):
            axes[i].imshow(img)
            axes[i].set_title(f'Step {idx}')
            axes[i].axis('off')
        fig.suptitle(f'{result["file"]} — Sample Images', fontsize=14)
        fig.savefig(os.path.join(output_dir, f'{prefix}_sample_images.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

    # --- 2. EE 轨迹 3D ---
    if '_ee_xyz' in result:
        ee_xyz = result['_ee_xyz']
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2],
                             c=np.arange(len(ee_xyz)), cmap='viridis', s=3)
        ax.plot(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2], alpha=0.3, linewidth=0.5)
        ax.scatter(*ee_xyz[0], c='green', s=80, marker='^', label='Start')
        ax.scatter(*ee_xyz[-1], c='red', s=80, marker='v', label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{result["file"]} — EE Trajectory')
        ax.legend()
        plt.colorbar(scatter, label='Step')
        fig.savefig(os.path.join(output_dir, f'{prefix}_ee_trajectory_3d.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

    # --- 3. XYZ + Gripper vs Time ---
    if '_ee_xyz' in result and '_timestamps' in result:
        ee_xyz = result['_ee_xyz']
        t = result['_timestamps'] - result['_timestamps'][0]

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        labels = ['X (m)', 'Y (m)', 'Z (m)']
        for i in range(3):
            axes[i].plot(t, ee_xyz[:, i], linewidth=0.8)
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True, alpha=0.3)

        if '_ee_gripper' in result:
            axes[3].plot(t, result['_ee_gripper'], linewidth=0.8, color='purple')
            axes[3].set_ylabel('Gripper')
            axes[3].grid(True, alpha=0.3)

        axes[3].set_xlabel('Time (s)')
        fig.suptitle(f'{result["file"]} — EE Position + Gripper', fontsize=14)
        fig.savefig(os.path.join(output_dir, f'{prefix}_ee_position.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

    # --- 4. Action vs Obs 对比 ---
    if '_action' in result and '_ee_xyz' in result and '_timestamps' in result:
        action = result['_action']
        ee_xyz = result['_ee_xyz']
        t = result['_timestamps'] - result['_timestamps'][0]

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            axes[i].plot(t, ee_xyz[:, i], linewidth=0.8, label='obs (actual)')
            axes[i].plot(t, action[:, i], linewidth=0.8, label='action (target)', linestyle='--')
            axes[i].set_ylabel(f'{labels[i]} (m)')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)

        axes[2].set_xlabel('Time (s)')
        fig.suptitle(f'{result["file"]} — Action (target) vs Observation (actual)', fontsize=14)
        fig.savefig(os.path.join(output_dir, f'{prefix}_action_vs_obs.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

    # --- 5. 帧间时间间隔直方图 ---
    if '_dt' in result:
        dt_ms = result['_dt'] * 1000
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(dt_ms, bins=50, edgecolor='black', alpha=0.7)
        target_dt = 1000.0 / result['info'].get('record_freq', 30)
        ax.axvline(target_dt, color='red', linestyle='--', label=f'Target ({target_dt:.1f}ms)')
        ax.axvline(dt_ms.mean(), color='green', linestyle='--', label=f'Mean ({dt_ms.mean():.1f}ms)')
        ax.set_xlabel('Frame interval (ms)')
        ax.set_ylabel('Count')
        ax.set_title(f'{result["file"]} — Frame Interval Distribution')
        ax.legend()
        fig.savefig(os.path.join(output_dir, f'{prefix}_frame_intervals.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

    # --- 6. Teleop scale 变化 ---
    if '_teleop_scale' in result and '_timestamps' in result:
        t = result['_timestamps'] - result['_timestamps'][0]
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(t, result['_teleop_scale'], linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Teleop Scale')
        ax.set_title(f'{result["file"]} — Teleop Scale Over Time')
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, f'{prefix}_teleop_scale.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

    # --- 7. Action-Obs 距离时间序列 ---
    if '_xyz_diff' in result and '_timestamps' in result:
        t = result['_timestamps'] - result['_timestamps'][0]
        diff_mm = result['_xyz_diff'] * 1000
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, diff_mm, linewidth=0.8)
        ax.axhline(diff_mm.mean(), color='red', linestyle='--',
                    label=f'Mean ({diff_mm.mean():.1f}mm)', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('|action_xyz - obs_xyz| (mm)')
        ax.set_title(f'{result["file"]} — Action-Observation XYZ Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, f'{prefix}_action_obs_diff.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

    # --- 8. 关节角度 ---
    if '_qpos_joint' in result and '_timestamps' in result:
        t = result['_timestamps'] - result['_timestamps'][0]
        qpos_joint = result['_qpos_joint']
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        for i in range(6):
            ax = axes[i // 2, i % 2]
            ax.plot(t, qpos_joint[:, i], linewidth=0.8)
            ax.set_ylabel(f'Joint {i+1} (rad)')
            ax.grid(True, alpha=0.3)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 1].set_xlabel('Time (s)')
        fig.suptitle(f'{result["file"]} — Joint Angles', fontsize=14)
        fig.savefig(os.path.join(output_dir, f'{prefix}_joint_angles.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)


def print_report(results: list, timeline_result: dict = None):
    """打印文本报告"""
    print("\n" + "=" * 80)
    print("  DATA ANALYSIS REPORT — record_data_ros v2 HDF5")
    print("=" * 80)

    for i, r in enumerate(results):
        print(f"\n{'─' * 60}")
        print(f"  Episode {i+1}: {r['file']}")
        print(f"{'─' * 60}")

        info = r['info']

        # 基本信息
        print(f"  Steps:       {info.get('num_steps', '?')}")
        print(f"  Record Freq: {info.get('record_freq', '?')} Hz")
        print(f"  Image Size:  {info.get('image_size', '?')}")
        print(f"  Version:     {info['attrs'].get('data_version', '?')}")

        # 时间
        if 'timing' in info:
            t = info['timing']
            print(f"\n  [Timing]")
            print(f"    Duration:      {t['duration_sec']:.2f}s")
            print(f"    Actual Freq:   {t['actual_freq_hz']:.1f} Hz (target: {info.get('record_freq', '?')})")
            print(f"    Frame dt:      mean={t['dt_mean_ms']:.1f}ms, std={t['dt_std_ms']:.1f}ms")
            print(f"                   min={t['dt_min_ms']:.1f}ms, max={t['dt_max_ms']:.1f}ms")
            print(f"                   P5={t['dt_p5_ms']:.1f}ms, P95={t['dt_p95_ms']:.1f}ms")

        # EE 状态
        if 'ee_xyz_range' in info:
            xyz = info['ee_xyz_range']
            print(f"\n  [End-Effector XYZ]")
            print(f"    X: [{xyz['min'][0]:.4f}, {xyz['max'][0]:.4f}] m")
            print(f"    Y: [{xyz['min'][1]:.4f}, {xyz['max'][1]:.4f}] m")
            print(f"    Z: [{xyz['min'][2]:.4f}, {xyz['max'][2]:.4f}] m")
        if 'ee_xyz_delta' in info:
            d = info['ee_xyz_delta']
            print(f"    Total path:  {d['total_distance']*1000:.1f}mm")
            print(f"    Step dist:   mean={d['mean_step_dist']*1000:.2f}mm, max={d['max_step_dist']*1000:.2f}mm")

        # Gripper
        if 'gripper_range' in info:
            g = info['gripper_range']
            print(f"\n  [Gripper]")
            print(f"    Range: [{g['min']:.4f}, {g['max']:.4f}]")
            print(f"    Changes: {g['num_changes']}")

        # Action vs Obs
        if 'action_vs_obs' in info:
            a = info['action_vs_obs']
            print(f"\n  [Action vs Observation (target vs actual)]")
            print(f"    XYZ diff:  mean={a['xyz_diff_mean_mm']:.1f}mm, max={a['xyz_diff_max_mm']:.1f}mm, "
                  f"P95={a['xyz_diff_p95_mm']:.1f}mm, std={a['xyz_diff_std_mm']:.1f}mm")
            print(f"    Quat diff: mean={a['quat_diff_mean_deg']:.2f}°, max={a['quat_diff_max_deg']:.2f}°")

        # Teleop scale
        if 'teleop_scale' in info:
            ts = info['teleop_scale']
            print(f"\n  [Teleop Scale]")
            print(f"    Active:   {ts['n_active']}/{ts['n_active']+ts['n_inactive']} "
                  f"({ts['active_ratio']*100:.1f}%)")
            print(f"    Values:   {ts['unique_values']}")

        # Redundancy
        if 'qpos_redundancy' in info:
            print(f"\n  [Redundancy Check]")
            print(f"    qpos: {info['qpos_redundancy']}")

        # Errors & Warnings
        if r['errors']:
            print(f"\n  ❌ ERRORS ({len(r['errors'])}):")
            for e in r['errors']:
                print(f"    - {e}")
        if r['warnings']:
            print(f"\n  ⚠️  WARNINGS ({len(r['warnings'])}):")
            for w in r['warnings']:
                print(f"    - {w}")

    # ---- 时间线分析 ----
    if timeline_result:
        print(f"\n{'=' * 80}")
        print("  TIMELINE LOG ANALYSIS")
        print(f"{'=' * 80}")
        print(f"  Total events: {timeline_result['total_events']}")
        print(f"  Event types:  {timeline_result['event_types']}")

        for ep_key, ep_info in sorted(timeline_result.get('episodes', {}).items()):
            print(f"\n  [{ep_key}]")
            print(f"    Steps: {ep_info.get('num_steps', '?')}")

            if 'sys_dt_mean_ms' in ep_info:
                print(f"    System dt: mean={ep_info['sys_dt_mean_ms']:.1f}ms, "
                      f"std={ep_info['sys_dt_std_ms']:.1f}ms, "
                      f"freq={ep_info['sys_actual_freq_hz']:.1f}Hz")

            if 'ros_dt_mean_ms' in ep_info:
                print(f"    ROS dt:    mean={ep_info['ros_dt_mean_ms']:.1f}ms, "
                      f"std={ep_info['ros_dt_std_ms']:.1f}ms, "
                      f"freq={ep_info['ros_actual_freq_hz']:.1f}Hz")
                if 'ros_non_monotonic_count' in ep_info:
                    print(f"    ⚠️  ROS non-monotonic: {ep_info['ros_non_monotonic_count']} occurrences")

            if 'delta_obs_mean_ms' in ep_info:
                print(f"    Obs latency (ROS→ready): mean={ep_info['delta_obs_mean_ms']:.1f}ms, "
                      f"std={ep_info['delta_obs_std_ms']:.1f}ms, "
                      f"range=[{ep_info['delta_obs_min_ms']:.1f}, {ep_info['delta_obs_max_ms']:.1f}]ms")

            if 'delta_action_obs_mean_ms' in ep_info:
                print(f"    Action query latency (ROS→query): "
                      f"mean={ep_info['delta_action_obs_mean_ms']:.1f}ms, "
                      f"std={ep_info['delta_action_obs_std_ms']:.1f}ms")

    # ---- 综合评估 ----
    print(f"\n{'=' * 80}")
    print("  CODE & PARAMETER ASSESSMENT")
    print(f"{'=' * 80}")

    # 代码冗余
    print(f"\n  [Code Redundancy Issues]")
    print(f"    1. `qpos` (15D) = concat(qpos_joint, qpos_end) — 完全冗余")
    print(f"       -> qpos_joint(7) + qpos_end(8) 已分别存储, qpos 可移除")
    print(f"    2. `gripper` 字段 = qpos_joint[-1] = qpos_end[-1] — 三重存储")
    print(f"       -> 仅需在 qpos_end[-1] 保留, 单独的 gripper 数据集可移除")
    print(f"    3. argparse 中 `--timeline_enabled` (store_true) 默认 False,")
    print(f"       但 main() 中 timeline_disabled=False 时强制设 True — 逻辑矛盾")
    print(f"       实际效果: timeline 始终开启(除非显式 --timeline_disabled)")
    print(f"    4. import os, sys 各出现两次 (line 23-24 和 line 40-41)")
    print(f"    5. `episode_data.copy()` (line 266) 是浅拷贝 — list 内的 np.array")
    print(f"       对象仍共享引用。虽不导致 bug(save 在 clear 之前执行),")
    print(f"       但语义上应使用 deepcopy 或直接传递引用更清晰")

    print(f"\n  [Parameter Issues]")
    for i, r in enumerate(results):
        info = r['info']
        if 'timing' in info:
            actual_freq = info['timing']['actual_freq_hz']
            target_freq = info.get('record_freq', 30)
            if abs(actual_freq - target_freq) > 2:
                print(f"    Ep{i+1}: Actual freq {actual_freq:.1f}Hz != target {target_freq}Hz "
                      f"(drift: {actual_freq-target_freq:+.1f}Hz)")


def main():
    parser = argparse.ArgumentParser(description='Analyze recorded HDF5 data')
    parser.add_argument('--data_dir', type=str, default='data/test_gap_fix2',
                        help='Directory containing HDF5 files')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output directory for plots (default: data_dir/analysis)')
    parser.add_argument('--no_plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir)

    output_dir = args.output_dir or os.path.join(data_dir, 'analysis')
    if not args.no_plot:
        os.makedirs(output_dir, exist_ok=True)

    # 找到 HDF5 文件
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))
    if not hdf5_files:
        print(f"[ERROR] No HDF5 files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(hdf5_files)} HDF5 files in {data_dir}")

    # 分析每个 episode
    results = []
    for filepath in hdf5_files:
        print(f"\nAnalyzing: {os.path.basename(filepath)}")
        r = analyze_single_episode(filepath)
        results.append(r)

    # 分析时间线日志
    timeline_result = None
    jsonl_files = sorted(glob.glob(os.path.join(data_dir, '*.jsonl')))
    if jsonl_files:
        print(f"\nAnalyzing timeline log: {os.path.basename(jsonl_files[0])}")
        timeline_result = analyze_timeline_log(jsonl_files[0])

    # 打印报告
    print_report(results, timeline_result)

    # 生成图表
    if not args.no_plot and HAS_MPL:
        print(f"\nGenerating plots to {output_dir}...")
        for i, r in enumerate(results):
            plot_episode(r, output_dir, i + 1)
        print(f"Done! {len(os.listdir(output_dir))} files generated.")


if __name__ == '__main__':
    main()
