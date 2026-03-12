#!/usr/bin/env python3
"""
GAP-1/2/3 修复前后数据对比分析

对比两种 action 记录方式的差异：
- v1 (旧): FK(get_plan_joint_pos()) — SDK 规划层输出的 FK 结果，接近实际位姿
- v2 (新): target_end_arm_pose — backend 发给 track_pose() 的遥操作目标位姿

分析维度：
1. 位置差异（mm）
2. 旋转差异（度）
3. 相对位姿（训练标签）的差异
4. 时序特性（领先/滞后）
"""

import os
import sys
import glob
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlft.utils.pose_utils import compute_relative_pose_transform, apply_relative_transform


def rotation_angle_deg(q1, q2):
    """计算两个四元数之间的旋转角度（度）"""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r_diff = r1.inv() * r2
    angle = r_diff.magnitude()  # radians
    return np.degrees(angle)


def analyze_episode(filepath):
    """分析单个 episode 的 v1 vs v2 差异"""
    with h5py.File(filepath, 'r') as f:
        action = np.array(f['action'])           # v2: [T, 8] target_pose + gripper
        qpos_end = np.array(f['observations']['qpos_end'])  # [T, 8] actual pose
        teleop_scale = np.array(f['teleop_scale'])
        timestamps = np.array(f['observations']['timestamps'])

    T = len(action)
    active_mask = teleop_scale > 0

    results = {
        'filename': os.path.basename(filepath),
        'total_frames': T,
        'active_frames': int(active_mask.sum()),
    }

    # ── 只分析 active frames ──
    act = action[active_mask]
    obs = qpos_end[active_mask]
    ts = timestamps[active_mask]
    N = len(act)

    # ═══════════════════════════════════════════════════════
    # 1. v2 target vs actual obs 的差异
    #    (v2 action = 遥操作目标, qpos_end = 实际位姿)
    # ═══════════════════════════════════════════════════════
    target_pos = act[:, :3]
    obs_pos = obs[:, :3]

    pos_diff_mm = np.linalg.norm(target_pos - obs_pos, axis=-1) * 1000  # mm
    pos_diff_per_axis_mm = (target_pos - obs_pos) * 1000  # [N, 3]

    rot_diff_deg = np.array([
        rotation_angle_deg(obs[t, 3:7], act[t, 3:7]) for t in range(N)
    ])

    results['v2_target_vs_obs'] = {
        'pos_diff_mm': {
            'mean': float(pos_diff_mm.mean()),
            'std': float(pos_diff_mm.std()),
            'max': float(pos_diff_mm.max()),
            'min': float(pos_diff_mm.min()),
            'p50': float(np.median(pos_diff_mm)),
            'p95': float(np.percentile(pos_diff_mm, 95)),
        },
        'pos_diff_per_axis_mm': {
            'x': {'mean': float(np.abs(pos_diff_per_axis_mm[:, 0]).mean()),
                   'max': float(np.abs(pos_diff_per_axis_mm[:, 0]).max())},
            'y': {'mean': float(np.abs(pos_diff_per_axis_mm[:, 1]).mean()),
                   'max': float(np.abs(pos_diff_per_axis_mm[:, 1]).max())},
            'z': {'mean': float(np.abs(pos_diff_per_axis_mm[:, 2]).mean()),
                   'max': float(np.abs(pos_diff_per_axis_mm[:, 2]).max())},
        },
        'rot_diff_deg': {
            'mean': float(rot_diff_deg.mean()),
            'std': float(rot_diff_deg.std()),
            'max': float(rot_diff_deg.max()),
            'p50': float(np.median(rot_diff_deg)),
            'p95': float(np.percentile(rot_diff_deg, 95)),
        },
    }

    # ═══════════════════════════════════════════════════════
    # 2. v1 模拟: 旧方法记录的 action ≈ qpos_end
    #    (因为 FK(get_plan_joint_pos()) 非常接近实际位姿)
    #    v1 的 relative pose 约等于 obs_diff（相邻帧差分）
    # ═══════════════════════════════════════════════════════

    # v2 relative pose: inv(T_obs) @ T_target
    v2_rel_poses = []
    for t in range(N):
        rel = compute_relative_pose_transform(obs[t, :7], act[t, :7])
        v2_rel_poses.append(rel)
    v2_rel_poses = np.array(v2_rel_poses)

    # v1 relative pose 模拟: inv(T_obs[t]) @ T_obs[t+1] (相邻帧差分)
    # 因为 v1 的 action ≈ actual pose, 所以 relative ≈ 二连帧差分
    v1_rel_poses = []
    for t in range(N - 1):
        rel = compute_relative_pose_transform(obs[t, :7], obs[t + 1, :7])
        v1_rel_poses.append(rel)
    v1_rel_poses = np.array(v1_rel_poses)

    # 对齐长度 (v1 少一帧)
    v2_rel_aligned = v2_rel_poses[:N - 1]

    # v1 vs v2 relative pose 差异
    rel_pos_diff_mm = np.linalg.norm(v2_rel_aligned[:, :3] - v1_rel_poses[:, :3], axis=-1) * 1000
    rel_rot_diff_deg = np.array([
        rotation_angle_deg(v1_rel_poses[t, 3:7], v2_rel_aligned[t, 3:7])
        for t in range(len(v1_rel_poses))
    ])

    results['v1_vs_v2_relative_pose'] = {
        'pos_diff_mm': {
            'mean': float(rel_pos_diff_mm.mean()),
            'std': float(rel_pos_diff_mm.std()),
            'max': float(rel_pos_diff_mm.max()),
            'p50': float(np.median(rel_pos_diff_mm)),
            'p95': float(np.percentile(rel_pos_diff_mm, 95)),
        },
        'rot_diff_deg': {
            'mean': float(rel_rot_diff_deg.mean()),
            'std': float(rel_rot_diff_deg.std()),
            'max': float(rel_rot_diff_deg.max()),
            'p50': float(np.median(rel_rot_diff_deg)),
            'p95': float(np.percentile(rel_rot_diff_deg, 95)),
        },
    }

    # ═══════════════════════════════════════════════════════
    # 3. v2 relative pose 的统计（训练标签特征）
    # ═══════════════════════════════════════════════════════
    v2_rel_pos_norm_mm = np.linalg.norm(v2_rel_poses[:, :3], axis=-1) * 1000
    v2_rel_rot_deg = np.array([
        rotation_angle_deg(np.array([0, 0, 0, 1.0]), v2_rel_poses[t, 3:7])
        for t in range(len(v2_rel_poses))
    ])

    results['v2_rel_pose_stats'] = {
        'pos_norm_mm': {
            'mean': float(v2_rel_pos_norm_mm.mean()),
            'std': float(v2_rel_pos_norm_mm.std()),
            'max': float(v2_rel_pos_norm_mm.max()),
        },
        'rot_deg': {
            'mean': float(v2_rel_rot_deg.mean()),
            'std': float(v2_rel_rot_deg.std()),
            'max': float(v2_rel_rot_deg.max()),
        },
    }

    v1_rel_pos_norm_mm = np.linalg.norm(v1_rel_poses[:, :3], axis=-1) * 1000
    v1_rel_rot_deg = np.array([
        rotation_angle_deg(np.array([0, 0, 0, 1.0]), v1_rel_poses[t, 3:7])
        for t in range(len(v1_rel_poses))
    ])

    results['v1_rel_pose_stats'] = {
        'pos_norm_mm': {
            'mean': float(v1_rel_pos_norm_mm.mean()),
            'std': float(v1_rel_pos_norm_mm.std()),
            'max': float(v1_rel_pos_norm_mm.max()),
        },
        'rot_deg': {
            'mean': float(v1_rel_rot_deg.mean()),
            'std': float(v1_rel_rot_deg.std()),
            'max': float(v1_rel_rot_deg.max()),
        },
    }

    # ═══════════════════════════════════════════════════════
    # 4. Roundtrip 精度验证
    #    v2: apply(rel, obs) ≈ target
    # ═══════════════════════════════════════════════════════
    roundtrip_errors = []
    for t in range(N):
        recovered = apply_relative_transform(v2_rel_poses[t], obs[t, :7])
        err = np.linalg.norm(recovered[:3] - act[t, :3]) * 1000
        roundtrip_errors.append(err)
    roundtrip_errors = np.array(roundtrip_errors)

    results['roundtrip_error_mm'] = {
        'mean': float(roundtrip_errors.mean()),
        'max': float(roundtrip_errors.max()),
    }

    # raw arrays for plotting
    results['_raw'] = {
        'pos_diff_mm': pos_diff_mm,
        'rot_diff_deg': rot_diff_deg,
        'rel_pos_diff_mm': rel_pos_diff_mm,
        'rel_rot_diff_deg': rel_rot_diff_deg,
        'v2_rel_pos_norm_mm': v2_rel_pos_norm_mm,
        'v1_rel_pos_norm_mm': v1_rel_pos_norm_mm,
        'timestamps': ts,
    }

    return results


def print_report(all_results):
    """打印综合报告到终端"""
    print("\n" + "=" * 70)
    print("GAP-1/2/3 修复前后数据对比分析报告")
    print("=" * 70)

    for r in all_results:
        print(f"\n{'─' * 60}")
        print(f"Episode: {r['filename']}")
        print(f"Total: {r['total_frames']} frames, Active: {r['active_frames']} frames")

        v2o = r['v2_target_vs_obs']
        print(f"\n  [1] v2 目标位姿 vs 实际位姿 (量化遥操作命令的 leading)")
        print(f"      位置差: mean={v2o['pos_diff_mm']['mean']:.1f}mm, "
              f"p50={v2o['pos_diff_mm']['p50']:.1f}mm, "
              f"p95={v2o['pos_diff_mm']['p95']:.1f}mm, "
              f"max={v2o['pos_diff_mm']['max']:.1f}mm")
        print(f"      旋转差: mean={v2o['rot_diff_deg']['mean']:.2f}°, "
              f"p95={v2o['rot_diff_deg']['p95']:.2f}°, "
              f"max={v2o['rot_diff_deg']['max']:.2f}°")
        ax = v2o['pos_diff_per_axis_mm']
        print(f"      各轴:   x={ax['x']['mean']:.1f}mm, y={ax['y']['mean']:.1f}mm, z={ax['z']['mean']:.1f}mm")

        vv = r['v1_vs_v2_relative_pose']
        print(f"\n  [2] v1 vs v2 训练标签差异 (relative pose)")
        print(f"      位置差: mean={vv['pos_diff_mm']['mean']:.1f}mm, "
              f"p95={vv['pos_diff_mm']['p95']:.1f}mm, "
              f"max={vv['pos_diff_mm']['max']:.1f}mm")
        print(f"      旋转差: mean={vv['rot_diff_deg']['mean']:.2f}°, "
              f"p95={vv['rot_diff_deg']['p95']:.2f}°, "
              f"max={vv['rot_diff_deg']['max']:.2f}°")

        v2s = r['v2_rel_pose_stats']
        v1s = r['v1_rel_pose_stats']
        print(f"\n  [3] 训练标签 (relative pose) 幅度对比")
        print(f"      v2 位移幅度: mean={v2s['pos_norm_mm']['mean']:.1f}mm, max={v2s['pos_norm_mm']['max']:.1f}mm")
        print(f"      v1 位移幅度: mean={v1s['pos_norm_mm']['mean']:.1f}mm, max={v1s['pos_norm_mm']['max']:.1f}mm")
        print(f"      v2 旋转幅度: mean={v2s['rot_deg']['mean']:.2f}°, max={v2s['rot_deg']['max']:.2f}°")
        print(f"      v1 旋转幅度: mean={v1s['rot_deg']['mean']:.2f}°, max={v1s['rot_deg']['max']:.2f}°")

        rt = r['roundtrip_error_mm']
        print(f"\n  [4] Roundtrip 精度: mean={rt['mean']:.4f}mm, max={rt['max']:.4f}mm")

    # 汇总
    print(f"\n{'=' * 70}")
    print("汇总")
    print(f"{'=' * 70}")

    all_pos_diff = np.concatenate([r['_raw']['pos_diff_mm'] for r in all_results])
    all_rot_diff = np.concatenate([r['_raw']['rot_diff_deg'] for r in all_results])
    all_rel_pos_diff = np.concatenate([r['_raw']['rel_pos_diff_mm'] for r in all_results])
    all_rel_rot_diff = np.concatenate([r['_raw']['rel_rot_diff_deg'] for r in all_results])

    print(f"\n  全局 v2_target vs actual:")
    print(f"    位置差 mean={all_pos_diff.mean():.1f}mm, p95={np.percentile(all_pos_diff, 95):.1f}mm")
    print(f"    旋转差 mean={all_rot_diff.mean():.2f}°, p95={np.percentile(all_rot_diff, 95):.2f}°")

    print(f"\n  全局 v1 vs v2 训练标签差异:")
    print(f"    位置差 mean={all_rel_pos_diff.mean():.1f}mm, p95={np.percentile(all_rel_pos_diff, 95):.1f}mm")
    print(f"    旋转差 mean={all_rel_rot_diff.mean():.2f}°, p95={np.percentile(all_rel_rot_diff, 95):.2f}°")

    return {
        'global_target_vs_obs_pos_mm': {'mean': float(all_pos_diff.mean()), 'p95': float(np.percentile(all_pos_diff, 95))},
        'global_target_vs_obs_rot_deg': {'mean': float(all_rot_diff.mean()), 'p95': float(np.percentile(all_rot_diff, 95))},
        'global_v1v2_rel_pos_mm': {'mean': float(all_rel_pos_diff.mean()), 'p95': float(np.percentile(all_rel_pos_diff, 95))},
        'global_v1v2_rel_rot_deg': {'mean': float(all_rel_rot_diff.mean()), 'p95': float(np.percentile(all_rel_rot_diff, 95))},
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, 'episode_*.hdf5')))
    if not files:
        print(f"No episode files found in {args.data_dir}")
        sys.exit(1)

    all_results = [analyze_episode(f) for f in files]
    summary = print_report(all_results)
