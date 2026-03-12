#!/usr/bin/env python3
"""
验证 HDF5 数据文件格式（v1 vs v2）

使用方法:
    python scripts/verify_hdf5_format.py <path_to_hdf5_or_directory>

功能:
    1. 检测数据版本（v1: 15D action, v2: 8D action）
    2. 打印字段维度和统计信息
    3. 验证数据质量（NaN、quaternion 归一化、合理范围等）
    4. 对比 action 和 observation 的差异（量化 GAP-1 修复效果）
"""

import argparse
import os
import glob
import sys
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlft.utils.pose_utils import compute_relative_pose_transform


def verify_episode(filepath: str, verbose: bool = True):
    """验证单个 episode HDF5 文件"""
    results = {
        'filepath': filepath,
        'valid': True,
        'warnings': [],
        'errors': [],
    }

    with h5py.File(filepath, 'r') as f:
        # 1. 检测版本
        data_version = f.attrs.get('data_version', 'v1')
        num_steps = f.attrs.get('num_steps', 0)
        record_freq = f.attrs.get('record_freq', 0)

        results['data_version'] = data_version
        results['num_steps'] = num_steps

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"File: {os.path.basename(filepath)}")
            print(f"  Version: {data_version}")
            print(f"  Steps: {num_steps}")
            print(f"  Freq: {record_freq} Hz")

        # 2. 检查 observations
        if 'observations' not in f:
            results['errors'].append('Missing observations group')
            results['valid'] = False
            return results

        obs = f['observations']
        for key in ['images', 'qpos_joint', 'qpos_end', 'timestamps']:
            if key not in obs:
                results['errors'].append(f'Missing observations/{key}')
                results['valid'] = False

        if not results['valid']:
            return results

        qpos_end = np.array(obs['qpos_end'])
        qpos_joint = np.array(obs['qpos_joint'])
        T = len(qpos_end)

        if verbose:
            print(f"  qpos_end shape: {qpos_end.shape}")
            print(f"  qpos_joint shape: {qpos_joint.shape}")

        # 3. 检查 action
        if 'action' not in f:
            results['warnings'].append('No action data')
            return results

        action = np.array(f['action'])
        action_dim = action.shape[-1]
        results['action_dim'] = action_dim

        if verbose:
            print(f"  action shape: {action.shape}")

        # 版本判定
        if data_version == 'v2':
            if action_dim != 8:
                results['errors'].append(f'v2 action should be 8D, got {action_dim}D')
                results['valid'] = False
        elif data_version == 'v1':
            if action_dim != 15:
                results['warnings'].append(f'v1 action expected 15D, got {action_dim}D')

        # 4. 检查 teleop_scale (v2 only)
        if data_version == 'v2':
            if 'teleop_scale' in f:
                teleop_scale = np.array(f['teleop_scale'])
                active_mask = teleop_scale > 0
                active_ratio = active_mask.mean()
                if verbose:
                    print(f"  teleop_scale: mean={teleop_scale[active_mask].mean():.3f}, "
                          f"active_ratio={active_ratio:.1%}")
                if active_ratio < 0.5:
                    results['warnings'].append(
                        f'Low active ratio ({active_ratio:.1%}): most frames have no teleop target'
                    )
            else:
                results['warnings'].append('v2 missing teleop_scale dataset')

        # 5. Quaternion validation
        for name, data in [('qpos_end', qpos_end), ('action', action)]:
            if data.shape[-1] >= 7:
                q_start = 3 if name != 'action' or data_version == 'v2' else 10
                q_end = q_start + 4
                if q_end <= data.shape[-1]:
                    quats = data[:, q_start:q_end]
                    norms = np.linalg.norm(quats, axis=-1)
                    bad_quats = np.abs(norms - 1.0) > 0.01
                    if bad_quats.any():
                        results['warnings'].append(
                            f'{name}: {bad_quats.sum()} non-unit quaternions '
                            f'(max deviation: {np.abs(norms - 1.0).max():.4f})'
                        )

        # 6. NaN check
        for name, data in [('qpos_end', qpos_end), ('qpos_joint', qpos_joint), ('action', action)]:
            if np.isnan(data).any():
                nan_count = np.isnan(data).sum()
                results['errors'].append(f'{name} contains {nan_count} NaN values')
                results['valid'] = False

        # 7. Action-observation difference analysis
        if verbose and action_dim >= 7:
            # Compute position difference between action target and observation
            if data_version == 'v2':
                target_pos = action[:, :3]
                obs_pos = qpos_end[:, :3]
            else:
                target_pos = action[:, 7:10]
                obs_pos = qpos_end[:, :3]

            pos_diff = np.linalg.norm(target_pos - obs_pos, axis=-1)
            print(f"  Action-Obs pos diff: mean={pos_diff.mean()*1000:.1f}mm, "
                  f"max={pos_diff.max()*1000:.1f}mm, min={pos_diff.min()*1000:.1f}mm")

        # 8. Position range check
        if verbose:
            for axis, name in enumerate(['x', 'y', 'z']):
                vals_obs = qpos_end[:, axis]
                print(f"  qpos_end.{name}: [{vals_obs.min():.4f}, {vals_obs.max():.4f}]")

    # Print warnings/errors
    if verbose:
        for w in results['warnings']:
            print(f"  WARNING: {w}")
        for e in results['errors']:
            print(f"  ERROR: {e}")
        status = "PASS" if results['valid'] and not results['warnings'] else \
                 "WARN" if results['valid'] else "FAIL"
        print(f"  Status: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Verify CARM HDF5 data format')
    parser.add_argument('path', type=str, help='Path to HDF5 file or directory')
    parser.add_argument('--quiet', action='store_true', help='Only print summary')
    args = parser.parse_args()

    path = os.path.expanduser(args.path)

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, 'episode_*.hdf5')))
        if not files:
            print(f"No episode_*.hdf5 files found in {path}")
            sys.exit(1)
    else:
        print(f"Path not found: {path}")
        sys.exit(1)

    print(f"Verifying {len(files)} file(s)...")

    all_results = []
    for f in files:
        result = verify_episode(f, verbose=not args.quiet)
        all_results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary: {len(files)} files")

    versions = {}
    for r in all_results:
        v = r.get('data_version', 'unknown')
        versions[v] = versions.get(v, 0) + 1

    valid_count = sum(1 for r in all_results if r['valid'])
    warn_count = sum(1 for r in all_results if r['warnings'])
    error_count = sum(1 for r in all_results if not r['valid'])

    print(f"  Versions: {versions}")
    print(f"  Valid: {valid_count}/{len(files)}")
    print(f"  With warnings: {warn_count}")
    print(f"  With errors: {error_count}")
    print(f"{'=' * 60}")

    if error_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
