#!/usr/bin/env python3
"""
验证 CARM HDF5 数据文件格式。

支持：
- teleop_v2
- inference_staging
- auto schema detection
"""

import argparse
import os
import glob
import sys
import time
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def infer_schema(f: h5py.File) -> str:
    dataset_type = f.attrs.get('dataset_type', '')
    if dataset_type == 'inference_staging':
        return 'inference_staging'
    if 'source_file' in f.attrs or 'filtered_intervention' in f.attrs:
        return 'inference_staging'
    return 'teleop_v2'


def _add_warning(results: dict, message: str) -> None:
    results['warnings'].append(message)


def _add_error(results: dict, message: str) -> None:
    results['errors'].append(message)
    results['valid'] = False


def verify_episode(filepath: str, schema: str = 'auto', verbose: bool = True):
    """验证单个 episode HDF5 文件"""
    results = {
        'filepath': filepath,
        'valid': True,
        'warnings': [],
        'errors': [],
    }

    retry_errors = (
        'unable to lock file',
        'resource temporarily unavailable',
        'bad object header version number',
    )
    last_error = None

    for attempt in range(6):
        try:
            with h5py.File(filepath, 'r') as f:
                active_schema = infer_schema(f) if schema == 'auto' else schema
                data_version = f.attrs.get('data_version', 'v1')
                num_steps = int(f.attrs.get('num_steps', 0))
                record_freq = f.attrs.get('record_freq', 0)

                results['schema'] = active_schema
                results['data_version'] = data_version
                results['num_steps'] = num_steps

                if verbose:
                    print(f"\n{'─' * 60}")
                    print(f"File: {os.path.basename(filepath)}")
                    print(f"  Schema: {active_schema}")
                    print(f"  Version: {data_version}")
                    print(f"  Steps: {num_steps}")
                    print(f"  Freq: {record_freq} Hz")

                if 'observations' not in f:
                    _add_error(results, 'Missing observations group')
                    return results

                obs = f['observations']
                for key in ['images', 'qpos_joint', 'qpos_end', 'timestamps']:
                    if key not in obs:
                        _add_error(results, f'Missing observations/{key}')

                if 'action' not in f:
                    _add_warning(results, 'No action data')
                    return results

                if not results['valid']:
                    return results

                qpos_end = np.array(obs['qpos_end'])
                qpos_joint = np.array(obs['qpos_joint'])
                action = np.array(f['action'])
                action_dim = action.shape[-1]
                results['action_dim'] = action_dim

                if verbose:
                    print(f"  qpos_end shape: {qpos_end.shape}")
                    print(f"  qpos_joint shape: {qpos_joint.shape}")
                    print(f"  action shape: {action.shape}")

                if active_schema == 'teleop_v2':
                    if data_version == 'v2':
                        if action_dim != 8:
                            _add_error(results, f'v2 action should be 8D, got {action_dim}D')
                        if 'teleop_scale' in f:
                            teleop_scale = np.array(f['teleop_scale'])
                            active_mask = teleop_scale > 0
                            active_ratio = active_mask.mean()
                            if verbose:
                                if active_mask.any():
                                    print(f"  teleop_scale: mean={teleop_scale[active_mask].mean():.3f}, active_ratio={active_ratio:.1%}")
                                else:
                                    print(f"  teleop_scale: active_ratio={active_ratio:.1%}")
                            if active_ratio < 0.5:
                                _add_warning(results, f'Low active ratio ({active_ratio:.1%}): most frames have no teleop target')
                        else:
                            _add_warning(results, 'v2 missing teleop_scale dataset')
                    elif data_version == 'v1' and action_dim != 15:
                        _add_warning(results, f'v1 action expected 15D, got {action_dim}D')
                elif active_schema == 'inference_staging':
                    if f.attrs.get('dataset_type', '') != 'inference_staging':
                        _add_error(results, 'inference_staging missing dataset_type=inference_staging')
                    if action_dim != 8:
                        _add_error(results, f'inference_staging action should be 8D, got {action_dim}D')
                    for attr_key in [
                        'staging_schema_version',
                        'source_file',
                        'filtered_intervention',
                        'kept_steps',
                        'dropped_steps',
                        'intervention_ratio_raw',
                        'intervention_ratio_kept',
                        'admission_label',
                        'admission_pass',
                        'admission_reason',
                        'policy_level',
                        'admission_policy',
                        'min_steps',
                        'max_intervention_ratio',
                        'timestamp_semantics',
                    ]:
                        if attr_key not in f.attrs:
                            _add_error(results, f'inference_staging missing attr {attr_key}')

                    if 'action_semantics_version' not in f.attrs:
                        _add_warning(results, 'inference_staging missing attr action_semantics_version')

                    kept_steps = int(f.attrs.get('kept_steps', -1))
                    dropped_steps = int(f.attrs.get('dropped_steps', -1))
                    if kept_steps != len(action):
                        _add_error(results, f'kept_steps={kept_steps} inconsistent with action length={len(action)}')
                    if num_steps != len(action):
                        _add_error(results, f'num_steps={num_steps} inconsistent with action length={len(action)}')
                    if kept_steps >= 0 and dropped_steps >= 0 and kept_steps + dropped_steps < kept_steps:
                        _add_error(results, 'kept_steps/dropped_steps overflow')

                    timestamp_semantics = f.attrs.get('timestamp_semantics', None)
                    if timestamp_semantics != 'obs_stamp_ros':
                        _add_warning(results, f"unexpected timestamp_semantics={timestamp_semantics!r}; expected 'obs_stamp_ros'")
                    action_semantics_version = f.attrs.get('action_semantics_version', None)
                    if action_semantics_version != 'absolute_ee_target_pose_v2':
                        _add_warning(results, f"unexpected action_semantics_version={action_semantics_version!r}")
                else:
                    _add_error(results, f'Unknown schema: {active_schema}')

                for name, data in [('qpos_end', qpos_end), ('action', action)]:
                    if data.shape[-1] >= 7:
                        q_start = 3 if (active_schema == 'inference_staging' or data_version == 'v2' or name != 'action') else 10
                        q_end = q_start + 4
                        if q_end <= data.shape[-1]:
                            quats = data[:, q_start:q_end]
                            norms = np.linalg.norm(quats, axis=-1)
                            bad_quats = np.abs(norms - 1.0) > 0.01
                            if bad_quats.any():
                                _add_warning(results, f'{name}: {bad_quats.sum()} non-unit quaternions (max deviation: {np.abs(norms - 1.0).max():.4f})')

                for name, data in [('qpos_end', qpos_end), ('qpos_joint', qpos_joint), ('action', action)]:
                    if np.isnan(data).any():
                        _add_error(results, f'{name} contains {np.isnan(data).sum()} NaN values')

                if verbose and action_dim >= 7:
                    target_pos = action[:, :3] if (active_schema == 'inference_staging' or data_version == 'v2') else action[:, 7:10]
                    obs_pos = qpos_end[:, :3]
                    pos_diff = np.linalg.norm(target_pos - obs_pos, axis=-1)
                    print(f"  Action-Obs pos diff: mean={pos_diff.mean()*1000:.1f}mm, max={pos_diff.max()*1000:.1f}mm, min={pos_diff.min()*1000:.1f}mm")
                    for axis, name in enumerate(['x', 'y', 'z']):
                        vals_obs = qpos_end[:, axis]
                        print(f"  qpos_end.{name}: [{vals_obs.min():.4f}, {vals_obs.max():.4f}]")

            if verbose:
                for w in results['warnings']:
                    print(f"  WARNING: {w}")
                for e in results['errors']:
                    print(f"  ERROR: {e}")
                status = 'PASS' if results['valid'] and not results['warnings'] else 'WARN' if results['valid'] else 'FAIL'
                print(f"  Status: {status}")

            return results
        except (BlockingIOError, OSError) as e:
            msg = str(e).lower()
            last_error = e
            if any(err in msg for err in retry_errors) and attempt < 5:
                time.sleep(0.2 * (attempt + 1))
                continue
            raise

    raise last_error


def main():
    parser = argparse.ArgumentParser(description='Verify CARM HDF5 data format')
    parser.add_argument('path', type=str, help='Path to HDF5 file or directory')
    parser.add_argument('--quiet', action='store_true', help='Only print summary')
    parser.add_argument('--schema', choices=['auto', 'teleop_v2', 'inference_staging'], default='auto', help='Schema to validate against')
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
        result = verify_episode(f, schema=args.schema, verbose=not args.quiet)
        all_results.append(result)

    print(f"\n{'=' * 60}")
    print(f"Summary: {len(files)} files")

    schemas = {}
    versions = {}
    for r in all_results:
        schema_name = r.get('schema', 'unknown')
        schemas[schema_name] = schemas.get(schema_name, 0) + 1
        version_name = r.get('data_version', 'unknown')
        versions[version_name] = versions.get(version_name, 0) + 1

    valid_count = sum(1 for r in all_results if r['valid'])
    warn_count = sum(1 for r in all_results if r['warnings'])
    error_count = sum(1 for r in all_results if not r['valid'])

    print(f"  Schemas: {schemas}")
    print(f"  Versions: {versions}")
    print(f"  Valid: {valid_count}/{len(files)}")
    print(f"  With warnings: {warn_count}")
    print(f"  With errors: {error_count}")
    print(f"{'=' * 60}")

    if error_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
