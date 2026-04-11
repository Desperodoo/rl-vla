#!/usr/bin/env python3
"""Compare teleop and inference data gaps for current CARM runs.

Outputs a JSON report covering:
- observation-side timing gap
- inference/chunk/control timing gap
- action-vs-state gap broken down by action source
- same-step and t+k alignment scans
- absolute and relative pose metrics
- intervention-free subset statistics for inference data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlft.utils.pose_utils import compute_relative_pose_transform


SourceData = dict[str, np.ndarray]


def _stats(arr: list[float] | np.ndarray) -> dict[str, Any] | None:
    if isinstance(arr, np.ndarray):
        if arr.size == 0:
            return None
        a = arr.astype(float)
    else:
        if not arr:
            return None
        a = np.array(arr, dtype=float)
    return {
        'count': int(a.size),
        'mean': float(a.mean()),
        'p50': float(np.percentile(a, 50)),
        'p90': float(np.percentile(a, 90)),
        'p95': float(np.percentile(a, 95)),
        'p99': float(np.percentile(a, 99)),
        'min': float(a.min()),
        'max': float(a.max()),
    }


def _timeline_stats(path: str, teleop: bool) -> dict[str, Any]:
    obs_delta: list[float] = []
    infer: list[float] = []
    chunk: list[float] = []
    control_send_minus_query: list[float] = []
    candidate_age: list[float] = []

    with open(path, 'r') as f:
        for line in f:
            e = json.loads(line)
            t = e.get('event')
            if teleop:
                if t == 'record_step' and e.get('delta_obs') is not None:
                    obs_delta.append(e['delta_obs'])
                continue

            if t == 'obs' and e.get('delta_obs') is not None:
                obs_delta.append(e['delta_obs'])
            elif t == 'inference' and e.get('inference_time') is not None:
                infer.append(e['inference_time'])
            elif t == 'chunk' and e.get('delta_chunk_obs') is not None:
                chunk.append(e['delta_chunk_obs'])
            elif t == 'control':
                if e.get('query_time') is not None and e.get('t_send_sys') is not None:
                    control_send_minus_query.append(e['t_send_sys'] - e['query_time'])
                cands = e.get('candidate_timestamps', [])
                qt = e.get('query_time')
                if cands and qt is not None:
                    candidate_age.append(qt - max(cands))

    return {
        'delta_obs': _stats(obs_delta),
        'inference_time': _stats(infer),
        'delta_chunk_obs': _stats(chunk),
        'control_send_minus_query': _stats(control_send_minus_query),
        'candidate_age': _stats(candidate_age),
    }


def _rotation_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r_diff = r1.inv() * r2
    return float(np.degrees(r_diff.magnitude()))


def _relative_rotation_deg(relative_quat: np.ndarray) -> float:
    return _rotation_angle_deg(np.array([0.0, 0.0, 0.0, 1.0]), relative_quat)


def _safe_bool_mask(mask: np.ndarray | None, length: int) -> np.ndarray:
    if mask is None:
        return np.ones(length, dtype=bool)
    return mask.astype(bool)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def _load_action_sources(path: str, kind: str) -> dict[str, Any]:
    with h5py.File(path, 'r') as f:
        obs = f['observations']
        qpos_end = obs['qpos_end'][:]
        timestamps = obs['timestamps'][:] if 'timestamps' in obs else None
        attrs = {k: _json_safe_value(f.attrs[k]) for k in f.attrs.keys()}
        action_semantics_version = attrs.get('action_semantics_version', None)

        sources: SourceData = {}
        subsets: dict[str, np.ndarray] = {}

        if 'action' in f:
            sources['compat_action'] = f['action'][:]

        if kind == 'inference':
            if 'action_model' in f:
                sources['model'] = f['action_model'][:, 0, :]
            if 'action_executed' in f:
                sources['executed'] = f['action_executed'][:, 0, :]
            subsets['all'] = np.ones(len(qpos_end), dtype=bool)
        else:
            subsets['all'] = np.ones(len(qpos_end), dtype=bool)
            if 'teleop_scale' in f:
                teleop_scale = f['teleop_scale'][:]
                subsets['active_teleop'] = teleop_scale > 0

        return {
            'qpos_end': qpos_end,
            'timestamps': timestamps,
            'sources': sources,
            'subsets': subsets,
            'attrs': attrs,
            'action_semantics_version': action_semantics_version,
        }


def _subset_summary(mask: np.ndarray | None) -> dict[str, Any] | None:
    if mask is None:
        return None
    return {
        'count': int(mask.sum()),
        'ratio': float(mask.mean()) if mask.size > 0 else 0.0,
    }


def _sample_indices(mask: np.ndarray, num_samples: int) -> list[int]:
    if num_samples <= 0:
        return []
    return np.flatnonzero(mask).astype(int).tolist()[:num_samples]


def _build_spot_check_samples(
    targets: np.ndarray,
    qpos_end: np.ndarray,
    mask: np.ndarray,
    k: int,
    num_samples: int,
) -> list[dict[str, Any]]:
    if num_samples <= 0 or len(targets) <= k:
        return []

    base_mask = mask[: len(targets) - k]
    indices = _sample_indices(base_mask, num_samples)
    return _build_samples_for_indices(targets, qpos_end, indices, k)


def _build_samples_for_indices(
    targets: np.ndarray,
    qpos_end: np.ndarray,
    indices: list[int],
    k: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])

    for t in indices:
        ref_idx = t + k
        if ref_idx >= len(qpos_end) or t >= len(targets):
            continue
        target = targets[t]
        ref = qpos_end[ref_idx]
        rel_pose = compute_relative_pose_transform(ref[:7], target[:7])
        abs_gap_xyz = target[:3] - ref[:3]
        sample = {
            't': int(t),
            'ref_t': int(ref_idx),
            'qpos_end_xyz': ref[:3].astype(float).tolist(),
            'qpos_end_quat': ref[3:7].astype(float).tolist(),
            'qpos_end_gripper': float(ref[-1]),
            'target_xyz': target[:3].astype(float).tolist(),
            'target_quat': target[3:7].astype(float).tolist(),
            'target_gripper': float(target[-1]),
            'absolute_gap_xyz': abs_gap_xyz.astype(float).tolist(),
            'absolute_gap_norm': float(np.linalg.norm(abs_gap_xyz)),
            'relative_translation': rel_pose[:3].astype(float).tolist(),
            'relative_translation_norm': float(np.linalg.norm(rel_pose[:3])),
            'relative_quat': rel_pose[3:7].astype(float).tolist(),
            'relative_rotation_deg': _rotation_angle_deg(identity_quat, rel_pose[3:7]),
            'gripper_gap': float(abs(target[-1] - ref[-1])),
        }
        samples.append(sample)

    return samples


def _rank_indices_by_metric(
    targets: np.ndarray,
    qpos_end: np.ndarray,
    mask: np.ndarray,
    k: int,
    descending: bool,
    top_n: int,
) -> list[int]:
    if top_n <= 0 or len(targets) <= k:
        return []
    valid_indices = np.flatnonzero(mask[: len(targets) - k]).astype(int)
    if valid_indices.size == 0:
        return []

    scores = []
    for t in valid_indices:
        ref = qpos_end[t + k]
        target = targets[t]
        score = float(np.linalg.norm(target[:3] - ref[:3]))
        scores.append((score, int(t)))

    scores.sort(key=lambda x: x[0], reverse=descending)
    return [t for _, t in scores[:top_n]]


def _rank_indices_by_improvement(
    targets: np.ndarray,
    qpos_end: np.ndarray,
    mask: np.ndarray,
    best_k: int,
    top_n: int,
) -> list[int]:
    if top_n <= 0 or best_k <= 0 or len(targets) <= best_k:
        return []
    valid_indices = np.flatnonzero(mask[: len(targets) - best_k]).astype(int)
    if valid_indices.size == 0:
        return []

    gains = []
    for t in valid_indices:
        same_gap = float(np.linalg.norm(targets[t, :3] - qpos_end[t, :3]))
        best_gap = float(np.linalg.norm(targets[t, :3] - qpos_end[t + best_k, :3]))
        gains.append((same_gap - best_gap, int(t)))

    gains.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in gains[:top_n]]


def _compute_absolute_gap(targets: np.ndarray, refs: np.ndarray) -> dict[str, Any]:
    pos_gap = np.linalg.norm(targets[:, :3] - refs[:, :3], axis=1)
    grip_gap = np.abs(targets[:, -1] - refs[:, -1])
    return {
        'pos_gap': _stats(pos_gap),
        'gripper_gap': _stats(grip_gap),
        'static_ratio': float(np.mean(pos_gap < 1e-6)),
    }


def _compute_relative_gap(targets: np.ndarray, refs: np.ndarray) -> dict[str, Any]:
    rel_trans_norm: list[float] = []
    rel_rot_deg: list[float] = []
    rel_grip_gap: list[float] = []

    for ref_pose, target_pose in zip(refs, targets, strict=False):
        rel_pose = compute_relative_pose_transform(ref_pose[:7], target_pose[:7])
        rel_trans_norm.append(float(np.linalg.norm(rel_pose[:3])))
        rel_rot_deg.append(_relative_rotation_deg(rel_pose[3:7]))
        rel_grip_gap.append(float(abs(target_pose[-1] - ref_pose[-1])))

    return {
        'translation_norm': _stats(rel_trans_norm),
        'rotation_deg': _stats(rel_rot_deg),
        'gripper_gap': _stats(rel_grip_gap),
    }


def _compute_metrics_for_arrays(targets: np.ndarray, refs: np.ndarray, include_relative: bool) -> dict[str, Any]:
    result = {'absolute': _compute_absolute_gap(targets, refs)}
    if include_relative:
        result['relative'] = _compute_relative_gap(targets, refs)
    return result


def _compute_dt(timestamps: np.ndarray | None, length: int) -> np.ndarray:
    if timestamps is None or len(timestamps) < 2:
        return np.ones(max(length - 1, 0), dtype=float)
    clipped = np.asarray(timestamps[:length], dtype=float)
    dt = np.diff(clipped)
    dt = np.where(dt > 1e-6, dt, np.nan)
    return dt


def _rotation_step_deg(quats: np.ndarray) -> np.ndarray:
    if len(quats) < 2:
        return np.array([], dtype=float)
    vals: list[float] = []
    for q1, q2 in zip(quats[:-1], quats[1:], strict=False):
        vals.append(_rotation_angle_deg(q1, q2))
    return np.array(vals, dtype=float)


def _compute_motion_series(poses: np.ndarray, dt: np.ndarray) -> dict[str, np.ndarray]:
    if len(poses) < 2:
        empty = np.array([], dtype=float)
        return {
            'translation_step': empty,
            'translation_velocity': empty,
            'rotation_step_deg': empty,
            'rotation_velocity_deg': empty,
            'gripper_step': empty,
            'gripper_velocity': empty,
        }

    translation_step = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    rotation_step_deg = _rotation_step_deg(poses[:, 3:7])
    gripper_step = np.abs(np.diff(poses[:, -1]))

    with np.errstate(divide='ignore', invalid='ignore'):
        translation_velocity = translation_step / dt
        rotation_velocity_deg = rotation_step_deg / dt
        gripper_velocity = gripper_step / dt

    return {
        'translation_step': translation_step,
        'translation_velocity': translation_velocity,
        'rotation_step_deg': rotation_step_deg,
        'rotation_velocity_deg': rotation_velocity_deg,
        'gripper_step': gripper_step,
        'gripper_velocity': gripper_velocity,
    }


def _safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if num.size == 0 or den.size == 0:
        return np.array([], dtype=float)
    valid = den > eps
    if not np.any(valid):
        return np.array([], dtype=float)
    return num[valid] / den[valid]


def _window_slices(length: int) -> dict[str, slice]:
    if length <= 0:
        return {}
    a = length // 3
    b = (2 * length) // 3
    return {
        'early': slice(0, a),
        'middle': slice(a, b),
        'late': slice(b, length),
    }


def _summarize_motion_series(series: dict[str, np.ndarray]) -> dict[str, Any]:
    return {name: _stats(values[np.isfinite(values)]) for name, values in series.items()}


def _compute_motion_analysis(
    targets: np.ndarray,
    qpos_end: np.ndarray,
    timestamps: np.ndarray | None,
    mask: np.ndarray,
    best_k: int,
) -> dict[str, Any]:
    source_len = min(len(targets), len(qpos_end))
    targets = targets[:source_len]
    qpos_end = qpos_end[:source_len]
    mask = mask[:source_len]
    if timestamps is not None:
        timestamps = timestamps[:source_len]

    if source_len < 2:
        return {
            'same_step_motion': None,
            'future_k_improvement': None,
            'windows': {},
        }

    step_mask = mask[:-1] & mask[1:]
    dt = _compute_dt(timestamps, source_len)
    valid_dt = np.isfinite(dt)
    step_mask = step_mask & valid_dt

    realized = _compute_motion_series(qpos_end, dt)
    target = _compute_motion_series(targets, dt)

    realized_valid = {
        name: values[step_mask]
        for name, values in realized.items()
    }
    target_valid = {
        name: values[step_mask]
        for name, values in target.items()
    }

    translation_ratio = _safe_ratio(
        realized['translation_step'][step_mask],
        target['translation_step'][step_mask],
    )
    translation_velocity_ratio = _safe_ratio(
        realized['translation_velocity'][step_mask],
        target['translation_velocity'][step_mask],
    )
    gripper_ratio = _safe_ratio(
        realized['gripper_step'][step_mask],
        target['gripper_step'][step_mask],
    )

    same_gap = np.linalg.norm(targets[:, :3] - qpos_end[:, :3], axis=1)
    future_gap = None
    improvement = None
    if best_k > 0 and len(targets) > best_k:
        future_base_mask = mask[: len(targets) - best_k]
        if np.any(future_base_mask):
            future_gap_arr = np.linalg.norm(targets[: len(targets) - best_k, :3] - qpos_end[best_k:, :3], axis=1)
            future_gap = _stats(future_gap_arr[future_base_mask])
            improvement_arr = same_gap[: len(targets) - best_k][future_base_mask] - future_gap_arr[future_base_mask]
            improvement = _stats(improvement_arr)

    window_results: dict[str, Any] = {}
    for name, sl in _window_slices(source_len - 1).items():
        local_mask = step_mask[sl]
        if local_mask.size == 0 or not np.any(local_mask):
            window_results[name] = None
            continue
        window_results[name] = {
            'realized_motion': {
                key: _stats(values[sl][local_mask])
                for key, values in realized.items()
            },
            'target_motion': {
                key: _stats(values[sl][local_mask])
                for key, values in target.items()
            },
            'ratios': {
                'translation_step': _stats(_safe_ratio(realized['translation_step'][sl][local_mask], target['translation_step'][sl][local_mask])),
                'translation_velocity': _stats(_safe_ratio(realized['translation_velocity'][sl][local_mask], target['translation_velocity'][sl][local_mask])),
                'gripper_step': _stats(_safe_ratio(realized['gripper_step'][sl][local_mask], target['gripper_step'][sl][local_mask])),
            },
        }

    return {
        'same_step_motion': {
            'realized_motion': _summarize_motion_series(realized_valid),
            'target_motion': _summarize_motion_series(target_valid),
            'realized_vs_target_ratio': {
                'translation_step': _stats(translation_ratio),
                'translation_velocity': _stats(translation_velocity_ratio),
                'gripper_step': _stats(gripper_ratio),
            },
        },
        'best_k_by_motion': best_k,
        'future_k_improvement': {
            'same_step_pos_gap': _stats(same_gap[mask]),
            'best_future_pos_gap': future_gap,
            'improvement': improvement,
        },
        'windows': window_results,
    }


def _scan_alignment(
    targets: np.ndarray,
    qpos_end: np.ndarray,
    mask: np.ndarray,
    max_k: int,
    include_relative: bool,
) -> dict[str, Any]:
    scan: dict[str, Any] = {}
    best_mean: tuple[int, float] | None = None
    best_p95: tuple[int, float] | None = None

    for k in range(max_k + 1):
        if len(targets) <= k:
            break
        valid_mask = mask[: len(targets) - k]
        if not np.any(valid_mask):
            continue

        aligned_targets = targets[: len(targets) - k][valid_mask]
        aligned_refs = qpos_end[k:][valid_mask]
        metrics = _compute_metrics_for_arrays(aligned_targets, aligned_refs, include_relative)
        scan[str(k)] = metrics

        pos_stats = metrics['absolute']['pos_gap']
        if pos_stats is None:
            continue
        mean_val = pos_stats['mean']
        p95_val = pos_stats['p95']
        if best_mean is None or mean_val < best_mean[1]:
            best_mean = (k, mean_val)
        if best_p95 is None or p95_val < best_p95[1]:
            best_p95 = (k, p95_val)

    return {
        'by_k': scan,
        'best_k_by_mean': None if best_mean is None else {'k': best_mean[0], 'mean': best_mean[1]},
        'best_k_by_p95': None if best_p95 is None else {'k': best_p95[0], 'p95': best_p95[1]},
    }


def _analyze_source(
    source_name: str,
    targets: np.ndarray,
    qpos_end: np.ndarray,
    timestamps: np.ndarray | None,
    subsets: dict[str, np.ndarray],
    max_k: int,
    include_relative: bool,
    include_intervention_subsets: bool,
    spot_check_samples: int,
) -> dict[str, Any]:
    source_len = min(len(targets), len(qpos_end))
    targets = targets[:source_len]
    qpos_end = qpos_end[:source_len]

    subset_names = ['all']
    if include_intervention_subsets:
        subset_names.extend([name for name in subsets.keys() if name != 'all'])

    subset_results: dict[str, Any] = {}
    for subset_name in subset_names:
        mask = _safe_bool_mask(subsets.get(subset_name), source_len)[:source_len]
        if not np.any(mask):
            subset_results[subset_name] = {
                'summary': _subset_summary(mask),
                'same_step': None,
                'scan_by_k': {'by_k': {}, 'best_k_by_mean': None, 'best_k_by_p95': None},
                'spot_check': {
                    'same_step': [],
                    'best_k_by_mean': [],
                    'worst_same_step': [],
                    'best_same_step': [],
                    'most_improved_to_best_k': [],
                },
            }
            continue

        same_step = _compute_metrics_for_arrays(targets[mask], qpos_end[mask], include_relative)
        scan = _scan_alignment(targets, qpos_end, mask, max_k=max_k, include_relative=include_relative)
        best_k = scan['best_k_by_mean']['k'] if scan['best_k_by_mean'] is not None else 0
        worst_same_step_indices = _rank_indices_by_metric(targets, qpos_end, mask, k=0, descending=True, top_n=spot_check_samples)
        best_same_step_indices = _rank_indices_by_metric(targets, qpos_end, mask, k=0, descending=False, top_n=spot_check_samples)
        most_improved_indices = _rank_indices_by_improvement(targets, qpos_end, mask, best_k=best_k, top_n=spot_check_samples)
        subset_results[subset_name] = {
            'summary': _subset_summary(mask),
            'same_step': same_step,
            'scan_by_k': scan,
            'motion_analysis': _compute_motion_analysis(
                targets=targets,
                qpos_end=qpos_end,
                timestamps=timestamps,
                mask=mask,
                best_k=best_k,
            ),
            'spot_check': {
                'same_step': _build_spot_check_samples(targets, qpos_end, mask, k=0, num_samples=spot_check_samples),
                'best_k_by_mean': _build_spot_check_samples(targets, qpos_end, mask, k=best_k, num_samples=spot_check_samples),
                'worst_same_step': _build_samples_for_indices(targets, qpos_end, worst_same_step_indices, k=0),
                'best_same_step': _build_samples_for_indices(targets, qpos_end, best_same_step_indices, k=0),
                'most_improved_to_best_k': _build_samples_for_indices(targets, qpos_end, most_improved_indices, k=best_k),
            },
        }

    return {
        'source_name': source_name,
        'steps': int(source_len),
        'subsets': subset_results,
    }


def _gap_analysis(
    path: str,
    kind: str,
    max_k: int,
    include_relative: bool,
    include_intervention_subsets: bool,
    spot_check_samples: int,
) -> dict[str, Any]:
    data = _load_action_sources(path, kind)
    qpos_end = data['qpos_end']
    timestamps = data.get('timestamps')
    sources = data['sources']
    subsets = data['subsets']

    source_aliases = {
        'compat_action': 'compat_action',
        'model': 'model',
        'executed': 'executed',
    }

    results: dict[str, Any] = {
        'kind': kind,
        'num_steps': int(len(qpos_end)),
        'available_sources': list(sources.keys()),
        'available_subsets': list(subsets.keys()),
        'action_semantics_version': data.get('action_semantics_version'),
        'attrs': data.get('attrs', {}),
        'sources': {},
    }

    for source_key, source_name in source_aliases.items():
        if source_key not in sources:
            continue
        results['sources'][source_name] = _analyze_source(
            source_name=source_name,
            targets=sources[source_key],
            qpos_end=qpos_end,
            timestamps=timestamps,
            subsets=subsets,
            max_k=max_k,
            include_relative=include_relative,
            include_intervention_subsets=include_intervention_subsets,
            spot_check_samples=spot_check_samples,
        )

    return results


def _legacy_action_gap(gap_analysis: dict[str, Any], preferred_source: str) -> dict[str, Any] | None:
    source = gap_analysis.get('sources', {}).get(preferred_source)
    if source is None:
        return None
    all_subset = source.get('subsets', {}).get('all')
    if all_subset is None or all_subset.get('same_step') is None:
        return None
    same_step = all_subset['same_step']
    return {
        'kind': gap_analysis.get('kind'),
        'steps': source['steps'],
        'pos_gap': same_step['absolute']['pos_gap'],
        'gripper_gap': same_step['absolute']['gripper_gap'],
        'static_ratio': same_step['absolute']['static_ratio'],
    }


def _source_summary_lines(label: str, source: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    all_subset = source.get('subsets', {}).get('all')
    if all_subset and all_subset.get('same_step'):
        pos = all_subset['same_step']['absolute']['pos_gap']
        if pos is not None:
            lines.append(f'{label} same-step position-gap mean≈{pos["mean"]:.4f}')
        motion = all_subset.get('motion_analysis', {})
        same_motion = motion.get('same_step_motion', {}) if motion else {}
        realized_vel = ((same_motion.get('realized_motion') or {}).get('translation_velocity') or {}).get('mean')
        target_vel = ((same_motion.get('target_motion') or {}).get('translation_velocity') or {}).get('mean')
        ratio_mean = ((same_motion.get('realized_vs_target_ratio') or {}).get('translation_velocity') or {}).get('mean')
        if realized_vel is not None:
            lines.append(f'{label} realized-pos-vel mean≈{realized_vel:.4f}')
        if target_vel is not None:
            lines.append(f'{label} target-pos-vel mean≈{target_vel:.4f}')
        if ratio_mean is not None:
            lines.append(f'{label} realized/target vel ratio mean≈{ratio_mean:.4f}')
        improvement = (motion.get('future_k_improvement', {}) or {}).get('improvement')
        if improvement is not None and improvement.get('mean') is not None:
            lines.append(f'{label} future-k improvement mean≈{improvement["mean"]:.4f}')
        best_k = all_subset['scan_by_k'].get('best_k_by_mean')
        if best_k is not None and best_k['k'] > 0:
            lines.append(f'{label} best aligned at k={best_k["k"]} with mean≈{best_k["mean"]:.4f}')
    return lines


def _semantic_notes(gap_analysis: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    kind = gap_analysis.get('kind')
    semantics = gap_analysis.get('action_semantics_version')
    if kind == 'inference':
        if semantics == 'absolute_ee_target_pose_v2':
            notes.append('Inference sample uses absolute EE target pose semantics (v2); same-step and t+k absolute-gap metrics are directly meaningful.')
        elif semantics is None:
            notes.append('Inference sample has no action_semantics_version; treat legacy results cautiously because action fields may be relative pose actions.')
        else:
            notes.append(f'Inference sample reports action_semantics_version={semantics!r}; verify semantics before comparing against qpos_end.')
    return notes


def _motion_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for kind_key in ['teleop_gap_analysis', 'inference_gap_analysis']:
        kind_report = report.get(kind_key, {})
        kind = kind_report.get('kind')
        for source_name, source in kind_report.get('sources', {}).items():
            all_subset = source.get('subsets', {}).get('all')
            if not all_subset:
                continue
            motion = all_subset.get('motion_analysis', {})
            same_step = motion.get('same_step_motion', {}) if motion else {}
            realized = same_step.get('realized_motion', {}) if same_step else {}
            target = same_step.get('target_motion', {}) if same_step else {}
            ratios = same_step.get('realized_vs_target_ratio', {}) if same_step else {}
            improvement = motion.get('future_k_improvement', {}) if motion else {}
            rows.append({
                'kind': kind,
                'source': source_name,
                'realized_pos_vel_mean': (realized.get('translation_velocity') or {}).get('mean'),
                'realized_pos_vel_p90': (realized.get('translation_velocity') or {}).get('p90'),
                'target_pos_vel_mean': (target.get('translation_velocity') or {}).get('mean'),
                'target_pos_vel_p90': (target.get('translation_velocity') or {}).get('p90'),
                'realized_vs_target_ratio_mean': (ratios.get('translation_velocity') or {}).get('mean'),
                'best_future_k': motion.get('best_k_by_motion'),
                'future_improvement_mean': (improvement.get('improvement') or {}).get('mean'),
            })
    return {'rows': rows}


    teleop_gap = report.get('teleop_action_gap')
    inference_gap = report.get('inference_action_gap')
    infer_chunk = report['inference_timeline']['delta_chunk_obs']['mean'] if report['inference_timeline']['delta_chunk_obs'] else None
    infer_time = report['inference_timeline']['inference_time']['mean'] if report['inference_timeline']['inference_time'] else None
    teleop_obs = report['teleop_timeline']['delta_obs']['mean'] if report['teleop_timeline']['delta_obs'] else None
    infer_obs = report['inference_timeline']['delta_obs']['mean'] if report['inference_timeline']['delta_obs'] else None

    conclusions = []
    if teleop_obs is not None and infer_obs is not None:
        conclusions.append(
            f'Observation freshness gap: teleop delta_obs≈{teleop_obs:.4f}s vs inference delta_obs≈{infer_obs:.4f}s'
        )
    if infer_time is not None and infer_chunk is not None:
        conclusions.append(
            f'Inference-side latency is dominated by model+chunk staging: inference_time≈{infer_time:.4f}s, delta_chunk_obs≈{infer_chunk:.4f}s'
        )

    if teleop_gap is not None and inference_gap is not None:
        teleop_pos = teleop_gap['pos_gap']['mean']
        infer_pos = inference_gap['pos_gap']['mean']
        conclusions.append(
            f'Legacy same-step behavior gap: inference position-gap mean≈{infer_pos:.4f} vs teleop≈{teleop_pos:.4f}'
        )
        if infer_pos > teleop_pos * 5:
            conclusions.append(
                'Inference actions are much farther from current state than teleop actions under the same-step absolute metric.'
            )

    inference_sources = report.get('inference_gap_analysis', {}).get('sources', {})
    conclusions.extend(_semantic_notes(report.get('inference_gap_analysis', {})))
    for source_name in ['model', 'executed', 'compat_action']:
        source = inference_sources.get(source_name)
        if source is not None:
            conclusions.extend(_source_summary_lines(f'Inference[{source_name}]', source))

    teleop_source = report.get('teleop_gap_analysis', {}).get('sources', {}).get('compat_action')
    if teleop_source is not None:
        conclusions.extend(_source_summary_lines('Teleop[compat_action]', teleop_source))

    return {'conclusions': conclusions}


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare teleop and inference gap for CARM data')
    parser.add_argument('--teleop_h5', required=True)
    parser.add_argument('--teleop_timeline', required=True)
    parser.add_argument('--inference_h5', required=True)
    parser.add_argument('--inference_timeline', required=True)
    parser.add_argument('--out', default='')
    parser.add_argument('--max_k', type=int, default=10, help='Maximum future offset k to scan for action-vs-state alignment')
    parser.add_argument('--include_relative', action='store_true', help='Include relative-pose metrics in the report')
    parser.add_argument('--include_intervention_subsets', action='store_true', help='Include intervention-free subset stats when available')
    parser.add_argument('--spot_check_samples', type=int, default=5, help='Number of per-source sample rows to dump for same-step and best-k views')
    args = parser.parse_args()

    teleop_gap_analysis = _gap_analysis(
        args.teleop_h5,
        kind='teleop',
        max_k=args.max_k,
        include_relative=args.include_relative,
        include_intervention_subsets=args.include_intervention_subsets,
        spot_check_samples=args.spot_check_samples,
    )
    inference_gap_analysis = _gap_analysis(
        args.inference_h5,
        kind='inference',
        max_k=args.max_k,
        include_relative=args.include_relative,
        include_intervention_subsets=args.include_intervention_subsets,
        spot_check_samples=args.spot_check_samples,
    )

    report = {
        'teleop_timeline': _timeline_stats(args.teleop_timeline, teleop=True),
        'inference_timeline': _timeline_stats(args.inference_timeline, teleop=False),
        'teleop_gap_analysis': teleop_gap_analysis,
        'inference_gap_analysis': inference_gap_analysis,
        'teleop_action_gap': _legacy_action_gap(teleop_gap_analysis, preferred_source='compat_action'),
        'inference_action_gap': _legacy_action_gap(inference_gap_analysis, preferred_source='compat_action'),
    }
    report['motion_summary'] = _motion_summary(report)
    report['summary'] = _summary(report)

    out = args.out or str(Path(args.inference_timeline).with_name(Path(args.inference_timeline).stem + '_gap_report.json'))
    Path(out).write_text(json.dumps(report, indent=2))
    print(f'Wrote gap report to {out}')
    print(json.dumps(report['summary'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
