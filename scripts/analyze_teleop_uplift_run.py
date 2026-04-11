#!/usr/bin/env python3
"""Quick analyzer for teleop uplift recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _safe_percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _safe_max(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.max(values))


def _safe_min(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.min(values))


def _timeline_stats(path: Path) -> dict:
    record_steps = 0
    applied_true = 0
    stale_count = 0
    active_steps = 0
    inactive_stale_count = 0
    timeout_stale_count = 0
    errors = []
    if not path.exists():
        return {
            "exists": False,
            "record_steps": 0,
            "applied_true": 0,
            "stale_count": 0,
            "active_steps": 0,
            "errors": [],
        }

    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("event") == "record_step":
                record_steps += 1
                applied_true += int(bool(obj.get("teleop_candidate_applied")))
                stale = bool(obj.get("teleop_candidate_stale"))
                active = bool(obj.get("teleop_active"))
                stale_count += int(stale)
                active_steps += int(active)
                inactive_stale_count += int(stale and (not active))
                timeout_stale_count += int(stale and active)
            elif obj.get("event") == "teleop_candidate_stale":
                errors.append("teleop_candidate_stale_event")
            elif obj.get("event") == "teleop_bridge_error":
                errors.append(obj.get("error", "teleop_bridge_error"))

    return {
        "exists": True,
        "record_steps": record_steps,
        "applied_true": applied_true,
        "stale_count": stale_count,
        "active_steps": active_steps,
        "inactive_stale_count": inactive_stale_count,
        "timeout_stale_count": timeout_stale_count,
        "errors": errors,
    }


def _describe_bool_ratio(name: str, values: np.ndarray) -> str:
    values = np.asarray(values).astype(bool)
    return f"{name}: true={int(values.sum())}/{len(values)} ({values.mean():.1%})"


def _describe_percentiles(name: str, values: np.ndarray) -> Iterable[str]:
    finite = np.asarray(values)[np.isfinite(values)]
    yield f"{name}: min={_fmt_float(_safe_min(finite))} p50={_fmt_float(_safe_percentile(finite, 50))} p95={_fmt_float(_safe_percentile(finite, 95))} max={_fmt_float(_safe_max(finite))}"
    nan_count = int(np.size(values) - finite.size)
    if nan_count:
        yield f"{name}: nan_count={nan_count}"


def analyze_hdf5(path: Path) -> list[str]:
    lines: list[str] = []
    with h5py.File(path, "r") as f:
        attrs = f.attrs
        lines.append(f"path: {path}")
        lines.append(f"teleop_bridge_mode: {attrs.get('teleop_bridge_mode', 'n/a')}")
        upper_control_enabled_attr = attrs.get("upper_control_enabled_at_start", "n/a")
        control_owner_attr = attrs.get("control_owner_at_start", "n/a")
        lines.append(f"upper_control_enabled_at_start: {upper_control_enabled_attr}")
        lines.append(f"control_owner_at_start: {control_owner_attr}")
        lines.append(f"num_steps: {attrs.get('num_steps', len(f['teleop_active']) if 'teleop_active' in f else 'n/a')}")

        if "teleop_active" in f:
            active = np.array(f["teleop_active"]).astype(bool)
            lines.append(_describe_bool_ratio("teleop_active", active))
        else:
            active = np.array([], dtype=bool)

        if "teleop_candidate_applied" in f:
            applied = np.array(f["teleop_candidate_applied"]).astype(bool)
            lines.append(_describe_bool_ratio("teleop_candidate_applied", applied))
        else:
            applied = np.array([], dtype=bool)

        if "teleop_candidate_stale" in f:
            stale = np.array(f["teleop_candidate_stale"]).astype(bool)
            lines.append(_describe_bool_ratio("teleop_candidate_stale", stale))
        else:
            stale = np.array([], dtype=bool)

        if stale.size and active.size:
            inactive_stale = stale & (~active)
            timeout_stale = stale & active
            lines.append(_describe_bool_ratio("inactive_stale", inactive_stale))
            lines.append(_describe_bool_ratio("timeout_stale", timeout_stale))

        for key in [
            "teleop_signal_age_ms",
            "teleop_candidate_loop_dt_ms",
            "upper_candidate_pos_error",
            "upper_candidate_rot_error",
            "teleop_abs_reconstruction_pos_error",
            "teleop_abs_reconstruction_rot_error",
        ]:
            if key in f:
                lines.extend(_describe_percentiles(key, np.array(f[key])))

        if "teleop_processed_sequence" in f:
            seq = np.array(f["teleop_processed_sequence"])
            lines.append(f"teleop_processed_sequence: min={int(seq.min())} max={int(seq.max())}")

        warnings = []
        if applied.size and attrs.get("teleop_bridge_mode") == "upper_control":
            live_enabled = attrs.get("upper_control_enabled_at_start", None)
            has_new_owner_metadata = "control_owner_at_start" in attrs
            if bool(live_enabled) and int(applied.sum()) == 0:
                warnings.append("live upper_control run recorded zero applied commands")
            if has_new_owner_metadata and (not bool(live_enabled)) and int(applied.sum()) > 0:
                warnings.append("candidate-only run unexpectedly recorded applied commands")
            elif (not has_new_owner_metadata) and (not bool(live_enabled)) and int(applied.sum()) > 0:
                warnings.append("legacy metadata mismatch: applied commands detected but upper_control_enabled_at_start=false")
        if stale.size and stale.mean() > 0.10:
            warnings.append("stale ratio exceeds 10%")
        if "teleop_candidate_loop_dt_ms" in f:
            loop_dt = np.array(f["teleop_candidate_loop_dt_ms"])
            p95 = _safe_percentile(loop_dt[np.isfinite(loop_dt)], 95)
            if p95 is not None and p95 > 35.0:
                warnings.append("candidate loop dt p95 exceeds 35ms")
        if "upper_candidate_rot_error" in f:
            rot = np.array(f["upper_candidate_rot_error"])
            nan_count = int(np.count_nonzero(~np.isfinite(rot)))
            if nan_count:
                warnings.append(f"upper_candidate_rot_error contains {nan_count} NaN/inf samples")

        if warnings:
            lines.append("warnings:")
            lines.extend(f"  - {warning}" for warning in warnings)
        else:
            lines.append("warnings: none")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a teleop uplift recording")
    parser.add_argument("--h5", required=True, help="Path to teleop uplift HDF5 file")
    parser.add_argument("--timeline", default="", help="Optional timeline JSONL path")
    args = parser.parse_args()

    h5_path = Path(args.h5).expanduser().resolve()
    lines = analyze_hdf5(h5_path)

    timeline_path = Path(args.timeline).expanduser().resolve() if args.timeline else None
    if timeline_path is not None:
        stats = _timeline_stats(timeline_path)
        lines.append(f"timeline_path: {timeline_path}")
        if stats["exists"]:
            lines.append(
                "timeline: "
                f"record_steps={stats['record_steps']} "
                f"active_steps={stats['active_steps']} "
                f"applied_true={stats['applied_true']} "
                f"stale_count={stats['stale_count']} "
                f"inactive_stale={stats['inactive_stale_count']} "
                f"timeout_stale={stats['timeout_stale_count']}"
            )
            if stats["errors"]:
                lines.append("timeline_errors:")
                lines.extend(f"  - {err}" for err in stats["errors"])
        else:
            lines.append("timeline: missing")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
