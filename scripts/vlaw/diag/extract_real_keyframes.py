#!/usr/bin/env python3
"""T-DIAG-SYN-002: Extract keyframes from real demo/rollout trajectories.

Picks 5 success + 5 failure trajectories from real data,
extracts keyframes at t=0, T/4, T/2, 3T/4, T, saves individual PNGs
and a strip image (2 rows: base_cam, render_cam).

Usage:
    python scripts/vlaw/diag/extract_real_keyframes.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from PIL import Image


def get_keyframe_indices(T: int) -> List[int]:
    """Return indices for t=0, T/4, T/2, 3T/4, T (last frame)."""
    if T <= 1:
        return [0]
    indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    # deduplicate while preserving order
    seen = set()
    result = []
    for i in indices:
        i = min(i, T - 1)
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result


def scan_trajectories(
    data_dirs: List[str], task: str = "LiftPegUpright-v1"
) -> Tuple[List[dict], List[dict]]:
    """Scan HDF5 files and classify trajectories into success/failure.

    Returns:
        (success_list, failure_list) where each entry is:
            {"file": path, "traj_key": str, "length": int, "success": bool}
    """
    success_list: List[dict] = []
    failure_list: List[dict] = []

    for data_dir in data_dirs:
        task_dir = os.path.join(data_dir, task)
        if not os.path.isdir(task_dir):
            # Maybe files are directly in data_dir
            task_dir = data_dir
        h5_files = sorted(
            [f for f in os.listdir(task_dir) if f.endswith(".h5")]
        )
        for h5_file in h5_files:
            h5_path = os.path.join(task_dir, h5_file)
            with h5py.File(h5_path, "r") as hf:
                traj_keys = sorted(
                    [k for k in hf.keys() if k.startswith("traj_")]
                )
                for tk in traj_keys:
                    grp = hf[tk]
                    env_success = grp["env_success"][:]
                    length = len(env_success)
                    is_success = bool(env_success[-1])
                    entry = {
                        "file": h5_path,
                        "traj_key": tk,
                        "length": length,
                        "success": is_success,
                        "source_dir": data_dir,
                    }
                    if is_success:
                        success_list.append(entry)
                    else:
                        failure_list.append(entry)

    return success_list, failure_list


def extract_keyframes(
    entry: dict,
    output_dir: str,
    traj_idx: int,
) -> dict:
    """Extract keyframes for one trajectory, save PNGs and strip image.

    Returns info dict about what was saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(entry["file"], "r") as hf:
        grp = hf[entry["traj_key"]]
        rgb_base = grp["rgb_base"][:]  # (T, 192, 192, 3) uint8
        rgb_render = grp["rgb_render"][:]  # (T, 192, 192, 3) uint8
        env_success = grp["env_success"][:]

    T = len(rgb_base)
    keyframe_indices = get_keyframe_indices(T)

    saved_files = []
    base_frames = []
    render_frames = []

    for t_idx in keyframe_indices:
        # Save individual PNGs
        for cam_name, rgb_array in [("base", rgb_base), ("render", rgb_render)]:
            img = Image.fromarray(rgb_array[t_idx])
            fname = f"traj_{traj_idx:03d}_t{t_idx:04d}_{cam_name}.png"
            fpath = os.path.join(output_dir, fname)
            img.save(fpath)
            saved_files.append(fname)

        base_frames.append(rgb_base[t_idx])
        render_frames.append(rgb_render[t_idx])

    # Create strip image: 2 rows (base on top, render on bottom), N columns
    n_frames = len(keyframe_indices)
    H, W = rgb_base.shape[1], rgb_base.shape[2]
    strip = np.zeros((2 * H, n_frames * W, 3), dtype=np.uint8)
    for i, (bf, rf) in enumerate(zip(base_frames, render_frames)):
        strip[0:H, i * W : (i + 1) * W, :] = bf
        strip[H : 2 * H, i * W : (i + 1) * W, :] = rf

    strip_fname = f"traj_{traj_idx:03d}_strip.png"
    strip_path = os.path.join(output_dir, strip_fname)
    Image.fromarray(strip).save(strip_path)
    saved_files.append(strip_fname)

    return {
        "traj_idx": traj_idx,
        "source_file": os.path.basename(entry["file"]),
        "traj_key": entry["traj_key"],
        "length": T,
        "success": entry["success"],
        "keyframe_indices": keyframe_indices,
        "num_files_saved": len(saved_files),
        "strip": strip_fname,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract real trajectory keyframes for diagnosis"
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=[
            "data/vlaw/rollouts/iter1_highsuc",
            "data/vlaw/rollouts/iter1",
        ],
        help="Directories containing HDF5 rollout data",
    )
    parser.add_argument(
        "--task", default="LiftPegUpright-v1", help="Task name"
    )
    parser.add_argument(
        "--output-base",
        default="results/vlaw/dsyn_diagnosis_frames",
        help="Base output directory",
    )
    parser.add_argument(
        "--num-success", type=int, default=5, help="Number of success trajs"
    )
    parser.add_argument(
        "--num-failure", type=int, default=5, help="Number of failure trajs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Minimum trajectory length to consider",
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    print(f"[T-DIAG-SYN-002] Scanning data directories: {args.data_dirs}")
    success_list, failure_list = scan_trajectories(args.data_dirs, args.task)
    print(
        f"  Found {len(success_list)} success, {len(failure_list)} failure trajectories"
    )

    # Filter by minimum length
    if args.min_length > 1:
        success_list = [e for e in success_list if e["length"] >= args.min_length]
        failure_list = [e for e in failure_list if e["length"] >= args.min_length]
        print(
            f"  After min_length={args.min_length} filter: "
            f"{len(success_list)} success, {len(failure_list)} failure"
        )

    # Sample
    if len(success_list) < args.num_success:
        print(
            f"  WARNING: Only {len(success_list)} success trajs available "
            f"(requested {args.num_success})"
        )
        selected_success = success_list
    else:
        idx = rng.choice(len(success_list), args.num_success, replace=False)
        selected_success = [success_list[i] for i in sorted(idx)]

    if len(failure_list) < args.num_failure:
        print(
            f"  WARNING: Only {len(failure_list)} failure trajs available "
            f"(requested {args.num_failure})"
        )
        selected_failure = failure_list
    else:
        idx = rng.choice(len(failure_list), args.num_failure, replace=False)
        selected_failure = [failure_list[i] for i in sorted(idx)]

    # Extract keyframes
    success_dir = os.path.join(args.output_base, "real_success")
    failure_dir = os.path.join(args.output_base, "real_failure")

    print(f"\n--- Extracting {len(selected_success)} success trajectories ---")
    success_results = []
    for i, entry in enumerate(selected_success):
        info = extract_keyframes(entry, success_dir, traj_idx=i)
        success_results.append(info)
        print(
            f"  [SUCCESS] traj_{i:03d}: {entry['traj_key']} from "
            f"{os.path.basename(entry['file'])}, T={info['length']}, "
            f"keyframes={info['keyframe_indices']}, "
            f"files={info['num_files_saved']}"
        )

    print(f"\n--- Extracting {len(selected_failure)} failure trajectories ---")
    failure_results = []
    for i, entry in enumerate(selected_failure):
        info = extract_keyframes(entry, failure_dir, traj_idx=i)
        failure_results.append(info)
        print(
            f"  [FAILURE] traj_{i:03d}: {entry['traj_key']} from "
            f"{os.path.basename(entry['file'])}, T={info['length']}, "
            f"keyframes={info['keyframe_indices']}, "
            f"files={info['num_files_saved']}"
        )

    # Summary
    total_success_files = sum(r["num_files_saved"] for r in success_results)
    total_failure_files = sum(r["num_files_saved"] for r in failure_results)
    print(f"\n=== Summary ===")
    print(f"  Success: {len(success_results)} trajs, {total_success_files} files → {success_dir}")
    print(f"  Failure: {len(failure_results)} trajs, {total_failure_files} files → {failure_dir}")
    print(f"  Total PNG files: {total_success_files + total_failure_files}")

    # Write manifest
    manifest_path = os.path.join(args.output_base, "real_keyframes_manifest.txt")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        f.write(f"# T-DIAG-SYN-002: Real Trajectory Keyframes Manifest\n")
        f.write(f"# Task: {args.task}\n")
        f.write(f"# Seed: {args.seed}\n\n")
        f.write(f"## Success ({len(success_results)} trajs)\n")
        for r in success_results:
            f.write(
                f"traj_{r['traj_idx']:03d}: {r['source_file']}/{r['traj_key']}, "
                f"T={r['length']}, keyframes={r['keyframe_indices']}\n"
            )
        f.write(f"\n## Failure ({len(failure_results)} trajs)\n")
        for r in failure_results:
            f.write(
                f"traj_{r['traj_idx']:03d}: {r['source_file']}/{r['traj_key']}, "
                f"T={r['length']}, keyframes={r['keyframe_indices']}\n"
            )
    print(f"  Manifest: {manifest_path}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
