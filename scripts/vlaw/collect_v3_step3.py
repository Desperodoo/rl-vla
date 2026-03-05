#!/usr/bin/env python3
"""VLAW v3 Step 3: 大规模采集 (mixed 1200 + eval 20) + high_suc 筛选.

Usage:
    CUDA_VISIBLE_DEVICES=4 conda run -n rlft_ms3 python scripts/vlaw/collect_v3_step3.py

遵循 VLAW_DATA_COLLECTION_PLAN_V3.md Step 3 规范。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------- Constants ----------
CHECKPOINT = "runs/fair_comparison/fair_comparison/awsc/best_s42__1772570560/checkpoints/final.pt"
TASK = "LiftPegUpright-v1"
FRAME_SKIP = 4
MAX_EPISODE_STEPS = 200
CAMERA_SIZE = 128
NUM_ENVS = 64
MIN_TRAJ_LENGTH = 5

MIXED_DIR = f"data/vlaw/rollouts/mixed/{TASK}"
EVAL_DIR = f"data/vlaw/rollouts/eval/{TASK}"
HIGH_SUC_DIR = f"data/vlaw/rollouts/high_suc/{TASK}"


def collect_dataset(
    name: str,
    num_episodes: int,
    output_dir: str,
    num_envs: int = NUM_ENVS,
) -> Path:
    """Run collection for a dataset."""
    from rlft.vlaw.data.collector import CollectorConfig, VLAWDataCollector

    print(f"\n{'='*60}")
    print(f"[v3-Step3] Collecting {name}: {num_episodes} episodes")
    print(f"  frame_skip={FRAME_SKIP} (MUST be 4, BUG-023)")
    print(f"  max_episode_steps={MAX_EPISODE_STEPS}")
    print(f"  checkpoint={CHECKPOINT}")
    print(f"  output_dir={output_dir}")
    print(f"{'='*60}")

    t0 = time.time()

    cfg = CollectorConfig(
        env_id=TASK,
        num_envs=num_envs,
        num_episodes=num_episodes,
        max_episode_steps=MAX_EPISODE_STEPS,
        camera_width=CAMERA_SIZE,
        camera_height=CAMERA_SIZE,
        checkpoint_path=str(project_root / CHECKPOINT),
        frame_skip=FRAME_SKIP,
        min_traj_length=MIN_TRAJ_LENGTH,
        gpu_id=0,  # After CUDA_VISIBLE_DEVICES=4, device 0 is physical GPU 4
        output_dir=str(project_root / output_dir),
        source_tag="real",
        task_instruction="Pick up the peg and lift it upright.",
        verbose=True,
        obs_horizon=2,
        act_steps=8,
    )

    # Double-check frame_skip
    assert cfg.frame_skip == 4, f"BUG-023: frame_skip={cfg.frame_skip}, expected 4!"

    collector = VLAWDataCollector(cfg)
    trajs = collector.collect_rollouts()
    elapsed = time.time() - t0

    # Stats
    n_total = len(trajs)
    n_success = sum(1 for t in trajs if bool(t["env_success"][-1]))
    sr = n_success / n_total * 100 if n_total > 0 else 0
    lengths = [t["actions"].shape[0] for t in trajs]

    print(f"\n[v3-Step3] {name} collection done:")
    print(f"  Trajectories: {n_total}")
    print(f"  Success (at_end): {n_success} ({sr:.1f}%)")
    print(f"  T: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")
    print(f"  Time: {elapsed:.1f}s")

    # Save HDF5
    h5_path = collector.save_hdf5(trajs)
    print(f"  HDF5: {h5_path}")

    # Also save stats JSON alongside
    stats = {
        "name": name,
        "num_trajectories": n_total,
        "success_at_end_count": n_success,
        "success_at_end_rate": sr / 100,
        "T_min": int(min(lengths)),
        "T_max": int(max(lengths)),
        "T_mean": float(np.mean(lengths)),
        "T_median": float(np.median(lengths)),
        "T_std": float(np.std(lengths)),
        "elapsed_sec": elapsed,
        "frame_skip": FRAME_SKIP,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "checkpoint": CHECKPOINT,
        "h5_path": str(h5_path),
    }
    stats_path = h5_path.parent / f"{name}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats_path}")

    return h5_path


def filter_high_suc(mixed_h5: Path, high_suc_dir: str) -> Path:
    """Filter success_at_end trajectories from mixed dataset."""
    print(f"\n{'='*60}")
    print(f"[v3-Step3] Filtering high_suc from mixed")
    print(f"{'='*60}")

    out_dir = project_root / high_suc_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5 = out_dir / mixed_h5.name.replace("mixed", "high_suc").replace("real", "high_suc_real")

    n_copied = 0
    n_total = 0

    with h5py.File(str(mixed_h5), "r") as src, h5py.File(str(out_h5), "w") as dst:
        # Copy meta
        if "meta" in src:
            src.copy("meta", dst)

        # Filter trajectories
        traj_keys = sorted([k for k in src.keys() if k.startswith("traj_")])
        n_total = len(traj_keys)

        for key in traj_keys:
            grp = src[key]
            success_arr = grp["env_success"][:]
            if bool(success_arr[-1]):  # success_at_end
                new_key = f"traj_{n_copied:04d}"
                src.copy(key, dst, name=new_key)
                n_copied += 1

        # Update meta
        if "meta" in dst:
            dst["meta"].attrs["num_trajectories"] = n_copied
            dst["meta"].attrs["success_rate"] = 1.0
            dst["meta"].attrs["source"] = "high_suc_filtered"

    # Compute stats for high_suc
    lengths = []
    with h5py.File(str(out_h5), "r") as f:
        for key in sorted(k for k in f.keys() if k.startswith("traj_")):
            T = f[key]["actions"].shape[0]
            lengths.append(T)

    print(f"  Total in mixed: {n_total}")
    print(f"  Filtered (success_at_end): {n_copied}")
    print(f"  Filter rate: {n_copied/n_total*100:.1f}%")
    if lengths:
        print(f"  T: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")
    print(f"  Output: {out_h5}")

    # Save stats
    stats = {
        "name": "high_suc",
        "num_trajectories": n_copied,
        "source_total": n_total,
        "filter_rate": n_copied / n_total if n_total > 0 else 0,
        "T_min": int(min(lengths)) if lengths else 0,
        "T_max": int(max(lengths)) if lengths else 0,
        "T_mean": float(np.mean(lengths)) if lengths else 0,
        "T_median": float(np.median(lengths)) if lengths else 0,
        "h5_path": str(out_h5),
    }
    stats_path = out_dir / "high_suc_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return out_h5


def main() -> None:
    print(f"[v3-Step3] Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[v3-Step3] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"[v3-Step3] frame_skip={FRAME_SKIP} (MUST be 4)")

    # Step 3a: Collect MIXED (1200 episodes)
    mixed_h5 = collect_dataset("mixed", 1200, MIXED_DIR)

    # Step 3b: Collect EVAL (20 episodes)
    eval_h5 = collect_dataset("eval", 20, EVAL_DIR, num_envs=20)

    # Step 3c: Filter high_suc from mixed
    high_suc_h5 = filter_high_suc(mixed_h5, HIGH_SUC_DIR)

    print(f"\n{'='*60}")
    print(f"[v3-Step3] ALL DONE at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  mixed:    {mixed_h5}")
    print(f"  eval:     {eval_h5}")
    print(f"  high_suc: {high_suc_h5}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
