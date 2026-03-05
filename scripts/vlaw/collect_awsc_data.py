"""Collect rollout data using AWSC fine-tuned model.

Step 3 of the AWSC data pipeline:
- mixed: 1200 trajectories for training
- eval: 20 trajectories for evaluation
- high_suc: filtered success_at_end trajectories

Usage:
    CUDA_VISIBLE_DEVICES=5 conda run -n rlft_ms3 python scripts/vlaw/collect_awsc_data.py \
        --checkpoint_path runs/vlaw_awsc_lr7e5__1772629860/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from rlft.vlaw.data.collector import CollectorConfig, VLAWDataCollector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to AWSC best.pt checkpoint")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU index (after CUDA_VISIBLE_DEVICES filtering)")
    parser.add_argument("--skip_mixed", action="store_true",
                        help="Skip mixed collection (for debugging)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip eval collection")
    args = parser.parse_args()

    ckpt = args.checkpoint_path
    assert Path(ckpt).exists(), f"Checkpoint not found: {ckpt}"

    task = "LiftPegUpright-v1"
    base_dir = "data/vlaw/rollouts_awsc"

    # ---- Step 3a: Collect MIXED (1200 episodes) ----
    if not args.skip_mixed:
        print("=" * 60)
        print("[Step 3a] Collecting MIXED data: 1200 episodes")
        print("=" * 60)
        t0 = time.time()

        cfg_mixed = CollectorConfig(
            env_id=task,
            num_envs=64,
            num_episodes=1200,
            max_episode_steps=100,
            camera_width=128,
            camera_height=128,
            checkpoint_path=ckpt,
            frame_skip=3,
            min_traj_length=10,
            gpu_id=args.gpu_id,
            output_dir=f"{base_dir}/mixed/{task}",
            source_tag="real",
            task_instruction="Pick up the peg and lift it upright.",
            verbose=True,
        )

        collector = VLAWDataCollector(cfg_mixed)
        trajs = collector.collect_rollouts()
        elapsed = time.time() - t0

        # Stats
        n_total = len(trajs)
        n_success = sum(1 for t in trajs if t.get("env_success", False))
        success_rate = n_success / n_total * 100 if n_total > 0 else 0
        avg_T = sum(t["actions"].shape[0] for t in trajs) / n_total if n_total else 0

        print(f"\n[Step 3a] Mixed collection done:")
        print(f"  Trajectories: {n_total}")
        print(f"  Success (at_end): {n_success} ({success_rate:.1f}%)")
        print(f"  Avg length: {avg_T:.1f}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Output: {base_dir}/mixed/{task}/")

    # ---- Step 3b: Collect EVAL (20 episodes) ----
    if not args.skip_eval:
        print("\n" + "=" * 60)
        print("[Step 3b] Collecting EVAL data: 20 episodes")
        print("=" * 60)
        t0 = time.time()

        cfg_eval = CollectorConfig(
            env_id=task,
            num_envs=20,
            num_episodes=20,
            max_episode_steps=100,
            camera_width=128,
            camera_height=128,
            checkpoint_path=ckpt,
            frame_skip=3,
            min_traj_length=10,
            gpu_id=args.gpu_id,
            output_dir=f"{base_dir}/eval/{task}",
            source_tag="real",
            task_instruction="Pick up the peg and lift it upright.",
            verbose=True,
        )

        collector = VLAWDataCollector(cfg_eval)
        trajs = collector.collect_rollouts()
        elapsed = time.time() - t0

        n_total = len(trajs)
        n_success = sum(1 for t in trajs if t.get("env_success", False))
        success_rate = n_success / n_total * 100 if n_total > 0 else 0

        print(f"\n[Step 3b] Eval collection done:")
        print(f"  Trajectories: {n_total}")
        print(f"  Success (at_end): {n_success} ({success_rate:.1f}%)")
        print(f"  Time: {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("ALL COLLECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
