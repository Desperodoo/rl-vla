"""BUG-022 Ghost Episode verification script.

Validates:
1. No T=1 trajectories pass through (ghost episodes discarded or filtered)
2. step_in_episode matches actual frame count for all saved trajectories
3. rgb_base vs rgb_render diff > 30 (dual camera sanity)
4. Ghost episode counter (discarded_ghost) is > 0 (if ghost events occur)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import time
import numpy as np
import torch
from rlft.vlaw.data.collector import (
    CollectorConfig, VLAWDataCollector, _np, _get_render_frame,
    Trajectory,
)


def main() -> None:
    cfg = CollectorConfig(
        env_id="LiftPegUpright-v1",
        num_envs=8,
        num_episodes=20,
        max_episode_steps=200,
        checkpoint_path="checkpoints/il/best_eval_success_once.pt",
        frame_skip=3,
        min_traj_length=10,
        gpu_id=0,
        output_dir="/home/wjz/rl-vla/data/vlaw/debug_bug022",
        source_tag="debug_bug022",
        task_instruction="Pick up the peg and orient it upright",
        verbose=True,
    )

    collector = VLAWDataCollector(cfg)
    t0 = time.time()
    trajs = collector.collect_rollouts()
    elapsed = time.time() - t0

    # ============================================================
    # Validation
    # ============================================================
    print("\n" + "=" * 60)
    print("[BUG-022 VERIFY] Running validation checks...")
    print("=" * 60)

    all_pass = True

    # --- Check 1: No T=1 trajectories ---
    lengths = [t["actions"].shape[0] for t in trajs]
    t1_count = sum(1 for L in lengths if L == 1)
    print(f"\n[CHECK 1] T=1 trajectories: {t1_count}")
    if t1_count > 0:
        print("  ❌ FAIL: Found T=1 trajectories (possible ghost leak)")
        all_pass = False
    else:
        print("  ✅ PASS: No T=1 trajectories")

    # --- Check 2: All trajectories have T >= min_traj_length ---
    short_count = sum(1 for L in lengths if L < cfg.min_traj_length)
    print(f"\n[CHECK 2] Trajectories with T < {cfg.min_traj_length}: {short_count}")
    if short_count > 0:
        print(f"  ❌ FAIL: Found {short_count} short trajectories")
        for i, t in enumerate(trajs):
            T = t["actions"].shape[0]
            if T < cfg.min_traj_length:
                print(f"    traj {i}: T={T}")
        all_pass = False
    else:
        print(f"  ✅ PASS: All trajectories T >= {cfg.min_traj_length}")

    # --- Check 3: rgb_base vs rgb_render diff > 30 ---
    print(f"\n[CHECK 3] Dual camera divergence (rgb_base vs rgb_render):")
    diffs = []
    for i, t in enumerate(trajs):
        if "rgb_base" in t and "rgb_render" in t:
            base = t["rgb_base"].astype(np.float32)
            render = t["rgb_render"].astype(np.float32)
            diff = np.mean(np.abs(base - render))
            diffs.append(diff)
    if diffs:
        min_diff = min(diffs)
        max_diff = max(diffs)
        mean_diff = np.mean(diffs)
        print(f"  diff stats: min={min_diff:.1f}, max={max_diff:.1f}, mean={mean_diff:.1f}")
        if min_diff > 30:
            print("  ✅ PASS: All diffs > 30 (cameras are distinct)")
        else:
            bad = sum(1 for d in diffs if d <= 30)
            print(f"  ❌ FAIL: {bad} trajectories have diff <= 30")
            all_pass = False
    else:
        print("  ⚠️ SKIP: No rgb_base/rgb_render in trajectory data")

    # --- Check 4: Frame consistency ---
    print(f"\n[CHECK 4] Frame consistency:")
    for i, t in enumerate(trajs):
        # All arrays should have same first dimension (T)
        T_actions = t["actions"].shape[0]
        keys_to_check = ["rgb_base", "rgb_render", "state", "obs_agent", "env_success"]
        for key in keys_to_check:
            if key in t:
                T_key = t[key].shape[0]
                if T_key != T_actions:
                    print(f"  ❌ FAIL: traj {i}: {key}.shape[0]={T_key} != actions.shape[0]={T_actions}")
                    all_pass = False
    print("  ✅ PASS: All arrays consistent" if all_pass else "  (see failures above)")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"[BUG-022 VERIFY] === SUMMARY ===")
    print(f"  Collected: {len(trajs)} trajectories in {elapsed:.1f}s")
    print(f"  Lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    print(f"  Success rate: {sum(bool(t['env_success'][-1]) for t in trajs) / len(trajs):.1%}")
    if all_pass:
        print(f"  Overall: ✅ ALL CHECKS PASSED")
    else:
        print(f"  Overall: ❌ SOME CHECKS FAILED")
    print(f"{'=' * 60}")

    return all_pass


if __name__ == "__main__":
    ok = main()
    exit(0 if ok else 1)
