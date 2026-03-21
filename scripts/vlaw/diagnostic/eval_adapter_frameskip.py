#!/usr/bin/env python3
"""Phase 0.2: Evaluate frame_skip impact on Dynamics Adapter accuracy.

Compares adapter V1 EE prediction accuracy on:
  1. mixed data (frame_skip=4) — matches WM training
  2. adapter_* data (frame_skip=3) — mismatched

No WM needed — purely evaluates adapter prediction quality.

Usage:
    conda run -n rlft_ms3 python scripts/vlaw/diagnostic/eval_adapter_frameskip.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from glob import glob
import numpy as np
import h5py

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "ctrl_world"))
sys.path.insert(0, str(_ROOT))

from ctrl_world.dataset.dataset_maniskill import state_to_ee_pose_7d


def load_chunks(h5_paths: list[str], K: int = 5) -> tuple[np.ndarray, ...]:
    """Load (state, actions, gt_ee) chunks from HDF5 files."""
    all_states, all_actions, all_targets = [], [], []
    total_traj = 0
    for h5_file in h5_paths:
        with h5py.File(h5_file, "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj_")]
            for key in traj_keys:
                grp = f[key]
                if "state" not in grp or "actions" not in grp:
                    continue
                state_arr = grp["state"][:].astype(np.float32)
                act_arr = grp["actions"][:].astype(np.float32)
                T = min(state_arr.shape[0], act_arr.shape[0])
                if T <= K:
                    continue
                for t in range(T - K):
                    all_states.append(state_arr[t])
                    all_actions.append(act_arr[t : t + K])
                    future_states = state_arr[t + 1 : t + K + 1]
                    all_targets.append(state_to_ee_pose_7d(future_states))
                total_traj += 1
    states = np.stack(all_states, axis=0)
    actions = np.stack(all_actions, axis=0)
    targets = np.stack(all_targets, axis=0)
    return states, actions, targets, total_traj


def eval_adapter(adapter, norm: dict, states: np.ndarray, actions: np.ndarray,
                 targets: np.ndarray, K: int = 5) -> dict:
    """Evaluate adapter on pre-loaded chunks. Returns per-step + aggregate metrics."""
    N = len(states)
    # Normalize states
    states_n = (states - norm["state_mean"]) / norm["state_std"]

    per_step_pos = [[] for _ in range(K)]
    per_step_euler = [[] for _ in range(K)]

    for i in range(N):
        pred = adapter.predict(states_n[i], actions[i])  # (K, 7)
        for step_i in range(K):
            pe = np.linalg.norm(pred[step_i, :3] - targets[i, step_i, :3])
            ee = np.abs(pred[step_i, 3:6] - targets[i, step_i, 3:6]).mean()
            per_step_pos[step_i].append(pe)
            per_step_euler[step_i].append(ee)

    result = {"n_samples": N, "per_step": {}}
    all_pos, all_euler = [], []
    for step_i in range(K):
        pm = float(np.mean(per_step_pos[step_i]) * 1000)
        em = float(np.mean(per_step_euler[step_i]))
        result["per_step"][f"step_{step_i+1}"] = {
            "pos_mae_mm": pm, "euler_mae_rad": em
        }
        all_pos.extend(per_step_pos[step_i])
        all_euler.extend(per_step_euler[step_i])
    result["aggregate"] = {
        "pos_mae_mm": float(np.mean(all_pos) * 1000),
        "euler_mae_rad": float(np.mean(all_euler)),
    }
    return result


def main():
    import json
    K = 5
    adapter_ckpt = str(_ROOT / "checkpoints/vlaw/dynamics_adapter/best.pt")

    # Load adapter
    from rlft.vlaw.world_model.dynamics_adapter import DynamicsAdapterTrainer
    adapter, norm = DynamicsAdapterTrainer.load_from_checkpoint(adapter_ckpt, device="cpu")
    print(f"[Adapter] Loaded V1 from {adapter_ckpt}")

    # Define data sources
    data_sources = {
        "mixed (fs=4)": glob(str(_ROOT / "data/vlaw/rollouts/mixed/LiftPegUpright-v1/*.h5")),
        "adapter_clean (fs=3)": glob(str(_ROOT / "data/vlaw/rollouts/adapter_clean/*.h5")),
        "adapter_teleop (fs=3)": glob(str(_ROOT / "data/vlaw/rollouts/adapter_teleop/*.h5")),
        "adapter_gaussian (fs=3)": glob(str(_ROOT / "data/vlaw/rollouts/adapter_gaussian/*.h5")),
        "adapter_random (fs=3)": glob(str(_ROOT / "data/vlaw/rollouts/adapter_random/*.h5")),
    }

    all_results = {}
    for name, h5_paths in data_sources.items():
        if not h5_paths:
            print(f"[SKIP] {name}: no H5 files found")
            continue
        print(f"\n[Eval] {name} ({len(h5_paths)} files)")
        states, actions, targets, n_traj = load_chunks(h5_paths, K)
        print(f"  Loaded {n_traj} trajs → {len(states)} chunks")
        result = eval_adapter(adapter, norm, states, actions, targets, K)
        all_results[name] = result

    # --- Summary report ---
    print("\n" + "=" * 80)
    print("Frame-Skip Impact on Adapter V1 Accuracy")
    print("=" * 80)
    print(f"\n{'Source':<30}  {'N chunks':>10}  {'pos_mae (mm)':>14}  {'euler_mae (rad)':>16}")
    print(f"{'-'*30}  {'-'*10}  {'-'*14}  {'-'*16}")
    for name, r in all_results.items():
        n = r["n_samples"]
        pm = r["aggregate"]["pos_mae_mm"]
        em = r["aggregate"]["euler_mae_rad"]
        print(f"{name:<30}  {n:>10}  {pm:>14.2f}  {em:>16.4f}")

    print("\nPer-Step Position MAE (mm):")
    print(f"{'Source':<30}", end="")
    for i in range(K):
        print(f"  {'S'+str(i+1):>8}", end="")
    print()
    print(f"{'-'*30}", end="")
    for _ in range(K):
        print(f"  {'-'*8}", end="")
    print()
    for name, r in all_results.items():
        print(f"{name:<30}", end="")
        for i in range(K):
            pm = r["per_step"][f"step_{i+1}"]["pos_mae_mm"]
            print(f"  {pm:>8.2f}", end="")
        print()

    print("\nPer-Step Euler MAE (rad):")
    print(f"{'Source':<30}", end="")
    for i in range(K):
        print(f"  {'S'+str(i+1):>8}", end="")
    print()
    print(f"{'-'*30}", end="")
    for _ in range(K):
        print(f"  {'-'*8}", end="")
    print()
    for name, r in all_results.items():
        print(f"{name:<30}", end="")
        for i in range(K):
            em = r["per_step"][f"step_{i+1}"]["euler_mae_rad"]
            print(f"  {em:>8.4f}", end="")
        print()
    print("=" * 80)

    # Save
    out_dir = _ROOT / "results" / "vlaw" / "adapter_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "frameskip_impact.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
