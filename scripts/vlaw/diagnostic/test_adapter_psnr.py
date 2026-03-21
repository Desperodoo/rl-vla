#!/usr/bin/env python3
"""Test Dynamics Adapter effect on WM PSNR.

Compares 3 EE pose variants:
  1. GT: Ground-truth future EE poses from state[t+1:t+K+1]
  2. Tiled: Tile current EE pose (baseline, = BUG-D)
  3. Adapter: Dynamics Adapter predicted future EE poses

Usage:
    CUDA_VISIBLE_DEVICES=8 conda run -n ctrl_world python scripts/vlaw/diagnostic/test_adapter_psnr.py \
        --data_h5 "data/vlaw/rollouts/mixed/LiftPegUpright-v1/*.h5" \
        --adapter_ckpt checkpoints/vlaw/dynamics_adapter/best.pt \
        --wm_checkpoint checkpoints/vlaw/world_model/iter1_v5/checkpoint-5000 \
        --num_samples 30  --gpu_id 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from glob import glob
import numpy as np
import torch
import h5py
from tqdm import tqdm

# Ensure paths
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "ctrl_world"))
sys.path.insert(0, str(_ROOT))

from ctrl_world.config import wm_args_maniskill
from ctrl_world.dataset.dataset_maniskill import state_to_ee_pose_7d


# Metrics
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between pred and target latents."""
    pred, target = pred.float().cpu(), target.float().cpu()
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 50.0
    data_range = target.max().item() - target.min().item()
    return 10 * np.log10(data_range ** 2 / mse)


class WMWrapper:
    """Minimal wrapper for CtrlWorldAdapter."""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        from rlft.vlaw.world_model.ctrl_world_adapter import CtrlWorldAdapter

        args = wm_args_maniskill()
        args.ckpt_path = ckpt_path

        # Fix relative paths
        root = str(_ROOT)
        for attr in ["svd_model_path", "clip_model_path", "ckpt_path",
                     "data_stat_path", "dataset_root_path", "val_dataset_dir",
                     "dataset_meta_info_path", "output_dir"]:
            val = getattr(args, attr, None)
            if val and val.startswith("../"):
                setattr(args, attr, os.path.join(root, val[3:]))

        self.adapter = CtrlWorldAdapter(args, ckpt_path=ckpt_path, device=device)
        self.num_history = args.num_history
        self.num_frames = args.num_frames

    def rollout(self, obs_latents, ee_poses, instruction="Lift the peg upright"):
        """obs_latents: (window, 4, 48, 24), ee_poses: (window, 7). Returns (2, T, 4, 24, 24)."""
        return self.adapter.rollout(obs_latents, ee_poses, instruction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_h5", type=str, nargs="+", required=True)
    parser.add_argument("--adapter_ckpt", type=str, required=True)
    parser.add_argument("--wm_checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--act_steps", type=int, default=5)
    parser.add_argument("--num_history", type=int, default=6)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    K = args.act_steps
    num_history = args.num_history

    # --- Load Dynamics Adapter ---
    from rlft.vlaw.world_model.dynamics_adapter import DynamicsAdapterTrainer, SingleStepDynamicsAdapter
    adapter, norm = DynamicsAdapterTrainer.load_from_checkpoint(args.adapter_ckpt, device=str(device))
    is_single_step = isinstance(adapter, SingleStepDynamicsAdapter)
    print(f"[Adapter] Loaded from {args.adapter_ckpt}")

    # --- Load World Model ---
    wm = WMWrapper(args.wm_checkpoint, device=str(device))
    print(f"[WM] Loaded from {args.wm_checkpoint}")

    # --- Collect samples from HDF5 ---
    h5_files = []
    for pattern in args.data_h5:
        h5_files.extend(glob(pattern))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found: {args.data_h5}")
    print(f"[Data] Found {len(h5_files)} HDF5 files")

    # WM requires window_len = num_history + K = 6 + 5 = 11 frames
    window_len = num_history + K
    samples: list[dict] = []  # {state, actions, latents, gt_ee}
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj_")]
            for key in traj_keys:
                grp = f[key]
                if "state" not in grp or "actions" not in grp or "latent_concat" not in grp:
                    continue
                states = grp["state"][:].astype(np.float32)
                actions = grp["actions"][:].astype(np.float32)
                latents = grp["latent_concat"][:].astype(np.float32)
                T = min(len(states), len(actions), len(latents))
                # Need at least window_len + 1 frames (for target comparison)
                if T < window_len + 1:
                    continue
                # Sample windows from trajectory
                for t in range(0, T - window_len, max(1, (T - window_len) // 3)):
                    samples.append({
                        "state_t": states[t + num_history],  # current state (at position num_history in window)
                        "actions_chunk": actions[t + num_history : t + window_len],  # (K, 7) future actions
                        "latent_window": latents[t : t + window_len],  # (window_len, 4, 48, 24) = 11 frames
                        "latent_future": latents[t + num_history + 1 : t + window_len + 1],  # (K, 4, 48, 24) GT future
                        "states_window": states[t : t + window_len],  # (window_len, 25) for EE extraction
                        "states_future": states[t + num_history + 1 : t + window_len + 1],  # for GT EE
                    })
        if len(samples) >= args.num_samples * 2:
            break

    np.random.shuffle(samples)
    samples = samples[: args.num_samples]
    print(f"[Data] Using {len(samples)} samples for evaluation")

    # --- Evaluate ---
    psnr_gt, psnr_tiled, psnr_adapter = [], [], []
    pos_errors, euler_errors = [], []
    # Per-step tracking (K steps)
    per_step_pos_err = [[] for _ in range(K)]  # per_step_pos_err[step_idx] = list of errors
    per_step_euler_err = [[] for _ in range(K)]
    per_step_psnr_gt = [[] for _ in range(K)]
    per_step_psnr_tiled = [[] for _ in range(K)]
    per_step_psnr_adapter = [[] for _ in range(K)]

    for sample in tqdm(samples, desc="Evaluating"):
        state_t = sample["state_t"]
        action_chunk = sample["actions_chunk"]  # (K, 7) delta actions
        latent_window = torch.from_numpy(sample["latent_window"]).to(device)  # (window_len, 4, 48, 24)
        latent_future = torch.from_numpy(sample["latent_future"]).to(device)  # (K, 4, 48, 24) GT target
        states_window = sample["states_window"]  # (window_len, 25)

        # GT EE poses for the full window
        gt_ee_window = state_to_ee_pose_7d(states_window)  # (window_len, 7)
        gt_ee_future = state_to_ee_pose_7d(sample["states_future"])  # (K, 7)
        current_ee = gt_ee_window[num_history]  # (7,)

        # Tiled EE (baseline = BUG-D): tile current EE for future steps
        tiled_ee_future = np.tile(current_ee[None, :], (K, 1))

        # Adapter EE: predict future EE from state + delta actions
        state_n = (state_t - norm["state_mean"]) / norm["state_std"]
        # SingleStepDynamicsAdapter needs current_ee explicitly
        if is_single_step:
            adapter_ee_future = adapter.predict(state_n, action_chunk, current_ee)  # (K, 7)
        else:
            adapter_ee_future = adapter.predict(state_n, action_chunk)  # (K, 7)

        # Compute per-step EE prediction errors (Adapter vs GT)
        for step_i in range(K):
            pe = np.linalg.norm(adapter_ee_future[step_i, :3] - gt_ee_future[step_i, :3])
            ee = np.abs(adapter_ee_future[step_i, 3:6] - gt_ee_future[step_i, 3:6]).mean()
            per_step_pos_err[step_i].append(pe)
            per_step_euler_err[step_i].append(ee)

        # Aggregate errors (backward compatible)
        pos_err = np.linalg.norm(adapter_ee_future[:, :3] - gt_ee_future[:, :3], axis=-1).mean()
        euler_err = np.abs(adapter_ee_future[:, 3:6] - gt_ee_future[:, 3:6]).mean()
        pos_errors.append(pos_err)
        euler_errors.append(euler_err)

        # Build WM input: history EE + future EE (window_len total)
        def build_full_ee(future_ee: np.ndarray) -> np.ndarray:
            """Combine history EE (from GT) + future EE (from method)."""
            return np.concatenate([gt_ee_window[:num_history + 1], future_ee[:-1]], axis=0)  # (window_len, 7)

        def run_wm(future_ee: np.ndarray) -> torch.Tensor:
            full_ee = build_full_ee(future_ee)  # (window_len, 7)
            pred = wm.rollout(latent_window, full_ee, instruction="lift the peg upright")
            # pred: (2, K, 4, 24, 24) for dual-camera
            if pred.shape[0] == 2:
                cam0, cam1 = pred[0], pred[1]
                pred_lat = torch.cat([cam0, cam1], dim=2)  # (K, 4, 48, 24)
            else:
                pred_lat = pred[0]
            return pred_lat

        try:
            pred_gt = run_wm(gt_ee_future)
            pred_tiled = run_wm(tiled_ee_future)
            pred_adapter = run_wm(adapter_ee_future)

            psnr_gt.append(compute_psnr(pred_gt, latent_future))
            psnr_tiled.append(compute_psnr(pred_tiled, latent_future))
            psnr_adapter.append(compute_psnr(pred_adapter, latent_future))

            # Per-step PSNR
            for step_i in range(K):
                per_step_psnr_gt[step_i].append(
                    compute_psnr(pred_gt[step_i], latent_future[step_i]))
                per_step_psnr_tiled[step_i].append(
                    compute_psnr(pred_tiled[step_i], latent_future[step_i]))
                per_step_psnr_adapter[step_i].append(
                    compute_psnr(pred_adapter[step_i], latent_future[step_i]))
        except Exception as e:
            print(f"[WARN] WM rollout failed: {e}")
            continue

    # --- Report ---
    print("\n" + "=" * 60)
    print("Dynamics Adapter PSNR Diagnostic Report")
    print("=" * 60)
    print(f"Samples evaluated: {len(psnr_gt)}")
    print()

    # Aggregate EE errors
    print("EE Pose Prediction Accuracy (Adapter vs GT):")
    print(f"  Position MAE:  {np.mean(pos_errors)*1000:.2f} mm")
    print(f"  Euler MAE:     {np.mean(euler_errors):.4f} rad ({np.degrees(np.mean(euler_errors)):.2f}°)")
    print()

    # Per-step EE error breakdown
    print("Per-Step EE Error Breakdown:")
    print(f"  {'Step':>6}  {'pos_mae (mm)':>14}  {'euler_mae (rad)':>16}  {'euler_mae (°)':>14}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*16}  {'-'*14}")
    for step_i in range(K):
        pm = np.mean(per_step_pos_err[step_i]) * 1000
        em = np.mean(per_step_euler_err[step_i])
        print(f"  {step_i+1:>6}  {pm:>14.2f}  {em:>16.4f}  {np.degrees(em):>14.2f}")
    print()

    # Aggregate PSNR
    print("WM PSNR — Aggregate (higher is better):")
    print(f"  GT EE:       {np.mean(psnr_gt):.2f} ± {np.std(psnr_gt):.2f} dB")
    print(f"  Tiled EE:    {np.mean(psnr_tiled):.2f} ± {np.std(psnr_tiled):.2f} dB  (baseline = BUG-D)")
    print(f"  Adapter EE:  {np.mean(psnr_adapter):.2f} ± {np.std(psnr_adapter):.2f} dB")
    print()
    improvement = np.mean(psnr_adapter) - np.mean(psnr_tiled)
    gap_to_gt = np.mean(psnr_gt) - np.mean(psnr_adapter)
    print(f"  Improvement over Tiled: +{improvement:.2f} dB")
    print(f"  Gap to GT:              -{gap_to_gt:.2f} dB")
    print()

    # Per-step PSNR breakdown
    print("WM PSNR — Per-Step Breakdown:")
    print(f"  {'Step':>6}  {'GT (dB)':>10}  {'Tiled (dB)':>12}  {'Adapter (dB)':>14}  {'Δ(Adpt-Tile)':>14}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*14}")
    for step_i in range(K):
        g = np.mean(per_step_psnr_gt[step_i])
        t = np.mean(per_step_psnr_tiled[step_i])
        a = np.mean(per_step_psnr_adapter[step_i])
        print(f"  {step_i+1:>6}  {g:>10.2f}  {t:>12.2f}  {a:>14.2f}  {a-t:>+14.2f}")
    print("=" * 60)

    # --- Save results to JSON ---
    import json
    results = {
        "adapter_ckpt": args.adapter_ckpt,
        "wm_checkpoint": args.wm_checkpoint,
        "num_samples": len(psnr_gt),
        "aggregate": {
            "psnr_gt": {"mean": float(np.mean(psnr_gt)), "std": float(np.std(psnr_gt))},
            "psnr_tiled": {"mean": float(np.mean(psnr_tiled)), "std": float(np.std(psnr_tiled))},
            "psnr_adapter": {"mean": float(np.mean(psnr_adapter)), "std": float(np.std(psnr_adapter))},
            "pos_mae_mm": float(np.mean(pos_errors) * 1000),
            "euler_mae_rad": float(np.mean(euler_errors)),
        },
        "per_step": {
            f"step_{i+1}": {
                "pos_mae_mm": float(np.mean(per_step_pos_err[i]) * 1000),
                "euler_mae_rad": float(np.mean(per_step_euler_err[i])),
                "psnr_gt": float(np.mean(per_step_psnr_gt[i])),
                "psnr_tiled": float(np.mean(per_step_psnr_tiled[i])),
                "psnr_adapter": float(np.mean(per_step_psnr_adapter[i])),
            }
            for i in range(K)
        },
    }
    out_dir = _ROOT / "results" / "vlaw" / "adapter_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "psnr_per_step.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
