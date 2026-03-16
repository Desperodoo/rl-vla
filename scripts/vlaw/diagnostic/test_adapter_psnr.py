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
    from rlft.vlaw.world_model.dynamics_adapter import DynamicsAdapterTrainer
    adapter, norm = DynamicsAdapterTrainer.load_from_checkpoint(args.adapter_ckpt, device=str(device))
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
                if T <= K + num_history:
                    continue
                # Sample from middle of trajectory
                for t in range(num_history, T - K - 1, max(1, (T - K - num_history) // 3)):
                    samples.append({
                        "state_t": states[t],
                        "actions_chunk": actions[t : t + K],  # (K, 7)
                        "latent_t": latents[t],               # (4, 48, 24)
                        "latent_future": latents[t + 1 : t + K + 1],  # (K, 4, 48, 24)
                        "states_future": states[t + 1 : t + K + 1],  # for GT EE
                        "latent_hist": latents[max(0, t - num_history * 2) : t : 2][-num_history:],  # sparse
                    })
        if len(samples) >= args.num_samples * 2:
            break

    np.random.shuffle(samples)
    samples = samples[: args.num_samples]
    print(f"[Data] Using {len(samples)} samples for evaluation")

    # --- Evaluate ---
    psnr_gt, psnr_tiled, psnr_adapter = [], [], []
    pos_errors, euler_errors = [], []

    for sample in tqdm(samples, desc="Evaluating"):
        state_t = sample["state_t"]
        action_chunk = sample["actions_chunk"]
        latent_t = torch.from_numpy(sample["latent_t"]).unsqueeze(0).to(device)  # (1, 4, 48, 24)
        latent_future = torch.from_numpy(sample["latent_future"]).to(device)     # (K, 4, 48, 24)
        latent_hist = sample["latent_hist"]
        if len(latent_hist) < num_history:
            latent_hist = np.concatenate([
                np.tile(latent_hist[:1], (num_history - len(latent_hist), 1, 1, 1)),
                latent_hist
            ], axis=0)
        latent_hist = torch.from_numpy(latent_hist).to(device)  # (num_history, 4, 48, 24)

        # GT EE poses
        gt_ee = state_to_ee_pose_7d(sample["states_future"])  # (K, 7)
        current_ee = state_to_ee_pose_7d(state_t[None, :])[0]  # (7,)

        # Tiled EE (baseline = BUG-D)
        tiled_ee = np.tile(current_ee[None, :], (K, 1))

        # Adapter EE
        state_n = (state_t - norm["state_mean"]) / norm["state_std"]
        adapter_ee = adapter.predict(state_n, action_chunk)  # (K, 7)

        # Compute EE prediction errors
        pos_err = np.linalg.norm(adapter_ee[:, :3] - gt_ee[:, :3], axis=-1).mean()
        euler_err = np.abs(adapter_ee[:, 3:6] - gt_ee[:, 3:6]).mean()
        pos_errors.append(pos_err)
        euler_errors.append(euler_err)

        # Build WM input: history + current
        hist_ee_gt = state_to_ee_pose_7d(
            np.concatenate([sample["state_t"][None, :]] * num_history, axis=0)
        )  # simplified: tile current for hist (matches batch path)

        def run_wm(future_ee: np.ndarray) -> torch.Tensor:
            full_ee = np.concatenate([hist_ee_gt, future_ee], axis=0)  # (num_history + K, 7)
            wm_input = torch.cat([latent_hist, latent_t], dim=0)  # (num_history + 1, 4, 48, 24)
            # WM expects (num_history, C, H, W) obs + (num_history + K, 7) ee_poses
            pred = wm.rollout(wm_input, full_ee, instruction="lift the peg upright")
            # pred: (2, K, 4, 24, 24) for dual-camera, or (1, K, ...)
            if pred.shape[0] == 2:
                cam0, cam1 = pred[0], pred[1]
                pred_lat = torch.cat([cam0, cam1], dim=2)  # (K, 4, 48, 24)
            else:
                pred_lat = pred[0]
            return pred_lat

        try:
            pred_gt = run_wm(gt_ee)
            pred_tiled = run_wm(tiled_ee)
            pred_adapter = run_wm(adapter_ee)

            psnr_gt.append(compute_psnr(pred_gt, latent_future))
            psnr_tiled.append(compute_psnr(pred_tiled, latent_future))
            psnr_adapter.append(compute_psnr(pred_adapter, latent_future))
        except Exception as e:
            print(f"[WARN] WM rollout failed: {e}")
            continue

    # --- Report ---
    print("\n" + "=" * 60)
    print("Dynamics Adapter PSNR Diagnostic Report")
    print("=" * 60)
    print(f"Samples evaluated: {len(psnr_gt)}")
    print()
    print("EE Pose Prediction Accuracy:")
    print(f"  Position MAE:  {np.mean(pos_errors)*1000:.2f} mm")
    print(f"  Euler MAE:     {np.mean(euler_errors):.4f} rad ({np.degrees(np.mean(euler_errors)):.2f}°)")
    print()
    print("WM PSNR (higher is better):")
    print(f"  GT EE:       {np.mean(psnr_gt):.2f} ± {np.std(psnr_gt):.2f} dB")
    print(f"  Tiled EE:    {np.mean(psnr_tiled):.2f} ± {np.std(psnr_tiled):.2f} dB  (baseline = BUG-D)")
    print(f"  Adapter EE:  {np.mean(psnr_adapter):.2f} ± {np.std(psnr_adapter):.2f} dB")
    print()
    improvement = np.mean(psnr_adapter) - np.mean(psnr_tiled)
    gap_to_gt = np.mean(psnr_gt) - np.mean(psnr_adapter)
    print(f"Improvement over Tiled: +{improvement:.2f} dB")
    print(f"Gap to GT:              -{gap_to_gt:.2f} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()
