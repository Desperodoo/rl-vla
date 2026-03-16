"""WM Imagination 诊断实验 — 因素隔离消融.

用已训练的 WM v5 checkpoint 运行受控推理实验，隔离 imagination 质量退化的每个因素。
无需重训 WM。

Usage:
    CUDA_VISIBLE_DEVICES=X conda run -n ctrl_world python scripts/vlaw/diagnostic/wm_diagnostic_battery.py \
        --checkpoint checkpoints/vlaw/world_model/iter1_v5/checkpoint-3800.pt \
        --output_dir results/vlaw/wm_diagnostic \
        --groups A B C D E F
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np
import torch
from PIL import Image

# Ensure ctrl_world is importable
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "ctrl_world"))
sys.path.insert(0, str(_ROOT))

from ctrl_world.config import wm_args_maniskill
from ctrl_world.dataset.dataset_maniskill import state_to_ee_pose_7d


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def latent_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR between two latent tensors. Higher = more similar."""
    pred, gt = pred.float().cpu(), gt.float().cpu()
    mse = (pred - gt).pow(2).mean().item()
    if mse < 1e-10:
        return 100.0
    # Estimate data range from GT
    data_range = gt.float().max().item() - gt.float().min().item()
    if data_range < 1e-10:
        data_range = 1.0
    return 10.0 * np.log10(data_range ** 2 / mse)


def latent_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized L2 distance between two latent tensors."""
    a, b = a.float().cpu(), b.float().cpu()
    return (a - b).pow(2).mean().sqrt().item()


def per_frame_psnr(pred: torch.Tensor, gt: torch.Tensor) -> list[float]:
    """Per-frame PSNR. pred/gt: (N_CAMS, T, 4, H, W) or (T, 4, H, W)."""
    pred, gt = pred.float().cpu(), gt.float().cpu()
    if pred.dim() == 5:
        pred = pred.reshape(-1, *pred.shape[2:])  # merge cams and frames
        gt = gt.reshape(-1, *gt.shape[2:])
    if pred.dim() == 4:
        # (T, C, H, W)
        results = []
        for t in range(pred.shape[0]):
            results.append(latent_psnr(pred[t], gt[t]))
        return results
    return [latent_psnr(pred, gt)]


# ---------------------------------------------------------------------------
# Data loading (bypass Dataset class, load raw from HDF5)
# ---------------------------------------------------------------------------

@dataclass
class GTSample:
    """A single GT sample from the training data."""
    latent: torch.Tensor       # (T, 4, 48, 24) -- full window
    state: np.ndarray          # (T, 25) -- raw state
    ee_pose_raw: np.ndarray    # (T, 7) -- unnormalized absolute EE pose
    text: str
    h5_path: str
    traj_key: str
    start_frame: int


def load_gt_samples(
    h5_path: str,
    num_history: int = 6,
    num_frames: int = 5,
    max_samples: int = 5,
    min_traj_len: int = 11,
) -> list[GTSample]:
    """Load GT samples directly from HDF5 for diagnostic experiments."""
    window_len = num_history + num_frames
    samples = []
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        for tkey in traj_keys:
            if len(samples) >= max_samples:
                break
            grp = f[tkey]
            if "latent_concat" not in grp:
                continue
            T = grp["latent_concat"].shape[0]
            if T < max(window_len, min_traj_len):
                continue
            # Take the first window
            frame_ids = list(range(0, window_len))
            latent = torch.from_numpy(grp["latent_concat"][frame_ids].astype(np.float32))
            state_key = "state" if "state" in grp else "obs_agent"
            state_raw = grp[state_key][frame_ids].astype(np.float32)
            ee_pose_raw = state_to_ee_pose_7d(state_raw)
            text = grp.attrs.get("task_instruction", "Lift the peg upright")
            samples.append(GTSample(
                latent=latent, state=state_raw, ee_pose_raw=ee_pose_raw,
                text=text, h5_path=h5_path, traj_key=tkey,
                start_frame=0,
            ))
    print(f"Loaded {len(samples)} GT samples from {h5_path}")
    return samples


def load_long_gt_trajectory(
    h5_path: str,
    min_length: int = 20,
) -> Optional[tuple[torch.Tensor, np.ndarray, np.ndarray, str]]:
    """Find the longest trajectory for multi-chunk experiments.

    Returns (latent_all, state_all, ee_pose_all, text) or None.
    """
    best = None
    best_len = 0
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        for tkey in traj_keys:
            grp = f[tkey]
            if "latent_concat" not in grp:
                continue
            T = grp["latent_concat"].shape[0]
            if T > best_len:
                best_len = T
                best = tkey
        if best is None or best_len < min_length:
            print(f"No trajectory with length >= {min_length} found")
            return None
        grp = f[best]
        latent = torch.from_numpy(grp["latent_concat"][:].astype(np.float32))
        state_key = "state" if "state" in grp else "obs_agent"
        state_raw = grp[state_key][:].astype(np.float32)
        ee_pose_raw = state_to_ee_pose_7d(state_raw)
        text = grp.attrs.get("task_instruction", "Lift the peg upright")
        print(f"Loaded long trajectory {best}: length={best_len}")
        return latent, state_raw, ee_pose_raw, text


# ---------------------------------------------------------------------------
# WM wrapper (loads adapter)
# ---------------------------------------------------------------------------

class WMDiagnostic:
    """Wraps CtrlWorldAdapter for diagnostic experiments."""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        from rlft.vlaw.world_model.ctrl_world_adapter import CtrlWorldAdapter

        args = wm_args_maniskill()
        args.ckpt_path = ckpt_path

        # Fix relative paths: config assumes cwd=ctrl_world/scripts/, we run from project root
        root = str(_ROOT)
        for attr in ["svd_model_path", "clip_model_path", "ckpt_path",
                      "data_stat_path", "dataset_root_path", "val_dataset_dir",
                      "dataset_meta_info_path", "output_dir"]:
            val = getattr(args, attr, None)
            if val and val.startswith("../"):
                setattr(args, attr, os.path.join(root, val[3:]))  # strip "../"

        self.args = args
        self.adapter = CtrlWorldAdapter(args, ckpt_path=ckpt_path, device=device)
        self.num_history = args.num_history
        self.num_frames = args.num_frames
        self.device = device

    def rollout(
        self,
        obs_latents: torch.Tensor,
        ee_poses_raw: np.ndarray,
        instruction: str = "Lift the peg upright",
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run WM rollout. Returns (N_CAMS, T_pred, 4, H_single, W).

        Args:
            obs_latents: (window_len, 4, 48, 24)
            ee_poses_raw: (window_len, 7) unnormalized absolute EE poses
            instruction: text
            num_inference_steps: override default (50 from config)
        """
        # Temporarily override num_inference_steps if requested
        orig_steps = getattr(self.adapter.args, "num_inference_steps", 50)
        if num_inference_steps is not None:
            self.adapter.args.num_inference_steps = num_inference_steps
        try:
            pred = self.adapter.rollout(obs_latents, ee_poses_raw, instruction)
        finally:
            self.adapter.args.num_inference_steps = orig_steps
        return pred  # (N_CAMS, T_pred, 4, 24, 24)

    def pred_to_combined(self, pred: torch.Tensor) -> torch.Tensor:
        """Convert (N_CAMS=2, T, 4, 24, 24) -> (T, 4, 48, 24) combined latent."""
        cam0, cam1 = pred[0], pred[1]  # each (T, 4, 24, 24)
        return torch.cat([cam0, cam1], dim=2)  # (T, 4, 48, 24)

    def decode_frames(self, latent: torch.Tensor) -> np.ndarray:
        """Decode (T, 4, 48, 24) latent -> (T, H, W, 3) uint8 RGB."""
        return self.adapter.decode_latents(latent)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_comparison_grid(
    images: dict[str, np.ndarray],
    out_path: str,
    title: str = "",
):
    """Save a comparison grid. images: {label: (T, H, W, 3) uint8}.

    Renders each variant as a row, frames as columns.
    """
    labels = list(images.keys())
    n_rows = len(labels)
    sample = next(iter(images.values()))
    T, H, W, _ = sample.shape
    n_cols = T

    pad = 2
    label_w = 120  # pixels for label text
    grid_h = n_rows * (H + pad) + pad
    grid_w = label_w + n_cols * (W + pad) + pad
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # dark gray bg

    for ri, label in enumerate(labels):
        frames = images[label]
        y = pad + ri * (H + pad)
        for ci in range(min(T, frames.shape[0])):
            x = label_w + pad + ci * (W + pad)
            grid[y:y+H, x:x+W] = frames[ci]

    Image.fromarray(grid).save(out_path)


def save_metrics_json(metrics: dict, out_path: str):
    """Save metrics dict as JSON."""
    # Convert numpy types to Python natives
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=convert)


# ---------------------------------------------------------------------------
# Group A: Action Sensitivity
# ---------------------------------------------------------------------------

def run_group_a(wm: WMDiagnostic, samples: list[GTSample], out_dir: str):
    """Group A: Does the WM actually use action conditioning?"""
    print("\n" + "=" * 60)
    print("GROUP A: Action Sensitivity Test")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for si, s in enumerate(samples[:3]):
        window_len = wm.num_history + wm.num_frames
        gt_latent = s.latent[:window_len]          # (11, 4, 48, 24)
        gt_future = gt_latent[wm.num_history:]     # (5, 4, 48, 24) GT future
        gt_ee = s.ee_pose_raw[:window_len]         # (11, 7) unnormalized

        # A1-gt: GT actions
        pred_gt = wm.rollout(gt_latent, gt_ee, s.text)
        pred_gt_combined = wm.pred_to_combined(pred_gt)  # (5, 4, 48, 24)

        # A1-zero: zero actions (all 11 frames)
        zero_ee = np.zeros_like(gt_ee)
        pred_zero = wm.rollout(gt_latent, zero_ee, s.text)
        pred_zero_combined = wm.pred_to_combined(pred_zero)

        # A1-rand: random actions in raw EE pose range
        rng = np.random.RandomState(42 + si)
        rand_ee = rng.uniform(-0.3, 0.3, size=gt_ee.shape).astype(np.float32)
        # Keep gripper in [0, 1]
        rand_ee[:, 6] = rng.uniform(0, 1, size=gt_ee.shape[0]).astype(np.float32)
        pred_rand = wm.rollout(gt_latent, rand_ee, s.text)
        pred_rand_combined = wm.pred_to_combined(pred_rand)

        # A1-tile: tiled current EE pose (BUG-D behavior)
        current_ee = gt_ee[wm.num_history - 1]  # frame just before future
        tile_ee = np.tile(current_ee[None, :], (window_len, 1))
        pred_tile = wm.rollout(gt_latent, tile_ee, s.text)
        pred_tile_combined = wm.pred_to_combined(pred_tile)

        # Metrics
        sample_metrics = {
            "sample_idx": si,
            "traj_key": s.traj_key,
            "psnr_gt": latent_psnr(pred_gt_combined, gt_future),
            "psnr_zero": latent_psnr(pred_zero_combined, gt_future),
            "psnr_rand": latent_psnr(pred_rand_combined, gt_future),
            "psnr_tile": latent_psnr(pred_tile_combined, gt_future),
            "l2_gt_vs_zero": latent_l2(pred_gt_combined, pred_zero_combined),
            "l2_gt_vs_rand": latent_l2(pred_gt_combined, pred_rand_combined),
            "l2_gt_vs_tile": latent_l2(pred_gt_combined, pred_tile_combined),
            "l2_zero_vs_tile": latent_l2(pred_zero_combined, pred_tile_combined),
            "per_frame_psnr_gt": per_frame_psnr(pred_gt_combined, gt_future),
            "per_frame_psnr_tile": per_frame_psnr(pred_tile_combined, gt_future),
        }
        results.append(sample_metrics)

        print(f"\n  Sample {si} ({s.traj_key}):")
        print(f"    PSNR  gt={sample_metrics['psnr_gt']:.2f}  zero={sample_metrics['psnr_zero']:.2f}  "
              f"rand={sample_metrics['psnr_rand']:.2f}  tile={sample_metrics['psnr_tile']:.2f}")
        print(f"    L2    gt-zero={sample_metrics['l2_gt_vs_zero']:.4f}  "
              f"gt-rand={sample_metrics['l2_gt_vs_rand']:.4f}  "
              f"gt-tile={sample_metrics['l2_gt_vs_tile']:.4f}")

        # Decode and save visual comparison for first sample
        if si == 0:
            try:
                vis = {}
                vis["GT_future"] = wm.decode_frames(gt_future)
                vis["pred_GT_actions"] = wm.decode_frames(pred_gt_combined)
                vis["pred_ZERO_actions"] = wm.decode_frames(pred_zero_combined)
                vis["pred_TILED_actions"] = wm.decode_frames(pred_tile_combined)
                vis["pred_RANDOM_actions"] = wm.decode_frames(pred_rand_combined)
                save_comparison_grid(vis, f"{out_dir}/group_a_visual.png", "Group A: Action Sensitivity")
                print(f"    Visual saved: {out_dir}/group_a_visual.png")
            except Exception as e:
                print(f"    Visual decode failed: {e}")

    # Summary
    summary = {
        "samples": results,
        "mean_psnr_gt": np.mean([r["psnr_gt"] for r in results]),
        "mean_psnr_zero": np.mean([r["psnr_zero"] for r in results]),
        "mean_psnr_rand": np.mean([r["psnr_rand"] for r in results]),
        "mean_psnr_tile": np.mean([r["psnr_tile"] for r in results]),
        "mean_l2_gt_vs_zero": np.mean([r["l2_gt_vs_zero"] for r in results]),
        "mean_l2_gt_vs_tile": np.mean([r["l2_gt_vs_tile"] for r in results]),
    }
    save_metrics_json(summary, f"{out_dir}/group_a_metrics.json")

    print(f"\n  >>> VERDICT: L2(gt-zero)={summary['mean_l2_gt_vs_zero']:.4f}, "
          f"L2(gt-tile)={summary['mean_l2_gt_vs_tile']:.4f}")
    if summary["mean_l2_gt_vs_zero"] > 0.05:
        print("  >>> WM IS responding to actions (L2 > 0.05)")
    else:
        print("  >>> WARNING: WM may NOT be using actions (L2 < 0.05)")
    return summary


# ---------------------------------------------------------------------------
# Group B: num_inference_steps
# ---------------------------------------------------------------------------

def run_group_b(wm: WMDiagnostic, samples: list[GTSample], out_dir: str):
    """Group B: Impact of num_inference_steps (NEW-BUG-1)."""
    print("\n" + "=" * 60)
    print("GROUP B: num_inference_steps Ablation (NEW-BUG-1)")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    s = samples[0]
    window_len = wm.num_history + wm.num_frames
    gt_latent = s.latent[:window_len]
    gt_future = gt_latent[wm.num_history:]
    gt_ee = s.ee_pose_raw[:window_len]

    steps_list = [10, 15, 20, 25, 50]
    results = {}
    vis = {"GT_future": None}  # will decode once

    for steps in steps_list:
        print(f"  Running with num_inference_steps={steps}...", end=" ", flush=True)
        pred = wm.rollout(gt_latent, gt_ee, s.text, num_inference_steps=steps)
        pred_combined = wm.pred_to_combined(pred)
        p = latent_psnr(pred_combined, gt_future)
        l2 = latent_l2(pred_combined, gt_future)
        pf = per_frame_psnr(pred_combined, gt_future)
        results[steps] = {"psnr": p, "l2": l2, "per_frame_psnr": pf}
        print(f"PSNR={p:.2f}  L2={l2:.4f}")

        # Decode for visual
        try:
            vis[f"steps={steps}"] = wm.decode_frames(pred_combined)
            if vis["GT_future"] is None:
                vis["GT_future"] = wm.decode_frames(gt_future)
        except Exception:
            pass

    # Save
    save_metrics_json(results, f"{out_dir}/group_b_metrics.json")
    if any(v is not None for v in vis.values()):
        vis = {k: v for k, v in vis.items() if v is not None}
        save_comparison_grid(vis, f"{out_dir}/group_b_visual.png", "Group B: Inference Steps")

    # Verdict
    psnr_25 = results[25]["psnr"]
    psnr_50 = results[50]["psnr"]
    delta = psnr_50 - psnr_25
    print(f"\n  >>> PSNR(50steps)={psnr_50:.2f}, PSNR(25steps)={psnr_25:.2f}, delta={delta:.2f} dB")
    if delta > 2.0:
        print(f"  >>> NEW-BUG-1 is SIGNIFICANT ({delta:.1f} dB gap)")
    elif delta > 0.5:
        print(f"  >>> NEW-BUG-1 has moderate impact ({delta:.1f} dB)")
    else:
        print(f"  >>> NEW-BUG-1 is negligible ({delta:.1f} dB)")
    return results


# ---------------------------------------------------------------------------
# Group C: Action Conditioning Strategies (BUG-D)
# ---------------------------------------------------------------------------

def run_group_c(wm: WMDiagnostic, samples: list[GTSample], out_dir: str):
    """Group C: Four action conditioning strategies (BUG-D isolation)."""
    print("\n" + "=" * 60)
    print("GROUP C: Action Conditioning (BUG-D)")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for si, s in enumerate(samples[:3]):
        window_len = wm.num_history + wm.num_frames
        gt_latent = s.latent[:window_len]
        gt_future = gt_latent[wm.num_history:]
        gt_ee = s.ee_pose_raw[:window_len]
        hist_ee = gt_ee[:wm.num_history]  # keep history GT
        future_gt_ee = gt_ee[wm.num_history:]

        # C1-gt: all GT
        pred_gt = wm.pred_to_combined(wm.rollout(gt_latent, gt_ee, s.text))

        # C1-tile: history GT + future tiled current
        current_ee = gt_ee[wm.num_history - 1]
        tile_future = np.tile(current_ee[None, :], (wm.num_frames, 1))
        tile_all = np.concatenate([hist_ee, tile_future], axis=0)
        pred_tile = wm.pred_to_combined(wm.rollout(gt_latent, tile_all, s.text))

        # C2: action magnitude scaling (alpha sweep)
        alpha_results = {}
        for alpha in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
            scaled_future = current_ee[None, :] + alpha * (future_gt_ee - current_ee[None, :])
            scaled_all = np.concatenate([hist_ee, scaled_future.astype(np.float32)], axis=0)
            pred_alpha = wm.pred_to_combined(wm.rollout(gt_latent, scaled_all, s.text))
            alpha_results[alpha] = {
                "psnr": latent_psnr(pred_alpha, gt_future),
                "l2_vs_gt_pred": latent_l2(pred_alpha, pred_gt),
            }

        sample_metrics = {
            "sample_idx": si,
            "psnr_gt": latent_psnr(pred_gt, gt_future),
            "psnr_tile": latent_psnr(pred_tile, gt_future),
            "l2_gt_vs_tile": latent_l2(pred_gt, pred_tile),
            "alpha_sweep": {str(k): v for k, v in alpha_results.items()},
            "per_frame_psnr_gt": per_frame_psnr(pred_gt, gt_future),
            "per_frame_psnr_tile": per_frame_psnr(pred_tile, gt_future),
        }
        results.append(sample_metrics)

        print(f"\n  Sample {si}: PSNR gt={sample_metrics['psnr_gt']:.2f}  "
              f"tile={sample_metrics['psnr_tile']:.2f}  "
              f"L2(gt-tile)={sample_metrics['l2_gt_vs_tile']:.4f}")
        print(f"    Alpha sweep PSNR: " +
              " ".join(f"α={a}:{v['psnr']:.1f}" for a, v in alpha_results.items()))

        if si == 0:
            try:
                vis = {
                    "GT_future": wm.decode_frames(gt_future),
                    "pred_GT_actions": wm.decode_frames(pred_gt),
                    "pred_TILED_actions": wm.decode_frames(pred_tile),
                }
                # Add a few alpha variants
                for alpha in [0.0, 0.5, 2.0]:
                    scaled_future = current_ee[None, :] + alpha * (future_gt_ee - current_ee[None, :])
                    scaled_all = np.concatenate([hist_ee, scaled_future.astype(np.float32)], axis=0)
                    p = wm.pred_to_combined(wm.rollout(gt_latent, scaled_all, s.text))
                    vis[f"alpha={alpha}"] = wm.decode_frames(p)
                save_comparison_grid(vis, f"{out_dir}/group_c_visual.png", "Group C: Actions")
            except Exception as e:
                print(f"    Visual decode failed: {e}")

    save_metrics_json({"samples": results}, f"{out_dir}/group_c_metrics.json")
    return results


# ---------------------------------------------------------------------------
# Group D3: History Noise Injection
# ---------------------------------------------------------------------------

def run_group_d3(wm: WMDiagnostic, samples: list[GTSample], out_dir: str):
    """Group D3: How sensitive is WM to history latent noise?"""
    print("\n" + "=" * 60)
    print("GROUP D3: History Noise Sensitivity")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    s = samples[0]
    window_len = wm.num_history + wm.num_frames
    gt_latent = s.latent[:window_len]
    gt_future = gt_latent[wm.num_history:]
    gt_ee = s.ee_pose_raw[:window_len]

    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results = {}

    for sigma in sigmas:
        noisy_latent = gt_latent.clone()
        if sigma > 0:
            noise = torch.randn_like(gt_latent[:wm.num_history]) * sigma
            noisy_latent[:wm.num_history] += noise
        pred = wm.pred_to_combined(wm.rollout(noisy_latent, gt_ee, s.text))
        p = latent_psnr(pred, gt_future)
        l2 = latent_l2(pred, gt_future)
        results[sigma] = {"psnr": p, "l2": l2}
        print(f"  sigma={sigma:.3f}: PSNR={p:.2f}  L2={l2:.4f}")

    save_metrics_json(results, f"{out_dir}/group_d3_metrics.json")

    # Verdict
    psnr_clean = results[0.0]["psnr"]
    psnr_005 = results[0.05]["psnr"]
    drop = psnr_clean - psnr_005
    print(f"\n  >>> PSNR drop at sigma=0.05: {drop:.2f} dB")
    if drop > 3.0:
        print("  >>> WM is VERY sensitive to history noise — AR error will be catastrophic")
    elif drop > 1.0:
        print("  >>> WM is moderately sensitive to history noise")
    else:
        print("  >>> WM is robust to history noise")
    return results


# ---------------------------------------------------------------------------
# Group E: History Sampling Strategy
# ---------------------------------------------------------------------------

def run_group_e(wm: WMDiagnostic, samples: list[GTSample], out_dir: str,
                h5_path: str):
    """Group E: Contiguous vs sparse history sampling."""
    print("\n" + "=" * 60)
    print("GROUP E: History Sampling Strategy")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    # Need a trajectory long enough for meaningful sparse sampling
    long = load_long_gt_trajectory(h5_path, min_length=20)
    if long is None:
        print("  SKIPPED: no trajectory long enough")
        return {}

    latent_all, state_all, ee_all, text = long
    T = latent_all.shape[0]
    nh = wm.num_history
    nf = wm.num_frames
    window_len = nh + nf

    # Pick a time point in the middle
    t = min(15, T - nf)  # start of future frames
    gt_future = latent_all[t:t+nf]  # (5, 4, 48, 24)
    gt_full = latent_all[t-nh:t+nf]
    gt_ee_full = ee_all[t-nh:t+nf]

    variants = {}

    # E1-contig: contiguous history [t-6..t-1] (training-like)
    contig_hist = latent_all[t-nh:t]
    contig_input = torch.cat([contig_hist, latent_all[t:t+nf]], dim=0)
    contig_ee = ee_all[t-nh:t+nf]
    pred_contig = wm.pred_to_combined(wm.rollout(contig_input, contig_ee, text))
    variants["contig"] = pred_contig

    # E1-sparse: indices [0, 0, t-12, t-9, t-6, t-3] (imagination-like)
    sparse_offsets = [0, 0] + [max(0, t + off) for off in [-12, -9, -6, -3]]
    sparse_hist = torch.stack([latent_all[min(i, T-1)] for i in sparse_offsets], dim=0)
    sparse_input = torch.cat([sparse_hist, latent_all[t:t+nf]], dim=0)
    sparse_ee_hist = np.stack([ee_all[min(i, T-1)] for i in sparse_offsets], axis=0)
    sparse_ee = np.concatenate([sparse_ee_hist, ee_all[t:t+nf]], axis=0)
    pred_sparse = wm.pred_to_combined(wm.rollout(sparse_input, sparse_ee, text))
    variants["sparse"] = pred_sparse

    # E1-same0: all history = frame 0 (simulates early imagination)
    same0_hist = latent_all[0:1].expand(nh, -1, -1, -1)
    same0_input = torch.cat([same0_hist, latent_all[t:t+nf]], dim=0)
    same0_ee_hist = np.tile(ee_all[0:1], (nh, 1))
    same0_ee = np.concatenate([same0_ee_hist, ee_all[t:t+nf]], axis=0)
    pred_same0 = wm.pred_to_combined(wm.rollout(same0_input, same0_ee, text))
    variants["same_frame0"] = pred_same0

    # E1-samet: all history = current frame t
    samet_hist = latent_all[t-1:t].expand(nh, -1, -1, -1)
    samet_input = torch.cat([samet_hist, latent_all[t:t+nf]], dim=0)
    samet_ee_hist = np.tile(ee_all[t-1:t], (nh, 1))
    samet_ee = np.concatenate([samet_ee_hist, ee_all[t:t+nf]], axis=0)
    pred_samet = wm.pred_to_combined(wm.rollout(samet_input, samet_ee, text))
    variants["same_current"] = pred_samet

    results = {}
    for name, pred in variants.items():
        p = latent_psnr(pred, gt_future)
        l2 = latent_l2(pred, gt_future)
        results[name] = {"psnr": p, "l2": l2, "per_frame_psnr": per_frame_psnr(pred, gt_future)}
        print(f"  {name:15s}: PSNR={p:.2f}  L2={l2:.4f}")

    save_metrics_json(results, f"{out_dir}/group_e_metrics.json")

    # Visual
    try:
        vis = {"GT_future": wm.decode_frames(gt_future)}
        for name, pred in variants.items():
            vis[name] = wm.decode_frames(pred)
        save_comparison_grid(vis, f"{out_dir}/group_e_visual.png", "Group E: History")
    except Exception as e:
        print(f"    Visual decode failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Group F2: Progressive Factor Introduction
# ---------------------------------------------------------------------------

def run_group_f2(wm: WMDiagnostic, samples: list[GTSample], out_dir: str):
    """Group F2: Progressively add imagination defects to quantify each factor."""
    print("\n" + "=" * 60)
    print("GROUP F2: Progressive Factor Introduction")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    s = samples[0]
    window_len = wm.num_history + wm.num_frames
    gt_latent = s.latent[:window_len]
    gt_future = gt_latent[wm.num_history:]
    gt_ee = s.ee_pose_raw[:window_len]
    hist_ee = gt_ee[:wm.num_history]
    current_ee = gt_ee[wm.num_history - 1]

    steps_results = {}

    # Step 0: Perfect baseline (GT everything, 50 steps)
    print("  F2-0: GT everything, 50 steps")
    pred = wm.pred_to_combined(wm.rollout(gt_latent, gt_ee, s.text, num_inference_steps=50))
    psnr_0 = latent_psnr(pred, gt_future)
    steps_results["F2-0_baseline"] = {"psnr": psnr_0, "description": "GT everything, 50 steps"}
    print(f"    PSNR={psnr_0:.2f}")

    # Step 1: Reduce to 25 steps (NEW-BUG-1)
    print("  F2-1: → 25 steps")
    pred = wm.pred_to_combined(wm.rollout(gt_latent, gt_ee, s.text, num_inference_steps=25))
    psnr_1 = latent_psnr(pred, gt_future)
    steps_results["F2-1_25steps"] = {"psnr": psnr_1, "delta": psnr_0 - psnr_1, "description": "25 steps"}
    print(f"    PSNR={psnr_1:.2f}  (Δ={psnr_0 - psnr_1:+.2f} from step reduction)")

    # Step 2: Tiled future actions (BUG-D)
    print("  F2-2: → tiled future actions")
    tile_future = np.tile(current_ee[None, :], (wm.num_frames, 1))
    tile_all = np.concatenate([hist_ee, tile_future], axis=0)
    pred = wm.pred_to_combined(wm.rollout(gt_latent, tile_all, s.text, num_inference_steps=25))
    psnr_2 = latent_psnr(pred, gt_future)
    steps_results["F2-2_tiled_actions"] = {"psnr": psnr_2, "delta": psnr_1 - psnr_2, "description": "tiled actions"}
    print(f"    PSNR={psnr_2:.2f}  (Δ={psnr_1 - psnr_2:+.2f} from action tiling)")

    save_metrics_json(steps_results, f"{out_dir}/group_f2_metrics.json")

    # Summary
    print("\n  === Factor Contribution Summary ===")
    deltas = []
    for k, v in steps_results.items():
        if "delta" in v:
            print(f"    {k}: Δ={v['delta']:+.2f} dB ({v['description']})")
            deltas.append((k, v["delta"]))
    if deltas:
        biggest = max(deltas, key=lambda x: abs(x[1]))
        print(f"\n  >>> Biggest factor: {biggest[0]} ({biggest[1]:+.2f} dB)")
    return steps_results


# ---------------------------------------------------------------------------
# Group D1/D2: Multi-chunk Autoregressive (oracle vs predicted history)
# ---------------------------------------------------------------------------

def run_group_d_multichunk(wm: WMDiagnostic, h5_path: str, out_dir: str,
                           max_chunks: int = 6):
    """Groups D1 & D2: Multi-chunk AR with oracle vs predicted history.

    D1: Each chunk uses GT history latents (oracle) — isolates per-chunk quality
    D2: Each chunk uses previous chunk's predictions (real AR) — measures error accumulation
    Both use GT EE poses throughout.
    """
    print("\n" + "=" * 60)
    print("GROUP D1/D2: Multi-chunk Autoregressive")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    long = load_long_gt_trajectory(h5_path, min_length=20)
    if long is None:
        print("  SKIPPED: no long trajectory")
        return {}

    latent_all, state_all, ee_all, text = long
    T = latent_all.shape[0]
    nh = wm.num_history
    nf = wm.num_frames
    window_len = nh + nf

    n_chunks = min(max_chunks, (T - nh) // nf)
    if n_chunks < 2:
        print(f"  SKIPPED: trajectory length {T} only supports {n_chunks} chunk(s)")
        return {}

    print(f"  Trajectory length={T}, running {n_chunks} chunks (each {nf} frames)")

    d1_psnrs = []  # oracle history
    d2_psnrs = []  # predicted history (AR)

    # For D2: maintain a buffer of predicted latents
    pred_buffer = []  # list of (nf, 4, 48, 24) predicted chunks

    for ci in range(n_chunks):
        chunk_start = nh + ci * nf  # start of future frames for this chunk
        chunk_end = chunk_start + nf

        if chunk_end > T:
            break

        gt_future = latent_all[chunk_start:chunk_end]
        gt_ee_chunk = ee_all[chunk_start - nh:chunk_end]  # (nh+nf, 7)

        # --- D1: Oracle history (always use GT) ---
        gt_hist = latent_all[chunk_start - nh:chunk_start]
        d1_input = torch.cat([gt_hist, latent_all[chunk_start:chunk_end]], dim=0)
        pred_d1 = wm.pred_to_combined(wm.rollout(d1_input, gt_ee_chunk, text))
        psnr_d1 = latent_psnr(pred_d1, gt_future)
        d1_psnrs.append(psnr_d1)

        # --- D2: Predicted history (real AR) ---
        if ci == 0:
            # First chunk: same as D1 (no prior predictions)
            d2_hist = gt_hist.clone()
        else:
            # Build history from predictions: use sparse-like approach
            # Take last nf frames from previous prediction + earlier GT/pred history
            all_pred_frames = torch.cat(pred_buffer, dim=0)  # all predicted so far
            n_available = all_pred_frames.shape[0]

            if n_available >= nh:
                # Enough predicted frames: use last nh frames
                d2_hist = all_pred_frames[-nh:]
            else:
                # Pad with GT initial frames
                n_gt_needed = nh - n_available
                d2_hist = torch.cat([
                    latent_all[:n_gt_needed],
                    all_pred_frames,
                ], dim=0)

        d2_input = torch.cat([d2_hist, latent_all[chunk_start:chunk_end]], dim=0)
        pred_d2 = wm.pred_to_combined(wm.rollout(d2_input, gt_ee_chunk, text))
        psnr_d2 = latent_psnr(pred_d2, gt_future)
        d2_psnrs.append(psnr_d2)

        # Store prediction for future AR chunks
        pred_buffer.append(pred_d2.cpu())

        print(f"  Chunk {ci}: D1(oracle)={psnr_d1:.2f} dB  D2(AR)={psnr_d2:.2f} dB  "
              f"gap={psnr_d1 - psnr_d2:+.2f} dB")

    results = {
        "n_chunks": n_chunks,
        "trajectory_length": int(T),
        "d1_oracle_psnr": d1_psnrs,
        "d2_ar_psnr": d2_psnrs,
        "d1_mean": float(np.mean(d1_psnrs)),
        "d2_mean": float(np.mean(d2_psnrs)),
        "ar_degradation": [d1 - d2 for d1, d2 in zip(d1_psnrs, d2_psnrs)],
    }
    save_metrics_json(results, f"{out_dir}/group_d_multichunk_metrics.json")

    # Verdict
    mean_gap = np.mean([d1 - d2 for d1, d2 in zip(d1_psnrs, d2_psnrs)])
    last_gap = d1_psnrs[-1] - d2_psnrs[-1] if d2_psnrs else 0
    print(f"\n  >>> D1(oracle) mean PSNR: {np.mean(d1_psnrs):.2f}")
    print(f"  >>> D2(AR) mean PSNR: {np.mean(d2_psnrs):.2f}")
    print(f"  >>> Mean gap (D1-D2): {mean_gap:.2f} dB")
    print(f"  >>> Last chunk gap: {last_gap:.2f} dB")
    if mean_gap > 3.0:
        print("  >>> AR error accumulation is SEVERE")
    elif mean_gap > 1.0:
        print("  >>> AR error accumulation is moderate")
    else:
        print("  >>> AR error accumulation is mild — problem is elsewhere")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WM Imagination Diagnostic Battery")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/vlaw/world_model/iter1_v5/checkpoint-3800.pt")
    parser.add_argument("--h5_path", type=str,
                        default="data/vlaw/encoded/train_v5/LiftPegUpright-v1/"
                                "LiftPegUpright-v1_real_1772643507.h5")
    parser.add_argument("--output_dir", type=str, default="results/vlaw/wm_diagnostic")
    parser.add_argument("--groups", nargs="+",
                        default=["A", "B", "C", "D3", "E", "F2", "D_MC"],
                        help="Which experiment groups to run")
    parser.add_argument("--max_samples", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load WM
    print(f"Loading WM from {args.checkpoint}...")
    wm = WMDiagnostic(args.checkpoint)
    print(f"WM loaded. num_history={wm.num_history}, num_frames={wm.num_frames}")

    # Load GT samples
    samples = load_gt_samples(
        args.h5_path,
        num_history=wm.num_history,
        num_frames=wm.num_frames,
        max_samples=args.max_samples,
    )
    if not samples:
        print("ERROR: No valid GT samples found!")
        return

    all_results = {}

    # Phase 1
    if "A" in args.groups:
        all_results["A"] = run_group_a(wm, samples, f"{args.output_dir}/group_a")
    if "B" in args.groups:
        all_results["B"] = run_group_b(wm, samples, f"{args.output_dir}/group_b")

    # Phase 2
    if "C" in args.groups:
        all_results["C"] = run_group_c(wm, samples, f"{args.output_dir}/group_c")
    if "D3" in args.groups:
        all_results["D3"] = run_group_d3(wm, samples, f"{args.output_dir}/group_d3")
    if "E" in args.groups:
        all_results["E"] = run_group_e(wm, samples, f"{args.output_dir}/group_e",
                                       args.h5_path)

    # Phase 4
    if "F2" in args.groups:
        all_results["F2"] = run_group_f2(wm, samples, f"{args.output_dir}/group_f2")

    # Phase 5: Multi-chunk AR
    if "D_MC" in args.groups:
        all_results["D_MC"] = run_group_d_multichunk(
            wm, args.h5_path, f"{args.output_dir}/group_d_multichunk")

    # Final report
    print("\n" + "=" * 60)
    print("DIAGNOSTIC BATTERY COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}/")

    save_metrics_json(all_results, f"{args.output_dir}/all_results.json")


if __name__ == "__main__":
    main()
