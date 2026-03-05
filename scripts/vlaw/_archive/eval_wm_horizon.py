#!/usr/bin/env python3
"""WM 多 horizon 评估脚本: pretrained vs Phase-A checkpoint.

对 LiftPegUpright 演示/Rollout 数据做 action replay 评估:
- 给定初始帧 + 真实动作序列 → WM 预测后续帧
- 自回归多步: 5 / 10 / 15 / 20 帧
- 指标: PSNR / SSIM / LPIPS (逐 horizon 分解)
- 可视化: GT vs Predicted 对比图

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_wm_horizon.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import einops
from PIL import Image, ImageDraw, ImageFont

# ---- Ctrl-World imports ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ctrl_world"))
from config import wm_args_maniskill
from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

# ---- Metrics ----
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

try:
    import lpips

    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    print("[WARN] lpips not installed, LPIPS metric will be skipped")


# =====================================================================
# Config
# =====================================================================
@dataclass
class EvalConfig:
    # ---- Data ----
    demo_h5: str = "data/vlaw/encoded/demos/LiftPegUpright-v1/LiftPegUpright-v1_demo_1771951465.h5"
    rollout_h5: str = "data/vlaw/encoded/rollouts/iter1/LiftPegUpright-v1/LiftPegUpright-v1_real_1772017887.h5"
    stat_path: str = "data/vlaw/meta_info/maniskill/stat.json"

    # ---- Checkpoints ----
    pretrained_ckpt: str = "checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt"
    phase_a_ckpt: str = "checkpoints/vlaw/world_model/phase_a/checkpoint-12000.pt"

    # ---- Model paths ----
    svd_model_path: str = "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"
    clip_model_path: str = "checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32"

    # ---- Eval params ----
    horizons: list[int] = field(default_factory=lambda: [5, 10, 15, 20])
    num_frames_per_step: int = 5  # WM predicts 5 frames per call
    num_history: int = 4
    max_trajs: int = 20  # max trajectories to evaluate
    vis_trajs: int = 5   # trajectories to visualize
    num_inference_steps: int = 50
    decode_chunk_size: int = 4

    # ---- Output ----
    output_dir: str = "results/vlaw/wm_baseline"
    report_path: str = "results/vlaw/wm_baseline_report.md"

    # ---- Device ----
    device: str = "cuda:0"


# =====================================================================
# Data loading
# =====================================================================
def load_norm_stats(stat_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(stat_path) as f:
        stat = json.load(f)
    p01 = np.array(stat["state_01"], dtype=np.float32)[None, :]
    p99 = np.array(stat["state_99"], dtype=np.float32)[None, :]
    return p01, p99


def normalize_action(action: np.ndarray, p01: np.ndarray, p99: np.ndarray) -> np.ndarray:
    eps = 1e-8
    ndata = 2.0 * (action - p01) / (p99 - p01 + eps) - 1.0
    return np.clip(ndata, -1.0, 1.0)


def load_trajectories(
    h5_path: str,
    min_length: int = 9,
    max_trajs: int = 50,
) -> list[dict]:
    """Load trajectories from HDF5, filtering by minimum length."""
    trajs = []
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        for key in traj_keys:
            grp = f[key]
            T = grp["latent_concat"].shape[0]
            if T < min_length:
                continue
            trajs.append({
                "key": key,
                "latent": torch.from_numpy(grp["latent_concat"][:].astype(np.float32)),  # (T,4,48,24)
                "actions": grp["actions"][:].astype(np.float32),  # (T,7)
                "length": T,
                "text": grp.attrs.get("task_instruction", "lift the peg upright"),
            })
            if len(trajs) >= max_trajs:
                break
    return trajs


# =====================================================================
# Model loading
# =====================================================================
def load_model(
    ckpt_path: str,
    svd_model_path: str,
    clip_model_path: str,
    device: str = "cuda:0",
) -> CrtlWorld:
    """Load Ctrl-World model from checkpoint."""
    args = wm_args_maniskill()
    args.svd_model_path = svd_model_path
    args.clip_model_path = clip_model_path
    args.ckpt_path = None  # Don't load during init

    model = CrtlWorld(args)
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# =====================================================================
# Inference: single-step prediction
# =====================================================================
@torch.no_grad()
def predict_one_step(
    model: CrtlWorld,
    history_latents: torch.Tensor,   # (1, num_history, 4, 48, 24)
    current_latent: torch.Tensor,    # (1, 4, 48, 24)
    actions: torch.Tensor,           # (1, num_history+num_frames, 7)
    text: str,
    args: wm_args_maniskill,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Run one WM prediction step, return predicted latents (1, num_frames, 4, 48, 24)."""
    pipeline = model.pipeline

    # Action encoding
    action_latent = model.action_encoder(
        actions, [text], model.tokenizer, model.text_encoder,
        args.frame_level_cond,
    )

    _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
        pipeline,
        image=current_latent,
        text=action_latent,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        history=history_latents,
        num_inference_steps=args.num_inference_steps,
        decode_chunk_size=args.decode_chunk_size,
        max_guidance_scale=args.guidance_scale,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        mask=None,
        output_type="latent",
        return_dict=False,
        frame_level_cond=args.frame_level_cond,
        his_cond_zero=args.his_cond_zero,
    )
    # pred_latents: (1, num_frames, 4, 48, 24)
    return pred_latents


# =====================================================================
# Autoregressive rollout
# =====================================================================
@torch.no_grad()
def autoregressive_rollout(
    model: CrtlWorld,
    traj: dict,
    max_horizon: int,
    p01: np.ndarray,
    p99: np.ndarray,
    args: wm_args_maniskill,
    device: str = "cuda:0",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run autoregressive rollout over a trajectory.
    
    Returns:
        pred_latents: (total_predicted_frames, 4, 48, 24)
        gt_latents:   (total_predicted_frames, 4, 48, 24)
    """
    num_h = args.num_history
    num_f = args.num_frames
    window = num_h + num_f

    full_latent = traj["latent"].to(device)  # (T, 4, 48, 24)
    full_actions = traj["actions"]  # (T, 7)
    text = traj["text"]
    T = full_latent.shape[0]

    # Maximum frames we can predict
    avail_future = T - num_h  # first num_h frames are initial history
    actual_horizon = min(max_horizon, avail_future)
    num_steps = (actual_horizon + num_f - 1) // num_f  # ceil division

    all_pred = []
    all_gt = []

    # Initial history from GT
    history = full_latent[:num_h].unsqueeze(0)  # (1, num_h, 4, 48, 24)

    for step_i in range(num_steps):
        start_frame = num_h + step_i * num_f
        end_frame = min(start_frame + num_f, T)
        if start_frame >= T:
            break

        # Current frame = first frame of prediction window
        current = full_latent[start_frame].unsqueeze(0)  # (1, 4, 48, 24)

        # Actions: need window of (num_h + num_f) actions
        # For action replay, we use GT actions
        act_start = start_frame - num_h
        act_end = act_start + window
        if act_end > T:
            # Pad with zeros
            act = np.zeros((window, 7), dtype=np.float32)
            avail = T - act_start
            act[:avail] = full_actions[act_start:T]
        else:
            act = full_actions[act_start:act_end]

        # Normalize actions
        act_norm = normalize_action(act, p01, p99)
        act_tensor = torch.tensor(act_norm, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict
        pred = predict_one_step(
            model, history, current, act_tensor, text, args, device,
        )  # (1, num_f, 4, 48, 24)

        actual_frames = end_frame - start_frame
        all_pred.append(pred[0, :actual_frames].cpu())
        all_gt.append(full_latent[start_frame:end_frame].cpu())

        # Update history for next step: use PREDICTED frames for autoregressive
        if pred.shape[1] >= num_h:
            history = pred[:, -num_h:]
        else:
            # Concatenate tail of old history with prediction
            keep = num_h - pred.shape[1]
            history = torch.cat([history[:, keep:], pred], dim=1)

    if all_pred:
        pred_latents = torch.cat(all_pred, dim=0)
        gt_latents = torch.cat(all_gt, dim=0)
    else:
        pred_latents = torch.empty(0, 4, 48, 24)
        gt_latents = torch.empty(0, 4, 48, 24)

    return pred_latents, gt_latents


# =====================================================================
# Decode latents to images
# =====================================================================
@torch.no_grad()
def decode_latents_to_images(
    latents: torch.Tensor,  # (N, 4, 48, 24)
    vae,
    device: str = "cuda:0",
    chunk_size: int = 4,
) -> np.ndarray:
    """Decode VAE latents back to images. Returns (N, H, W, 3) uint8."""
    N = latents.shape[0]
    all_imgs = []
    for i in range(0, N, chunk_size):
        chunk = latents[i : i + chunk_size].to(device) / vae.config.scaling_factor
        decoded = vae.decode(chunk, num_frames=chunk.shape[0]).sample
        # decoded: (chunk, 3, H, W)
        imgs = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255).cpu().numpy()
        imgs = imgs.transpose(0, 2, 3, 1).astype(np.uint8)  # (chunk, H, W, 3)
        all_imgs.append(imgs)
    return np.concatenate(all_imgs, axis=0)


# =====================================================================
# Metrics computation
# =====================================================================
def compute_metrics_per_frame(
    pred_images: np.ndarray,  # (N, H, W, 3)
    gt_images: np.ndarray,    # (N, H, W, 3)
    lpips_model=None,
) -> dict:
    """Compute PSNR, SSIM, LPIPS for each frame. Return per-frame dict."""
    N = pred_images.shape[0]
    psnrs, ssims, lpips_vals = [], [], []

    for i in range(N):
        p = pred_images[i]
        g = gt_images[i]

        psnr = compute_psnr(g, p, data_range=255)
        ssim = compute_ssim(g, p, data_range=255, channel_axis=2, win_size=7)
        psnrs.append(psnr)
        ssims.append(ssim)

        if lpips_model is not None:
            # Convert to [-1, 1] tensor
            p_t = torch.from_numpy(p).permute(2, 0, 1).float() / 127.5 - 1.0
            g_t = torch.from_numpy(g).permute(2, 0, 1).float() / 127.5 - 1.0
            with torch.no_grad():
                lp = lpips_model(
                    p_t.unsqueeze(0).cuda(), g_t.unsqueeze(0).cuda()
                ).item()
            lpips_vals.append(lp)

    result = {
        "psnr": psnrs,
        "ssim": ssims,
    }
    if lpips_vals:
        result["lpips"] = lpips_vals
    return result


def aggregate_by_horizon(
    all_frame_metrics: list[dict],
    horizons: list[int],
) -> dict:
    """Aggregate per-frame metrics into per-horizon means.
    
    all_frame_metrics: list of per-trajectory dicts with 'psnr', 'ssim', 'lpips' lists.
    Returns: {horizon: {metric: mean_value}}
    """
    result = {}
    for h in horizons:
        psnrs, ssims, lpipss = [], [], []
        for fm in all_frame_metrics:
            n_frames = len(fm["psnr"])
            end = min(h, n_frames)
            psnrs.extend(fm["psnr"][:end])
            ssims.extend(fm["ssim"][:end])
            if "lpips" in fm:
                lpipss.extend(fm["lpips"][:end])

        result[h] = {
            "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
            "ssim": float(np.mean(ssims)) if ssims else 0.0,
            "n_frames": len(psnrs),
        }
        if lpipss:
            result[h]["lpips"] = float(np.mean(lpipss))
    return result


def per_step_metrics(all_frame_metrics: list[dict], max_frames: int = 20) -> dict:
    """Compute per-step (individual frame) mean metrics across trajectories."""
    result = {}
    for t in range(max_frames):
        psnrs, ssims, lpipss = [], [], []
        for fm in all_frame_metrics:
            if t < len(fm["psnr"]):
                psnrs.append(fm["psnr"][t])
                ssims.append(fm["ssim"][t])
                if "lpips" in fm and t < len(fm["lpips"]):
                    lpipss.append(fm["lpips"][t])
        if psnrs:
            result[t + 1] = {
                "psnr": float(np.mean(psnrs)),
                "ssim": float(np.mean(ssims)),
                "n_trajs": len(psnrs),
            }
            if lpipss:
                result[t + 1]["lpips"] = float(np.mean(lpipss))
    return result


# =====================================================================
# Visualization
# =====================================================================
def save_comparison_grid(
    pred_images: np.ndarray,  # (N, H, W, 3)
    gt_images: np.ndarray,    # (N, H, W, 3)
    save_path: str,
    traj_key: str,
    max_frames: int = 20,
    n_cams: int = 2,
) -> None:
    """Save a GT vs Predicted comparison grid image.
    
    Images are vertically concatenated (2 cameras), so we split them for display.
    Grid: rows = {GT_cam0, Pred_cam0, GT_cam1, Pred_cam1}, cols = frames.
    """
    N = min(pred_images.shape[0], gt_images.shape[0], max_frames)
    if N == 0:
        return

    H, W = pred_images.shape[1], pred_images.shape[2]
    cam_h = H // n_cams  # Height per camera

    # Select frames to display (evenly spaced if too many)
    if N > 10:
        indices = np.linspace(0, N - 1, 10, dtype=int)
    else:
        indices = np.arange(N)
    n_cols = len(indices)

    # Grid layout: 4 rows (GT_cam0, Pred_cam0, GT_cam1, Pred_cam1) x n_cols
    cell_w = W
    cell_h = cam_h
    margin = 2
    label_h = 20

    grid_w = n_cols * (cell_w + margin) + margin
    grid_h = (n_cams * 2) * (cell_h + margin) + margin + label_h
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240  # light gray bg

    row_labels = []
    for cam_i in range(n_cams):
        row_labels.append(f"GT cam{cam_i}")
        row_labels.append(f"Pred cam{cam_i}")

    for col, frame_idx in enumerate(indices):
        x = margin + col * (cell_w + margin)
        for cam_i in range(n_cams):
            y_start = cam_i * cam_h
            y_end = y_start + cam_h

            # GT row
            row_gt = cam_i * 2
            y_grid = margin + label_h + row_gt * (cell_h + margin)
            grid[y_grid : y_grid + cell_h, x : x + cell_w] = gt_images[frame_idx, y_start:y_end, :, :]

            # Pred row
            row_pred = cam_i * 2 + 1
            y_grid = margin + label_h + row_pred * (cell_h + margin)
            grid[y_grid : y_grid + cell_h, x : x + cell_w] = pred_images[frame_idx, y_start:y_end, :, :]

    img = Image.fromarray(grid)
    draw = ImageDraw.Draw(img)
    # Add frame index labels at top
    for col, frame_idx in enumerate(indices):
        x = margin + col * (cell_w + margin) + cell_w // 4
        draw.text((x, 2), f"t={frame_idx + 1}", fill=(0, 0, 0))

    # Add row labels on the left (encoded in filename since PIL font may not be available)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    print(f"  Saved: {save_path}")


# =====================================================================
# Main evaluation
# =====================================================================
def evaluate_checkpoint(
    ckpt_path: str,
    ckpt_name: str,
    trajs: list[dict],
    p01: np.ndarray,
    p99: np.ndarray,
    cfg: EvalConfig,
    lpips_model=None,
    save_vis: bool = True,
) -> dict:
    """Evaluate a single checkpoint on all trajectories."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {ckpt_name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Trajectories: {len(trajs)}")
    print(f"{'='*60}")

    # Load model
    model = load_model(ckpt_path, cfg.svd_model_path, cfg.clip_model_path, cfg.device)
    args = wm_args_maniskill()
    args.num_inference_steps = cfg.num_inference_steps
    args.decode_chunk_size = cfg.decode_chunk_size

    max_horizon = max(cfg.horizons)
    all_frame_metrics = []
    vis_count = 0

    for i, traj in enumerate(trajs):
        t0 = time.time()
        print(f"  [{i+1}/{len(trajs)}] {traj['key']} (len={traj['length']})...", end=" ", flush=True)

        pred_latents, gt_latents = autoregressive_rollout(
            model, traj, max_horizon, p01, p99, args, cfg.device,
        )

        if pred_latents.shape[0] == 0:
            print("SKIP (too short)")
            continue

        # Decode to images
        pred_images = decode_latents_to_images(
            pred_latents, model.vae, cfg.device, cfg.decode_chunk_size,
        )
        gt_images = decode_latents_to_images(
            gt_latents, model.vae, cfg.device, cfg.decode_chunk_size,
        )

        # Compute metrics
        fm = compute_metrics_per_frame(pred_images, gt_images, lpips_model)
        all_frame_metrics.append(fm)
        elapsed = time.time() - t0

        # Quick summary
        n = len(fm["psnr"])
        avg_psnr = np.mean(fm["psnr"])
        avg_ssim = np.mean(fm["ssim"])
        print(f"{n} frames, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, {elapsed:.1f}s")

        # Save visualization
        if save_vis and vis_count < cfg.vis_trajs:
            vis_path = os.path.join(cfg.output_dir, ckpt_name, f"vis_{traj['key']}.png")
            save_comparison_grid(pred_images, gt_images, vis_path, traj["key"], max_frames=max_horizon)
            vis_count += 1

    # Aggregate
    horizon_metrics = aggregate_by_horizon(all_frame_metrics, cfg.horizons)
    step_metrics = per_step_metrics(all_frame_metrics, max_horizon)

    result = {
        "ckpt_name": ckpt_name,
        "ckpt_path": ckpt_path,
        "n_trajs": len(all_frame_metrics),
        "horizons": horizon_metrics,
        "per_step": step_metrics,
    }

    # Print summary
    print(f"\n  {ckpt_name} Summary:")
    print(f"  {'Horizon':<10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'#frames':>8}")
    for h in cfg.horizons:
        m = horizon_metrics[h]
        lp = f"{m.get('lpips', -1):.4f}" if "lpips" in m else "N/A"
        print(f"  {h:<10} {m['psnr']:>8.2f} {m['ssim']:>8.4f} {lp:>8} {m['n_frames']:>8}")

    # Cleanup model
    del model
    torch.cuda.empty_cache()

    return result


def generate_report(
    results: dict,
    cfg: EvalConfig,
) -> str:
    """Generate markdown report."""
    lines = [
        "# WM Horizon Baseline Report",
        "",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> Task: LiftPegUpright-v1",
        f"> Data: {cfg.demo_h5} + {cfg.rollout_h5}",
        "",
        "## 概述",
        "",
        "本报告评估 Ctrl-World 世界模型在 LiftPegUpright 数据上的视频预测质量，",
        "按不同 horizon (5/10/15/20 帧) 分解指标，建立后续迭代对照基准。",
        "",
        "## Checkpoints",
        "",
    ]

    for name, r in results.items():
        lines.append(f"- **{name}**: `{r['ckpt_path']}` ({r['n_trajs']} trajs)")
    lines.append("")

    # Comparison table
    lines.extend([
        "## Horizon 分解对比",
        "",
        "| Horizon | " + " | ".join(f"{name} PSNR" for name in results) + " | " +
        " | ".join(f"{name} SSIM" for name in results) + " | " +
        " | ".join(f"{name} LPIPS" for name in results) + " |",
        "| ------- | " + " | ".join("------" for _ in results) + " | " +
        " | ".join("------" for _ in results) + " | " +
        " | ".join("------" for _ in results) + " |",
    ])

    for h in cfg.horizons:
        row = f"| {h} |"
        for name, r in results.items():
            m = r["horizons"][h]
            row += f" {m['psnr']:.2f} |"
        for name, r in results.items():
            m = r["horizons"][h]
            row += f" {m['ssim']:.4f} |"
        for name, r in results.items():
            m = r["horizons"][h]
            lp = f"{m['lpips']:.4f}" if "lpips" in m else "N/A"
            row += f" {lp} |"
        lines.append(row)
    lines.append("")

    # Per-step decay curve (text)
    lines.extend([
        "## 逐帧衰减曲线",
        "",
        "| Frame |",
    ])
    header = "| Frame |"
    sep = "| ----- |"
    for name in results:
        header += f" {name} PSNR | {name} SSIM |"
        sep += " ------ | ------ |"
    lines[-1] = header
    lines.append(sep)

    max_step = max(max(int(s) for s in r["per_step"]) for r in results.values() if r["per_step"])
    for t in range(1, max_step + 1):
        row = f"| {t} |"
        for name, r in results.items():
            if t in r["per_step"]:
                m = r["per_step"][t]
                row += f" {m['psnr']:.2f} | {m['ssim']:.4f} |"
            else:
                row += " - | - |"
        lines.append(row)
    lines.append("")

    # Delta analysis
    if len(results) >= 2:
        names = list(results.keys())
        lines.extend([
            "## 对比分析",
            "",
            f"### {names[1]} vs {names[0]} (Delta)",
            "",
            "| Horizon | ΔPSNR | ΔSSIM | ΔLPIPS |",
            "| ------- | ----- | ----- | ------ |",
        ])
        for h in cfg.horizons:
            m0 = results[names[0]]["horizons"][h]
            m1 = results[names[1]]["horizons"][h]
            dp = m1["psnr"] - m0["psnr"]
            ds = m1["ssim"] - m0["ssim"]
            if "lpips" in m0 and "lpips" in m1:
                dl = m1["lpips"] - m0["lpips"]
                dl_str = f"{dl:+.4f}"
            else:
                dl_str = "N/A"
            lines.append(f"| {h} | {dp:+.2f} | {ds:+.4f} | {dl_str} |")
        lines.append("")

    # Visualization links
    lines.extend([
        "## 可视化",
        "",
        f"GT vs Predicted 对比图保存在: `{cfg.output_dir}/`",
        "",
    ])
    for name in results:
        vis_dir = os.path.join(cfg.output_dir, name)
        if os.path.exists(vis_dir):
            vis_files = sorted(os.listdir(vis_dir))
            lines.append(f"### {name}")
            for vf in vis_files:
                lines.append(f"- `{name}/{vf}`")
            lines.append("")

    # Conclusion
    lines.extend([
        "## 结论",
        "",
        "- PSNR > 18 为通过标准 (P2.3)",
    ])
    for name, r in results.items():
        h5 = r["horizons"][5]
        h20 = r["horizons"].get(20, r["horizons"].get(15, h5))
        decay = h5["psnr"] - h20["psnr"]
        lines.append(
            f"- **{name}**: horizon-5 PSNR={h5['psnr']:.2f}, "
            f"horizon-{list(r['horizons'].keys())[-1]} PSNR={h20['psnr']:.2f}, "
            f"衰减={decay:.2f}dB"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    cfg = EvalConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("WM Multi-Horizon Evaluation")
    print("=" * 60)

    # Load normalization stats
    p01, p99 = load_norm_stats(cfg.stat_path)
    print(f"Loaded norm stats from {cfg.stat_path}")

    # Load LPIPS model
    lpips_model = None
    if _LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net="alex").cuda()
            print("LPIPS model loaded")
        except Exception as e:
            print(f"Failed to load LPIPS: {e}")

    # Load trajectories from both demo and rollout data
    # Use rollout data primarily (longer trajectories for multi-horizon)
    print(f"\nLoading rollout data: {cfg.rollout_h5}")
    rollout_trajs = load_trajectories(
        cfg.rollout_h5, min_length=9, max_trajs=cfg.max_trajs,
    )
    print(f"  Loaded {len(rollout_trajs)} rollout trajectories")

    print(f"\nLoading demo data: {cfg.demo_h5}")
    demo_trajs = load_trajectories(
        cfg.demo_h5, min_length=9, max_trajs=10,
    )
    print(f"  Loaded {len(demo_trajs)} demo trajectories")

    # Combine: use rollout (longer) for horizon evaluation
    trajs = rollout_trajs
    if demo_trajs:
        # Add some demo trajs for comparison
        trajs = trajs + demo_trajs[:5]
    print(f"\nTotal evaluation trajectories: {len(trajs)}")

    # Evaluate both checkpoints
    results = {}

    # 1. Pretrained
    results["pretrained"] = evaluate_checkpoint(
        cfg.pretrained_ckpt, "pretrained", trajs, p01, p99, cfg, lpips_model,
    )

    # 2. Phase-A
    results["phase_a_step12000"] = evaluate_checkpoint(
        cfg.phase_a_ckpt, "phase_a_step12000", trajs, p01, p99, cfg, lpips_model,
    )

    # Generate report
    report = generate_report(results, cfg)
    os.makedirs(os.path.dirname(cfg.report_path), exist_ok=True)
    with open(cfg.report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {cfg.report_path}")

    # Save raw results as JSON
    json_path = os.path.join(cfg.output_dir, "horizon_metrics.json")
    # Convert int keys to str for JSON
    json_results = {}
    for name, r in results.items():
        jr = {
            "ckpt_name": r["ckpt_name"],
            "ckpt_path": r["ckpt_path"],
            "n_trajs": r["n_trajs"],
            "horizons": {str(k): v for k, v in r["horizons"].items()},
            "per_step": {str(k): v for k, v in r["per_step"].items()},
        }
        json_results[name] = jr
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Raw metrics saved to: {json_path}")


if __name__ == "__main__":
    main()
