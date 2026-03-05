#!/usr/bin/env python3
"""Standard WM evaluation script using fixed eval set (eval_fixed/).

Evaluates one or more WM checkpoints against the fixed evaluation set,
always including the pretrained model as a baseline. Outputs standardized
JSON + Markdown report.

Usage:
    # Evaluate pretrained only:
    CUDA_VISIBLE_DEVICES=0 conda run -n ctrl_world python scripts/vlaw/eval/eval_wm_standard.py

    # Evaluate specific checkpoints:
    CUDA_VISIBLE_DEVICES=0 conda run -n ctrl_world python scripts/vlaw/eval/eval_wm_standard.py \
        --checkpoint-paths "checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt,checkpoints/vlaw/world_model/ablation_optimal_steps/checkpoint-1000.pt"

    # With tyro:
    CUDA_VISIBLE_DEVICES=0 conda run -n ctrl_world python scripts/vlaw/eval/eval_wm_standard.py \
        --eval-h5-path data/vlaw/encoded/eval_fixed/eval_set.h5 \
        --num-frames-per-step 5
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

# ---- Ctrl-World imports ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "ctrl_world"))
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

try:
    import tyro

    _TYRO_AVAILABLE = True
except ImportError:
    _TYRO_AVAILABLE = False


# =====================================================================
# Config
# =====================================================================
@dataclass
class EvalStandardConfig:
    """Standard WM evaluation config using fixed eval set."""

    # ---- Data ----
    eval_h5_path: str = "data/vlaw/encoded/eval_fixed/eval_set.h5"
    """Path to the fixed evaluation H5 file."""

    stat_path: str = "data/vlaw/meta_info/maniskill/stat.json"
    """Path to action normalization statistics."""

    # ---- Checkpoints to evaluate (comma-separated) ----
    checkpoint_paths: str = ""
    """Comma-separated list of checkpoint paths to evaluate. Empty = pretrained only."""

    pretrained_path: str = (
        "checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt"
    )
    """Path to pretrained checkpoint (always included as baseline)."""

    # ---- Model dependencies ----
    svd_model_path: str = (
        "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"
    )
    clip_model_path: str = (
        "checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32"
    )

    # ---- Eval params ----
    num_frames_per_step: int = 5
    """Number of frames to predict per step (should match WM training num_frames)."""

    num_history: int = 6
    """Number of history frames for conditioning."""

    num_inference_steps: int = 50
    """Diffusion inference steps."""

    decode_chunk_size: int = 4
    """Batch size for VAE decoding."""

    min_traj_length: int = 9
    """Minimum trajectory length to include in evaluation."""

    # ---- Output ----
    output_dir: str = "results/vlaw/wm_eval"
    """Output directory for results."""

    # ---- Device ----
    device: str = "cuda:0"


# =====================================================================
# Data loading
# =====================================================================
def load_norm_stats(stat_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load action normalization statistics."""
    with open(stat_path) as f:
        stat = json.load(f)
    p01 = np.array(stat["state_01"], dtype=np.float32)[None, :]
    p99 = np.array(stat["state_99"], dtype=np.float32)[None, :]
    return p01, p99


def normalize_action(
    action: np.ndarray, p01: np.ndarray, p99: np.ndarray
) -> np.ndarray:
    """Normalize actions to [-1, 1] using percentile stats."""
    eps = 1e-8
    ndata = 2.0 * (action - p01) / (p99 - p01 + eps) - 1.0
    return np.clip(ndata, -1.0, 1.0)


def load_eval_trajectories(
    h5_path: str,
    min_length: int = 9,
) -> list[dict]:
    """Load all trajectories from the fixed eval H5 (no splitting)."""
    trajs: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        print(f"[VLAW] Loading eval set: {h5_path} ({len(traj_keys)} trajectories)")

        for key in traj_keys:
            grp = f[key]
            if "latent_concat" not in grp:
                print(f"  SKIP {key}: no latent_concat")
                continue
            T = grp["latent_concat"].shape[0]
            if T < min_length:
                print(f"  SKIP {key}: T={T} < min_length={min_length}")
                continue
            trajs.append(
                {
                    "key": key,
                    "latent": torch.from_numpy(
                        grp["latent_concat"][:].astype(np.float32)
                    ),
                    "actions": grp["actions"][:].astype(np.float32),
                    "length": T,
                    "text": grp.attrs.get("task_instruction", "lift the peg upright"),
                    "source": grp.attrs.get("original_source", "unknown"),
                }
            )

    print(f"[VLAW] Loaded {len(trajs)} eval trajectories (min_length={min_length})")
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
    """Load a Ctrl-World model from checkpoint."""
    args = wm_args_maniskill()
    args.svd_model_path = svd_model_path
    args.clip_model_path = clip_model_path
    args.ckpt_path = None

    model = CrtlWorld(args)
    print(f"[VLAW] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# =====================================================================
# Inference (single-step, non-autoregressive)
# =====================================================================
@torch.no_grad()
def predict_one_step(
    model: CrtlWorld,
    history_latents: torch.Tensor,
    current_latent: torch.Tensor,
    actions: torch.Tensor,
    text: str,
    args,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Run one-step WM prediction."""
    pipeline = model.pipeline
    action_latent = model.action_encoder(
        actions,
        [text],
        model.tokenizer,
        model.text_encoder,
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
    return pred_latents


@torch.no_grad()
def single_step_predict(
    model: CrtlWorld,
    traj: dict,
    p01: np.ndarray,
    p99: np.ndarray,
    args,
    device: str = "cuda:0",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step (non-autoregressive) prediction.

    Uses first num_history frames as history, predicts the next num_frames frames.
    """
    num_h = args.num_history
    num_f = args.num_frames
    window = num_h + num_f

    full_latent = traj["latent"].to(device)
    full_actions = traj["actions"]
    text = traj["text"]
    T = full_latent.shape[0]

    if T < num_h + num_f:
        return torch.empty(0, 4, 48, 24), torch.empty(0, 4, 48, 24)

    # History: frames 0..num_h-1
    history = full_latent[:num_h].unsqueeze(0)
    # Current: frame num_h (first frame to predict from)
    current = full_latent[num_h].unsqueeze(0)

    # Actions for the entire window
    act = full_actions[:window]
    act_norm = normalize_action(act, p01, p99)
    act_tensor = (
        torch.tensor(act_norm, dtype=torch.float32).unsqueeze(0).to(device)
    )

    pred = predict_one_step(model, history, current, act_tensor, text, args, device)

    # GT: frames num_h .. num_h+num_f-1
    end = min(num_h + num_f, T)
    gt = full_latent[num_h:end].cpu()
    pred_out = pred[0, : end - num_h].cpu()

    return pred_out, gt


# =====================================================================
# Decode latents to images
# =====================================================================
@torch.no_grad()
def decode_latents_to_images(
    latents: torch.Tensor,
    vae,
    device: str = "cuda:0",
    chunk_size: int = 4,
) -> np.ndarray:
    """Decode VAE latents to uint8 images."""
    N = latents.shape[0]
    all_imgs: list[np.ndarray] = []
    for i in range(0, N, chunk_size):
        chunk = latents[i : i + chunk_size].to(device) / vae.config.scaling_factor
        decoded = vae.decode(chunk, num_frames=chunk.shape[0]).sample
        imgs = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255).cpu().numpy()
        imgs = imgs.transpose(0, 2, 3, 1).astype(np.uint8)
        all_imgs.append(imgs)
    return np.concatenate(all_imgs, axis=0)


# =====================================================================
# Metrics
# =====================================================================
def compute_metrics_per_frame(
    pred_images: np.ndarray,
    gt_images: np.ndarray,
    lpips_model=None,
) -> dict:
    """Compute PSNR, SSIM, LPIPS per frame."""
    N = pred_images.shape[0]
    psnrs: list[float] = []
    ssims: list[float] = []
    lpips_vals: list[float] = []

    for i in range(N):
        p = pred_images[i]
        g = gt_images[i]

        psnr = compute_psnr(g, p, data_range=255)
        ssim = compute_ssim(g, p, data_range=255, channel_axis=2, win_size=7)
        psnrs.append(float(psnr))
        ssims.append(float(ssim))

        if lpips_model is not None:
            p_t = torch.from_numpy(p).permute(2, 0, 1).float() / 127.5 - 1.0
            g_t = torch.from_numpy(g).permute(2, 0, 1).float() / 127.5 - 1.0
            with torch.no_grad():
                lp = lpips_model(
                    p_t.unsqueeze(0).cuda(), g_t.unsqueeze(0).cuda()
                ).item()
            lpips_vals.append(lp)

    result: dict = {"psnr": psnrs, "ssim": ssims}
    if lpips_vals:
        result["lpips"] = lpips_vals
    return result


# =====================================================================
# Evaluate one checkpoint
# =====================================================================
def evaluate_checkpoint(
    ckpt_path: str,
    ckpt_label: str,
    trajs: list[dict],
    p01: np.ndarray,
    p99: np.ndarray,
    cfg: EvalStandardConfig,
    lpips_model=None,
) -> dict:
    """Evaluate a single checkpoint against the eval set."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {ckpt_label}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Trajectories: {len(trajs)}")
    print(f"{'=' * 60}")

    model = load_model(
        ckpt_path, cfg.svd_model_path, cfg.clip_model_path, cfg.device
    )
    args = wm_args_maniskill()
    args.num_frames = cfg.num_frames_per_step
    args.num_history = cfg.num_history
    args.num_inference_steps = cfg.num_inference_steps
    args.decode_chunk_size = cfg.decode_chunk_size

    all_psnrs: list[float] = []
    all_ssims: list[float] = []
    all_lpips: list[float] = []
    per_frame_psnrs: dict[int, list[float]] = {}  # frame_idx -> [psnr values]
    per_frame_ssims: dict[int, list[float]] = {}
    per_frame_lpips: dict[int, list[float]] = {}
    per_traj_results: list[dict] = []

    for i, traj in enumerate(trajs):
        t0 = time.time()
        print(
            f"  [{i + 1}/{len(trajs)}] {traj['key']} (len={traj['length']}, src={traj['source']})...",
            end=" ",
            flush=True,
        )

        pred_latents, gt_latents = single_step_predict(
            model, traj, p01, p99, args, cfg.device
        )

        if pred_latents.shape[0] == 0:
            print("SKIP (too short)")
            continue

        # Decode
        pred_images = decode_latents_to_images(
            pred_latents, model.vae, cfg.device, cfg.decode_chunk_size
        )
        gt_images = decode_latents_to_images(
            gt_latents, model.vae, cfg.device, cfg.decode_chunk_size
        )

        # Metrics
        fm = compute_metrics_per_frame(pred_images, gt_images, lpips_model)
        elapsed = time.time() - t0

        avg_psnr = np.mean(fm["psnr"])
        avg_ssim = np.mean(fm["ssim"])
        all_psnrs.extend(fm["psnr"])
        all_ssims.extend(fm["ssim"])
        if "lpips" in fm:
            all_lpips.extend(fm["lpips"])

        # Track per-frame temporal decay
        for fi, pv in enumerate(fm["psnr"]):
            per_frame_psnrs.setdefault(fi, []).append(pv)
        for fi, sv in enumerate(fm["ssim"]):
            per_frame_ssims.setdefault(fi, []).append(sv)
        if "lpips" in fm:
            for fi, lv in enumerate(fm["lpips"]):
                per_frame_lpips.setdefault(fi, []).append(lv)

        lp_str = (
            f", LPIPS={np.mean(fm['lpips']):.4f}" if "lpips" in fm else ""
        )
        per_traj_results.append(
            {
                "key": traj["key"],
                "source": traj["source"],
                "length": traj["length"],
                "psnr": float(avg_psnr),
                "ssim": float(avg_ssim),
                "lpips": float(np.mean(fm.get("lpips", [0]))),
                "n_frames": len(fm["psnr"]),
                "frame_psnrs": fm["psnr"],
                "frame_ssims": fm["ssim"],
            }
        )

        print(
            f"{len(fm['psnr'])}f, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}{lp_str}, {elapsed:.1f}s"
        )

    # Build temporal decay curve
    temporal_decay: list[dict] = []
    for fi in sorted(per_frame_psnrs.keys()):
        entry: dict = {
            "frame": fi,
            "psnr_mean": float(np.mean(per_frame_psnrs[fi])),
            "psnr_std": float(np.std(per_frame_psnrs[fi])),
            "ssim_mean": float(np.mean(per_frame_ssims.get(fi, [0]))),
            "ssim_std": float(np.std(per_frame_ssims.get(fi, [0]))),
            "n_samples": len(per_frame_psnrs[fi]),
        }
        if fi in per_frame_lpips:
            entry["lpips_mean"] = float(np.mean(per_frame_lpips[fi]))
            entry["lpips_std"] = float(np.std(per_frame_lpips[fi]))
        temporal_decay.append(entry)

    overall: dict = {
        "ckpt_label": ckpt_label,
        "ckpt_path": ckpt_path,
        "n_trajs": len(per_traj_results),
        "n_frames": len(all_psnrs),
        "psnr_mean": float(np.mean(all_psnrs)) if all_psnrs else 0.0,
        "psnr_std": float(np.std(all_psnrs)) if all_psnrs else 0.0,
        "ssim_mean": float(np.mean(all_ssims)) if all_ssims else 0.0,
        "ssim_std": float(np.std(all_ssims)) if all_ssims else 0.0,
        "temporal_decay": temporal_decay,
        "per_traj": per_traj_results,
    }
    if all_lpips:
        overall["lpips_mean"] = float(np.mean(all_lpips))
        overall["lpips_std"] = float(np.std(all_lpips))

    print(
        f"\n  {ckpt_label} Overall: PSNR={overall['psnr_mean']:.2f}±{overall['psnr_std']:.2f}, "
        f"SSIM={overall['ssim_mean']:.4f}±{overall['ssim_std']:.4f}, "
        f"{overall['n_trajs']} trajs, {overall['n_frames']} frames"
    )
    if all_lpips:
        print(f"  LPIPS={overall['lpips_mean']:.4f}±{overall['lpips_std']:.4f}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return overall


# =====================================================================
# Report generation
# =====================================================================
def generate_report(results: dict[str, dict], cfg: EvalStandardConfig) -> str:
    """Generate a Markdown report from evaluation results."""
    lines: list[str] = [
        "# WM 标准评估报告",
        "",
        f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> Task: LiftPegUpright-v1",
        f"> 评估集: `{cfg.eval_h5_path}` (固定集, 15 条轨迹)",
        f"> num_frames: {cfg.num_frames_per_step}, num_history: {cfg.num_history}",
        "",
    ]

    # Summary table
    labels = list(results.keys())
    header = ["指标"] + labels
    lines.append("## 总览")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["----"] * len(header)) + " |")

    # Get pretrained metrics as baseline for delta
    pretrained_key = next(
        (k for k in labels if "pretrained" in k.lower()), labels[0]
    )
    pretrained_psnr = results[pretrained_key]["psnr_mean"]

    # PSNR row
    row = ["PSNR"]
    for label in labels:
        r = results[label]
        dp = r["psnr_mean"] - pretrained_psnr
        delta_str = f" ({dp:+.2f})" if label != pretrained_key else ""
        row.append(f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}{delta_str}")
    lines.append("| " + " | ".join(row) + " |")

    # SSIM row
    row = ["SSIM"]
    for label in labels:
        r = results[label]
        row.append(f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}")
    lines.append("| " + " | ".join(row) + " |")

    # LPIPS row
    has_lpips = any("lpips_mean" in r for r in results.values())
    if has_lpips:
        row = ["LPIPS ↓"]
        for label in labels:
            r = results[label]
            if "lpips_mean" in r:
                row.append(f"{r['lpips_mean']:.4f} ± {r['lpips_std']:.4f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    # Frames & trajs
    row = ["#trajs"]
    for label in labels:
        row.append(str(results[label]["n_trajs"]))
    lines.append("| " + " | ".join(row) + " |")

    row = ["#frames"]
    for label in labels:
        row.append(str(results[label]["n_frames"]))
    lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Gate check
    lines.append("## 门控检查 (PSNR > 18.0)")
    lines.append("")
    for label in labels:
        r = results[label]
        gate_pass = r["psnr_mean"] >= 18.0
        emoji = "✅" if gate_pass else "❌"
        lines.append(
            f"- **{label}**: PSNR = {r['psnr_mean']:.2f} → {emoji} "
            f"{'通过' if gate_pass else '未通过'}"
        )
    lines.append("")

    # Temporal decay
    lines.append("## 帧级时序衰减曲线")
    lines.append("")
    lines.append("逐帧 (frame_0 → frame_N) 的预测质量衰减:")
    lines.append("")

    for label in labels:
        r = results[label]
        td = r.get("temporal_decay", [])
        if not td:
            continue
        lines.append(f"### {label}")
        lines.append("")
        td_header = ["Frame", "PSNR", "SSIM"]
        if any("lpips_mean" in d for d in td):
            td_header.append("LPIPS ↓")
        td_header.append("#samples")
        lines.append("| " + " | ".join(td_header) + " |")
        lines.append("| " + " | ".join(["----"] * len(td_header)) + " |")

        for d in td:
            row = [
                f"frame_{d['frame']}",
                f"{d['psnr_mean']:.2f} ± {d['psnr_std']:.2f}",
                f"{d['ssim_mean']:.4f} ± {d['ssim_std']:.4f}",
            ]
            if "lpips_mean" in d:
                row.append(f"{d['lpips_mean']:.4f} ± {d['lpips_std']:.4f}")
            elif any("lpips_mean" in dd for dd in td):
                row.append("N/A")
            row.append(str(d["n_samples"]))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Per-traj details
    lines.append("## 逐轨迹详情")
    lines.append("")
    for label in labels:
        r = results[label]
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Traj | Source | T | PSNR | SSIM | LPIPS | #frames |")
        lines.append("| ---- | ------ | - | ---- | ---- | ----- | ------- |")
        for tr in r["per_traj"]:
            lp = (
                f"{tr['lpips']:.4f}" if tr.get("lpips", 0) > 0 else "N/A"
            )
            lines.append(
                f"| {tr['key']} | {tr['source']} | {tr['length']} | "
                f"{tr['psnr']:.2f} | {tr['ssim']:.4f} | {lp} | {tr['n_frames']} |"
            )
        lines.append("")

    return "\n".join(lines)


# =====================================================================
# Main
# =====================================================================
def main(cfg: Optional[EvalStandardConfig] = None) -> None:
    """Run standard WM evaluation."""
    if cfg is None:
        if _TYRO_AVAILABLE:
            cfg = tyro.cli(EvalStandardConfig)
        else:
            cfg = EvalStandardConfig()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[VLAW] Standard WM Evaluation")
    print(f"  Eval set: {cfg.eval_h5_path}")
    print(f"  Pretrained: {cfg.pretrained_path}")
    print(f"  Checkpoints: {cfg.checkpoint_paths or '(none, pretrained only)'}")
    print(f"  num_frames: {cfg.num_frames_per_step}, num_history: {cfg.num_history}")
    print(f"  Output: {cfg.output_dir}")
    print("=" * 60)

    # Validate paths
    if not Path(cfg.eval_h5_path).exists():
        print(f"ERROR: Eval H5 not found: {cfg.eval_h5_path}")
        return
    if not Path(cfg.pretrained_path).exists():
        print(f"ERROR: Pretrained checkpoint not found: {cfg.pretrained_path}")
        return

    # Load norm stats
    p01, p99 = load_norm_stats(cfg.stat_path)
    print(f"[VLAW] Loaded norm stats from {cfg.stat_path}")

    # Load LPIPS
    lpips_model = None
    if _LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net="alex").cuda()
            print("[VLAW] LPIPS model loaded")
        except Exception as e:
            print(f"[VLAW] Failed to load LPIPS: {e}")

    # Load eval trajectories (no splitting — use all)
    trajs = load_eval_trajectories(cfg.eval_h5_path, min_length=cfg.min_traj_length)
    if not trajs:
        print("ERROR: No valid trajectories found!")
        return

    # Build checkpoint list: pretrained (baseline) + user-specified
    checkpoints: list[tuple[str, str]] = [
        ("pretrained", cfg.pretrained_path),
    ]
    if cfg.checkpoint_paths.strip():
        for p in cfg.checkpoint_paths.split(","):
            p = p.strip()
            if not p:
                continue
            if not Path(p).exists():
                print(f"WARNING: Checkpoint not found, skipping: {p}")
                continue
            # Generate label from path
            label = Path(p).stem
            parent = Path(p).parent.name
            if parent and parent != ".":
                label = f"{parent}/{label}"
            checkpoints.append((label, p))

    # Evaluate each checkpoint
    results: dict[str, dict] = {}
    for label, ckpt_path in checkpoints:
        res = evaluate_checkpoint(
            ckpt_path, label, trajs, p01, p99, cfg, lpips_model
        )
        results[label] = res

        # Save per-checkpoint JSON
        safe_label = label.replace("/", "_").replace(" ", "_")
        json_path = output_dir / f"metrics_{safe_label}.json"
        with open(json_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Saved: {json_path}")

    if not results:
        print("ERROR: No checkpoints evaluated!")
        return

    # Generate report
    report = generate_report(results, cfg)
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n[VLAW] Report saved: {report_path}")

    # Save combined JSON (without per_traj for summary)
    summary: dict = {}
    for label, res in results.items():
        summary[label] = {
            k: v for k, v in res.items() if k not in ("per_traj", "temporal_decay")
        }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[VLAW] Summary saved: {summary_path}")

    # Save full results JSON (with per_traj + temporal_decay)
    full_path = output_dir / "full_results.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[VLAW] Full results saved: {full_path}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Label':<30} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10} {'#trajs':>8} {'Gate':>6}"
    )
    print(f"{'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 6}")

    for label in results:
        r = results[label]
        gate = "PASS" if r["psnr_mean"] >= 18.0 else "FAIL"
        lp = f"{r['lpips_mean']:.4f}" if "lpips_mean" in r else "N/A"
        print(
            f"{label:<30} {r['psnr_mean']:>10.2f} {r['ssim_mean']:>10.4f} "
            f"{lp:>10} {r['n_trajs']:>8} {gate:>6}"
        )

    print(f"{'=' * 80}")
    all_pass = all(r["psnr_mean"] >= 18.0 for r in results.values())
    print(f"\nOverall gate: {'✅ ALL PASS' if all_pass else '❌ SOME FAIL'}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
