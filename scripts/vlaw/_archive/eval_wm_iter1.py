#!/usr/bin/env python3
"""WM iter1 质量验证: 计算 PSNR/SSIM/LPIPS，对比 pretrained baseline.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n ctrl_world python scripts/eval_wm_iter1.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch

# ---- Ctrl-World imports ----
# VLAW MODIFICATION: fix path — script moved from scripts/ to scripts/vlaw/eval/
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
    iter1_ckpt: str = "checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt"

    # ---- Model paths ----
    svd_model_path: str = "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"
    clip_model_path: str = "checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32"

    # ---- Eval params ----
    horizons: list[int] = field(default_factory=lambda: [5])
    num_frames_per_step: int = 5
    num_history: int = 4
    max_trajs: int = 20
    vis_trajs: int = 3
    num_inference_steps: int = 50
    decode_chunk_size: int = 4

    # ---- Output ----
    output_dir: str = "results/vlaw/wm_iter1_eval"
    report_path: str = "results/vlaw/wm_iter1_eval_report.md"

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
    val_split_ratio: float = 0.2,
    use_val: bool = True,
) -> list[dict]:
    """Load trajectories from HDF5. If use_val=True, take the last val_split_ratio fraction."""
    trajs = []
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        
        # Val split: use last 20% trajectories
        if use_val:
            n_total = len(traj_keys)
            val_start = int(n_total * (1 - val_split_ratio))
            traj_keys = traj_keys[val_start:]
            print(f"  Using val split: trajs [{val_start}:{n_total}] ({len(traj_keys)} trajs)")
        
        for key in traj_keys:
            grp = f[key]
            if "latent_concat" not in grp:
                continue
            T = grp["latent_concat"].shape[0]
            if T < min_length:
                continue
            trajs.append({
                "key": key,
                "latent": torch.from_numpy(grp["latent_concat"][:].astype(np.float32)),
                "actions": grp["actions"][:].astype(np.float32),
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
    args = wm_args_maniskill()
    args.svd_model_path = svd_model_path
    args.clip_model_path = clip_model_path
    args.ckpt_path = None

    model = CrtlWorld(args)
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# =====================================================================
# Inference
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
    pipeline = model.pipeline
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
    """Single-step (non-autoregressive) prediction: predict frames 5-9 from frames 0-4."""
    num_h = args.num_history  # 4
    num_f = args.num_frames   # 5
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

    # Actions
    act = full_actions[:window]
    act_norm = normalize_action(act, p01, p99)
    act_tensor = torch.tensor(act_norm, dtype=torch.float32).unsqueeze(0).to(device)

    pred = predict_one_step(model, history, current, act_tensor, text, args, device)
    
    # GT: frames num_h .. num_h+num_f-1
    end = min(num_h + num_f, T)
    gt = full_latent[num_h:end].cpu()
    pred_out = pred[0, :end - num_h].cpu()

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
    N = latents.shape[0]
    all_imgs = []
    for i in range(0, N, chunk_size):
        chunk = latents[i:i + chunk_size].to(device) / vae.config.scaling_factor
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
            p_t = torch.from_numpy(p).permute(2, 0, 1).float() / 127.5 - 1.0
            g_t = torch.from_numpy(g).permute(2, 0, 1).float() / 127.5 - 1.0
            with torch.no_grad():
                lp = lpips_model(
                    p_t.unsqueeze(0).cuda(), g_t.unsqueeze(0).cuda()
                ).item()
            lpips_vals.append(lp)

    result = {"psnr": psnrs, "ssim": ssims}
    if lpips_vals:
        result["lpips"] = lpips_vals
    return result


# =====================================================================
# Evaluate one checkpoint
# =====================================================================
def evaluate_checkpoint(
    ckpt_path: str,
    ckpt_name: str,
    trajs: list[dict],
    p01: np.ndarray,
    p99: np.ndarray,
    cfg: EvalConfig,
    lpips_model=None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {ckpt_name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Trajectories: {len(trajs)}")
    print(f"{'='*60}")

    model = load_model(ckpt_path, cfg.svd_model_path, cfg.clip_model_path, cfg.device)
    args = wm_args_maniskill()
    args.num_inference_steps = cfg.num_inference_steps
    args.decode_chunk_size = cfg.decode_chunk_size

    all_psnrs = []
    all_ssims = []
    all_lpips = []
    per_traj_results = []

    for i, traj in enumerate(trajs):
        t0 = time.time()
        print(f"  [{i+1}/{len(trajs)}] {traj['key']} (len={traj['length']})...", end=" ", flush=True)

        pred_latents, gt_latents = single_step_predict(
            model, traj, p01, p99, args, cfg.device,
        )

        if pred_latents.shape[0] == 0:
            print("SKIP (too short)")
            continue

        # Decode
        pred_images = decode_latents_to_images(pred_latents, model.vae, cfg.device, cfg.decode_chunk_size)
        gt_images = decode_latents_to_images(gt_latents, model.vae, cfg.device, cfg.decode_chunk_size)

        # Metrics
        fm = compute_metrics_per_frame(pred_images, gt_images, lpips_model)
        elapsed = time.time() - t0

        avg_psnr = np.mean(fm["psnr"])
        avg_ssim = np.mean(fm["ssim"])
        all_psnrs.extend(fm["psnr"])
        all_ssims.extend(fm["ssim"])
        if "lpips" in fm:
            all_lpips.extend(fm["lpips"])

        per_traj_results.append({
            "key": traj["key"],
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "lpips": np.mean(fm.get("lpips", [0])),
            "n_frames": len(fm["psnr"]),
        })

        print(f"{len(fm['psnr'])} frames, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, {elapsed:.1f}s")

    overall = {
        "ckpt_name": ckpt_name,
        "ckpt_path": ckpt_path,
        "n_trajs": len(per_traj_results),
        "psnr_mean": float(np.mean(all_psnrs)) if all_psnrs else 0.0,
        "psnr_std": float(np.std(all_psnrs)) if all_psnrs else 0.0,
        "ssim_mean": float(np.mean(all_ssims)) if all_ssims else 0.0,
        "ssim_std": float(np.std(all_ssims)) if all_ssims else 0.0,
        "n_frames": len(all_psnrs),
        "per_traj": per_traj_results,
    }
    if all_lpips:
        overall["lpips_mean"] = float(np.mean(all_lpips))
        overall["lpips_std"] = float(np.std(all_lpips))

    print(f"\n  {ckpt_name} Overall: PSNR={overall['psnr_mean']:.2f}±{overall['psnr_std']:.2f}, "
          f"SSIM={overall['ssim_mean']:.4f}±{overall['ssim_std']:.4f}")
    if all_lpips:
        print(f"  LPIPS={overall['lpips_mean']:.4f}±{overall['lpips_std']:.4f}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return overall


# =====================================================================
# Report
# =====================================================================
def generate_report(pretrained_res: dict, iter1_res: dict, cfg: EvalConfig) -> str:
    lines = [
        "# WM iter1 评估报告",
        "",
        f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> Task: LiftPegUpright-v1",
        f"> 验证数据: demo (val split) + rollout (val split)",
        "",
        "## 概述",
        "",
        "| 指标 | pretrained (ckpt-10000) | iter1 (ckpt-2000) | Delta |",
        "| ---- | ---------------------- | ----------------- | ----- |",
    ]

    dp = iter1_res["psnr_mean"] - pretrained_res["psnr_mean"]
    ds = iter1_res["ssim_mean"] - pretrained_res["ssim_mean"]
    lines.append(
        f"| PSNR | {pretrained_res['psnr_mean']:.2f} ± {pretrained_res['psnr_std']:.2f} "
        f"| {iter1_res['psnr_mean']:.2f} ± {iter1_res['psnr_std']:.2f} "
        f"| {dp:+.2f} |"
    )
    lines.append(
        f"| SSIM | {pretrained_res['ssim_mean']:.4f} ± {pretrained_res['ssim_std']:.4f} "
        f"| {iter1_res['ssim_mean']:.4f} ± {iter1_res['ssim_std']:.4f} "
        f"| {ds:+.4f} |"
    )
    if "lpips_mean" in pretrained_res and "lpips_mean" in iter1_res:
        dl = iter1_res["lpips_mean"] - pretrained_res["lpips_mean"]
        lines.append(
            f"| LPIPS ↓ | {pretrained_res['lpips_mean']:.4f} ± {pretrained_res['lpips_std']:.4f} "
            f"| {iter1_res['lpips_mean']:.4f} ± {iter1_res['lpips_std']:.4f} "
            f"| {dl:+.4f} |"
        )
    lines.append(
        f"| #frames | {pretrained_res['n_frames']} | {iter1_res['n_frames']} | - |"
    )
    lines.append(
        f"| #trajs | {pretrained_res['n_trajs']} | {iter1_res['n_trajs']} | - |"
    )
    lines.append("")

    # Gate check
    gate_pass = iter1_res["psnr_mean"] >= 18.0
    lines.extend([
        "## 门控检查",
        "",
        f"- iter1 PSNR: **{iter1_res['psnr_mean']:.2f}**",
        f"- 门控阈值: PSNR > 18.0",
        f"- 结果: **{'✅ 通过' if gate_pass else '❌ 未通过'}**",
        "",
    ])

    # Per-traj details
    lines.extend([
        "## 逐轨迹详情 (iter1)",
        "",
        "| Traj | PSNR | SSIM | #frames |",
        "| ---- | ---- | ---- | ------- |",
    ])
    for tr in iter1_res["per_traj"]:
        lines.append(f"| {tr['key']} | {tr['psnr']:.2f} | {tr['ssim']:.4f} | {tr['n_frames']} |")
    lines.append("")

    # Comparison analysis
    lines.extend([
        "## 分析",
        "",
        f"- pretrained 基线 PSNR: {pretrained_res['psnr_mean']:.2f} (论文值 22.35)",
        f"- iter1 finetuned PSNR: {iter1_res['psnr_mean']:.2f}",
        f"- Delta: {dp:+.2f} dB",
        "",
    ])
    if dp < -2:
        lines.append("⚠️ iter1 PSNR 比 pretrained 下降超过 2dB，可能存在过拟合或数据问题。")
    elif dp > 0:
        lines.append("✅ iter1 PSNR 相比 pretrained 有提升，微调有效。")
    else:
        lines.append("iter1 PSNR 与 pretrained 接近，微调对视频质量影响不大。")
    lines.append("")

    # Checkpoints
    lines.extend([
        "## Checkpoints",
        "",
        f"- pretrained: `{pretrained_res['ckpt_path']}`",
        f"- iter1: `{iter1_res['ckpt_path']}`",
        "",
    ])

    return "\n".join(lines)


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    cfg = EvalConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("WM iter1 Quality Evaluation")
    print("=" * 60)

    # Load norm stats
    p01, p99 = load_norm_stats(cfg.stat_path)
    print(f"Loaded norm stats from {cfg.stat_path}")

    # LPIPS
    lpips_model = None
    if _LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net="alex").cuda()
            print("LPIPS model loaded")
        except Exception as e:
            print(f"Failed to load LPIPS: {e}")

    # Load validation data
    # Demo data: use val split (last 20%)
    print(f"\nLoading demo data (val split): {cfg.demo_h5}")
    demo_trajs = load_trajectories(cfg.demo_h5, min_length=9, max_trajs=10, use_val=True)
    print(f"  Loaded {len(demo_trajs)} demo val trajectories")

    # Rollout data: use val split (last 20%)
    print(f"\nLoading rollout data (val split): {cfg.rollout_h5}")
    rollout_trajs = load_trajectories(cfg.rollout_h5, min_length=9, max_trajs=15, use_val=True)
    print(f"  Loaded {len(rollout_trajs)} rollout val trajectories")

    trajs = demo_trajs + rollout_trajs
    print(f"\nTotal val trajectories: {len(trajs)}")

    if len(trajs) == 0:
        print("ERROR: No valid trajectories found!")
        return

    # Evaluate pretrained
    pretrained_res = evaluate_checkpoint(
        cfg.pretrained_ckpt, "pretrained_ckpt10000", trajs, p01, p99, cfg, lpips_model,
    )

    # Evaluate iter1
    iter1_res = evaluate_checkpoint(
        cfg.iter1_ckpt, "iter1_ckpt2000", trajs, p01, p99, cfg, lpips_model,
    )

    # Generate report
    report = generate_report(pretrained_res, iter1_res, cfg)
    os.makedirs(os.path.dirname(cfg.report_path), exist_ok=True)
    with open(cfg.report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {cfg.report_path}")

    # Save raw JSON
    json_path = os.path.join(cfg.output_dir, "iter1_metrics.json")
    with open(json_path, "w") as f:
        json.dump({"pretrained": pretrained_res, "iter1": iter1_res}, f, indent=2)
    print(f"Raw metrics saved to: {json_path}")

    # Gate check
    gate = iter1_res["psnr_mean"] >= 18.0
    print(f"\n{'='*60}")
    print(f"GATE CHECK: iter1 PSNR = {iter1_res['psnr_mean']:.2f} {'≥' if gate else '<'} 18.0")
    print(f"Result: {'✅ PASS' if gate else '❌ FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
