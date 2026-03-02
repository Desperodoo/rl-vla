#!/usr/bin/env python3
"""WM optimal-steps evaluation: PSNR/SSIM/LPIPS across 500/1000/1500/2000 checkpoints.

Evaluates 4 checkpoints from ablation_optimal_steps/ to find the optimal training step.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n ctrl_world python scripts/vlaw/eval/eval_wm_optimal_steps.py
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

    # ---- Ablation checkpoints ----
    ablation_dir: str = "checkpoints/vlaw/world_model/ablation_optimal_steps"
    ablation_steps: list[int] = field(default_factory=lambda: [500, 1000, 1500, 2000])

    # ---- Model paths ----
    svd_model_path: str = "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"
    clip_model_path: str = "checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32"

    # ---- Eval params ----
    num_frames_per_step: int = 5
    num_history: int = 4
    max_trajs: int = 20
    num_inference_steps: int = 50
    decode_chunk_size: int = 4

    # ---- Output ----
    output_dir: str = "results/vlaw/wm_optimal_steps_eval"
    report_path: str = "results/vlaw/wm_optimal_steps_eval/report.md"

    # ---- Device ----
    device: str = "cuda:0"

    # ---- Known baselines (from previous runs) ----
    iter1_baseline_psnr: float = 23.40
    iter1_baseline_ssim: float = 0.7913
    iter1_baseline_lpips: float = 0.1200
    pretrained_baseline_psnr: float = 23.39
    pretrained_baseline_ssim: float = 0.8116
    pretrained_baseline_lpips: float = 0.1148
    # 4000-step ablation baselines (2000 step from that run)
    ablation4k_2000_psnr: float = 24.11
    ablation4k_2000_ssim: float = 0.8408
    ablation4k_2000_lpips: float = 0.0977


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

        lp_str = f", LPIPS={np.mean(fm['lpips']):.4f}" if "lpips" in fm else ""
        per_traj_results.append({
            "key": traj["key"],
            "psnr": float(avg_psnr),
            "ssim": float(avg_ssim),
            "lpips": float(np.mean(fm.get("lpips", [0]))),
            "n_frames": len(fm["psnr"]),
        })

        print(f"{len(fm['psnr'])} frames, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}{lp_str}, {elapsed:.1f}s")

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
def generate_report(results: dict[str, dict], cfg: EvalConfig) -> str:
    lines = [
        "# WM 最优训练步数消融评估报告",
        "",
        f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> Task: LiftPegUpright-v1",
        f"> 验证数据: demo (val split) + rollout (val split)",
        f"> Checkpoint 来源: `{cfg.ablation_dir}/`",
        "",
        "## 概述",
        "",
    ]

    # Build comparison table
    header_cols = ["指标", "pretrained*", "iter1 (2000步)*"]
    step_keys = sorted(results.keys(), key=lambda x: int(x.split("_")[0].replace("step", "")))
    for name in step_keys:
        header_cols.append(name)
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["----"] * len(header_cols)) + " |")

    # PSNR row
    row = ["PSNR", f"{cfg.pretrained_baseline_psnr:.2f}", f"{cfg.iter1_baseline_psnr:.2f}"]
    for name in step_keys:
        res = results[name]
        dp = res["psnr_mean"] - cfg.iter1_baseline_psnr
        row.append(f"{res['psnr_mean']:.2f} ± {res['psnr_std']:.2f} ({dp:+.2f})")
    lines.append("| " + " | ".join(row) + " |")

    # SSIM row
    row = ["SSIM", f"{cfg.pretrained_baseline_ssim:.4f}", f"{cfg.iter1_baseline_ssim:.4f}"]
    for name in step_keys:
        res = results[name]
        ds = res["ssim_mean"] - cfg.iter1_baseline_ssim
        row.append(f"{res['ssim_mean']:.4f} ± {res['ssim_std']:.4f} ({ds:+.4f})")
    lines.append("| " + " | ".join(row) + " |")

    # LPIPS row
    has_lpips = any("lpips_mean" in r for r in results.values())
    if has_lpips:
        row = ["LPIPS ↓", f"{cfg.pretrained_baseline_lpips:.4f}", f"{cfg.iter1_baseline_lpips:.4f}"]
        for name in step_keys:
            res = results[name]
            if "lpips_mean" in res:
                dl = res["lpips_mean"] - cfg.iter1_baseline_lpips
                row.append(f"{res['lpips_mean']:.4f} ± {res['lpips_std']:.4f} ({dl:+.4f})")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    # Frames & trajs
    row = ["#frames", "-", "-"]
    for name in step_keys:
        row.append(str(results[name]["n_frames"]))
    lines.append("| " + " | ".join(row) + " |")

    row = ["#trajs", "-", "-"]
    for name in step_keys:
        row.append(str(results[name]["n_trajs"]))
    lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("> *pretrained 和 iter1 数值来自之前的评估运行。")
    lines.append("")

    # Gate check
    lines.extend([
        "## 门控检查 (PSNR > 18.0)",
        "",
    ])
    for name in step_keys:
        res = results[name]
        gate_pass = res["psnr_mean"] >= 18.0
        emoji = "✅" if gate_pass else "❌"
        lines.append(f"- **{name}**: PSNR = {res['psnr_mean']:.2f} → {emoji} {'通过' if gate_pass else '未通过'}")
    lines.append("")

    # Step-metric curve (text-based)
    lines.extend([
        "## Step→Metric 曲线",
        "",
        "| Step | PSNR | SSIM | LPIPS ↓ | Δ PSNR (vs iter1) |",
        "| ---- | ---- | ---- | ------- | ----------------- |",
    ])
    for name in step_keys:
        res = results[name]
        step = name.split("_")[0].replace("step", "")
        dp = res["psnr_mean"] - cfg.iter1_baseline_psnr
        lp = f"{res['lpips_mean']:.4f}" if "lpips_mean" in res else "N/A"
        lines.append(f"| {step} | {res['psnr_mean']:.2f} | {res['ssim_mean']:.4f} | {lp} | {dp:+.2f} |")
    lines.append("")

    # Optimal step analysis
    lines.extend([
        "## 最优步数分析",
        "",
    ])

    # Find best by each metric
    best_psnr_name = max(step_keys, key=lambda k: results[k]["psnr_mean"])
    best_ssim_name = max(step_keys, key=lambda k: results[k]["ssim_mean"])
    best_lpips_name = min(step_keys, key=lambda k: results[k].get("lpips_mean", float("inf")))

    lines.append(f"- **最佳 PSNR**: {best_psnr_name} = {results[best_psnr_name]['psnr_mean']:.2f} dB")
    lines.append(f"- **最佳 SSIM**: {best_ssim_name} = {results[best_ssim_name]['ssim_mean']:.4f}")
    if has_lpips:
        lines.append(f"- **最佳 LPIPS**: {best_lpips_name} = {results[best_lpips_name]['lpips_mean']:.4f}")
    lines.append("")

    # Check if 500/1000 steps are sufficient
    lines.extend([
        "## 训练步数缩短可行性分析",
        "",
    ])
    for name in step_keys:
        res = results[name]
        step = name.split("_")[0].replace("step", "")
        dp = res["psnr_mean"] - cfg.iter1_baseline_psnr
        gate = res["psnr_mean"] >= 18.0
        ssim_ok = res["ssim_mean"] >= 0.75

        if gate and ssim_ok:
            lines.append(f"- **{step}步**: PSNR={res['psnr_mean']:.2f} (>18 ✅), SSIM={res['ssim_mean']:.4f} (>0.75 ✅) — 可用")
        elif gate:
            lines.append(f"- **{step}步**: PSNR={res['psnr_mean']:.2f} (>18 ✅), SSIM={res['ssim_mean']:.4f} (<0.75 ⚠️) — 勉强可用")
        else:
            lines.append(f"- **{step}步**: PSNR={res['psnr_mean']:.2f} (<18 ❌) — 不可用")
    lines.append("")

    # Recommendation
    lines.extend([
        "## 推荐",
        "",
    ])

    # Find the sweet spot: earliest step where PSNR is within 0.5 dB of the best
    best_psnr = max(results[k]["psnr_mean"] for k in step_keys)
    recommended = None
    for name in step_keys:
        if results[name]["psnr_mean"] >= best_psnr - 0.5:
            recommended = name
            break

    if recommended:
        rec_step = recommended.split("_")[0].replace("step", "")
        lines.append(f"**推荐训练步数: {rec_step} 步**")
        lines.append(f"- 理由: 在最佳 PSNR ({best_psnr:.2f}) 的 0.5 dB 容差内的最早步数")
        lines.append(f"- PSNR={results[recommended]['psnr_mean']:.2f}, SSIM={results[recommended]['ssim_mean']:.4f}")
        if "lpips_mean" in results[recommended]:
            lines.append(f"- LPIPS={results[recommended]['lpips_mean']:.4f}")
    lines.append("")

    # Per-traj details
    for name in step_keys:
        res = results[name]
        lines.extend([
            f"## 逐轨迹详情: {name}",
            "",
            "| Traj | PSNR | SSIM | LPIPS | #frames |",
            "| ---- | ---- | ---- | ----- | ------- |",
        ])
        for tr in res["per_traj"]:
            lp = f"{tr['lpips']:.4f}" if tr.get("lpips", 0) > 0 else "N/A"
            lines.append(f"| {tr['key']} | {tr['psnr']:.2f} | {tr['ssim']:.4f} | {lp} | {tr['n_frames']} |")
        lines.append("")

    return "\n".join(lines)


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    cfg = EvalConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("WM Optimal Training Steps Evaluation")
    print(f"Steps to evaluate: {cfg.ablation_steps}")
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
    print(f"\nLoading demo data (val split): {cfg.demo_h5}")
    demo_trajs = load_trajectories(cfg.demo_h5, min_length=9, max_trajs=10, use_val=True)
    print(f"  Loaded {len(demo_trajs)} demo val trajectories")

    print(f"\nLoading rollout data (val split): {cfg.rollout_h5}")
    rollout_trajs = load_trajectories(cfg.rollout_h5, min_length=9, max_trajs=15, use_val=True)
    print(f"  Loaded {len(rollout_trajs)} rollout val trajectories")

    trajs = demo_trajs + rollout_trajs
    print(f"\nTotal val trajectories: {len(trajs)}")

    if len(trajs) == 0:
        print("ERROR: No valid trajectories found!")
        return

    # Evaluate each checkpoint
    results: dict[str, dict] = {}
    for step in cfg.ablation_steps:
        ckpt_name = f"checkpoint-{step}.pt"
        ckpt_path = os.path.join(cfg.ablation_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found, skipping")
            continue
        label = f"step{step}_ckpt"
        res = evaluate_checkpoint(
            ckpt_path, label, trajs, p01, p99, cfg, lpips_model,
        )
        results[label] = res

        # Save intermediate JSON after each checkpoint
        json_path = os.path.join(cfg.output_dir, f"metrics_step{step}.json")
        with open(json_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Intermediate results saved: {json_path}")

    if not results:
        print("ERROR: No checkpoints evaluated!")
        return

    # Generate report
    report = generate_report(results, cfg)
    with open(cfg.report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {cfg.report_path}")

    # Save combined JSON
    json_path = os.path.join(cfg.output_dir, "all_metrics.json")
    # Strip per_traj for the combined summary
    summary = {}
    for name, res in results.items():
        summary[name] = {k: v for k, v in res.items() if k != "per_traj"}
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Combined metrics saved to: {json_path}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY — Step → Metric Comparison")
    print(f"{'='*80}")
    print(f"{'Step':<8} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10} {'ΔPSNR':>10} {'Gate':>6}")
    print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    # Print baselines
    print(f"{'pretrained':<8} {cfg.pretrained_baseline_psnr:>10.2f} {cfg.pretrained_baseline_ssim:>10.4f} {cfg.pretrained_baseline_lpips:>10.4f} {'(ref)':>10} {'PASS':>6}")
    print(f"{'iter1':<8} {cfg.iter1_baseline_psnr:>10.2f} {cfg.iter1_baseline_ssim:>10.4f} {cfg.iter1_baseline_lpips:>10.4f} {'(ref)':>10} {'PASS':>6}")

    step_keys = sorted(results.keys(), key=lambda x: int(x.split("_")[0].replace("step", "")))
    for name in step_keys:
        res = results[name]
        step = name.split("_")[0].replace("step", "")
        dp = res["psnr_mean"] - cfg.iter1_baseline_psnr
        gate = "PASS" if res["psnr_mean"] >= 18.0 else "FAIL"
        lp = f"{res['lpips_mean']:.4f}" if "lpips_mean" in res else "N/A"
        print(f"{step:<8} {res['psnr_mean']:>10.2f} {res['ssim_mean']:>10.4f} {lp:>10} {dp:>+10.2f} {gate:>6}")

    print(f"{'='*80}")

    # Final recommendation
    best_psnr = max(results[k]["psnr_mean"] for k in step_keys)
    best_name = max(step_keys, key=lambda k: results[k]["psnr_mean"])
    rec = None
    for name in step_keys:
        if results[name]["psnr_mean"] >= best_psnr - 0.5:
            rec = name
            break

    print(f"\nBest PSNR: {best_name} = {best_psnr:.2f}")
    if rec:
        rec_step = rec.split("_")[0].replace("step", "")
        print(f"Recommended step (within 0.5dB of best): {rec_step}")

    all_pass = all(r["psnr_mean"] >= 18.0 for r in results.values())
    print(f"\nOverall gate: {'✅ ALL PASS' if all_pass else '❌ SOME FAIL'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
