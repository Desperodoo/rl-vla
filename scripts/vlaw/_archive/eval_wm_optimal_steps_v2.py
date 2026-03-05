#!/usr/bin/env python3
"""Batch evaluation of WM optimal_steps_v2 checkpoints (100-2000, step=100).

Evaluates all 20 checkpoints + pretrained baseline, aggregates results,
plots step→PSNR/SSIM/LPIPS curves, finds optimal inflection point.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n ctrl_world python scripts/vlaw/eval/eval_wm_optimal_steps_v2.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Import the standard eval machinery
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_wm_standard import (
    EvalStandardConfig,
    evaluate_checkpoint,
    load_eval_trajectories,
    load_model,
    load_norm_stats,
)

try:
    import lpips

    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


@dataclass
class OptimalStepsV2Config:
    ckpt_dir: str = "checkpoints/vlaw/world_model/optimal_steps_v2"
    pretrained_path: str = (
        "checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt"
    )
    iter1_path: str = "checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt"
    eval_h5_path: str = "data/vlaw/encoded/eval_fixed/eval_set.h5"
    stat_path: str = "data/vlaw/meta_info/maniskill/stat.json"
    svd_model_path: str = (
        "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"
    )
    clip_model_path: str = (
        "checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32"
    )
    output_dir: str = "results/vlaw/wm_optimal_steps_v2_eval"
    device: str = "cuda:0"
    num_frames_per_step: int = 5
    num_history: int = 6
    num_inference_steps: int = 50
    decode_chunk_size: int = 4
    min_traj_length: int = 9
    steps_range: tuple[int, ...] = tuple(range(100, 2100, 100))


def main() -> None:
    cfg = OptimalStepsV2Config()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print banner
    print("=" * 70)
    print("[VLAW] WM Optimal Steps v2 — Batch Evaluation")
    print(f"  Checkpoints: {cfg.ckpt_dir}")
    print(f"  Steps: {cfg.steps_range}")
    print(f"  Eval set: {cfg.eval_h5_path}")
    print(f"  Output: {cfg.output_dir}")
    print("=" * 70)

    # Build EvalStandardConfig for reuse
    eval_cfg = EvalStandardConfig(
        eval_h5_path=cfg.eval_h5_path,
        stat_path=cfg.stat_path,
        pretrained_path=cfg.pretrained_path,
        svd_model_path=cfg.svd_model_path,
        clip_model_path=cfg.clip_model_path,
        num_frames_per_step=cfg.num_frames_per_step,
        num_history=cfg.num_history,
        num_inference_steps=cfg.num_inference_steps,
        decode_chunk_size=cfg.decode_chunk_size,
        min_traj_length=cfg.min_traj_length,
        output_dir=cfg.output_dir,
        device=cfg.device,
    )

    # Load data
    p01, p99 = load_norm_stats(cfg.stat_path)
    trajs = load_eval_trajectories(cfg.eval_h5_path, min_length=cfg.min_traj_length)
    if not trajs:
        print("ERROR: No valid trajectories!")
        return

    # Load LPIPS
    lpips_model = None
    if _LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net="alex").cuda()
            print("[VLAW] LPIPS model loaded")
        except Exception as e:
            print(f"[WARN] LPIPS load failed: {e}")

    # Build checkpoint list: pretrained + iter1 + 20 v2 steps
    all_ckpts: list[tuple[str, str, int | None]] = [
        ("pretrained-10000", cfg.pretrained_path, None),
    ]
    if Path(cfg.iter1_path).exists():
        all_ckpts.append(("iter1-2000", cfg.iter1_path, None))

    for step in cfg.steps_range:
        ckpt_path = os.path.join(cfg.ckpt_dir, f"checkpoint-{step}.pt")
        if Path(ckpt_path).exists():
            all_ckpts.append((f"v2-{step}", ckpt_path, step))
        else:
            print(f"[WARN] Missing checkpoint: {ckpt_path}")

    print(f"\n[VLAW] Total checkpoints to evaluate: {len(all_ckpts)}")

    # Evaluate each checkpoint, with incremental saving
    all_results: dict[str, dict] = {}
    progress_path = out_dir / "progress.json"

    # Resume from previous run if exists
    if progress_path.exists():
        with open(progress_path) as f:
            saved = json.load(f)
        all_results = saved
        print(f"[VLAW] Resuming: {len(all_results)} checkpoints already evaluated")

    for i, (label, ckpt_path, step) in enumerate(all_ckpts):
        if label in all_results:
            print(f"\n[{i+1}/{len(all_ckpts)}] SKIP {label} (already evaluated)")
            continue

        print(f"\n[{i+1}/{len(all_ckpts)}] Evaluating {label}...")
        t0 = time.time()

        try:
            res = evaluate_checkpoint(
                ckpt_path, label, trajs, p01, p99, eval_cfg, lpips_model
            )
            if step is not None:
                res["training_step"] = step
            all_results[label] = res
            elapsed = time.time() - t0

            # Save per-checkpoint JSON
            safe_label = label.replace("/", "_").replace(" ", "_")
            with open(out_dir / f"metrics_{safe_label}.json", "w") as f:
                json.dump(res, f, indent=2)

            # Save progress (for resume)
            with open(progress_path, "w") as f:
                json.dump(
                    {k: {kk: vv for kk, vv in v.items() if kk != "per_traj"}
                     for k, v in all_results.items()},
                    f, indent=2,
                )

            psnr = res["psnr_mean"]
            ssim = res["ssim_mean"]
            lp_str = f", LPIPS={res['lpips_mean']:.4f}" if "lpips_mean" in res else ""
            print(
                f"  ✅ {label}: PSNR={psnr:.2f}, SSIM={ssim:.4f}{lp_str} ({elapsed:.1f}s)"
            )

        except Exception as e:
            print(f"  ❌ {label}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            continue

    # ---- Aggregation & Analysis ----
    print("\n" + "=" * 70)
    print("[VLAW] All evaluations complete. Aggregating results...")
    print("=" * 70)

    # Extract v2 step results for curve plotting
    v2_steps: list[int] = []
    v2_psnrs: list[float] = []
    v2_ssims: list[float] = []
    v2_lpips: list[float] = []
    v2_psnr_stds: list[float] = []

    for label, res in all_results.items():
        if label.startswith("v2-"):
            step = int(label.split("-")[1])
            v2_steps.append(step)
            v2_psnrs.append(res["psnr_mean"])
            v2_ssims.append(res["ssim_mean"])
            v2_psnr_stds.append(res["psnr_std"])
            if "lpips_mean" in res:
                v2_lpips.append(res["lpips_mean"])

    # Sort by step
    sort_idx = np.argsort(v2_steps)
    v2_steps = [v2_steps[i] for i in sort_idx]
    v2_psnrs = [v2_psnrs[i] for i in sort_idx]
    v2_ssims = [v2_ssims[i] for i in sort_idx]
    v2_psnr_stds = [v2_psnr_stds[i] for i in sort_idx]
    if v2_lpips:
        v2_lpips = [v2_lpips[i] for i in sort_idx]

    # Baselines
    pretrained_psnr = all_results.get("pretrained-10000", {}).get("psnr_mean", 23.39)
    iter1_psnr = all_results.get("iter1-2000", {}).get("psnr_mean", 23.34)

    # Find best step
    best_idx = int(np.argmax(v2_psnrs))
    best_step = v2_steps[best_idx]
    best_psnr = v2_psnrs[best_idx]

    # Find 95% threshold inflection point
    threshold_95 = best_psnr * 0.95
    inflection_step = None
    for s, p in zip(v2_steps, v2_psnrs):
        if p >= threshold_95:
            inflection_step = s
            break

    print(f"\n--- Key Results ---")
    print(f"  Best v2 step: {best_step} (PSNR={best_psnr:.2f})")
    print(f"  95% threshold: PSNR≥{threshold_95:.2f}")
    print(f"  Inflection point (min step ≥ 95% best): step={inflection_step}")
    print(f"  Pretrained baseline: PSNR={pretrained_psnr:.2f}")
    print(f"  Iter1 baseline: PSNR={iter1_psnr:.2f}")

    # ---- Generate summary table ----
    summary_lines: list[str] = [
        "# WM Optimal Steps v2 — 评估报告",
        "",
        f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> Task: LiftPegUpright-v1",
        f"> 评估集: {cfg.eval_h5_path} (15条固定轨迹)",
        f"> 训练配置: num_frames=5, demos data",
        "",
        "## 关键结论",
        "",
        f"- **最佳步数**: step={best_step}, PSNR={best_psnr:.2f}",
        f"- **95%拐点**: step={inflection_step} (首次PSNR≥{threshold_95:.2f})",
        f"- **Pretrained baseline**: PSNR={pretrained_psnr:.2f}",
        f"- **Iter1 baseline (2000步)**: PSNR={iter1_psnr:.2f}",
        "",
        "## Step→PSNR/SSIM 完整表",
        "",
        "| Step | PSNR | PSNR_std | SSIM | Δ vs pretrained | Δ vs iter1 | ≥95%best |",
        "| ---- | ---- | -------- | ---- | --------------- | ---------- | -------- |",
    ]

    for s, p, ps, ss in zip(v2_steps, v2_psnrs, v2_psnr_stds, v2_ssims):
        dp = p - pretrained_psnr
        di = p - iter1_psnr
        ge95 = "✅" if p >= threshold_95 else ""
        best_mark = " 🏆" if s == best_step else ""
        summary_lines.append(
            f"| {s} | {p:.2f}{best_mark} | {ps:.2f} | {ss:.4f} | {dp:+.2f} | {di:+.2f} | {ge95} |"
        )

    if v2_lpips:
        summary_lines.extend([
            "",
            "## Step→LPIPS (↓ better)",
            "",
            "| Step | LPIPS |",
            "| ---- | ----- |",
        ])
        for s, lp in zip(v2_steps, v2_lpips):
            summary_lines.append(f"| {s} | {lp:.4f} |")

    # Baselines section
    summary_lines.extend([
        "",
        "## 基线对比",
        "",
        "| Model | PSNR | SSIM | LPIPS |",
        "| ----- | ---- | ---- | ----- |",
    ])
    for label in ["pretrained-10000", "iter1-2000"]:
        if label in all_results:
            r = all_results[label]
            lp = f"{r['lpips_mean']:.4f}" if "lpips_mean" in r else "N/A"
            summary_lines.append(
                f"| {label} | {r['psnr_mean']:.2f} | {r['ssim_mean']:.4f} | {lp} |"
            )

    # ASCII curve
    summary_lines.extend(["", "## PSNR 曲线 (ASCII)", ""])
    if v2_psnrs:
        min_p = min(v2_psnrs) - 0.5
        max_p = max(max(v2_psnrs), pretrained_psnr) + 0.5
        h = 20  # rows
        w = len(v2_steps)
        for row in range(h, -1, -1):
            val = min_p + (max_p - min_p) * row / h
            line = f"{val:6.1f} |"
            for j in range(w):
                p_val = v2_psnrs[j]
                p_row = int((p_val - min_p) / (max_p - min_p) * h)
                if p_row == row:
                    line += "●"
                elif row == int((pretrained_psnr - min_p) / (max_p - min_p) * h):
                    line += "-"
                else:
                    line += " "
            summary_lines.append(f"```")
            summary_lines.append(line)
            summary_lines.append(f"```")
        summary_lines.append(f"```")
        x_labels = "       " + "".join([str(s // 100 % 10) for s in v2_steps])
        summary_lines.append(x_labels)
        summary_lines.append(f"       (×100 steps)")
        summary_lines.append(f"```")
        summary_lines.append(f"  `—` = pretrained baseline PSNR={pretrained_psnr:.2f}")

    # Recommendation
    summary_lines.extend([
        "",
        "## 建议",
        "",
        f"- 推荐使用 **step={inflection_step}** (95%拐点, 训练时间最短达同等质量)",
        f"- 如追求最高 PSNR, 使用 step={best_step} (PSNR={best_psnr:.2f})",
    ])

    report_text = "\n".join(summary_lines)
    report_path = out_dir / "optimal_steps_v2_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n[VLAW] Report saved: {report_path}")

    # Save combined summary JSON
    summary_json = {
        "best_step": best_step,
        "best_psnr": best_psnr,
        "threshold_95_psnr": threshold_95,
        "inflection_step": inflection_step,
        "pretrained_psnr": pretrained_psnr,
        "iter1_psnr": iter1_psnr,
        "steps": v2_steps,
        "psnrs": v2_psnrs,
        "ssims": v2_ssims,
        "lpips": v2_lpips if v2_lpips else None,
        "psnr_stds": v2_psnr_stds,
        "all_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_traj"}
            for k, v in all_results.items()
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"[VLAW] Summary JSON saved: {out_dir / 'summary.json'}")

    # ---- Matplotlib plot (if available) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3 if v2_lpips else 2, figsize=(14, 5))

        # PSNR
        ax = axes[0]
        ax.plot(v2_steps, v2_psnrs, "b-o", label="v2 fine-tuned", markersize=4)
        ax.fill_between(
            v2_steps,
            [p - s for p, s in zip(v2_psnrs, v2_psnr_stds)],
            [p + s for p, s in zip(v2_psnrs, v2_psnr_stds)],
            alpha=0.2, color="blue",
        )
        ax.axhline(pretrained_psnr, color="red", linestyle="--", label=f"pretrained ({pretrained_psnr:.2f})")
        ax.axhline(iter1_psnr, color="green", linestyle=":", label=f"iter1 ({iter1_psnr:.2f})")
        ax.axhline(threshold_95, color="gray", linestyle="-.", alpha=0.5, label=f"95% best ({threshold_95:.2f})")
        if inflection_step:
            ax.axvline(inflection_step, color="orange", linestyle="--", alpha=0.7, label=f"inflection @ {inflection_step}")
        ax.scatter([best_step], [best_psnr], color="gold", s=100, zorder=5, marker="*", label=f"best @ {best_step}")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("PSNR vs Training Steps")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # SSIM
        ax = axes[1]
        ax.plot(v2_steps, v2_ssims, "g-o", label="v2 fine-tuned", markersize=4)
        pretrained_ssim = all_results.get("pretrained-10000", {}).get("ssim_mean", 0)
        iter1_ssim = all_results.get("iter1-2000", {}).get("ssim_mean", 0)
        if pretrained_ssim:
            ax.axhline(pretrained_ssim, color="red", linestyle="--", label=f"pretrained ({pretrained_ssim:.4f})")
        if iter1_ssim:
            ax.axhline(iter1_ssim, color="green", linestyle=":", label=f"iter1 ({iter1_ssim:.4f})")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("SSIM")
        ax.set_title("SSIM vs Training Steps")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # LPIPS
        if v2_lpips:
            ax = axes[2]
            ax.plot(v2_steps, v2_lpips, "r-o", label="v2 fine-tuned", markersize=4)
            pretrained_lpips = all_results.get("pretrained-10000", {}).get("lpips_mean", 0)
            iter1_lpips = all_results.get("iter1-2000", {}).get("lpips_mean", 0)
            if pretrained_lpips:
                ax.axhline(pretrained_lpips, color="red", linestyle="--", label=f"pretrained ({pretrained_lpips:.4f})")
            if iter1_lpips:
                ax.axhline(iter1_lpips, color="green", linestyle=":", label=f"iter1 ({iter1_lpips:.4f})")
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("LPIPS (↓)")
            ax.set_title("LPIPS vs Training Steps")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / "optimal_steps_v2_curves.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[VLAW] Plot saved: {fig_path}")

    except ImportError:
        print("[WARN] matplotlib not available, skipping plot generation")

    # Print final summary table
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY — WM Optimal Steps v2")
    print(f"{'=' * 80}")
    print(f"{'Step':>6} {'PSNR':>8} {'±std':>6} {'SSIM':>8} {'LPIPS':>8} {'Δpre':>6} {'Mark':>8}")
    print(f"{'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")

    # Baselines first
    for label in ["pretrained-10000", "iter1-2000"]:
        if label in all_results:
            r = all_results[label]
            lp = f"{r['lpips_mean']:.4f}" if "lpips_mean" in r else "N/A"
            print(
                f"{'base':>6} {r['psnr_mean']:>8.2f} {r['psnr_std']:>6.2f} "
                f"{r['ssim_mean']:>8.4f} {lp:>8} {'':>6} {label}"
            )

    for s, p, ps, ss in zip(v2_steps, v2_psnrs, v2_psnr_stds, v2_ssims):
        dp = p - pretrained_psnr
        mark = ""
        if s == best_step:
            mark = "🏆best"
        elif s == inflection_step:
            mark = "📍95%"
        lp_val = "N/A"
        if v2_lpips:
            idx = v2_steps.index(s)
            lp_val = f"{v2_lpips[idx]:.4f}"
        print(
            f"{s:>6} {p:>8.2f} {ps:>6.2f} {ss:>8.4f} {lp_val:>8} {dp:>+6.2f} {mark:>8}"
        )

    print(f"{'=' * 80}")
    print(f"Best: step={best_step}, PSNR={best_psnr:.2f}")
    print(f"95% inflection: step={inflection_step}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
