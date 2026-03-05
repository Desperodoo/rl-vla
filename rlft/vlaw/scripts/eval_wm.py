#!/usr/bin/env python3
"""WM 定量评估 — 稳定入口脚本.

评估一个或多个 Ctrl-World checkpoint 的 PSNR/SSIM/LPIPS，
始终包含 pretrained 作为 baseline。输出标准化 JSON + Markdown。
合并自: eval_wm_standard.py / eval_wm_iter1.py / eval_wm_optimal_steps.py 等。

用法:
    # 评估 pretrained baseline
    CUDA_VISIBLE_DEVICES=4 conda run -n ctrl_world python \\
        rlft/vlaw/scripts/eval_wm.py

    # 评估特定 checkpoint
    CUDA_VISIBLE_DEVICES=4 conda run -n ctrl_world python \\
        rlft/vlaw/scripts/eval_wm.py \\
        --checkpoint_paths "checkpoints/vlaw/world_model/iter1_v3/checkpoint-2000.pt"

    # 自定义 eval 集
    CUDA_VISIBLE_DEVICES=4 conda run -n ctrl_world python \\
        rlft/vlaw/scripts/eval_wm.py \\
        --eval_h5 data/vlaw/encoded/eval/eval_set.h5 \\
        --checkpoint_paths "ckpt1.pt,ckpt2.pt"
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

# ── 路径设置 ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "ctrl_world"))
os.chdir(str(WORKSPACE))

from config import wm_args_maniskill
from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ── 可视化辅助 ──────────────────────────────────────────────────────────────

def _get_font(size: int = 14) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """尝试加载 TTF 字体，失败则回退到默认."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _make_traj_comparison(
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    per_frame_psnr: list[float],
    per_frame_ssim: list[float],
    traj_key: str,
    mean_psnr: float,
    mean_ssim: float,
) -> "PILImage.Image":
    """生成单条轨迹的 GT vs Pred 对比图.

    Layout:
        Title bar: "traj_XXXX | mean PSNR=XX.XX  SSIM=X.XXXX"
        Row 1 (GT):   [frame0] [frame1] ... [frameN]
        Row 2 (Pred):  [frame0] [frame1] ... [frameN]  标注 per-frame PSNR

    Args:
        gt_frames: (N, H, W, 3) uint8
        pred_frames: (N, H, W, 3) uint8
    """
    n_frames = gt_frames.shape[0]
    h, w = gt_frames.shape[1], gt_frames.shape[2]
    pad, label_h, title_h, row_label_h = 4, 28, 36, 22

    total_w = pad + n_frames * (w + pad)
    total_h = title_h + 2 * (row_label_h + h + pad) + pad

    canvas = PILImage.new("RGB", (total_w, total_h), color=(40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(14)
    small_font = _get_font(11)
    title_font = _get_font(16)

    title_text = f"{traj_key}  |  mean PSNR={mean_psnr:.2f} dB   SSIM={mean_ssim:.4f}"
    draw.text((pad + 4, 8), title_text, fill=(255, 255, 255), font=title_font)

    y_offset = title_h
    # Row 1: GT
    draw.text((pad + 4, y_offset + 2), "GT (Ground Truth)", fill=(100, 255, 100), font=font)
    y_offset += row_label_h
    for i in range(n_frames):
        x = pad + i * (w + pad)
        canvas.paste(PILImage.fromarray(gt_frames[i]), (x, y_offset))
        draw.text((x + 2, y_offset + h + 1), f"f{i}", fill=(180, 180, 180), font=small_font)
    y_offset += h + pad

    # Row 2: Pred
    draw.text((pad + 4, y_offset + 2), "Pred (Generated)", fill=(255, 150, 100), font=font)
    y_offset += row_label_h
    for i in range(n_frames):
        x = pad + i * (w + pad)
        canvas.paste(PILImage.fromarray(pred_frames[i]), (x, y_offset))
        pval = per_frame_psnr[i]
        color = (100, 255, 100) if pval >= 25 else ((255, 255, 100) if pval >= 20 else (255, 100, 100))
        draw.text(
            (x + 2, y_offset + h + 1),
            f"PSNR={pval:.1f}  SSIM={per_frame_ssim[i]:.3f}",
            fill=color, font=small_font,
        )

    return canvas


def _make_all_trajs_grid(traj_images: "list[PILImage.Image]") -> "PILImage.Image":
    """将所有轨迹对比图垂直拼接为大图."""
    if not traj_images:
        return PILImage.new("RGB", (400, 100), (40, 40, 40))
    max_w = max(img.width for img in traj_images)
    gap = 6
    total_h = sum(img.height + gap for img in traj_images)
    canvas = PILImage.new("RGB", (max_w, total_h), color=(30, 30, 30))
    y = 0
    for img in traj_images:
        canvas.paste(img, (0, y))
        y += img.height + gap
    return canvas


def _make_diff_map(gt_frame: np.ndarray, pred_frame: np.ndarray) -> np.ndarray:
    """生成差异热力图 (uint8, H x W x 3)."""
    diff = np.abs(gt_frame.astype(np.float32) - pred_frame.astype(np.float32))
    diff_norm = np.clip(diff.mean(axis=2) * 3, 0, 255).astype(np.uint8)
    r = diff_norm
    g = np.clip(diff_norm.astype(np.int16) - 80, 0, 255).astype(np.uint8)
    b = np.clip(diff_norm.astype(np.int16) - 160, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=2)


# ── 配置 ────────────────────────────────────────────────────────────────────

@dataclass
class EvalWMConfig:
    """WM 评估配置."""
    eval_h5: str = "data/vlaw/encoded/eval/eval_set.h5"
    stat_path: str = "data/vlaw/meta_info/maniskill/stat.json"
    checkpoint_paths: str = ""
    pretrained_path: str = "checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt"
    svd_model_path: str = "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"
    clip_model_path: str = "checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32"
    num_frames: int = 5
    num_history: int = 6
    num_inference_steps: int = 50
    decode_chunk_size: int = 4
    min_traj_length: int = 9
    output_dir: str = "results/vlaw/wm_eval"
    device: str = "cuda:0"


# ── 数据 ────────────────────────────────────────────────────────────────────

def load_norm_stats(stat_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(stat_path) as f:
        stat = json.load(f)
    p01 = np.array(stat["state_01"], dtype=np.float32)[None, :]
    p99 = np.array(stat["state_99"], dtype=np.float32)[None, :]
    return p01, p99


def normalize_action(action: np.ndarray, p01: np.ndarray, p99: np.ndarray) -> np.ndarray:
    ndata = 2.0 * (action - p01) / (p99 - p01 + 1e-8) - 1.0
    return np.clip(ndata, -1.0, 1.0)


def load_eval_trajectories(h5_path: str, min_length: int = 9) -> list[dict]:
    trajs: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        print(f"[EVAL-WM] Loading: {h5_path} ({len(traj_keys)} trajectories)")
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
                "source": grp.attrs.get("original_source", "unknown"),
            })
    print(f"[EVAL-WM] Loaded {len(trajs)} trajectories (min_length={min_length})")
    return trajs


# ── 模型 ────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, svd_path: str, clip_path: str,
               device: str = "cuda:0") -> CrtlWorld:
    args = wm_args_maniskill()
    args.svd_model_path = svd_path
    args.clip_model_path = clip_path
    args.ckpt_path = None
    model = CrtlWorld(args)
    print(f"[EVAL-WM] Loading: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


# ── 推理 ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_one_step(model: CrtlWorld, history: torch.Tensor,
                     current: torch.Tensor, actions: torch.Tensor,
                     text: str, args, device: str) -> torch.Tensor:
    action_latent = model.action_encoder(
        actions, [text], model.tokenizer, model.text_encoder, args.frame_level_cond,
    )
    _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
        model.pipeline, image=current, text=action_latent,
        width=args.width, height=args.height, num_frames=args.num_frames,
        history=history, num_inference_steps=args.num_inference_steps,
        decode_chunk_size=args.decode_chunk_size,
        max_guidance_scale=args.guidance_scale, fps=args.fps,
        motion_bucket_id=args.motion_bucket_id, mask=None,
        output_type="latent", return_dict=False,
        frame_level_cond=args.frame_level_cond, his_cond_zero=args.his_cond_zero,
    )
    return pred_latents


@torch.no_grad()
def single_step_predict(model: CrtlWorld, traj: dict, p01: np.ndarray,
                        p99: np.ndarray, args, device: str):
    num_h, num_f = args.num_history, args.num_frames
    full_latent = traj["latent"].to(device)
    T = full_latent.shape[0]
    if T < num_h + num_f:
        return torch.empty(0, 4, 48, 24), torch.empty(0, 4, 48, 24)

    history = full_latent[:num_h].unsqueeze(0)
    current = full_latent[num_h].unsqueeze(0)
    act = traj["actions"][:num_h + num_f]
    act_norm = normalize_action(act, p01, p99)
    act_tensor = torch.tensor(act_norm, dtype=torch.float32).unsqueeze(0).to(device)

    pred = predict_one_step(model, history, current, act_tensor, traj["text"], args, device)
    end = min(num_h + num_f, T)
    return pred[0, :end - num_h].cpu(), full_latent[num_h:end].cpu()


@torch.no_grad()
def decode_latents(latents: torch.Tensor, vae, device: str, chunk: int = 4) -> np.ndarray:
    all_imgs = []
    for i in range(0, latents.shape[0], chunk):
        c = latents[i:i+chunk].to(device) / vae.config.scaling_factor
        out = vae.decode(c, num_frames=c.shape[0]).sample
        imgs = ((out / 2.0 + 0.5).clamp(0, 1) * 255).cpu().numpy()
        all_imgs.append(imgs.transpose(0, 2, 3, 1).astype(np.uint8))
    return np.concatenate(all_imgs, axis=0)


# ── 评估 ────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(ckpt_path: str, ckpt_label: str, trajs: list[dict],
                        p01: np.ndarray, p99: np.ndarray, cfg: EvalWMConfig,
                        lpips_model=None, *, visualize: bool = False) -> dict:
    """评估一个 checkpoint 的 PSNR/SSIM/LPIPS.

    当 ``visualize=True`` 时，额外:
      - 收集 per-frame PSNR/SSIM 到 ``result["per_traj"][*]["frame_psnrs"]``
      - 收集 pred/gt 图像, 生成 GT vs Pred 对比图 + all_trajs_grid
    """
    print(f"\n{'='*60}\n  Evaluating: {ckpt_label}\n  {ckpt_path}\n{'='*60}")
    model = load_model(ckpt_path, cfg.svd_model_path, cfg.clip_model_path, cfg.device)
    args = wm_args_maniskill()
    args.num_frames = cfg.num_frames
    args.num_history = cfg.num_history
    args.num_inference_steps = cfg.num_inference_steps
    args.decode_chunk_size = cfg.decode_chunk_size

    all_psnrs, all_ssims, all_lpips = [], [], []
    per_traj = []
    traj_comparison_images: list = []  # PIL Images (only when visualize=True)

    for i, traj in enumerate(trajs):
        pred_lat, gt_lat = single_step_predict(model, traj, p01, p99, args, cfg.device)
        if pred_lat.shape[0] == 0:
            continue
        pred_img = decode_latents(pred_lat, model.vae, cfg.device, cfg.decode_chunk_size)
        gt_img = decode_latents(gt_lat, model.vae, cfg.device, cfg.decode_chunk_size)

        psnrs, ssims, lps = [], [], []
        for j in range(pred_img.shape[0]):
            psnrs.append(float(compute_psnr(gt_img[j], pred_img[j], data_range=255)))
            ssims.append(float(compute_ssim(gt_img[j], pred_img[j], data_range=255, channel_axis=2, win_size=7)))
            if lpips_model is not None:
                p_t = torch.from_numpy(pred_img[j]).permute(2, 0, 1).float() / 127.5 - 1.0
                g_t = torch.from_numpy(gt_img[j]).permute(2, 0, 1).float() / 127.5 - 1.0
                lps.append(float(lpips_model(p_t.unsqueeze(0).cuda(), g_t.unsqueeze(0).cuda()).item()))

        all_psnrs.extend(psnrs); all_ssims.extend(ssims); all_lpips.extend(lps)
        traj_entry: dict = {
            "key": traj["key"], "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)), "n_frames": len(psnrs),
        }
        if visualize:
            traj_entry["frame_psnrs"] = psnrs
            traj_entry["frame_ssims"] = ssims
            if lps:
                traj_entry["frame_lpips"] = lps
        per_traj.append(traj_entry)

        # ── 深度可视化: GT vs Pred 帧对比 ──
        if visualize and _PIL_AVAILABLE:
            comp_img = _make_traj_comparison(
                gt_frames=gt_img, pred_frames=pred_img,
                per_frame_psnr=psnrs, per_frame_ssim=ssims,
                traj_key=traj["key"],
                mean_psnr=float(np.mean(psnrs)),
                mean_ssim=float(np.mean(ssims)),
            )
            traj_comparison_images.append(comp_img)
            # 保存单条轨迹对比图
            viz_dir = Path(cfg.output_dir) / "viz" / ckpt_label
            viz_dir.mkdir(parents=True, exist_ok=True)
            comp_img.save(str(viz_dir / f"{traj['key']}_compare.png"))

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(trajs)}] PSNR={np.mean(psnrs):.2f} SSIM={np.mean(ssims):.4f}")

    # ── all_trajs_grid ──
    if visualize and _PIL_AVAILABLE and traj_comparison_images:
        grid_img = _make_all_trajs_grid(traj_comparison_images)
        grid_path = Path(cfg.output_dir) / "viz" / ckpt_label / "all_trajs_grid.png"
        grid_img.save(str(grid_path))
        print(f"[EVAL-WM] 🖼️ Grid: {grid_path} ({grid_img.width}x{grid_img.height})")

    result = {
        "ckpt_label": ckpt_label, "ckpt_path": ckpt_path,
        "n_trajs": len(per_traj), "n_frames": len(all_psnrs),
        "psnr_mean": float(np.mean(all_psnrs)) if all_psnrs else 0.0,
        "psnr_std": float(np.std(all_psnrs)) if all_psnrs else 0.0,
        "ssim_mean": float(np.mean(all_ssims)) if all_ssims else 0.0,
        "ssim_std": float(np.std(all_ssims)) if all_ssims else 0.0,
        "per_traj": per_traj,
    }
    if all_lpips:
        result["lpips_mean"] = float(np.mean(all_lpips))
        result["lpips_std"] = float(np.std(all_lpips))

    print(f"\n  {ckpt_label}: PSNR={result['psnr_mean']:.2f}±{result['psnr_std']:.2f}, "
          f"SSIM={result['ssim_mean']:.4f}±{result['ssim_std']:.4f}")
    del model; torch.cuda.empty_cache()
    return result


def generate_report(results: dict[str, dict], output_dir: str) -> str:
    lines = ["# WM Evaluation Report\n", f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n",
             "| Model | PSNR | SSIM | LPIPS | #Trajs | #Frames |",
             "|-------|------|------|-------|--------|---------|"]
    for label, r in results.items():
        lp = f"{r.get('lpips_mean', 0):.4f}" if "lpips_mean" in r else "N/A"
        lines.append(f"| {label} | {r['psnr_mean']:.2f}±{r['psnr_std']:.2f} | "
                     f"{r['ssim_mean']:.4f}±{r['ssim_std']:.4f} | {lp} | "
                     f"{r['n_trajs']} | {r['n_frames']} |")
    report = "\n".join(lines) + "\n"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write(report)
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="VLAW WM 定量评估 (稳定版)")
    parser.add_argument("--eval_h5", default="data/vlaw/encoded/eval/eval_set.h5")
    parser.add_argument("--stat_path", default="data/vlaw/meta_info/maniskill/stat.json")
    parser.add_argument("--checkpoint_paths", default="", help="逗号分隔的 ckpt 路径")
    parser.add_argument("--pretrained_path",
                        default="checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt")
    parser.add_argument("--output_dir", default="results/vlaw/wm_eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--num_history", type=int, default=6)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="生成 GT vs Pred 关键帧 + 指标条形图")
    args = parser.parse_args()

    cfg = EvalWMConfig(
        eval_h5=args.eval_h5, stat_path=args.stat_path,
        checkpoint_paths=args.checkpoint_paths, pretrained_path=args.pretrained_path,
        output_dir=args.output_dir, device=args.device,
        num_frames=args.num_frames, num_history=args.num_history,
        num_inference_steps=args.num_inference_steps,
    )

    trajs = load_eval_trajectories(cfg.eval_h5, cfg.min_traj_length)
    p01, p99 = load_norm_stats(cfg.stat_path)

    lpips_model = None
    if _LPIPS_AVAILABLE and not args.no_lpips:
        lpips_model = lpips.LPIPS(net="alex").to(cfg.device).eval()

    # 总是评估 pretrained baseline
    do_viz = args.visualize
    all_results: dict[str, dict] = {}
    if Path(cfg.pretrained_path).exists():
        all_results["pretrained"] = evaluate_checkpoint(
            cfg.pretrained_path, "pretrained", trajs, p01, p99, cfg, lpips_model,
            visualize=do_viz)

    # 额外 checkpoints
    if cfg.checkpoint_paths:
        for cp in cfg.checkpoint_paths.split(","):
            cp = cp.strip()
            if not cp:
                continue
            label = Path(cp).stem
            if not Path(cp).exists():
                print(f"[WARN] {cp} not found, skipping")
                continue
            all_results[label] = evaluate_checkpoint(
                cp, label, trajs, p01, p99, cfg, lpips_model, visualize=do_viz)

    # 保存
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    report = generate_report(all_results, cfg.output_dir)
    print(f"\n{report}")

    if args.visualize:
        _visualize_results(all_results, cfg.output_dir)

    print(f"[EVAL-WM] ✅ 结果保存至 {cfg.output_dir}")


# ── 可视化 ────────────────────────────────────────────────────────────────────

def _visualize_results(all_results: dict[str, dict], output_dir: str) -> None:
    """生成 GT vs Pred 关键帧对比 + PSNR/SSIM 汇总条形图.

    保存到 {output_dir}/viz/.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    labels = list(all_results.keys())
    if not labels:
        print("[EVAL-WM] ⚠️ 无结果可视化")
        return

    # ── 1. PSNR / SSIM 条形图 ──
    psnrs = [all_results[l]["psnr_mean"] for l in labels]
    psnr_stds = [all_results[l]["psnr_std"] for l in labels]
    ssims = [all_results[l]["ssim_mean"] for l in labels]
    ssim_stds = [all_results[l]["ssim_std"] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    x = np.arange(len(labels))

    axes[0].bar(x, psnrs, yerr=psnr_stds, capsize=4, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR Comparison")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(psnrs):
        axes[0].text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=7)

    axes[1].bar(x, ssims, yerr=ssim_stds, capsize=4, color="seagreen")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM Comparison")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(ssims):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=7)

    plt.tight_layout()
    fig_path = viz_dir / "psnr_ssim_bar.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EVAL-WM] 📊 条形图: {fig_path}")

    # ── 2. Per-traj PSNR 散点图 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in labels:
        per_traj = all_results[label].get("per_traj", [])
        if per_traj:
            traj_psnrs = [t["psnr"] for t in per_traj]
            ax.scatter(range(len(traj_psnrs)), traj_psnrs, label=label, alpha=0.7, s=20)
    ax.set_xlabel("Trajectory Index")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-trajectory PSNR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = viz_dir / "per_traj_psnr.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EVAL-WM] 📊 散点图: {fig_path}")

    # ── 3. Per-frame PSNR 分解 (仅当 frame_psnrs 存在时) ──
    has_frame_data = any(
        "frame_psnrs" in t
        for r in all_results.values()
        for t in r.get("per_traj", [])
    )
    if has_frame_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label in labels:
            per_traj = all_results[label].get("per_traj", [])
            # Collect per-frame across all trajs
            frame_lists: dict[int, list[float]] = {}
            for t in per_traj:
                for fi, pval in enumerate(t.get("frame_psnrs", [])):
                    frame_lists.setdefault(fi, []).append(pval)
            if frame_lists:
                frame_ids = sorted(frame_lists.keys())
                means = [float(np.mean(frame_lists[fi])) for fi in frame_ids]
                stds = [float(np.std(frame_lists[fi])) for fi in frame_ids]
                ax.errorbar(
                    frame_ids, means, yerr=stds, marker="o", capsize=3,
                    label=label, linewidth=2, markersize=5,
                )
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Per-frame PSNR Decay (F0=current frame, F1-F4=predicted)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = viz_dir / "per_frame_psnr_decay.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[EVAL-WM] 📊 Per-frame decay: {fig_path}")


if __name__ == "__main__":
    main()
