#!/usr/bin/env python3
"""Visualize WM inference results: compare timing_test (num_interact=4, 20 frames)
vs 002b baseline (num_interact=12, 60 frames).

Decodes latent trajectories to RGB images for human inspection.
For each dataset, selects:
  - Top-K p_yes trajectories (best quality according to VLM)
  - Bottom-K p_yes trajectories (worst quality)
  - Random sample

Creates:
  1. Per-trajectory strip images (all frames side-by-side)
  2. Comparison grid (timing vs baseline, matched by VLM rank)
  3. Frame-by-frame quality progression

Usage:
    CUDA_VISIBLE_DEVICES=4 conda run -n rlft_ms3 python scripts/vlaw/diag/visualize_wm_comparison.py \
        --output_dir results/vlaw/wm_visual_comparison/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="results/vlaw/wm_visual_comparison/")
    p.add_argument("--timing_h5", type=str,
                   default="data/vlaw/synthetic/iter1_timing_test/synthetic_iter1_final_1772601046.h5")
    p.add_argument("--timing_scores", type=str,
                   default="data/vlaw/labeled/synthetic_iter1_timing_test/vlm_scores.json")
    p.add_argument("--baseline_h5", type=str,
                   default="data/vlaw/synthetic/iter1_002b_sliding/synthetic_iter1_final_1772382989.h5")
    p.add_argument("--baseline_scores", type=str,
                   default="data/vlaw/labeled/synthetic_iter1_002b_sliding/LiftPegUpright-v1_vlm_rewards.json")
    p.add_argument("--top_k", type=int, default=5, help="Top/bottom K trajectories to visualize")
    p.add_argument("--n_random", type=int, default=3, help="Number of random trajectories")
    p.add_argument("--max_frames_strip", type=int, default=20, help="Max frames to show in a strip")
    p.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_vae(model_name: str, device: torch.device):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_name, torch_dtype=torch.float16)
    vae = vae.to(device).eval()
    return vae


def decode_latent(vae, latent: np.ndarray, device: torch.device) -> np.ndarray:
    """Decode (4, 48, 24) latent -> (H, W, 3) uint8 RGB."""
    lat = torch.from_numpy(latent).unsqueeze(0).to(device=device, dtype=torch.float16)
    lat = lat / 0.18215  # SVD VAE scaling
    with torch.no_grad():
        decoded = vae.decode(lat).sample
    img = decoded[0].float().clamp(-1, 1)
    img = ((img + 1) / 2 * 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    return img


def get_font():
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except OSError:
            return ImageFont.load_default()


def make_strip(frames: list[np.ndarray], labels: list[str], title: str = "",
               max_frames: int = 20) -> Image.Image:
    """Create a horizontal strip of frames with labels."""
    # Subsample if too many frames
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
        labels = [labels[i] for i in indices]

    font = get_font()
    gap = 2
    title_h = 22 if title else 0
    label_h = 18
    h = frames[0].shape[0]
    w = frames[0].shape[1]
    strip_w = len(frames) * w + (len(frames) - 1) * gap
    strip_h = h + title_h + label_h

    strip = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
    draw = ImageDraw.Draw(strip)

    if title:
        draw.text((4, 2), title, fill=(0, 0, 0), font=font)

    for i, (frame, label) in enumerate(zip(frames, labels)):
        x = i * (w + gap)
        strip.paste(Image.fromarray(frame), (x, title_h))
        draw.text((x + 2, title_h + h + 2), label, fill=(80, 80, 80), font=font)

    return strip


def process_dataset(name: str, h5_path: str, scores_path: str, vae, device,
                    out_dir: Path, top_k: int, n_random: int, max_frames: int,
                    seed: int) -> dict:
    """Process one dataset: decode top/bottom/random trajectories."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"  H5: {h5_path}")
    print(f"  Scores: {scores_path}")

    # Load scores (handle both formats: p_yes and vlm_yes_prob)
    with open(scores_path) as f:
        scores = json.load(f)

    # Normalize score key
    for s in scores:
        if "p_yes" not in s and "vlm_yes_prob" in s:
            s["p_yes"] = s["vlm_yes_prob"]
        if "T" not in s:
            s["T"] = 0

    # Sort by p_yes
    scores_sorted = sorted(scores, key=lambda x: x["p_yes"], reverse=True)

    # Select trajectories
    top_trajs = scores_sorted[:top_k]
    bottom_trajs = scores_sorted[-top_k:]

    rng = np.random.RandomState(seed)
    mid_indices = rng.choice(range(top_k, max(top_k + 1, len(scores) - top_k)),
                             size=min(n_random, len(scores) - 2 * top_k), replace=False)
    random_trajs = [scores_sorted[i] for i in sorted(mid_indices)]

    dataset_dir = out_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    f = h5py.File(h5_path, "r")

    # Stats
    all_p_yes = [s["p_yes"] for s in scores]
    stats = {
        "name": name,
        "n_total": len(scores),
        "p_yes_mean": float(np.mean(all_p_yes)),
        "p_yes_median": float(np.median(all_p_yes)),
        "p_yes_max": float(np.max(all_p_yes)),
        "p_yes_min": float(np.min(all_p_yes)),
        "d_syn_plus_04": sum(1 for x in all_p_yes if x >= 0.4),
    }

    def decode_and_save(traj_info: dict, category: str, rank: int):
        key = traj_info["traj_key"]
        p_yes = traj_info["p_yes"]
        T = traj_info.get("T", 0)

        if key not in f:
            print(f"  WARNING: {key} not in H5, skipping")
            return None

        latents = f[key]["latent"][:]  # (T, 4, 48, 24)
        T = latents.shape[0]

        print(f"  {category}#{rank}: {key}, T={T}, p_yes={p_yes:.4f}")

        frames_full = []
        frames_base = []
        frames_render = []
        labels = []

        for t in range(T):
            rgb_full = decode_latent(vae, latents[t], device)
            H_full = rgb_full.shape[0]
            H_half = H_full // 2
            base = rgb_full[:H_half]
            render = rgb_full[H_half:]

            frames_full.append(rgb_full)
            frames_base.append(base)
            frames_render.append(render)
            labels.append(f"t={t}")

            # Save individual frames for first/mid/last
            if t in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
                cat_dir = dataset_dir / category
                cat_dir.mkdir(exist_ok=True)
                Image.fromarray(base).save(cat_dir / f"rank{rank}_{key}_t{t:03d}_base.png")
                Image.fromarray(render).save(cat_dir / f"rank{rank}_{key}_t{t:03d}_render.png")

        # Create strips (base camera — more informative)
        title_base = f"{name} | {category}#{rank} | {key} | T={T} | p_yes={p_yes:.4f} [BASE CAM]"
        strip_base = make_strip(frames_base, labels, title_base, max_frames)
        strip_base.save(dataset_dir / f"{category}_rank{rank:02d}_{key}_base_strip.png")

        title_render = f"{name} | {category}#{rank} | {key} | T={T} | p_yes={p_yes:.4f} [RENDER CAM]"
        strip_render = make_strip(frames_render, labels, title_render, max_frames)
        strip_render.save(dataset_dir / f"{category}_rank{rank:02d}_{key}_render_strip.png")

        return {"key": key, "T": T, "p_yes": p_yes, "category": category, "rank": rank}

    results = []

    print(f"\n  --- Top {top_k} (highest p_yes) ---")
    for i, traj in enumerate(top_trajs):
        r = decode_and_save(traj, "top", i + 1)
        if r:
            results.append(r)

    print(f"\n  --- Bottom {top_k} (lowest p_yes) ---")
    for i, traj in enumerate(bottom_trajs):
        r = decode_and_save(traj, "bottom", i + 1)
        if r:
            results.append(r)

    print(f"\n  --- Random {n_random} ---")
    for i, traj in enumerate(random_trajs):
        r = decode_and_save(traj, "random", i + 1)
        if r:
            results.append(r)

    f.close()

    stats["decoded_trajs"] = results
    return stats


def create_comparison_grid(out_dir: Path, timing_stats: dict, baseline_stats: dict):
    """Create a side-by-side comparison of top trajectories from both datasets."""
    font = get_font()
    
    # Find top strip images from each
    timing_dir = out_dir / timing_stats["name"]
    baseline_dir = out_dir / baseline_stats["name"]
    
    timing_tops = sorted(timing_dir.glob("top_rank*_base_strip.png"))[:5]
    baseline_tops = sorted(baseline_dir.glob("top_rank*_base_strip.png"))[:5]
    
    if not timing_tops or not baseline_tops:
        print("WARNING: No strip images found for comparison grid")
        return
    
    # Load all strips
    all_strips = []
    for tp, bp in zip(timing_tops, baseline_tops):
        t_img = Image.open(tp)
        b_img = Image.open(bp)
        all_strips.append((t_img, b_img))
    
    if not all_strips:
        return
    
    # Stack vertically: timing_top1, baseline_top1, timing_top2, baseline_top2, ...
    gap = 6
    max_w = max(max(t.width, b.width) for t, b in all_strips)
    total_h = sum(t.height + b.height + gap * 2 for t, b in all_strips)
    
    grid = Image.new("RGB", (max_w, total_h), (240, 240, 240))
    draw = ImageDraw.Draw(grid)
    y = 0
    
    for i, (t_img, b_img) in enumerate(all_strips):
        # Paste timing strip
        grid.paste(t_img, (0, y))
        y += t_img.height + gap
        # Paste baseline strip
        grid.paste(b_img, (0, y))
        y += b_img.height + gap
    
    grid.save(out_dir / "comparison_grid_top5.png")
    print(f"\nComparison grid saved: {out_dir / 'comparison_grid_top5.png'}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VAE on {device} ...")
    vae = load_vae(args.vae_model, device)
    print("VAE loaded.\n")

    # Check which files exist
    timing_h5 = args.timing_h5
    baseline_h5 = args.baseline_h5
    
    # Auto-detect baseline H5 if exact path doesn't exist
    if not os.path.exists(baseline_h5):
        import glob
        candidates = glob.glob("data/vlaw/synthetic/iter1_002b_sliding/synthetic_iter1_final_*.h5")
        if candidates:
            baseline_h5 = candidates[0]
            print(f"Auto-detected baseline H5: {baseline_h5}")

    # Process both datasets
    results = {}

    if os.path.exists(timing_h5) and os.path.exists(args.timing_scores):
        timing_stats = process_dataset(
            "timing_test_interact4",
            timing_h5, args.timing_scores, vae, device, out_dir,
            args.top_k, args.n_random, args.max_frames_strip, args.seed
        )
        results["timing"] = timing_stats
    else:
        print(f"WARNING: timing test files not found, skipping")

    if os.path.exists(baseline_h5) and os.path.exists(args.baseline_scores):
        baseline_stats = process_dataset(
            "baseline_002b_interact12",
            baseline_h5, args.baseline_scores, vae, device, out_dir,
            args.top_k, args.n_random, args.max_frames_strip, args.seed
        )
        results["baseline"] = baseline_stats
    else:
        print(f"WARNING: baseline files not found, skipping")

    # Create comparison grid if both available
    if "timing" in results and "baseline" in results:
        create_comparison_grid(out_dir, results["timing"], results["baseline"])

    # Save summary
    summary = {
        "timing": {k: v for k, v in results.get("timing", {}).items() if k != "decoded_trajs"},
        "baseline": {k: v for k, v in results.get("baseline", {}).items() if k != "decoded_trajs"},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Timing (interact=4)':<25} {'Baseline (interact=12)':<25}")
    print("-" * 70)
    for key in ["n_total", "p_yes_mean", "p_yes_median", "p_yes_max", "p_yes_min", "d_syn_plus_04"]:
        t_val = results.get("timing", {}).get(key, "N/A")
        b_val = results.get("baseline", {}).get(key, "N/A")
        if isinstance(t_val, float):
            t_val = f"{t_val:.4f}"
            b_val = f"{b_val:.4f}" if isinstance(b_val, float) else b_val
        print(f"{key:<25} {str(t_val):<25} {str(b_val):<25}")

    print(f"\nOutput directory: {out_dir}")
    print("Files created:")
    for f_path in sorted(out_dir.rglob("*strip*.png")):
        print(f"  {f_path.relative_to(out_dir)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
