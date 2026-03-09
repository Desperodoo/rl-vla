#!/usr/bin/env python3
"""Dual-camera visualization for imagination ablation comparison.

Takes multiple experiment output dirs and generates side-by-side comparison
strips showing both top and side camera views across all configurations.

Usage:
    python scripts/viz_ablation_comparison.py \
        --dirs data/vlaw/synthetic/wm_eval_step1200_short \
               data/vlaw/synthetic/ablation_steps50 \
               data/vlaw/synthetic/ablation_cfg5 \
               data/vlaw/synthetic/ablation_cfg1 \
        --labels "baseline(steps25,cfg3)" "steps50" "cfg5.0" "cfg1.0" \
        --output_dir data/vlaw/synthetic/ablation_comparison \
        --num_trajs 5 --num_frames 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

WORKSPACE = Path("/home/wjz/rl-vla")


@torch.inference_mode()
def decode_latent_dual(
    vae, latents: np.ndarray, frame_indices: np.ndarray,
    device: str, chunk_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode VAE latents into top-camera and side-camera RGB arrays.

    Returns:
        top_rgb: (N, 192, 192, 3) uint8
        side_rgb: (N, 192, 192, 3) uint8
    """
    selected = torch.from_numpy(latents[frame_indices]).to(device).to(torch.float16)
    decoded_list = []
    for i in range(0, selected.shape[0], chunk_size):
        chunk = selected[i:i + chunk_size] / vae.config.scaling_factor
        out = vae.decode(chunk, num_frames=chunk.shape[0]).sample
        decoded_list.append(out)
    decoded = torch.cat(decoded_list, dim=0)
    decoded = (decoded / 2.0 + 0.5).clamp(0, 1) * 255
    rgb = decoded.float().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    # latent height 48 -> pixel 384 (2 cameras stacked), width 24 -> 192
    top_rgb = rgb[:, :192, :, :]   # top camera
    side_rgb = rgb[:, 192:, :, :]  # side camera
    return top_rgb, side_rgb


def find_h5(exp_dir: Path) -> Path | None:
    """Find the H5 file in an experiment directory."""
    candidates = list(exp_dir.glob("synthetic_*.h5"))
    if candidates:
        return candidates[0]
    candidates = list(exp_dir.glob("*.h5"))
    if candidates:
        return candidates[0]
    return None


def create_comparison_image(
    all_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    traj_key: str,
    labels: list[str],
    num_frames: int = 6,
) -> np.ndarray:
    """Create a comparison image for one trajectory across all experiments.

    Layout:
        Row 0: Label header
        Row 1: Exp1 top camera frames
        Row 2: Exp1 side camera frames
        Row 3: Exp2 top camera frames
        Row 4: Exp2 side camera frames
        ...

    Returns: (H, W, 3) uint8 array
    """
    from PIL import Image, ImageDraw, ImageFont

    frame_h, frame_w = 192, 192
    label_h = 30
    gap = 4
    n_exps = len(labels)

    total_w = num_frames * frame_w + (num_frames - 1) * gap
    # Each exp: label_h + frame_h (top) + 2px gap + frame_h (side) + gap
    total_h = n_exps * (2 * frame_h + label_h + 2 + gap) + gap

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240  # light gray bg

    y = gap
    for i, label in enumerate(labels):
        exp_key = label
        # Draw label
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((10, y + 5), f"{label} | {traj_key}", fill=(0, 0, 0), font=font)
        canvas = np.array(img)
        y += label_h

        if exp_key in all_data and traj_key in all_data[exp_key]:
            top_rgb, side_rgb = all_data[exp_key][traj_key]
            T = top_rgb.shape[0]
            # Select evenly spaced frames
            if T >= num_frames:
                idxs = np.linspace(0, T - 1, num_frames, dtype=int)
            else:
                idxs = np.arange(T)

            # Top camera row
            for j, idx in enumerate(idxs):
                x = j * (frame_w + gap)
                canvas[y:y + frame_h, x:x + frame_w] = top_rgb[idx]
            y += frame_h + 2

            # Side camera row
            for j, idx in enumerate(idxs):
                x = j * (frame_w + gap)
                canvas[y:y + frame_h, x:x + frame_w] = side_rgb[idx]
            y += frame_h + gap
        else:
            # Missing data
            y += 2 * frame_h + gap + 2

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Ablation comparison visualization")
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="Experiment output directories")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each experiment")
    parser.add_argument("--output_dir", type=str,
                        default="data/vlaw/synthetic/ablation_comparison")
    parser.add_argument("--num_trajs", type=int, default=5,
                        help="Number of trajectories to visualize")
    parser.add_argument("--num_frames", type=int, default=6,
                        help="Number of frames per trajectory strip")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    from PIL import Image
    from diffusers.models import AutoencoderKLTemporalDecoder

    device = f"cuda:{args.gpu_id}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert len(args.dirs) == len(args.labels), "dirs and labels must match"

    # Load VAE
    vae_path = str(WORKSPACE / "checkpoints/vlaw/world_model/pretrained"
                   "/stable-video-diffusion-img2vid/vae")
    if not Path(vae_path).exists():
        vae_path = str(WORKSPACE / "checkpoints/vlaw/world_model/pretrained"
                       "/stable-video-diffusion-img2vid")
    print(f"Loading VAE from {vae_path}...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        vae_path, torch_dtype=torch.float16).to(device).eval()

    # Load all H5 data
    all_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    all_traj_keys: set[str] = set()

    for exp_dir_str, label in zip(args.dirs, args.labels):
        exp_dir = Path(exp_dir_str)
        h5_file = find_h5(exp_dir)
        if h5_file is None:
            print(f"[WARN] No H5 found in {exp_dir}, skipping")
            continue

        print(f"Processing {label}: {h5_file}")
        exp_data = {}
        with h5py.File(str(h5_file), "r") as f:
            traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
            n = min(args.num_trajs, len(traj_keys))
            for i in range(n):
                key = traj_keys[i]
                latents = f[key]["latent"][:].astype(np.float32)
                T = latents.shape[0]
                all_idxs = np.arange(T)
                top_rgb, side_rgb = decode_latent_dual(vae, latents, all_idxs, device)
                exp_data[key] = (top_rgb, side_rgb)
                all_traj_keys.add(key)
                print(f"  Decoded {key}: {T} frames")

        all_data[label] = exp_data

    # Generate comparison images
    common_keys = sorted(all_traj_keys)[:args.num_trajs]
    for key in common_keys:
        canvas = create_comparison_image(all_data, key, args.labels, args.num_frames)
        save_path = out_dir / f"{key}_comparison.png"
        Image.fromarray(canvas).save(str(save_path))
        print(f"Saved {save_path}")

    # Also generate per-experiment dual-camera strips
    for label in args.labels:
        if label not in all_data:
            continue
        exp_viz_dir = out_dir / label.replace("(", "_").replace(")", "_").replace(",", "_")
        exp_viz_dir.mkdir(parents=True, exist_ok=True)
        for key, (top_rgb, side_rgb) in all_data[label].items():
            T = top_rgb.shape[0]
            num_f = min(args.num_frames, T)
            idxs = np.linspace(0, T - 1, num_f, dtype=int) if T >= num_f else np.arange(T)
            # Top row + side row
            top_strip = np.concatenate([top_rgb[i] for i in idxs], axis=1)
            side_strip = np.concatenate([side_rgb[i] for i in idxs], axis=1)
            dual = np.concatenate([top_strip, side_strip], axis=0)
            Image.fromarray(dual).save(str(exp_viz_dir / f"{key}_dual.png"))
        print(f"Saved per-exp dual strips for {label} -> {exp_viz_dir}")

    del vae
    torch.cuda.empty_cache()
    print(f"\nAll comparison images saved to {out_dir}")


if __name__ == "__main__":
    main()
