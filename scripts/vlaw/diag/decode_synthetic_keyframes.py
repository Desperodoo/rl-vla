#!/usr/bin/env python3
"""T-DIAG-SYN-001: Decode key frames from synthetic trajectories.

Randomly sample 8 trajectories from 200 synthetic rollouts,
extract key frames at t=0, T/4, T/2, 3T/4, T-1,
decode latent → RGB via VAE, split top/bottom (base_cam / render_cam),
save individual PNGs and per-trajectory strip images.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/vlaw/diag/decode_synthetic_keyframes.py \
        --h5_path data/vlaw/synthetic/iter1_wm_real/synthetic_iter1_final_1772339809.h5 \
        --output_dir results/vlaw/dsyn_diagnosis_frames/synthetic/ \
        --num_sample 8 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode keyframes from synthetic latent trajectories")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the HDF5 file with synthetic data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for decoded PNGs")
    parser.add_argument("--num_sample", type=int, default=8, help="Number of trajectories to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="VAE model name or local path")
    return parser.parse_args()


def load_vae(model_name: str, device: torch.device) -> torch.nn.Module:
    """Load the VAE decoder."""
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_name, torch_dtype=torch.float16)
    vae = vae.to(device)
    vae.eval()
    return vae


def decode_latent(vae: torch.nn.Module, latent: np.ndarray, device: torch.device) -> np.ndarray:
    """Decode a latent tensor (4, 48, 24) to RGB image(s).

    The latent is (4, H_lat, W_lat) where H_lat=48 encodes a vertically
    concatenated pair of cameras (base_cam top 24 rows, render_cam bottom 24 rows).
    We decode the full latent and then split the resulting image.

    Returns:
        rgb_full: (H_full, W_full, 3) uint8 image (full decoded, before split)
    """
    # VAE expects (B, C, H, W)
    lat_tensor = torch.from_numpy(latent).unsqueeze(0).to(device=device, dtype=torch.float16)

    # SVD VAE scaling factor
    scaling_factor = 0.18215
    lat_tensor = lat_tensor / scaling_factor

    with torch.no_grad():
        decoded = vae.decode(lat_tensor).sample  # (1, 3, H, W)

    # Convert to uint8 image
    img = decoded[0].float().clamp(-1, 1)
    img = ((img + 1) / 2 * 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    return img


def get_keyframe_indices(T: int) -> list[int]:
    """Get key frame indices: t=0, T/4, T/2, 3T/4, T-1."""
    indices = [
        0,
        T // 4,
        T // 2,
        3 * T // 4,
        T - 1,
    ]
    # Deduplicate while preserving order (for very short trajectories)
    seen = set()
    unique = []
    for idx in indices:
        idx = min(idx, T - 1)
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


def main() -> None:
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open HDF5
    print(f"Opening {args.h5_path} ...")
    f = h5py.File(args.h5_path, "r")
    traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
    print(f"  Found {len(traj_keys)} trajectories")

    # Sample
    sampled_keys = sorted(random.sample(traj_keys, min(args.num_sample, len(traj_keys))))
    print(f"  Sampled {len(sampled_keys)} trajectories: {sampled_keys}")

    # Load VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VAE ({args.vae_model}) on {device} ...")
    vae = load_vae(args.vae_model, device)
    print("  VAE loaded.")

    summary_lines: list[str] = []

    for traj_key in sampled_keys:
        traj_idx = int(traj_key.split("_")[1])
        latents = f[traj_key]["latent"][:]  # (T, 4, 48, 24) float16
        T = latents.shape[0]
        keyframes = get_keyframe_indices(T)

        print(f"\n{traj_key}: T={T}, keyframes={keyframes}")
        traj_info = f"  traj_{traj_idx:04d}: T={T}, keyframes={keyframes}"

        strip_images: list[Image.Image] = []

        for t_idx in keyframes:
            lat = latents[t_idx]  # (4, 48, 24)
            rgb_full = decode_latent(vae, lat, device)  # (H, W, 3)
            H_full, W_full = rgb_full.shape[:2]

            # Split top/bottom: base_cam (top half), render_cam (bottom half)
            H_half = H_full // 2
            base_cam = rgb_full[:H_half]
            render_cam = rgb_full[H_half:]

            # Save individual frames
            # base cam
            base_path = out_dir / f"traj_{traj_idx:04d}_t{t_idx:03d}_base.png"
            Image.fromarray(base_cam).save(base_path)
            # render cam
            render_path = out_dir / f"traj_{traj_idx:04d}_t{t_idx:03d}_render.png"
            Image.fromarray(render_cam).save(render_path)
            # combined (stacked vertically — original decoded image)
            combined_path = out_dir / f"traj_{traj_idx:04d}_t{t_idx:03d}.png"
            Image.fromarray(rgb_full).save(combined_path)

            print(f"  t={t_idx}: decoded {H_full}x{W_full} → base {base_cam.shape[:2]}, render {render_cam.shape[:2]}")

            # For strip: combine base and render side by side for this timestep
            # Create a panel: base on top, render on bottom, with label
            panel = Image.fromarray(rgb_full)
            strip_images.append(panel)

        # Create strip: all keyframes side by side
        if strip_images:
            widths = [img.width for img in strip_images]
            heights = [img.height for img in strip_images]
            gap = 4  # pixel gap between frames
            strip_w = sum(widths) + gap * (len(strip_images) - 1)
            strip_h = max(heights) + 30  # extra space for labels
            strip = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))

            x_offset = 0
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(strip)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except OSError:
                font = ImageFont.load_default()

            for i, (img, t_idx) in enumerate(zip(strip_images, keyframes)):
                strip.paste(img, (x_offset, 0))
                # Label
                label = f"t={t_idx}"
                draw.text((x_offset + 2, max(heights) + 4), label, fill=(0, 0, 0), font=font)
                x_offset += img.width + gap

            strip_path = out_dir / f"traj_{traj_idx:04d}_strip.png"
            strip.save(strip_path)
            print(f"  Strip saved: {strip_path} ({strip_w}x{strip_h})")

        summary_lines.append(traj_info)

    f.close()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print(f"Sampled trajectories: {len(sampled_keys)}")
    for line in summary_lines:
        print(line)
    total_files = len(list(out_dir.glob("*.png")))
    print(f"Total PNG files: {total_files}")
    print("Done!")


if __name__ == "__main__":
    main()
