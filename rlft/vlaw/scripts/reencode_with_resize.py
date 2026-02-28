#!/usr/bin/env python3
"""
reencode_with_resize.py — 重编码 HDF5 轨迹，支持 128×128 → 192×192 resize

将 rgb_base/rgb_render 帧 resize 到统一分辨率后 VAE 编码，
输出 latent shape (4, 48, 24)。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

WORKSPACE = Path(__file__).parents[3].resolve()
sys.path.insert(0, str(WORKSPACE))


def resize_frames(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize (T, H, W, 3) uint8 frames to (T, target_h, target_w, 3)."""
    T, H, W, C = frames.shape
    if H == target_h and W == target_w:
        return frames
    out = np.empty((T, target_h, target_w, C), dtype=np.uint8)
    for i in range(T):
        img = Image.fromarray(frames[i])
        img = img.resize((target_w, target_h), Image.BILINEAR)
        out[i] = np.array(img)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-encode HDF5 with resize")
    parser.add_argument("--input_dirs", type=str, nargs="+", required=True,
                        help="Input directories containing HDF5 files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for encoded HDF5")
    parser.add_argument("--gpu_id", type=int, default=4)
    parser.add_argument("--target_h", type=int, default=192)
    parser.add_argument("--target_w", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--vae_path", type=str,
                        default="checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid/vae")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda")

    # Load VAE
    from diffusers import AutoencoderKL
    vae_path = WORKSPACE / args.vae_path
    print(f"Loading VAE from {vae_path}")
    vae = AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    tgt_h, tgt_w = args.target_h, args.target_w
    concat_h = tgt_h * 2  # vertical concat
    concat_w = tgt_w
    lat_h, lat_w = concat_h // 8, concat_w // 8
    expected_lat_shape = (4, lat_h, lat_w)
    print(f"Target: {tgt_h}×{tgt_w} → concat {concat_h}×{concat_w} → latent {expected_lat_shape}")

    output_base = Path(args.output_dir)
    total_trajs = 0
    total_resized = 0
    total_files = 0

    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        h5_files = sorted(input_path.glob("**/*.h5"))
        print(f"\nProcessing {input_dir}: {len(h5_files)} files")

        for h5_file in h5_files:
            # Determine output path preserving task subfolder
            # e.g., data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1/file.h5
            # → output_dir/LiftPegUpright-v1/file.h5
            # Find task folder name
            rel_parts = h5_file.relative_to(input_path).parts
            if len(rel_parts) > 1:
                task_dir = rel_parts[0]
                out_dir = output_base / task_dir
            else:
                out_dir = output_base
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / h5_file.name

            # Copy source file first
            shutil.copy2(str(h5_file), str(out_path))

            with h5py.File(str(h5_file), "r") as f_src:
                traj_keys = sorted([k for k in f_src.keys() if k.startswith("traj_")])

            print(f"\n  {h5_file.name}: {len(traj_keys)} trajs")

            with h5py.File(str(out_path), "a") as f_dst:
                for tkey in traj_keys:
                    with h5py.File(str(h5_file), "r") as f_src:
                        grp = f_src[tkey]
                        rgb_base = grp["rgb_base"][:]      # (T, H, W, 3)
                        rgb_render = grp["rgb_render"][:]   # (T, H, W, 3)

                    orig_h, orig_w = rgb_base.shape[1], rgb_base.shape[2]
                    needs_resize = (orig_h != tgt_h or orig_w != tgt_w)

                    if needs_resize:
                        rgb_base = resize_frames(rgb_base, tgt_h, tgt_w)
                        rgb_render = resize_frames(rgb_render, tgt_h, tgt_w)
                        total_resized += 1

                    # Vertical concat
                    concat = np.concatenate([rgb_base, rgb_render], axis=1)
                    # (T, 2*tgt_h, tgt_w, 3)

                    # VAE encode in batches
                    T = concat.shape[0]
                    latents = []
                    for start in range(0, T, args.batch_size):
                        batch = concat[start:start + args.batch_size]
                        x = torch.from_numpy(batch).float().to(device)
                        x = x.permute(0, 3, 1, 2) / 255.0
                        x = x * 2.0 - 1.0
                        with torch.no_grad():
                            dist = vae.encode(x).latent_dist
                            z = dist.sample() * vae.config.scaling_factor
                        latents.append(z.cpu().to(torch.float16).numpy())

                    latent = np.concatenate(latents, axis=0)  # (T, 4, lat_h, lat_w)
                    assert latent.shape[1:] == expected_lat_shape, \
                        f"Latent shape mismatch: {latent.shape[1:]} vs {expected_lat_shape}"

                    # Write to HDF5
                    grp_dst = f_dst[tkey]

                    # If resized, also update rgb in the output file
                    if needs_resize:
                        if "rgb_base" in grp_dst:
                            del grp_dst["rgb_base"]
                        grp_dst.create_dataset("rgb_base", data=resize_frames(
                            h5py.File(str(h5_file), "r")[tkey]["rgb_base"][:], tgt_h, tgt_w))
                        if "rgb_render" in grp_dst:
                            del grp_dst["rgb_render"]
                        grp_dst.create_dataset("rgb_render", data=resize_frames(
                            h5py.File(str(h5_file), "r")[tkey]["rgb_render"][:], tgt_h, tgt_w))

                    if "latent_concat" in grp_dst:
                        del grp_dst["latent_concat"]
                    grp_dst.create_dataset(
                        "latent_concat", data=latent,
                        chunks=True, compression="gzip", compression_opts=1,
                    )
                    grp_dst.attrs["latent_shape"] = str(latent.shape)
                    total_trajs += 1

            total_files += 1
            print(f"  → {out_path} ({total_trajs} trajs so far, {total_resized} resized)")

    print(f"\n{'='*60}")
    print(f"Total: {total_files} files, {total_trajs} trajs, {total_resized} resized")
    print(f"Output: {output_base}")


if __name__ == "__main__":
    main()
