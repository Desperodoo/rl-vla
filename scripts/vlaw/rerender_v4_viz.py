#!/usr/bin/env python3
"""Re-render WM v4 imagination visualizations with dual-camera strips.

Also computes basic quality metrics (brightness, sharpness) per checkpoint.
Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/vlaw/rerender_v4_viz.py --gpu_id 0
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image as PILImage

WORKSPACE = Path("/home/wjz/rl-vla")
SYNTH_BASE = WORKSPACE / "data/vlaw/synthetic"
STEPS = list(range(200, 4001, 200))  # 200, 400, ..., 4000
VIS_COUNT = 5  # trajectories per checkpoint


def compute_sharpness(img: np.ndarray) -> float:
    """Laplacian variance as sharpness metric."""
    gray = img.mean(axis=-1) if img.ndim == 3 else img
    lap = np.zeros_like(gray, dtype=np.float64)
    lap[1:-1, 1:-1] = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(np.var(lap))


@torch.inference_mode()
def decode_latents(vae, latents: np.ndarray, frame_indices: np.ndarray,
                   device: str) -> np.ndarray:
    """Decode (N, 4, 48, 24) latents → (N, H_full, W, 3) uint8."""
    selected = torch.from_numpy(latents[frame_indices]).to(device).to(torch.float16)
    chunks = []
    for i in range(0, selected.shape[0], 4):
        chunk = selected[i:i+4] / vae.config.scaling_factor
        out = vae.decode(chunk, num_frames=chunk.shape[0]).sample
        chunks.append(out)
    decoded = torch.cat(chunks, dim=0)
    decoded = (decoded / 2.0 + 0.5).clamp(0, 1) * 255
    return decoded.float().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Checkpoint steps to render (default: all 20)")
    parser.add_argument("--vis_count", type=int, default=VIS_COUNT)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    steps = args.steps or STEPS

    # Load VAE
    from diffusers.models import AutoencoderKLTemporalDecoder
    vae_path = str(WORKSPACE / "checkpoints/vlaw/world_model/pretrained"
                   "/stable-video-diffusion-img2vid/vae")
    if not Path(vae_path).exists():
        vae_path = str(WORKSPACE / "checkpoints/vlaw/world_model/pretrained"
                       "/stable-video-diffusion-img2vid")
    print(f"Loading VAE from {vae_path} ...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        vae_path, torch_dtype=torch.float16).to(device).eval()

    metrics_all = {}

    for step in steps:
        synth_dir = SYNTH_BASE / f"v4_eval_step{step}"
        h5_files = list(synth_dir.glob("synthetic_final_*.h5"))
        if not h5_files:
            print(f"[step-{step}] No HDF5 found, skipping")
            continue

        h5_path = h5_files[0]
        viz_dir = synth_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        step_metrics = {"sharpness_base": [], "sharpness_render": [],
                        "brightness_base": [], "brightness_render": []}

        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
            n = min(args.vis_count, len(traj_keys))

            for i in range(n):
                grp = f[traj_keys[i]]
                latents = grp["latent"][:].astype(np.float32)
                T = latents.shape[0]
                idxs = np.array([0, T // 2, T - 1]) if T >= 3 else np.arange(T)
                rgb = decode_latents(vae, latents, idxs, device)  # (N, H_full, W, 3)

                H_half = rgb.shape[1] // 2
                base = rgb[:, :H_half, :, :]
                render = rgb[:, H_half:, :, :]

                # Dual-camera strip: base on top, render on bottom
                strip_base = np.concatenate([base[j] for j in range(base.shape[0])], axis=1)
                strip_render = np.concatenate([render[j] for j in range(render.shape[0])], axis=1)
                strip = np.concatenate([strip_base, strip_render], axis=0)

                save_path = viz_dir / f"{traj_keys[i]}_strip.png"
                PILImage.fromarray(strip).save(str(save_path))

                # Metrics on last frame (most evolved)
                for cam_name, cam_data in [("base", base[-1]), ("render", render[-1])]:
                    step_metrics[f"sharpness_{cam_name}"].append(compute_sharpness(cam_data))
                    step_metrics[f"brightness_{cam_name}"].append(float(cam_data.mean()))

        avg_metrics = {k: float(np.mean(v)) for k, v in step_metrics.items()}
        metrics_all[step] = avg_metrics
        print(f"[step-{step:4d}] sharp_base={avg_metrics['sharpness_base']:.1f}  "
              f"sharp_render={avg_metrics['sharpness_render']:.1f}  "
              f"bright_base={avg_metrics['brightness_base']:.1f}  "
              f"bright_render={avg_metrics['brightness_render']:.1f}  "
              f"({n} trajs re-rendered)")

    # Save metrics summary
    out_path = SYNTH_BASE / "v4_checkpoint_metrics.json"
    with open(out_path, "w") as fp:
        json.dump(metrics_all, fp, indent=2)
    print(f"\nMetrics saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Step':>6} | {'Sharp(base)':>12} | {'Sharp(render)':>13} | {'Bright(base)':>13} | {'Bright(render)':>14}")
    print("-" * 80)
    for step in sorted(metrics_all.keys()):
        m = metrics_all[step]
        print(f"{step:>6} | {m['sharpness_base']:>12.1f} | {m['sharpness_render']:>13.1f} | "
              f"{m['brightness_base']:>13.1f} | {m['brightness_render']:>14.1f}")


if __name__ == "__main__":
    main()
