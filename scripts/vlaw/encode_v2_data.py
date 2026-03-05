"""Encode v2 rollout data with VAE for VLAW training.

Phase 0.2: VAE encode all train (mixed + high_suc) and eval data.
- Input: 128x128 dual-camera RGB -> resize 192x192 -> vertical concat 384x192
- Output: latent_concat (T, 4, 48, 24) float16

Usage:
    CUDA_VISIBLE_DEVICES=4 conda run -n rlft_ms3 python scripts/vlaw/encode_v2_data.py
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from rlft.vlaw.data.pipeline import VLAWDataPipeline, PipelineConfig


def main() -> None:
    vae_local = str(project_root / "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid/vae")

    base_cfg = dict(
        vae_local_path=vae_local,
        camera_height=192,
        camera_width=192,
        concat_mode="vertical",
        batch_size=32,
        gpu_id=4,
        verbose=True,
        dry_run=False,
    )

    task = "LiftPegUpright-v1"
    train_out = f"data/vlaw/encoded/train/{task}"
    eval_out = "data/vlaw/encoded/eval"

    # ---- Step 1: Encode mixed data ----
    print("=" * 60)
    print("[Step 2a] Encoding MIXED data...")
    print("=" * 60)
    t0 = time.time()
    cfg_mixed = PipelineConfig(
        input_dir=f"data/vlaw/rollouts/mixed/{task}",
        output_dir=train_out,
        **base_cfg,  # type: ignore[arg-type]
    )
    pipe = VLAWDataPipeline(cfg_mixed)
    mixed_paths = pipe.encode_trajectories()
    t_mixed = time.time() - t0
    print(f"[Step 2a] Mixed encoding done: {len(mixed_paths)} files, {t_mixed:.1f}s")

    # ---- Step 2: Encode high_suc data ----
    print("=" * 60)
    print("[Step 2b] Encoding HIGH_SUC data...")
    print("=" * 60)
    t0 = time.time()
    cfg_hs = PipelineConfig(
        input_dir=f"data/vlaw/rollouts/high_suc/{task}",
        output_dir=train_out,
        **base_cfg,  # type: ignore[arg-type]
    )
    # Reuse VAE from previous pipeline
    pipe_hs = VLAWDataPipeline(cfg_hs)
    pipe_hs._vae = pipe._vae  # Share loaded VAE
    hs_paths = pipe_hs.encode_trajectories()
    t_hs = time.time() - t0
    print(f"[Step 2b] High_suc encoding done: {len(hs_paths)} files, {t_hs:.1f}s")

    # ---- Step 3: Encode eval data ----
    print("=" * 60)
    print("[Step 3] Encoding EVAL data...")
    print("=" * 60)
    t0 = time.time()
    cfg_eval = PipelineConfig(
        input_dir=f"data/vlaw/rollouts/eval/{task}",
        output_dir=eval_out,
        **base_cfg,  # type: ignore[arg-type]
    )
    pipe_eval = VLAWDataPipeline(cfg_eval)
    pipe_eval._vae = pipe._vae  # Share loaded VAE
    eval_paths = pipe_eval.encode_trajectories()
    t_eval = time.time() - t0
    print(f"[Step 3] Eval encoding done: {len(eval_paths)} files, {t_eval:.1f}s")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("ENCODING COMPLETE")
    print("=" * 60)
    all_train = mixed_paths + hs_paths
    print(f"Train files: {len(all_train)}")
    for p in all_train:
        print(f"  {p}")
    print(f"Eval files: {len(eval_paths)}")
    for p in eval_paths:
        print(f"  {p}")
    print(f"Time: mixed={t_mixed:.1f}s, high_suc={t_hs:.1f}s, eval={t_eval:.1f}s")
    print(f"Total: {t_mixed + t_hs + t_eval:.1f}s")


if __name__ == "__main__":
    main()
