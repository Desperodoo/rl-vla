"""Encode AWSC rollout data with VAE for VLAW training.

Step 4: VAE encode all train (mixed + high_suc) and eval data from AWSC model.
- Input: 128x128 dual-camera RGB -> resize 192x192 -> vertical concat 384x192
- Output: latent_concat (T, 4, 48, 24) float16

Usage:
    CUDA_VISIBLE_DEVICES=5 conda run -n rlft_ms3 python scripts/vlaw/encode_awsc_data.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from rlft.vlaw.data.pipeline import VLAWDataPipeline, PipelineConfig


def filter_high_suc(
    input_dir: str,
    output_dir: str,
    task: str,
) -> int:
    """Filter success_at_end trajectories from mixed data.
    
    Copies only trajectories where env_success[-1] == True.
    Returns count of filtered trajectories.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir) / task
    out_path.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(in_path.glob("*.h5")) + sorted(in_path.glob("*.hdf5"))
    total_suc = 0
    total_traj = 0

    for h5_file in h5_files:
        out_file = out_path / h5_file.name
        with h5py.File(str(h5_file), "r") as fin:
            traj_keys = [k for k in fin.keys() if k.startswith("traj_")]
            suc_keys = []
            for k in traj_keys:
                total_traj += 1
                if "env_success" in fin[k]:
                    success = fin[k]["env_success"][:]
                    if len(success) > 0 and success[-1]:
                        suc_keys.append(k)
                        total_suc += 1

            if suc_keys:
                with h5py.File(str(out_file), "w") as fout:
                    for i, k in enumerate(suc_keys):
                        new_key = f"traj_{i:04d}"
                        fin.copy(k, fout, name=new_key)
                print(f"  {h5_file.name}: {len(suc_keys)}/{len(traj_keys)} success")

    print(f"[high_suc] Total: {total_suc}/{total_traj} success_at_end trajectories")
    return total_suc


def compute_stat_json(
    encoded_dir: str,
    output_path: str,
) -> None:
    """Recompute stat.json (action p01/p99) from encoded HDF5 files."""
    all_actions: list[np.ndarray] = []
    h5_files = sorted(Path(encoded_dir).glob("**/*.h5")) + sorted(Path(encoded_dir).glob("**/*.hdf5"))
    print(f"[stat.json] Scanning {len(h5_files)} HDF5 files...")

    for h5_path in h5_files:
        with h5py.File(str(h5_path), "r") as f:
            for key in sorted(f.keys()):
                if key.startswith("traj_") and "actions" in f[key]:
                    all_actions.append(f[key]["actions"][:].astype(np.float32))

    if not all_actions:
        print(f"[stat.json] WARNING: No actions found in {encoded_dir}")
        return

    actions = np.concatenate(all_actions, axis=0)
    p01 = np.percentile(actions, 1, axis=0).tolist()
    p99 = np.percentile(actions, 99, axis=0).tolist()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"state_01": p01, "state_99": p99}, f, indent=2)

    print(f"[stat.json] Saved → {output_path}")
    print(f"[stat.json] {actions.shape[0]} total frames")
    print(f"[stat.json] p01: {[f'{x:.4f}' for x in p01]}")
    print(f"[stat.json] p99: {[f'{x:.4f}' for x in p99]}")


def main() -> None:
    task = "LiftPegUpright-v1"
    vae_local = str(project_root / "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid/vae")

    base_cfg = dict(
        vae_local_path=vae_local,
        camera_height=192,
        camera_width=192,
        concat_mode="vertical",
        batch_size=32,
        gpu_id=0,  # after CUDA_VISIBLE_DEVICES
        verbose=True,
        dry_run=False,
    )

    rollout_base = "data/vlaw/rollouts_awsc"
    encoded_base = "data/vlaw/encoded_awsc"
    train_out = f"{encoded_base}/train/{task}"
    eval_out = f"{encoded_base}/eval"

    # ---- Filter high_suc ----
    print("=" * 60)
    print("[Step 4.0] Filtering high_suc trajectories from mixed data...")
    print("=" * 60)
    n_suc = filter_high_suc(
        input_dir=f"{rollout_base}/mixed/{task}",
        output_dir=f"{rollout_base}/high_suc",
        task=task,
    )

    # ---- Encode mixed ----
    print("\n" + "=" * 60)
    print("[Step 4.1] Encoding MIXED data...")
    print("=" * 60)
    t0 = time.time()
    cfg_mixed = PipelineConfig(
        input_dir=f"{rollout_base}/mixed/{task}",
        output_dir=train_out,
        **base_cfg,
    )
    pipe = VLAWDataPipeline(cfg_mixed)
    mixed_paths = pipe.encode_trajectories()
    t_mixed = time.time() - t0
    print(f"[Step 4.1] Mixed encoding done: {len(mixed_paths)} files, {t_mixed:.1f}s")

    # ---- Encode high_suc ----
    print("\n" + "=" * 60)
    print("[Step 4.2] Encoding HIGH_SUC data...")
    print("=" * 60)
    t0 = time.time()
    cfg_hs = PipelineConfig(
        input_dir=f"{rollout_base}/high_suc/{task}",
        output_dir=train_out,
        **base_cfg,
    )
    pipe_hs = VLAWDataPipeline(cfg_hs)
    pipe_hs._vae = pipe._vae  # Reuse VAE
    hs_paths = pipe_hs.encode_trajectories()
    t_hs = time.time() - t0
    print(f"[Step 4.2] High_suc encoding done: {len(hs_paths)} files, {t_hs:.1f}s")

    # ---- Encode eval ----
    print("\n" + "=" * 60)
    print("[Step 4.3] Encoding EVAL data...")
    print("=" * 60)
    t0 = time.time()
    cfg_eval = PipelineConfig(
        input_dir=f"{rollout_base}/eval/{task}",
        output_dir=eval_out,
        **base_cfg,
    )
    pipe_eval = VLAWDataPipeline(cfg_eval)
    pipe_eval._vae = pipe._vae
    eval_paths = pipe_eval.encode_trajectories()
    t_eval = time.time() - t0
    print(f"[Step 4.3] Eval encoding done: {len(eval_paths)} files, {t_eval:.1f}s")

    # ---- Compute stat.json ----
    print("\n" + "=" * 60)
    print("[Step 4.4] Computing stat.json from AWSC encoded data...")
    print("=" * 60)
    compute_stat_json(
        encoded_dir=train_out,
        output_path=f"data/vlaw/meta_info/maniskill_awsc/stat.json",
    )

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
    print(f"High_suc count: {n_suc}")
    print(f"Time: mixed={t_mixed:.1f}s, high_suc={t_hs:.1f}s, eval={t_eval:.1f}s")
    print(f"Total: {t_mixed + t_hs + t_eval:.1f}s")
    print(f"stat.json: data/vlaw/meta_info/maniskill_awsc/stat.json")


if __name__ == "__main__":
    main()
