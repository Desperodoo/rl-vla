#!/usr/bin/env python3
"""
encode_rollouts_iter.py — P1.2 单次迭代 VAE 编码包装脚本

从 train_vlaw.py Step 2 调用，将 D_real HDF5 rollouts VAE 编码为 latent。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parents[3].resolve()
sys.path.insert(0, str(WORKSPACE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="D_real rollout 目录，含 {task}/*.h5")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出 latent 目录")
    parser.add_argument("--tasks", type=str,
                        default="LiftPegUpright-v1,PickCube-v1,StackCube-v1")
    parser.add_argument("--iter_id", type=int, default=1)
    parser.add_argument("--vae_model_path", type=str,
                        default="checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)
    vae_model_path = WORKSPACE / args.vae_model_path

    print(f"[Step2] encode_rollouts_iter iter={args.iter_id} tasks={task_list}")

    if args.dry_run:
        print("[DRY RUN] 跳过实际编码")
        return

    from rlft.vlaw.data import PipelineConfig, VLAWDataPipeline

    for task_id in task_list:
        task_in = input_base / task_id
        task_out = output_base / task_id
        task_out.mkdir(parents=True, exist_ok=True)

        h5_files = sorted(task_in.glob("*.h5"))
        if not h5_files:
            print(f"[Step2] [WARN] {task_id}: 未找到 HDF5 文件")
            continue

        print(f"\n[Step2] 编码 {task_id} ({len(h5_files)} 文件)...")
        cfg = PipelineConfig(
            input_dir=str(task_in),
            output_dir=str(task_out),
            vae_local_path=str(vae_model_path / "vae") if (vae_model_path / "vae").exists() else "",
            gpu_id=4,
            batch_size=16,
            verbose=True,
        )
        pipeline = VLAWDataPipeline(cfg)
        for h5_path in h5_files:
            out_path = task_out / h5_path.name
            pipeline.encode_single_hdf5(h5_path, out_path)
            print(f"  {h5_path.name} → {out_path.name}")

    print(f"\n[Step2] VAE 编码全部完成 → {output_base}")


if __name__ == "__main__":
    main()
