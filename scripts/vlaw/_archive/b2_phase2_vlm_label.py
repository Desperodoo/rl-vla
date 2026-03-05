#!/usr/bin/env python3
"""B2 Phase 2: 用微调后的 VLM 标注合成轨迹.

从 Phase 1 生成的 RGB 帧 HDF5 中读取 16 帧图像，
使用 fine-tuned VLM (Qwen3-VL + LoRA) 进行二分类标注。

输出: data/vlaw/labeled/synthetic_iter1_pretrained/vlm_labels.h5
  每条轨迹附加 p_yes, vlm_reward (p_yes > 0.8 → 1), source 信息

Usage:
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python scripts/b2_phase2_vlm_label.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


TASK_INSTRUCTIONS = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
}


def main() -> None:
    t0 = time.time()

    # Paths
    rgb_path = Path("data/vlaw/labeled/synthetic_iter1_pretrained/rgb_frames_for_vlm.h5")
    output_dir = Path("data/vlaw/labeled/synthetic_iter1_pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / "vlm_labels.h5"

    model_path = "checkpoints/vlaw/reward_model/qwen_vl"
    lora_path = "checkpoints/vlaw/reward_model/lora_iter1_16frame"

    if not rgb_path.exists():
        print(f"[B2-P2] ❌ RGB 文件不存在: {rgb_path}")
        print("[B2-P2] 请先运行 b2_phase1_vae_decode.py")
        sys.exit(1)

    # 加载 VLM
    print("[B2-P2] 加载 VLM (Qwen3-VL + LoRA fine-tuned) ...")
    from rlft.vlaw.reward.reward_model import VLAWRewardConfig, VLAWRewardModel
    from PIL import Image

    cfg = VLAWRewardConfig(
        model_path=model_path,
        num_frames=16,
        threshold=0.8,
        use_video_format=True,
        video_fps=2.0,
    )
    model = VLAWRewardModel(cfg)
    model.load_model(lora_path=lora_path)
    print("[B2-P2] ✅ VLM 加载完成")

    # 读取 RGB 帧并标注
    instruction = TASK_INSTRUCTIONS["LiftPegUpright-v1"]
    results = []

    with h5py.File(str(rgb_path), "r") as f:
        num_traj = f.attrs["num_trajectories"]
        print(f"[B2-P2] 共 {num_traj} 条轨迹待标注")

        for i in range(num_traj):
            traj_key = f"traj_{i:04d}"
            grp = f[traj_key]
            rgb_base = grp["rgb_base_16"][:]  # (16, 192, 192, 3) uint8

            # 转为 PIL Image 列表
            frames = [Image.fromarray(rgb_base[j]) for j in range(rgb_base.shape[0])]

            # VLM 评分
            score = model.score_trajectory(frames, instruction)

            result = {
                "traj_idx": i,
                "source_file": grp.attrs["source_file"],
                "source_key": grp.attrs["source_key"],
                "total_frames": int(grp.attrs["total_frames"]),
                "p_yes": score["p_yes"],
                "vlm_reward": score["reward"],
                "threshold": score["threshold"],
            }
            results.append(result)

            tag = "✅" if score["reward"] == 1 else "❌"
            print(
                f"  [{i+1:02d}/{num_traj}] {tag} p_yes={score['p_yes']:.4f} "
                f"reward={score['reward']} ({grp.attrs['source_file']}:{grp.attrs['source_key']})"
            )

    # 保存标注结果
    p_values = [r["p_yes"] for r in results]
    rewards = [r["vlm_reward"] for r in results]
    n_pos = sum(rewards)

    with h5py.File(str(label_path), "w") as out_f:
        for r in results:
            key = f"traj_{r['traj_idx']:04d}"
            grp = out_f.create_group(key)
            grp.attrs["p_yes"] = r["p_yes"]
            grp.attrs["vlm_reward"] = r["vlm_reward"]
            grp.attrs["threshold"] = r["threshold"]
            grp.attrs["source_file"] = r["source_file"]
            grp.attrs["source_key"] = r["source_key"]
            grp.attrs["total_frames"] = r["total_frames"]

        out_f.attrs["task_id"] = "LiftPegUpright-v1"
        out_f.attrs["num_trajectories"] = len(results)
        out_f.attrs["num_positive"] = n_pos
        out_f.attrs["num_negative"] = len(results) - n_pos
        out_f.attrs["positive_rate"] = n_pos / len(results) if results else 0.0
        out_f.attrs["p_yes_mean"] = float(np.mean(p_values))
        out_f.attrs["p_yes_std"] = float(np.std(p_values))
        out_f.attrs["p_yes_min"] = float(np.min(p_values))
        out_f.attrs["p_yes_max"] = float(np.max(p_values))
        out_f.attrs["lora_path"] = lora_path
        out_f.attrs["model_path"] = model_path

    # 同时保存 JSON 报告
    report = {
        "task_id": "LiftPegUpright-v1",
        "num_trajectories": len(results),
        "num_positive (vlm_reward=1)": n_pos,
        "num_negative (vlm_reward=0)": len(results) - n_pos,
        "positive_rate": round(n_pos / len(results), 4) if results else 0.0,
        "p_yes_stats": {
            "mean": round(float(np.mean(p_values)), 4),
            "std": round(float(np.std(p_values)), 4),
            "min": round(float(np.min(p_values)), 4),
            "max": round(float(np.max(p_values)), 4),
            "median": round(float(np.median(p_values)), 4),
        },
        "threshold": 0.8,
        "model_path": model_path,
        "lora_path": lora_path,
        "per_trajectory": results,
    }
    report_path = output_dir / "vlm_label_report.json"
    with open(report_path, "w") as jf:
        json.dump(report, jf, indent=2)

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"[B2-P2] ✅ 标注完成!")
    print(f"  轨迹总数: {len(results)}")
    print(f"  vlm_reward=1: {n_pos}/{len(results)} ({n_pos/len(results)*100:.1f}%)")
    print(f"  vlm_reward=0: {len(results)-n_pos}/{len(results)} ({(len(results)-n_pos)/len(results)*100:.1f}%)")
    print(f"  p_yes: mean={np.mean(p_values):.4f}, std={np.std(p_values):.4f}")
    print(f"  p_yes: min={np.min(p_values):.4f}, max={np.max(p_values):.4f}, median={np.median(p_values):.4f}")
    print(f"  HDF5 标签: {label_path}")
    print(f"  JSON 报告: {report_path}")
    print(f"  耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
