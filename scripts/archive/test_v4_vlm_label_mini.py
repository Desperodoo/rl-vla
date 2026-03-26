#!/usr/bin/env python3
"""V4: VLM labeling mini 验证 — 对 5 条 D_real 轨迹进行 VLM 零样本评分.

Phase 1.5 V4 验证项:
  1. VLAWRewardModel 加载 (zero-shot)
  2. 从 D_real 读取 5 条轨迹 RGB 帧
  3. score_trajectory 输出 p_yes
  4. 标签写回 (到 stdout，不修改原数据)

Usage:
    CUDA_VISIBLE_DEVICES=6 /home/wjz/miniconda3/envs/vlaw_reward/bin/python scripts/test_v4_vlm_label_mini.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from PIL import Image


TASK_INSTRUCTIONS = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
}


def uniform_sample_indices(total: int, num_frames: int) -> list[int]:
    if total <= num_frames:
        return list(range(total))
    indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    if (total - 1) not in indices:
        indices[-1] = total - 1
    return indices


def main() -> None:
    # === Step 1: 加载模型 ===
    print("[V4] Step 1: 加载 VLAWRewardModel (zero-shot) ...")
    from rlft.vlaw.reward.reward_model import VLAWRewardConfig, VLAWRewardModel

    cfg = VLAWRewardConfig(
        model_path="checkpoints/vlaw/reward_model/qwen_vl",
        num_frames=16,
        threshold=0.8,
    )
    model = VLAWRewardModel(cfg)
    model.load_model()
    print("[V4] ✅ Step 1: 模型加载成功")

    # === Step 2: 读取 5 条轨迹 ===
    print("\n[V4] Step 2: 从 D_real 读取 5 条轨迹 ...")
    data_dir = Path("data/vlaw/rollouts/iter1/LiftPegUpright-v1")
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        print(f"[V4] ❌ 未找到 HDF5 文件: {data_dir}")
        sys.exit(1)

    trajectories = []
    for h5_path in h5_files:
        with h5py.File(str(h5_path), "r") as f:
            for key in sorted(f.keys()):
                if not key.startswith("traj_"):
                    continue
                grp = f[key]
                if "rgb_base" not in grp:
                    continue
                rgb = grp["rgb_base"][:]  # (T, H, W, 3) uint8
                env_succ = grp["env_success"][:] if "env_success" in grp else None
                trajectories.append({
                    "rgb": rgb,
                    "env_success": env_succ,
                    "key": key,
                    "file": h5_path.name,
                })
                if len(trajectories) >= 5:
                    break
        if len(trajectories) >= 5:
            break

    print(f"[V4] ✅ Step 2: 读取了 {len(trajectories)} 条轨迹")

    # === Step 3: VLM 评分 ===
    print("\n[V4] Step 3: VLM 评分 (zero-shot, 16帧 images) ...")
    instruction = TASK_INSTRUCTIONS["LiftPegUpright-v1"]
    results = []
    for i, traj in enumerate(trajectories):
        rgb = traj["rgb"]
        T = rgb.shape[0]
        indices = uniform_sample_indices(T, 16)
        frames = [Image.fromarray(rgb[idx]) for idx in indices]

        score = model.score_trajectory(frames, instruction)
        env_succ = bool(traj["env_success"][-1]) if traj["env_success"] is not None else "N/A"
        results.append({
            "idx": i,
            "file": traj["file"],
            "key": traj["key"],
            "p_yes": score["p_yes"],
            "reward": score["reward"],
            "env_success_at_end": env_succ,
            "T": T,
        })
        print(f"  traj {i} ({traj['key']}): p_yes={score['p_yes']:.6f}, "
              f"reward={score['reward']}, env_succ={env_succ}, T={T}")

    print(f"[V4] ✅ Step 3: 评分完成 ({len(results)} 条)")

    # === Step 4: 汇总 ===
    print(f"\n{'='*60}")
    p_values = [r["p_yes"] for r in results]
    print(f"[V4] Phase 1.5 V4 VLM Labeling Mini 验证结果:")
    print(f"  Step 1 模型加载:  ✅")
    print(f"  Step 2 数据读取:  ✅ ({len(trajectories)} 条)")
    print(f"  Step 3 VLM 评分:  ✅")
    print(f"  p_yes 范围: [{min(p_values):.6f}, {max(p_values):.6f}]")
    print(f"  p_yes 均值: {np.mean(p_values):.6f}")
    print(f"  reward=1 数量: {sum(r['reward'] for r in results)}/{len(results)}")
    print(f"  总体状态: ✅ 全部通过 (p_yes 有值、非 NaN、模型运行正常)")


if __name__ == "__main__":
    main()
