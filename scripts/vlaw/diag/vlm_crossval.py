#!/usr/bin/env python3
"""
T-DIAG-SYN-003 — VLM 交叉验证: 用 fine-tuned VLM (16帧 LoRA) 对真实轨迹评分。

遍历 HDF5 轨迹，均匀采样 16 帧 RGB → VLM score (p_yes)，
按 env_success[-1] 分成成功/失败两组，统计 p_yes 分布。

Usage:
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python scripts/vlaw/diag/vlm_crossval.py

Outputs:
    results/vlaw/dsyn_diagnosis_vlm_crossval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

# 把项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rlft.vlaw.reward.reward_model import VLAWRewardConfig, VLAWRewardModel

# ── 任务指令 (与训练时一致) ────────────────────────────────────────────────────
TASK_INSTRUCTIONS: Dict[str, str] = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
    "PickCube-v1": "Pick up the cube and place it on the target location.",
    "StackCube-v1": "Stack the red cube on top of the green cube.",
}


def collect_trajectories(
    data_dir: str, task: str
) -> List[Dict]:
    """
    从 HDF5 中收集所有轨迹信息。

    Returns:
        List[Dict] with keys: h5_path, traj_key, env_success_last, num_steps
    """
    task_dir = os.path.join(data_dir, task)
    h5_files = sorted(glob.glob(os.path.join(task_dir, "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files in {task_dir}")

    trajectories = []
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
            for tk in traj_keys:
                env_success = f[tk]["env_success"][:]
                trajectories.append({
                    "h5_path": h5_path,
                    "traj_key": tk,
                    "env_success_last": bool(env_success[-1]),
                    "num_steps": len(env_success),
                })
    return trajectories


def score_all_trajectories(
    model: VLAWRewardModel,
    trajectories: List[Dict],
    instruction: str,
    num_frames: int = 16,
) -> List[Dict]:
    """
    对所有轨迹评分，返回带 p_yes 的结果列表。
    """
    results = []
    total = len(trajectories)

    for i, traj_info in enumerate(trajectories):
        h5_path = traj_info["h5_path"]
        traj_key = traj_info["traj_key"]

        # 读取 RGB 帧
        with h5py.File(h5_path, "r") as f:
            rgb = f[traj_key]["rgb_base"][:]  # (T, H, W, C) uint8

        # 评分
        t0 = time.time()
        result = model.score_trajectory(rgb, instruction)
        elapsed = time.time() - t0

        record = {
            "h5_file": os.path.basename(h5_path),
            "traj_key": traj_key,
            "env_success_last": traj_info["env_success_last"],
            "num_steps": traj_info["num_steps"],
            "p_yes": result["p_yes"],
            "reward": result["reward"],
            "num_frames_used": result["num_frames"],
            "inference_time_s": round(elapsed, 2),
        }
        results.append(record)

        status = "SUCCESS" if traj_info["env_success_last"] else "FAIL"
        print(
            f"[{i+1:3d}/{total}] {traj_key} ({status}): "
            f"p_yes={result['p_yes']:.4f}  reward={result['reward']}  "
            f"({elapsed:.1f}s)"
        )

    return results


def compute_stats(values: List[float]) -> Dict:
    """计算分布统计量。"""
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "std": round(float(arr.std()), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="VLM cross-validation on real trajectories")
    parser.add_argument("--data-dir", default="data/vlaw/rollouts/iter1_highsuc",
                        help="Real rollout data directory")
    parser.add_argument("--task", default="LiftPegUpright-v1")
    parser.add_argument("--model-path", default="checkpoints/vlaw/reward_model/qwen_vl",
                        help="Base VLM model path")
    parser.add_argument("--lora-path", default="checkpoints/vlaw/reward_model/lora_iter1_16frame",
                        help="LoRA adapter path")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output", default="results/vlaw/dsyn_diagnosis_vlm_crossval.json")
    parser.add_argument("--device", default="cuda:0", help="Device (use cuda:0 with CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--video-mode", action="store_true", default=False,
                        help="Use video format instead of multi-image (for A/B comparison)")
    args = parser.parse_args()

    print("=" * 70)
    print("T-DIAG-SYN-003: VLM 交叉验证 (fine-tuned 16帧 LoRA)")
    print("=" * 70)
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Task      : {args.task}")
    print(f"  Model     : {args.model_path}")
    print(f"  LoRA      : {args.lora_path}")
    print(f"  Frames    : {args.num_frames}")
    print(f"  Threshold : {args.threshold}")
    print(f"  Video mode: {args.video_mode}")
    print(f"  Output    : {args.output}")
    print()

    # Step 1: 收集轨迹信息
    print("[Step 1] Collecting trajectories...")
    trajectories = collect_trajectories(args.data_dir, args.task)
    n_success = sum(1 for t in trajectories if t["env_success_last"])
    n_fail = sum(1 for t in trajectories if not t["env_success_last"])
    print(f"  Found {len(trajectories)} trajectories: {n_success} success, {n_fail} fail")
    print()

    # Step 2: 加载 VLM
    print("[Step 2] Loading VLM with LoRA...")
    config = VLAWRewardConfig(
        model_path=args.model_path,
        device=args.device,
        num_frames=args.num_frames,
        threshold=args.threshold,
        # --video-mode 切换 video vs multi-image 格式
        use_video_format=args.video_mode,
        use_flash_attention=True,
    )
    model = VLAWRewardModel(config)
    model.load_model(lora_path=args.lora_path)
    print()

    # Step 3: 评分
    print("[Step 3] Scoring trajectories...")
    instruction = TASK_INSTRUCTIONS.get(args.task, "Complete the manipulation task successfully.")
    print(f"  Instruction: '{instruction}'")
    print()

    t_start = time.time()
    scored = score_all_trajectories(model, trajectories, instruction, args.num_frames)
    total_time = time.time() - t_start
    print(f"\n  Total scoring time: {total_time:.1f}s ({total_time/len(scored):.1f}s/traj)")
    print()

    # Step 4: 分组统计
    print("[Step 4] Computing statistics...")
    success_pyes = [r["p_yes"] for r in scored if r["env_success_last"]]
    fail_pyes = [r["p_yes"] for r in scored if not r["env_success_last"]]

    success_stats = compute_stats(success_pyes)
    fail_stats = compute_stats(fail_pyes)

    # 分类指标 (threshold)
    tp = sum(1 for r in scored if r["env_success_last"] and r["p_yes"] > args.threshold)
    fn = sum(1 for r in scored if r["env_success_last"] and r["p_yes"] <= args.threshold)
    fp = sum(1 for r in scored if not r["env_success_last"] and r["p_yes"] > args.threshold)
    tn = sum(1 for r in scored if not r["env_success_last"] and r["p_yes"] <= args.threshold)

    accuracy = (tp + tn) / len(scored) if scored else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    classification = {
        "threshold": args.threshold,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp_rate": round(fp_rate, 4),
    }

    # 打印
    print(f"\n{'='*50}")
    print("SUCCESS group p_yes distribution:")
    for k, v in success_stats.items():
        print(f"  {k}: {v}")
    print(f"\nFAIL group p_yes distribution:")
    for k, v in fail_stats.items():
        print(f"  {k}: {v}")
    print(f"\nClassification @ threshold={args.threshold}:")
    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"  Accuracy={accuracy:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
    print(f"  FP Rate={fp_rate:.4f}")
    print(f"{'='*50}")

    # Step 5: 保存结果
    output_data = {
        "task": "T-DIAG-SYN-003",
        "description": "VLM cross-validation with fine-tuned 16-frame LoRA on real trajectories",
        "config": {
            "data_dir": args.data_dir,
            "task": args.task,
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "num_frames": args.num_frames,
            "threshold": args.threshold,
            "use_video_format": False,
        },
        "summary": {
            "total_trajectories": len(scored),
            "success_count": success_stats["count"],
            "fail_count": fail_stats["count"],
            "total_scoring_time_s": round(total_time, 1),
            "avg_time_per_traj_s": round(total_time / len(scored), 2) if scored else 0,
        },
        "success_group_pyes": success_stats,
        "fail_group_pyes": fail_stats,
        "classification": classification,
        "per_trajectory": scored,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
