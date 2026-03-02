#!/usr/bin/env python3
"""T-DIAG-SYN-003 — VLM 交叉验证: fine-tuned VLM 对真实成功/失败 demo 评分

用 fine-tuned VLM (16帧 LoRA) 对真实数据 (iter1_highsuc) 中的成功/失败
轨迹分别打分，评估 VLM 在真实数据上的分离度。

输出:
    results/vlaw/dsyn_diagnosis_vlm_crossval.json
    - 每条轨迹: p_yes, env_success, vlm_reward, file, traj_key
    - 成功/失败组统计: mean, median, min, max, std
    - 分离度分析

用法:
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python scripts/vlaw/eval/vlm_crossval_real.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── 配置 ──────────────────────────────────────────────────────────────────────

MODEL_PATH = "checkpoints/vlaw/reward_model/qwen_vl"
LORA_PATH = "checkpoints/vlaw/reward_model/lora_iter1_16frame"
DATA_DIR = "data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1"
TASK = "LiftPegUpright-v1"
INSTRUCTION = "Lift the peg and insert it upright into the holder."
NUM_FRAMES = 16
THRESHOLD = 0.8
OUTPUT_PATH = "results/vlaw/dsyn_diagnosis_vlm_crossval.json"
DEVICE = "cuda:0"


def load_trajectories(data_dir: str) -> List[Dict]:
    """从 HDF5 文件加载所有轨迹的 RGB 帧 + env_success."""
    trajectories = []
    data_path = Path(data_dir)

    for h5_file in sorted(data_path.glob("*.h5")):
        with h5py.File(h5_file, "r") as f:
            traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
            for tk in traj_keys:
                grp = f[tk]
                rgb = grp["rgb_base"][:]  # (T, H, W, C) uint8
                env_success = bool(grp["env_success"][-1])
                trajectories.append({
                    "rgb": rgb,
                    "env_success": env_success,
                    "file": h5_file.name,
                    "traj_key": tk,
                    "num_steps": rgb.shape[0],
                })

    return trajectories


def main():
    t_start = time.time()

    # 加载所有轨迹
    full_data_dir = str(ROOT / DATA_DIR)
    print(f"[Cross-Val] 加载数据: {full_data_dir}")
    trajectories = load_trajectories(full_data_dir)

    n_suc = sum(1 for t in trajectories if t["env_success"])
    n_fail = sum(1 for t in trajectories if not t["env_success"])
    print(f"[Cross-Val] 总计: {len(trajectories)} 条 (成功={n_suc}, 失败={n_fail})")

    # 加载 VLM
    from rlft.vlaw.reward.reward_model import VLAWRewardConfig, VLAWRewardModel

    config = VLAWRewardConfig(
        model_path=str(ROOT / MODEL_PATH),
        device=DEVICE,
        num_frames=NUM_FRAMES,
        threshold=THRESHOLD,
        use_video_format=False,  # 与训练一致：多图模式
    )
    model = VLAWRewardModel(config)
    lora_full_path = str(ROOT / LORA_PATH)
    print(f"[Cross-Val] 加载 VLM + LoRA: {lora_full_path}")
    model.load_model(lora_path=lora_full_path)

    # 对每条轨迹评分
    results = []
    for i, traj in enumerate(trajectories):
        result = model.score_trajectory(traj["rgb"], INSTRUCTION)
        entry = {
            "idx": i,
            "file": traj["file"],
            "traj_key": traj["traj_key"],
            "num_steps": traj["num_steps"],
            "env_success": traj["env_success"],
            "p_yes": round(result["p_yes"], 6),
            "vlm_reward": result["reward"],
            "threshold": result["threshold"],
            "num_frames_used": result["num_frames"],
        }
        results.append(entry)

        status = "✓" if traj["env_success"] else "✗"
        print(
            f"  [{i+1:3d}/{len(trajectories)}] {status} p_yes={result['p_yes']:.4f} "
            f"vlm={result['reward']} | {traj['file']}:{traj['traj_key']} "
            f"(steps={traj['num_steps']})"
        )

    # 分组统计
    suc_pyes = [r["p_yes"] for r in results if r["env_success"]]
    fail_pyes = [r["p_yes"] for r in results if not r["env_success"]]

    def compute_stats(values: List[float]) -> Dict:
        if not values:
            return {"count": 0}
        arr = np.array(values)
        return {
            "count": len(values),
            "mean": round(float(arr.mean()), 6),
            "median": round(float(np.median(arr)), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "std": round(float(arr.std()), 6),
            "p25": round(float(np.percentile(arr, 25)), 6),
            "p75": round(float(np.percentile(arr, 75)), 6),
        }

    suc_stats = compute_stats(suc_pyes)
    fail_stats = compute_stats(fail_pyes)

    # 分离度分析
    separation = {}
    if suc_pyes and fail_pyes:
        # Cohen's d
        pooled_std = np.sqrt(
            (np.std(suc_pyes) ** 2 + np.std(fail_pyes) ** 2) / 2
        )
        if pooled_std > 0:
            cohens_d = (np.mean(suc_pyes) - np.mean(fail_pyes)) / pooled_std
        else:
            cohens_d = float("inf")

        # 分离: min(success) > max(failure)?
        clean_separation = min(suc_pyes) > max(fail_pyes)

        # 阈值分析 @ α=0.8
        tp = sum(1 for p in suc_pyes if p > THRESHOLD)
        fn = sum(1 for p in suc_pyes if p <= THRESHOLD)
        fp = sum(1 for p in fail_pyes if p > THRESHOLD)
        tn = sum(1 for p in fail_pyes if p <= THRESHOLD)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # 最优阈值 (Youden's J)
        all_pyes = sorted(set(suc_pyes + fail_pyes))
        best_j, best_thresh = -1, 0.5
        for thresh_candidate in np.arange(0.0, 1.001, 0.005):
            tpr = sum(1 for p in suc_pyes if p > thresh_candidate) / len(suc_pyes)
            fpr = sum(1 for p in fail_pyes if p > thresh_candidate) / len(fail_pyes)
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_thresh = thresh_candidate

        separation = {
            "cohens_d": round(float(cohens_d), 4),
            "clean_separation": clean_separation,
            "mean_gap": round(float(np.mean(suc_pyes) - np.mean(fail_pyes)), 6),
            "confusion_matrix_at_0.8": {
                "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
                "FP_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
            },
            "optimal_threshold_youden": {
                "threshold": round(float(best_thresh), 4),
                "youden_j": round(float(best_j), 4),
            },
        }

    elapsed = time.time() - t_start
    output = {
        "task": TASK,
        "model_path": MODEL_PATH,
        "lora_path": LORA_PATH,
        "data_dir": DATA_DIR,
        "num_frames": NUM_FRAMES,
        "threshold": THRESHOLD,
        "video_format": False,
        "total_trajectories": len(results),
        "success_stats": suc_stats,
        "failure_stats": fail_stats,
        "separation_analysis": separation,
        "elapsed_seconds": round(elapsed, 1),
        "trajectories": results,
    }

    # 保存
    out_path = ROOT / OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"[Cross-Val] 完成 | 耗时: {elapsed:.1f}s")
    print(f"[Cross-Val] 结果保存: {out_path}")
    print(f"\n=== 成功组 (n={suc_stats['count']}) ===")
    if suc_stats["count"] > 0:
        print(f"  mean={suc_stats['mean']:.4f}  median={suc_stats['median']:.4f}  "
              f"min={suc_stats['min']:.4f}  max={suc_stats['max']:.4f}  std={suc_stats['std']:.4f}")
    print(f"\n=== 失败组 (n={fail_stats['count']}) ===")
    if fail_stats["count"] > 0:
        print(f"  mean={fail_stats['mean']:.4f}  median={fail_stats['median']:.4f}  "
              f"min={fail_stats['min']:.4f}  max={fail_stats['max']:.4f}  std={fail_stats['std']:.4f}")

    if separation:
        print(f"\n=== 分离度 ===")
        print(f"  Cohen's d: {separation['cohens_d']:.4f}")
        print(f"  Mean gap: {separation['mean_gap']:.4f}")
        print(f"  Clean separation: {separation['clean_separation']}")
        cm = separation["confusion_matrix_at_0.8"]
        print(f"  @ α=0.8: TP={cm['TP']} FN={cm['FN']} FP={cm['FP']} TN={cm['TN']}")
        print(f"           Acc={cm['accuracy']:.4f} Prec={cm['precision']:.4f} "
              f"Rec={cm['recall']:.4f} F1={cm['f1']:.4f} FP_rate={cm['FP_rate']:.4f}")
        ot = separation["optimal_threshold_youden"]
        print(f"  Youden optimal: thresh={ot['threshold']:.4f} J={ot['youden_j']:.4f}")


if __name__ == "__main__":
    main()
