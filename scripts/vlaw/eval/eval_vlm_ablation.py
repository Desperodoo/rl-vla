#!/usr/bin/env python3
"""VLM 帧数消融评估脚本

对 LoRA fine-tuned 模型在指定 num_frames 下评估 ROC-AUC / Acc / FP 等指标。

用法:
    CUDA_VISIBLE_DEVICES=6 python scripts/vlaw/eval/eval_vlm_ablation.py \
        --lora_path checkpoints/vlaw/reward_model/ablation_4frame/final \
        --num_frames 4 \
        --task LiftPegUpright-v1
"""
import json
import os
import sys
import glob
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

TASK_INSTRUCTIONS = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
    "PickCube-v1": "Pick up the cube and place it on the target location.",
    "StackCube-v1": "Stack the red cube on top of the green cube.",
}


def load_trajectories(
    rollout_dirs: list[str],
    max_frames: int = 16,
) -> list[dict]:
    """从 rollout HDF5 加载轨迹: 均匀采样 max_frames 帧."""
    trajs = []
    for rollout_dir in rollout_dirs:
        for fp in sorted(glob.glob(os.path.join(rollout_dir, "*.h5"))):
            with h5py.File(fp, "r") as f:
                for k in sorted(f.keys()):
                    if not k.startswith("traj"):
                        continue
                    grp = f[k]
                    rgb = grp["rgb_base"][:]
                    success = bool(grp["env_success"][-1])
                    T = rgb.shape[0]
                    n = min(max_frames, T)
                    idxs = np.linspace(0, T - 1, n, dtype=int)
                    frames = rgb[idxs]
                    trajs.append({
                        "frames": frames,
                        "n_frames_total": T,
                        "n_frames_sampled": n,
                        "success": success,
                        "source": f"{os.path.basename(fp)}:{k}",
                    })
    return trajs


def evaluate_model(model, trajs: list[dict], instruction: str) -> list[float]:
    """对所有轨迹评分."""
    p_yes_list = []
    t0 = time.time()
    for i, t in enumerate(trajs):
        result = model.score_trajectory(t["frames"], instruction)
        p_yes_list.append(result["p_yes"])
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(trajs)}] p_yes={result['p_yes']:.4f} "
                  f"gt={t['success']} nf={result['num_frames']} "
                  f"({elapsed:.0f}s)")
    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s, {elapsed/len(trajs):.2f}s/traj)")
    return p_yes_list


def compute_metrics(y_true: np.ndarray, p_yes: np.ndarray,
                    threshold: float = 0.8) -> dict:
    """计算 ROC-AUC, 最优阈值, confusion matrix."""
    try:
        from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    except ImportError:
        print("[WARN] sklearn not found")
        return _compute_basic(y_true, p_yes, threshold)

    auc = roc_auc_score(y_true, p_yes)
    fpr, tpr, thresholds = roc_curve(y_true, p_yes)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])

    preds_alpha = (p_yes > threshold).astype(int)
    cm_alpha = confusion_matrix(y_true, preds_alpha, labels=[0, 1])

    preds_best = (p_yes > best_thresh).astype(int)
    cm_best = confusion_matrix(y_true, preds_best, labels=[0, 1])

    # FP rate at different thresholds
    fp_alpha = int(cm_alpha[0][1])
    tn_alpha = int(cm_alpha[0][0])
    fp_rate_alpha = fp_alpha / max(fp_alpha + tn_alpha, 1)

    fp_youd = int(cm_best[0][1])
    tn_youd = int(cm_best[0][0])
    fp_rate_youden = fp_youd / max(fp_youd + tn_youd, 1)

    acc_alpha = (int(cm_alpha[0][0]) + int(cm_alpha[1][1])) / max(y_true.shape[0], 1)
    acc_youden = (int(cm_best[0][0]) + int(cm_best[1][1])) / max(y_true.shape[0], 1)

    return {
        "auc": auc,
        "youden_threshold": best_thresh,
        "youden_j": float(j_scores[best_idx]),
        "acc_alpha": acc_alpha,
        "acc_youden": acc_youden,
        "fp_rate_alpha": fp_rate_alpha,
        "fp_rate_youden": fp_rate_youden,
        "cm_alpha": cm_alpha.tolist(),
        "cm_youden": cm_best.tolist(),
        "p_yes_mean_success": float(p_yes[y_true == 1].mean()) if (y_true == 1).any() else 0.0,
        "p_yes_mean_fail": float(p_yes[y_true == 0].mean()) if (y_true == 0).any() else 0.0,
        "p_yes_std_success": float(p_yes[y_true == 1].std()) if (y_true == 1).any() else 0.0,
        "p_yes_std_fail": float(p_yes[y_true == 0].std()) if (y_true == 0).any() else 0.0,
        "p_yes_max": float(p_yes.max()),
        "p_yes_min": float(p_yes.min()),
        # Youden recall
        "youden_recall": float(tpr[best_idx]),
        "youden_fpr": float(fpr[best_idx]),
    }


def _compute_basic(y_true, p_yes, threshold):
    preds = (p_yes > threshold).astype(int)
    tp = int(((y_true == 1) & (preds == 1)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    tn = int(((y_true == 0) & (preds == 0)).sum())
    fn = int(((y_true == 1) & (preds == 0)).sum())
    total = max(tp + fp + tn + fn, 1)
    return {
        "auc": -1.0,
        "acc_alpha": (tp + tn) / total,
        "fp_rate_alpha": fp / max(fp + tn, 1),
        "cm_alpha": [[tn, fp], [fn, tp]],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="LiftPegUpright-v1")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--model_path", default=str(ROOT / "checkpoints/vlaw/reward_model/qwen_vl"))
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output_json", default=None, help="Save metrics as JSON")
    args = parser.parse_args()

    rollout_dirs = [
        str(ROOT / f"data/vlaw/rollouts/iter1/{args.task}"),
        str(ROOT / f"data/vlaw/rollouts/iter1_highsuc/{args.task}"),
    ]
    rollout_dirs = [d for d in rollout_dirs if os.path.isdir(d)]

    instruction = TASK_INSTRUCTIONS.get(args.task, "Complete the task.")

    print(f"=== VLM Ablation Eval: num_frames={args.num_frames} ===")
    print(f"LoRA: {args.lora_path}")
    print(f"Task: {args.task}")

    trajs = load_trajectories(rollout_dirs, args.num_frames)
    y_true = np.array([int(t["success"]) for t in trajs])
    n_succ = int(y_true.sum())
    n_fail = len(y_true) - n_succ
    print(f"Loaded {len(trajs)} trajs: {n_succ} success, {n_fail} fail")
    if len(trajs) == 0:
        print("ERROR: No trajectories found!")
        return

    from rlft.vlaw.reward.reward_model import VLAWRewardModel, VLAWRewardConfig
    cfg = VLAWRewardConfig(
        model_path=args.model_path,
        threshold=args.threshold,
        device="cuda:0",
        num_frames=args.num_frames,
    )
    model = VLAWRewardModel(cfg)
    model.load_model(lora_path=args.lora_path)

    p_yes = np.array(evaluate_model(model, trajs, instruction))
    metrics = compute_metrics(y_true, p_yes, args.threshold)

    print(f"\n{'='*50}")
    print(f"  num_frames = {args.num_frames}")
    print(f"  ROC-AUC    = {metrics['auc']:.4f}")
    print(f"  Acc@α=0.8  = {metrics['acc_alpha']:.4f}")
    print(f"  FP@α=0.8   = {metrics['fp_rate_alpha']:.4f}")
    print(f"  Acc@Youden = {metrics.get('acc_youden', 'N/A')}")
    print(f"  FP@Youden  = {metrics.get('fp_rate_youden', 'N/A')}")
    print(f"  Youden θ   = {metrics.get('youden_threshold', 'N/A')}")
    print(f"  p_yes(+)   = {metrics['p_yes_mean_success']:.4f} ± {metrics['p_yes_std_success']:.4f}")
    print(f"  p_yes(-)   = {metrics['p_yes_mean_fail']:.4f} ± {metrics['p_yes_std_fail']:.4f}")
    print(f"{'='*50}")

    metrics["num_frames"] = args.num_frames
    metrics["n_trajs"] = len(trajs)
    metrics["n_success"] = n_succ
    metrics["n_fail"] = n_fail
    metrics["lora_path"] = args.lora_path

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as jf:
            json.dump(metrics, jf, indent=2)
        print(f"Saved → {args.output_json}")

    model.unload_model()
    return metrics


if __name__ == "__main__":
    main()
