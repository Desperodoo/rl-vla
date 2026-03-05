#!/usr/bin/env python3
"""VLM LoRA v2 评估脚本

在指定的 eval HDF5 数据目录上评估 fine-tuned VLM，计算 ROC-AUC、FP rate 等。

用法:
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python scripts/vlaw/eval/eval_vlm_v2.py \
        --lora_path checkpoints/vlaw/reward_model/lora_v2/final \
        --eval_dir data/vlaw/rollouts/eval/LiftPegUpright-v1 \
        --task LiftPegUpright-v1 \
        --output_json results/vlaw/vlm_v2_eval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

TASK_INSTRUCTIONS = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
    "PickCube-v1": "Pick up the cube and place it on the target location.",
    "StackCube-v1": "Stack the red cube on top of the green cube.",
}


def load_trajectories(eval_dirs: list[str], max_frames: int = 16) -> list[dict]:
    """从 HDF5 加载轨迹，均匀采样 max_frames 帧。"""
    trajs = []
    for eval_dir in eval_dirs:
        for fp in sorted(glob.glob(os.path.join(eval_dir, "*.h5"))):
            with h5py.File(fp, "r") as f:
                for k in sorted(f.keys()):
                    if not k.startswith("traj"):
                        continue
                    grp = f[k]
                    if "rgb_base" not in grp:
                        continue
                    rgb = grp["rgb_base"][:]
                    # success_at_end
                    if "env_success" in grp:
                        success = bool(grp["env_success"][-1])
                    elif "success" in grp:
                        success = bool(grp["success"][-1])
                    else:
                        continue
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
    """对所有轨迹评分，返回 p_yes 列表。"""
    p_yes_list = []
    t0 = time.time()
    for i, t in enumerate(trajs):
        result = model.score_trajectory(t["frames"], instruction)
        p_yes_list.append(result["p_yes"])
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(trajs)}] p_yes={result['p_yes']:.4f} "
              f"gt={'suc' if t['success'] else 'fail'} "
              f"nf={result['num_frames']} ({elapsed:.0f}s)")
    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s, {elapsed/max(len(trajs),1):.2f}s/traj)")
    return p_yes_list


def compute_metrics(y_true: np.ndarray, p_yes: np.ndarray,
                    threshold: float = 0.8) -> dict:
    """计算 ROC-AUC, confusion matrix, FP rate 等完整指标。"""
    try:
        from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    except ImportError:
        return _compute_basic(y_true, p_yes, threshold)

    # ROC-AUC
    if len(np.unique(y_true)) < 2:
        auc = -1.0
        best_thresh = threshold
        best_j = 0.0
        best_tpr = 0.0
        best_fpr = 0.0
    else:
        auc = roc_auc_score(y_true, p_yes)
        fpr_arr, tpr_arr, thresholds = roc_curve(y_true, p_yes)
        j_scores = tpr_arr - fpr_arr
        best_idx = np.argmax(j_scores)
        best_thresh = float(thresholds[best_idx])
        best_j = float(j_scores[best_idx])
        best_tpr = float(tpr_arr[best_idx])
        best_fpr = float(fpr_arr[best_idx])

    # @ alpha threshold
    preds_alpha = (p_yes > threshold).astype(int)
    cm_alpha = confusion_matrix(y_true, preds_alpha, labels=[0, 1])
    tn_a, fp_a, fn_a, tp_a = cm_alpha[0][0], cm_alpha[0][1], cm_alpha[1][0], cm_alpha[1][1]

    # @ Youden optimal threshold
    preds_best = (p_yes > best_thresh).astype(int)
    cm_best = confusion_matrix(y_true, preds_best, labels=[0, 1])
    tn_y, fp_y, fn_y, tp_y = cm_best[0][0], cm_best[0][1], cm_best[1][0], cm_best[1][1]

    total = max(len(y_true), 1)
    return {
        "auc": float(auc),
        "youden_threshold": best_thresh,
        "youden_j": best_j,
        "youden_recall": best_tpr,
        "youden_fpr": best_fpr,
        # @ alpha=0.8
        "acc_alpha": (int(tp_a) + int(tn_a)) / total,
        "fp_rate_alpha": int(fp_a) / max(int(fp_a) + int(tn_a), 1),
        "recall_alpha": int(tp_a) / max(int(tp_a) + int(fn_a), 1),
        "cm_alpha": cm_alpha.tolist(),
        # @ Youden
        "acc_youden": (int(tp_y) + int(tn_y)) / total,
        "fp_rate_youden": int(fp_y) / max(int(fp_y) + int(tn_y), 1),
        "recall_youden": int(tp_y) / max(int(tp_y) + int(fn_y), 1),
        "cm_youden": cm_best.tolist(),
        # p_yes statistics
        "p_yes_mean_success": float(p_yes[y_true == 1].mean()) if (y_true == 1).any() else 0.0,
        "p_yes_mean_fail": float(p_yes[y_true == 0].mean()) if (y_true == 0).any() else 0.0,
        "p_yes_std_success": float(p_yes[y_true == 1].std()) if (y_true == 1).any() else 0.0,
        "p_yes_std_fail": float(p_yes[y_true == 0].std()) if (y_true == 0).any() else 0.0,
        "p_yes_max": float(p_yes.max()),
        "p_yes_min": float(p_yes.min()),
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
        "recall_alpha": tp / max(tp + fn, 1),
        "cm_alpha": [[tn, fp], [fn, tp]],
    }


def main():
    parser = argparse.ArgumentParser(description="VLM LoRA v2 eval")
    parser.add_argument("--task", default="LiftPegUpright-v1")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--model_path", default=str(ROOT / "checkpoints/vlaw/reward_model/qwen_vl"))
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--eval_dir", required=True, help="Eval rollout directory with *.h5")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    eval_dirs = [d.strip() for d in args.eval_dir.split(",") if d.strip()]
    eval_dirs = [d for d in eval_dirs if os.path.isdir(d)]
    if not eval_dirs:
        print(f"ERROR: No valid eval dirs: {args.eval_dir}")
        return

    instruction = TASK_INSTRUCTIONS.get(args.task, "Complete the task.")

    print(f"{'='*60}")
    print(f"  VLM v2 Eval: {args.task}")
    print(f"  LoRA:     {args.lora_path}")
    print(f"  Eval dir: {eval_dirs}")
    print(f"  Frames:   {args.num_frames}")
    print(f"  α:        {args.threshold}")
    print(f"{'='*60}")

    trajs = load_trajectories(eval_dirs, args.num_frames)
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

    print(f"\n{'='*60}")
    print(f"  RESULTS — VLM v2 ({args.task})")
    print(f"{'='*60}")
    print(f"  ROC-AUC      = {metrics['auc']:.4f}")
    print(f"  Acc@α=0.8    = {metrics['acc_alpha']:.4f}")
    print(f"  FP@α=0.8     = {metrics['fp_rate_alpha']:.4f}")
    print(f"  Recall@α=0.8 = {metrics.get('recall_alpha', 'N/A')}")
    print(f"  Acc@Youden   = {metrics.get('acc_youden', 'N/A')}")
    print(f"  FP@Youden    = {metrics.get('fp_rate_youden', 'N/A')}")
    print(f"  Youden θ     = {metrics.get('youden_threshold', 'N/A')}")
    print(f"  p_yes(+)     = {metrics['p_yes_mean_success']:.4f} ± {metrics['p_yes_std_success']:.4f}")
    print(f"  p_yes(-)     = {metrics['p_yes_mean_fail']:.4f} ± {metrics['p_yes_std_fail']:.4f}")
    print(f"  CM@α=0.8: {metrics['cm_alpha']}")
    print(f"{'='*60}")

    metrics["num_frames"] = args.num_frames
    metrics["n_trajs"] = len(trajs)
    metrics["n_success"] = n_succ
    metrics["n_fail"] = n_fail
    metrics["lora_path"] = args.lora_path
    metrics["eval_dirs"] = eval_dirs
    metrics["task"] = args.task

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as jf:
            json.dump(metrics, jf, indent=2)
        print(f"Saved → {args.output_json}")

    # Also evaluate on training data for comparison (if mixed dir exists)
    mixed_dir = str(ROOT / f"data/vlaw/rollouts/mixed/{args.task}")
    if os.path.isdir(mixed_dir):
        print(f"\n--- Also evaluating on training data ({mixed_dir}) ---")
        train_trajs = load_trajectories([mixed_dir], args.num_frames)
        if train_trajs:
            y_true_train = np.array([int(t["success"]) for t in train_trajs])
            p_yes_train = np.array(evaluate_model(model, train_trajs, instruction))
            train_metrics = compute_metrics(y_true_train, p_yes_train, args.threshold)
            print(f"  Train AUC={train_metrics['auc']:.4f}  "
                  f"Acc@α={train_metrics['acc_alpha']:.4f}  "
                  f"FP@α={train_metrics['fp_rate_alpha']:.4f}")
            if args.output_json:
                train_json = args.output_json.replace(".json", "_train.json")
                with open(train_json, "w") as jf:
                    json.dump(train_metrics, jf, indent=2)
                print(f"Saved train metrics → {train_json}")

    model.unload_model()
    print("\nDone.")


if __name__ == "__main__":
    main()
