"""
VLM 奖励模型外部评估脚本 — 在独立 eval 集上评估 LoRA 微调模型。

评估内容:
  1. 混淆矩阵 (TP/FP/TN/FN)
  2. ROC-AUC
  3. FP rate @ α=0.8
  4. 对比 zero-shot vs fine-tuned

用法:
  CUDA_VISIBLE_DEVICES=6 python scripts/vlaw/eval_reward_model_v3.py \
      --eval_data data/vlaw/rollouts/eval/LiftPegUpright-v1 \
      --lora_path checkpoints/vlaw/reward_model/lora_v3 \
      --output_dir results/vlaw/reward_eval_v3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# 添加项目根路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlft.vlaw.reward.reward_model import VLAWRewardConfig, VLAWRewardModel


TASK_INSTRUCTIONS = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
    "PickCube-v1": "Pick up the cube and place it on the target location.",
    "StackCube-v1": "Stack the red cube on top of the green cube.",
}


def load_eval_data(eval_dir: str, task_id: str = "LiftPegUpright-v1"):
    """从 HDF5 加载评估数据。"""
    eval_path = Path(eval_dir)
    h5_files = sorted(eval_path.glob("*.h5")) + sorted(eval_path.glob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files in {eval_dir}")

    trajectories = []
    labels = []
    for h5_path in h5_files:
        with h5py.File(str(h5_path), "r") as f:
            for key in sorted(f.keys()):
                if key == "meta":
                    continue
                grp = f[key]
                if "rgb_base" not in grp or "env_success" not in grp:
                    continue
                rgb = grp["rgb_base"][()]  # [T, H, W, C]
                env_success = grp["env_success"][()]
                label = bool(env_success[-1]) if isinstance(env_success, np.ndarray) else bool(env_success)
                trajectories.append(rgb)
                labels.append(int(label))

    print(f"[EVAL] 加载 {len(trajectories)} 条轨迹 (pos={sum(labels)}, neg={len(labels)-sum(labels)})")
    return trajectories, labels


def compute_metrics(p_yes_list, labels, threshold=0.8):
    """计算评估指标。"""
    tp = fp = tn = fn = 0
    for p_yes, label in zip(p_yes_list, labels):
        pred = 1 if p_yes >= threshold else 0
        if label == 1 and pred == 1: tp += 1
        elif label == 0 and pred == 1: fp += 1
        elif label == 0 and pred == 0: tn += 1
        else: fn += 1

    total = max(tp + fp + tn + fn, 1)
    metrics = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / total,
        "fp_rate": fp / max(fp + tn, 1),
        "fn_rate": fn / max(fn + tp, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "threshold": threshold,
    }

    # ROC-AUC (manual computation to avoid sklearn dependency)
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(labels, p_yes_list))
        else:
            metrics["roc_auc"] = float("nan")
    except ImportError:
        # Manual AUC computation
        if len(set(labels)) > 1:
            metrics["roc_auc"] = _manual_roc_auc(labels, p_yes_list)
        else:
            metrics["roc_auc"] = float("nan")

    return metrics


def _manual_roc_auc(labels, scores):
    """手动计算 ROC-AUC (无 sklearn 依赖)。"""
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    
    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr
    
    return auc


def evaluate_model(model, trajectories, labels, instruction, tag=""):
    """对所有轨迹评分并返回指标。"""
    p_yes_list = []
    for i, traj in enumerate(trajectories):
        result = model.score_trajectory(traj, instruction)
        p_yes_list.append(result["p_yes"])
        label_str = "✓" if labels[i] else "✗"
        pred_str = "YES" if result["p_yes"] >= 0.8 else "no"
        print(f"  [{tag}] traj {i:3d}: p_yes={result['p_yes']:.4f} → {pred_str}  (GT={label_str})")

    metrics = compute_metrics(p_yes_list, labels)
    metrics["p_yes_list"] = p_yes_list
    metrics["mean_p_yes_pos"] = float(np.mean([p for p, l in zip(p_yes_list, labels) if l == 1])) if any(l == 1 for l in labels) else 0.0
    metrics["mean_p_yes_neg"] = float(np.mean([p for p, l in zip(p_yes_list, labels) if l == 0])) if any(l == 0 for l in labels) else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="VLM 奖励模型评估")
    parser.add_argument("--eval_data", default="data/vlaw/rollouts/eval/LiftPegUpright-v1",
                       help="评估数据目录")
    parser.add_argument("--model_path", default="checkpoints/vlaw/reward_model/qwen_vl",
                       help="基座模型路径")
    parser.add_argument("--lora_path", default="checkpoints/vlaw/reward_model/lora_v3",
                       help="LoRA adapter 路径")
    parser.add_argument("--task", default="LiftPegUpright-v1")
    parser.add_argument("--output_dir", default="results/vlaw/reward_eval_v3")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--skip_zeroshot", action="store_true",
                       help="跳过 zero-shot 评估")
    args = parser.parse_args()

    instruction = TASK_INSTRUCTIONS.get(args.task, "Complete the manipulation task successfully.")
    trajectories, labels = load_eval_data(args.eval_data, args.task)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── Zero-shot 评估 ──
    if not args.skip_zeroshot:
        print("\n" + "="*60)
        print("  Zero-shot 评估 (无 LoRA)")
        print("="*60)
        config_zs = VLAWRewardConfig(
            model_path=args.model_path,
            threshold=args.threshold,
            use_video_format=True,
            video_fps=2.0,
        )
        model_zs = VLAWRewardModel(config_zs)
        model_zs.load_model(lora_path=None)
        metrics_zs = evaluate_model(model_zs, trajectories, labels, instruction, tag="ZS")
        results["zero_shot"] = {k: v for k, v in metrics_zs.items() if k != "p_yes_list"}
        results["zero_shot"]["p_yes_list"] = metrics_zs["p_yes_list"]

        print(f"\n[ZS] Accuracy={metrics_zs['accuracy']:.3f}  FP_rate={metrics_zs['fp_rate']:.3f}  "
              f"ROC-AUC={metrics_zs.get('roc_auc', 'N/A')}")
        print(f"[ZS] TP={metrics_zs['tp']}  FP={metrics_zs['fp']}  TN={metrics_zs['tn']}  FN={metrics_zs['fn']}")
        print(f"[ZS] mean_p_yes: pos={metrics_zs['mean_p_yes_pos']:.4f}  neg={metrics_zs['mean_p_yes_neg']:.4f}")
        model_zs.unload_model()

    # ── Fine-tuned 评估 ──
    lora_path = Path(args.lora_path)
    # 检查多个可能的路径
    lora_candidates = [lora_path, lora_path / "final", lora_path / "step_200"]
    actual_lora = None
    for candidate in lora_candidates:
        if (candidate / "adapter_config.json").exists():
            actual_lora = candidate
            break
    
    if actual_lora is None:
        print(f"\n[WARN] 未找到 LoRA adapter: {args.lora_path}")
        print("  跳过 fine-tuned 评估")
    else:
        print(f"\n{'='*60}")
        print(f"  Fine-tuned 评估 (LoRA: {actual_lora})")
        print(f"{'='*60}")
        config_ft = VLAWRewardConfig(
            model_path=args.model_path,
            threshold=args.threshold,
            use_video_format=True,
            video_fps=2.0,
        )
        model_ft = VLAWRewardModel(config_ft)
        model_ft.load_model(lora_path=str(actual_lora))
        metrics_ft = evaluate_model(model_ft, trajectories, labels, instruction, tag="FT")
        results["fine_tuned"] = {k: v for k, v in metrics_ft.items() if k != "p_yes_list"}
        results["fine_tuned"]["p_yes_list"] = metrics_ft["p_yes_list"]
        results["fine_tuned"]["lora_path"] = str(actual_lora)

        print(f"\n[FT] Accuracy={metrics_ft['accuracy']:.3f}  FP_rate={metrics_ft['fp_rate']:.3f}  "
              f"ROC-AUC={metrics_ft.get('roc_auc', 'N/A')}")
        print(f"[FT] TP={metrics_ft['tp']}  FP={metrics_ft['fp']}  TN={metrics_ft['tn']}  FN={metrics_ft['fn']}")
        print(f"[FT] mean_p_yes: pos={metrics_ft['mean_p_yes_pos']:.4f}  neg={metrics_ft['mean_p_yes_neg']:.4f}")
        model_ft.unload_model()

    # ── 保存结果 ──
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── 对比摘要 ──
    print("\n" + "="*60)
    print("  评估摘要")
    print("="*60)
    if "zero_shot" in results:
        zs = results["zero_shot"]
        print(f"  Zero-shot : Acc={zs['accuracy']:.3f}  FP={zs['fp_rate']:.3f}  AUC={zs.get('roc_auc', 'N/A')}")
    if "fine_tuned" in results:
        ft = results["fine_tuned"]
        print(f"  Fine-tuned: Acc={ft['accuracy']:.3f}  FP={ft['fp_rate']:.3f}  AUC={ft.get('roc_auc', 'N/A')}")
    print("="*60)

    print(f"\n结果已保存到: {output_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
