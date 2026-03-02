#!/usr/bin/env python3
"""VLM 1-frame vs 16-frame 对比评估脚本

对比 4 种配置的 ROC-AUC:
  1. Zero-shot + 1 帧 (当前基线)
  2. Zero-shot + 全帧 (uniform, paper Appendix C)
  3. LoRA + 1 帧
  4. LoRA + 全帧 (uniform)

用法:
    CUDA_VISIBLE_DEVICES=6 python scripts/vlaw/eval/eval_vlm_16frame.py
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


# ── 短指令 (与训练一致) ──
TASK_INSTRUCTIONS = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
    "PickCube-v1": "Pick up the cube and place it on the target location.",
    "StackCube-v1": "Stack the red cube on top of the green cube.",
}


def load_trajectories_multiframe(
    rollout_dirs: list[str],
    max_frames: int = 16,
) -> list[dict]:
    """从 rollout HDF5 加载轨迹: 均匀采样 max_frames 帧 + 最后1帧."""
    trajs = []
    for rollout_dir in rollout_dirs:
        for fp in sorted(glob.glob(os.path.join(rollout_dir, "*.h5"))):
            with h5py.File(fp, "r") as f:
                for k in sorted(f.keys()):
                    if not k.startswith("traj"):
                        continue
                    grp = f[k]
                    rgb = grp["rgb_base"][:]           # (T, H, W, 3) uint8
                    success = bool(grp["env_success"][-1])
                    T = rgb.shape[0]

                    # 均匀采样 min(max_frames, T) 帧
                    n = min(max_frames, T)
                    uniform_idxs = np.linspace(0, T - 1, n, dtype=int)
                    multi_frames = rgb[uniform_idxs]   # (n, H, W, 3)

                    # 最后1帧
                    last_frame = rgb[-1:]               # (1, H, W, 3)

                    trajs.append({
                        "multi_frames": multi_frames,
                        "last_frame": last_frame,
                        "n_frames_total": T,
                        "n_frames_sampled": n,
                        "success": success,
                        "source": f"{os.path.basename(fp)}:{k}",
                    })
    return trajs


def evaluate_model(model, trajs: list[dict], instruction: str,
                   mode: str = "multi") -> list[float]:
    """对所有轨迹评分.

    Args:
        mode: "multi" = 使用均匀采样的多帧; "single" = 仅用最后1帧
    """
    p_yes_list = []
    t0 = time.time()
    for i, t in enumerate(trajs):
        frames = t["multi_frames"] if mode == "multi" else t["last_frame"]
        result = model.score_trajectory(frames, instruction)
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
                    threshold: float = 0.8):
    """计算 ROC-AUC, 最优阈值, confusion matrix."""
    try:
        from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    except ImportError:
        print("[WARN] sklearn not found, computing basic metrics only")
        return _compute_basic_metrics(y_true, p_yes, threshold)

    auc = roc_auc_score(y_true, p_yes)

    fpr, tpr, thresholds = roc_curve(y_true, p_yes)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])
    best_j = float(j_scores[best_idx])

    preds_alpha = (p_yes > threshold).astype(int)
    cm_alpha = confusion_matrix(y_true, preds_alpha, labels=[0, 1])

    preds_best = (p_yes > best_thresh).astype(int)
    cm_best = confusion_matrix(y_true, preds_best, labels=[0, 1])

    return {
        "auc": auc,
        "best_threshold": best_thresh,
        "best_j": best_j,
        "cm_alpha": cm_alpha.tolist(),
        "cm_best": cm_best.tolist(),
        "p_yes_mean_success": float(p_yes[y_true == 1].mean()),
        "p_yes_mean_fail": float(p_yes[y_true == 0].mean()),
        "p_yes_std_success": float(p_yes[y_true == 1].std()),
        "p_yes_std_fail": float(p_yes[y_true == 0].std()),
        "p_yes_max": float(p_yes.max()),
        "p_yes_min": float(p_yes.min()),
    }


def _compute_basic_metrics(y_true, p_yes, threshold):
    preds = (p_yes > threshold).astype(int)
    tp = int(((y_true == 1) & (preds == 1)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    tn = int(((y_true == 0) & (preds == 0)).sum())
    fn = int(((y_true == 1) & (preds == 0)).sum())
    return {
        "auc": -1.0,
        "best_threshold": threshold,
        "cm_alpha": [[tn, fp], [fn, tp]],
        "cm_best": [[tn, fp], [fn, tp]],
        "p_yes_mean_success": float(p_yes[y_true == 1].mean()),
        "p_yes_mean_fail": float(p_yes[y_true == 0].mean()),
        "p_yes_std_success": float(p_yes[y_true == 1].std()),
        "p_yes_std_fail": float(p_yes[y_true == 0].std()),
    }


def format_cm(cm) -> str:
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    return (
        f"|  | Pred Fail | Pred Succ |\n"
        f"|--|-----------|----------|\n"
        f"| GT Fail    | {tn} (TN) | {fp} (FP) |\n"
        f"| GT Succ    | {fn} (FN) | {tp} (TP) |\n\n"
        f"Acc={acc:.1%}, FP rate={fp_rate:.1%}"
    )


def find_optimal_thresholds(y_true: np.ndarray, p_yes: np.ndarray,
                            max_fp_rate: float = 0.20) -> dict:
    """查找使 FP rate < max_fp_rate 的最优阈值 + Youden's J 最优阈值."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, p_yes)

    # Youden's J
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)

    # FP rate < max_fp_rate 的最高 recall 阈值
    valid = fpr <= max_fp_rate
    if valid.any():
        valid_recalls = tpr[valid]
        best_constrained_idx = np.where(valid)[0][np.argmax(valid_recalls)]
        constrained_thresh = float(thresholds[best_constrained_idx])
        constrained_recall = float(tpr[best_constrained_idx])
        constrained_fp_rate = float(fpr[best_constrained_idx])
    else:
        constrained_thresh = 1.0
        constrained_recall = 0.0
        constrained_fp_rate = 0.0

    return {
        "youden_threshold": float(thresholds[best_j_idx]),
        "youden_j": float(j_scores[best_j_idx]),
        "youden_recall": float(tpr[best_j_idx]),
        "youden_fp_rate": float(fpr[best_j_idx]),
        "constrained_threshold": constrained_thresh,
        "constrained_recall": constrained_recall,
        "constrained_fp_rate": constrained_fp_rate,
        "max_fp_rate_constraint": max_fp_rate,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VLM 1-frame vs 16-frame evaluation")
    parser.add_argument("--task", default="LiftPegUpright-v1")
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--model_path", default=str(ROOT / "checkpoints/vlaw/reward_model/qwen_vl"))
    parser.add_argument("--lora_path", default=str(ROOT / "checkpoints/vlaw/reward_model/lora_iter1_16frame/final"))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--skip_zeroshot", action="store_true",
                        help="Skip zero-shot evaluation to save time")
    args = parser.parse_args()

    rollout_dirs = [
        str(ROOT / f"data/vlaw/rollouts/iter1/{args.task}"),
        str(ROOT / f"data/vlaw/rollouts/iter1_highsuc/{args.task}"),
    ]
    rollout_dirs = [d for d in rollout_dirs if os.path.isdir(d)]

    out_dir = ROOT / "results" / "vlaw"
    out_dir.mkdir(parents=True, exist_ok=True)

    instruction = TASK_INSTRUCTIONS.get(args.task, "Complete the task.")

    # ── 加载轨迹 ──
    print("=== Loading trajectories ===")
    trajs = load_trajectories_multiframe(rollout_dirs, args.max_frames)
    y_true = np.array([int(t["success"]) for t in trajs])
    n_succ = y_true.sum()
    n_fail = len(y_true) - n_succ
    print(f"Loaded {len(trajs)} trajs: {n_succ} success, {n_fail} fail")
    if len(trajs) == 0:
        print("ERROR: No trajectories found!")
        return

    # 帧数统计
    frame_counts = [t["n_frames_sampled"] for t in trajs]
    print(f"Frame counts: min={min(frame_counts)}, max={max(frame_counts)}, "
          f"mean={np.mean(frame_counts):.1f}")

    from rlft.vlaw.reward.reward_model import VLAWRewardModel, VLAWRewardConfig

    all_results = {}
    report_parts = []

    # ── 配置矩阵: (model_type, frame_mode) ──
    configs = []
    if not args.skip_zeroshot:
        configs.extend([
            ("zero_shot", "single", None),
            ("zero_shot", "multi", None),
        ])

    has_lora = os.path.isfile(os.path.join(args.lora_path, "adapter_config.json"))
    if has_lora:
        configs.extend([
            ("lora", "single", args.lora_path),
            ("lora", "multi", args.lora_path),
        ])
    else:
        print(f"⚠️ LoRA adapter not found at {args.lora_path}")

    for model_type, frame_mode, lora_path in configs:
        label = f"{model_type}_{frame_mode}"
        n_frames_desc = "1 frame (last)" if frame_mode == "single" else f"≤{args.max_frames} frames (uniform)"
        print(f"\n{'='*60}")
        print(f"=== {label}: {n_frames_desc} ===")
        print(f"{'='*60}")

        cfg = VLAWRewardConfig(
            model_path=args.model_path,
            threshold=args.threshold,
            device="cuda:0",
            num_frames=args.max_frames,
        )
        model = VLAWRewardModel(cfg)
        model.load_model(lora_path=lora_path)

        p_yes = np.array(evaluate_model(model, trajs, instruction, mode=frame_mode))
        metrics = compute_metrics(y_true, p_yes, args.threshold)
        thresholds = find_optimal_thresholds(y_true, p_yes)

        all_results[label] = {
            **metrics,
            **thresholds,
            "p_yes_all": p_yes.tolist(),
        }

        print(f"\n  AUC = {metrics['auc']:.4f}")
        print(f"  p_yes (success): {metrics['p_yes_mean_success']:.4f} ± {metrics['p_yes_std_success']:.4f}")
        print(f"  p_yes (fail):    {metrics['p_yes_mean_fail']:.4f} ± {metrics['p_yes_std_fail']:.4f}")
        print(f"  p_yes range: [{metrics.get('p_yes_min', 0):.4f}, {metrics.get('p_yes_max', 0):.4f}]")
        print(f"  Youden threshold: {thresholds['youden_threshold']:.4f} "
              f"(J={thresholds['youden_j']:.4f}, recall={thresholds['youden_recall']:.2%}, "
              f"FP={thresholds['youden_fp_rate']:.2%})")
        print(f"  FP<20% threshold: {thresholds['constrained_threshold']:.4f} "
              f"(recall={thresholds['constrained_recall']:.2%}, "
              f"FP={thresholds['constrained_fp_rate']:.2%})")

        report_parts.append(f"""### {label} ({n_frames_desc})
- **ROC-AUC: {metrics['auc']:.4f}**
- p_yes (success): {metrics['p_yes_mean_success']:.4f} ± {metrics['p_yes_std_success']:.4f}
- p_yes (fail): {metrics['p_yes_mean_fail']:.4f} ± {metrics['p_yes_std_fail']:.4f}
- Youden's J threshold: {thresholds['youden_threshold']:.4f} (recall={thresholds['youden_recall']:.1%}, FP={thresholds['youden_fp_rate']:.1%})
- FP<20% threshold: {thresholds['constrained_threshold']:.4f} (recall={thresholds['constrained_recall']:.1%}, FP={thresholds['constrained_fp_rate']:.1%})

**CM @ α={args.threshold}:**
{format_cm(metrics['cm_alpha'])}

**CM @ Youden threshold={thresholds['youden_threshold']:.4f}:**
{format_cm(metrics['cm_best'])}
""")
        model.unload_model()

    # ── 对比摘要 ──
    print(f"\n{'='*60}")
    print("=== COMPARISON SUMMARY ===")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'AUC':>8} {'p_yes(+)':>10} {'p_yes(-)':>10} {'Best_θ':>8} {'Recall@FP20%':>12}")
    for key, m in all_results.items():
        print(f"{key:<25} {m['auc']:>8.4f} "
              f"{m['p_yes_mean_success']:>10.4f} {m['p_yes_mean_fail']:>10.4f} "
              f"{m['youden_threshold']:>8.4f} "
              f"{m['constrained_recall']:>12.1%}")

    # ── 保存报告 ──
    report_md = f"""# VLM 1-Frame vs 16-Frame 评估报告 — {args.task}

## 数据
- 轨迹: {len(trajs)} (success={n_succ}, fail={n_fail})
- 来源: {', '.join(rollout_dirs)}
- Instruction: "{instruction}"
- 帧数: min={min(frame_counts)}, max={max(frame_counts)}

## 结果

{''.join(report_parts)}

## 对比表

| Config | AUC | p_yes(+) | p_yes(-) | Youden_θ | Recall@FP<20% |
|--------|-----|----------|----------|----------|---------------|
""" + "\n".join(
    f"| {k} | {m['auc']:.4f} | {m['p_yes_mean_success']:.4f} | "
    f"{m['p_yes_mean_fail']:.4f} | {m['youden_threshold']:.4f} | "
    f"{m['constrained_recall']:.1%} |"
    for k, m in all_results.items()
) + f"""

## 结论
{'16帧显著优于单帧' if any(
    all_results.get(k, {}).get('auc', 0) - all_results.get(k.replace('multi', 'single'), {}).get('auc', 0) > 0.05
    for k in all_results if 'multi' in k
) else 'AUC 差异不显著，需进一步调查'}
"""

    report_path = out_dir / "vlm_16frame_comparison.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\n✅ Report → {report_path}")

    json_path = out_dir / "vlm_16frame_metrics.json"
    json_results = {}
    for mode, m in all_results.items():
        json_results[mode] = {k: v for k, v in m.items()}
    with open(json_path, "w") as jf:
        json.dump(json_results, jf, indent=2)
    print(f"✅ Metrics → {json_path}")


if __name__ == "__main__":
    main()
