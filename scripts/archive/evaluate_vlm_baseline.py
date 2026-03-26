#!/usr/bin/env python3
"""
VLM 奖励模型 Zero-shot vs LoRA 基线评估脚本

评估内容:
1. Zero-shot Qwen3-VL-4B 对所有轨迹的 p_yes 评分
2. 与已有的 LoRA 评分对比
3. 分析: ROC-AUC, PR-AUC, 最优阈值, Confusion Matrix, 分布直方图
4. 输出: results/vlaw/vlm_baseline_report.md

数据来源:
- 160 条 LiftPegUpright-v1 轨迹 (63 成功 + 97 失败)
- 原始 rollout: data/vlaw/rollouts/iter1_lift_only/
- LoRA 评分: data/vlaw/labeled/iter1_lift_only/.../LiftPegUpright-v1_vlm_rewards.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# Add rlft to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class EvalConfig:
    """评估配置"""
    # 数据路径
    rollout_dir: str = "data/vlaw/rollouts/iter1_lift_only/LiftPegUpright-v1"
    lora_json: str = "data/vlaw/labeled/iter1_lift_only/LiftPegUpright-v1/LiftPegUpright-v1_vlm_rewards.json"

    # 模型路径
    base_model: str = "checkpoints/vlaw/reward_model/qwen_vl"
    lora_adapter: str = "checkpoints/vlaw/reward_model/lora_iter1/final"

    # 输出
    output_dir: str = "results/vlaw"
    output_json: str = "results/vlaw/vlm_baseline_scores.json"
    output_report: str = "results/vlaw/vlm_baseline_report.md"
    output_plots_dir: str = "results/vlaw/vlm_plots"

    # GPU
    device: str = "cuda:6"

    # VLM 参数
    num_frames: int = 16
    rgb_key: str = "rgb_base"  # HDF5 中的 RGB 数据 key

    # 指令 (与 LoRA labeling 时一致)
    instruction: str = (
        "Look at the FINAL frame of this robot manipulation sequence. "
        "The robot's goal is to lift the peg and insert it upright into the holder. "
        "Based on the last frame, has the robot fully completed this task with the peg standing upright in the holder? "
        "Answer only 'yes' or 'no'."
    )


def load_trajectories(config: EvalConfig) -> list[dict]:
    """
    加载所有轨迹的 RGB 帧和 GT 标签。

    Returns:
        list of dict: 每个 dict 包含:
            - file: 源文件名
            - traj_key: 轨迹 key
            - global_idx: 全局索引
            - rgb_frames: np.ndarray [T, H, W, C]
            - env_success_at_end: bool
    """
    rollout_dir = Path(config.rollout_dir)
    h5_files = sorted(rollout_dir.glob("*.h5"))
    print(f"[EVAL] 发现 {len(h5_files)} 个 HDF5 文件")

    trajectories = []
    global_idx = 0
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            traj_keys = sorted([k for k in f.keys() if k.startswith("traj")])
            for traj_key in traj_keys:
                grp = f[traj_key]
                rgb = grp[config.rgb_key][:]  # [T, H, W, C]
                env_success = grp["env_success"][:]  # [T]
                success_at_end = bool(env_success[-1])

                trajectories.append({
                    "file": h5_file.name,
                    "traj_key": traj_key,
                    "global_idx": global_idx,
                    "rgb_frames": rgb,
                    "env_success_at_end": success_at_end,
                })
                global_idx += 1

    n_succ = sum(1 for t in trajectories if t["env_success_at_end"])
    n_fail = len(trajectories) - n_succ
    print(f"[EVAL] 共 {len(trajectories)} 条轨迹: {n_succ} 成功, {n_fail} 失败")
    return trajectories


def run_zero_shot_scoring(
    trajectories: list[dict],
    config: EvalConfig,
) -> list[float]:
    """
    使用 zero-shot (无 LoRA) 模型对所有轨迹评分。

    Returns:
        list[float]: 每条轨迹的 p_yes
    """
    from rlft.vlaw.reward.reward_model import VLAWRewardConfig, VLAWRewardModel

    rm_config = VLAWRewardConfig(
        model_path=config.base_model,
        device=config.device,
        num_frames=config.num_frames,
        threshold=0.8,
    )
    model = VLAWRewardModel(rm_config)
    model.load_model(lora_path=None)  # 不加载 LoRA

    p_yes_list = []
    t0 = time.time()
    for i, traj in enumerate(trajectories):
        result = model.score_trajectory(traj["rgb_frames"], config.instruction)
        p_yes_list.append(result["p_yes"])
        if (i + 1) % 20 == 0 or (i + 1) == len(trajectories):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(trajectories) - i - 1)
            print(f"  [Zero-shot] {i+1}/{len(trajectories)} | "
                  f"p_yes={result['p_yes']:.4f} | "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

    model.unload_model()
    return p_yes_list


def run_lora_scoring(
    trajectories: list[dict],
    config: EvalConfig,
) -> list[float]:
    """
    使用 LoRA fine-tuned 模型对所有轨迹评分。

    Returns:
        list[float]: 每条轨迹的 p_yes
    """
    from rlft.vlaw.reward.reward_model import VLAWRewardConfig, VLAWRewardModel

    rm_config = VLAWRewardConfig(
        model_path=config.base_model,
        device=config.device,
        num_frames=config.num_frames,
        threshold=0.8,
    )
    model = VLAWRewardModel(rm_config)
    model.load_model(lora_path=config.lora_adapter)

    p_yes_list = []
    t0 = time.time()
    for i, traj in enumerate(trajectories):
        result = model.score_trajectory(traj["rgb_frames"], config.instruction)
        p_yes_list.append(result["p_yes"])
        if (i + 1) % 20 == 0 or (i + 1) == len(trajectories):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(trajectories) - i - 1)
            print(f"  [LoRA] {i+1}/{len(trajectories)} | "
                  f"p_yes={result['p_yes']:.4f} | "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

    model.unload_model()
    return p_yes_list


def load_existing_lora_scores(config: EvalConfig) -> Optional[list[float]]:
    """
    从已有的标注 JSON 加载 LoRA p_yes 分数。

    Returns:
        list[float] | None
    """
    json_path = Path(config.lora_json)
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    return [entry["vlm_yes_prob"] for entry in data]


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    model_name: str,
) -> dict:
    """
    计算二分类评估指标。

    Returns:
        dict with ROC-AUC, PR-AUC, optimal threshold, confusion matrix, etc.
    """
    from sklearn.metrics import (
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    results = {"model": model_name}

    # 基本统计
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    results["n_total"] = len(labels)
    results["n_positive"] = n_pos
    results["n_negative"] = n_neg

    # p_yes 分布统计
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    results["p_yes_mean_pos"] = float(pos_scores.mean()) if len(pos_scores) > 0 else 0.0
    results["p_yes_std_pos"] = float(pos_scores.std()) if len(pos_scores) > 0 else 0.0
    results["p_yes_mean_neg"] = float(neg_scores.mean()) if len(neg_scores) > 0 else 0.0
    results["p_yes_std_neg"] = float(neg_scores.std()) if len(neg_scores) > 0 else 0.0
    results["p_yes_min"] = float(scores.min())
    results["p_yes_max"] = float(scores.max())
    results["p_yes_median"] = float(np.median(scores))

    # ROC-AUC
    try:
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        roc_auc = roc_auc_score(labels, scores)
        results["roc_auc"] = float(roc_auc)
        results["roc_fpr"] = fpr.tolist()
        results["roc_tpr"] = tpr.tolist()
        results["roc_thresholds"] = roc_thresholds.tolist()

        # Youden's J 最优阈值
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        results["youden_threshold"] = float(roc_thresholds[best_j_idx])
        results["youden_j"] = float(j_scores[best_j_idx])
        results["youden_tpr"] = float(tpr[best_j_idx])
        results["youden_fpr"] = float(fpr[best_j_idx])
    except Exception as e:
        results["roc_auc"] = 0.5
        results["roc_error"] = str(e)

    # PR-AUC
    try:
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        results["pr_auc"] = float(pr_auc)
        results["pr_precision"] = precision.tolist()
        results["pr_recall"] = recall.tolist()
        results["pr_thresholds"] = pr_thresholds.tolist()

        # F1 最优阈值
        f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
        best_f1_idx = np.argmax(f1s)
        results["f1_threshold"] = float(pr_thresholds[best_f1_idx])
        results["f1_max"] = float(f1s[best_f1_idx])
    except Exception as e:
        results["pr_auc"] = 0.0
        results["pr_error"] = str(e)

    # 在各阈值下的 Confusion Matrix
    thresholds_to_eval = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]
    # 添加最优阈值
    if "youden_threshold" in results:
        thresholds_to_eval.append(results["youden_threshold"])
    if "f1_threshold" in results:
        thresholds_to_eval.append(results["f1_threshold"])
    thresholds_to_eval = sorted(set(thresholds_to_eval))

    results["threshold_analysis"] = []
    for thresh in thresholds_to_eval:
        preds = (scores >= thresh).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results["threshold_analysis"].append({
            "threshold": float(thresh),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "accuracy": float(acc),
            "precision": float(precision_val),
            "recall": float(recall_val),
            "f1": float(f1),
            "fpr": float(fpr_val),
        })

    return results


def generate_plots(
    labels: np.ndarray,
    zs_scores: np.ndarray,
    lora_scores: np.ndarray,
    zs_metrics: dict,
    lora_metrics: dict,
    output_dir: str,
) -> list[str]:
    """
    生成可视化图表。

    Returns:
        list of saved plot paths
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # ── 1. p_yes 分布直方图 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Zero-shot
    ax = axes[0]
    pos_mask = labels == 1
    ax.hist(zs_scores[pos_mask], bins=30, alpha=0.7, label=f"Success (n={pos_mask.sum()})", color="green", density=True)
    ax.hist(zs_scores[~pos_mask], bins=30, alpha=0.7, label=f"Failure (n=(~pos_mask).sum())", color="red", density=True)
    ax.set_xlabel("p_yes")
    ax.set_ylabel("Density")
    ax.set_title(f"Zero-shot p_yes Distribution\nAUC={zs_metrics.get('roc_auc', 0):.3f}")
    ax.legend()
    ax.axvline(zs_metrics.get("youden_threshold", 0), color="blue", linestyle="--", label=f"Youden={zs_metrics.get('youden_threshold', 0):.4f}")
    ax.legend()

    # LoRA
    ax = axes[1]
    ax.hist(lora_scores[pos_mask], bins=30, alpha=0.7, label=f"Success (n={pos_mask.sum()})", color="green", density=True)
    ax.hist(lora_scores[~pos_mask], bins=30, alpha=0.7, label=f"Failure (n=(~pos_mask).sum())", color="red", density=True)
    ax.set_xlabel("p_yes")
    ax.set_ylabel("Density")
    ax.set_title(f"LoRA p_yes Distribution\nAUC={lora_metrics.get('roc_auc', 0):.3f}")
    ax.axvline(lora_metrics.get("youden_threshold", 0), color="blue", linestyle="--", label=f"Youden={lora_metrics.get('youden_threshold', 0):.4f}")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "p_yes_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    saved_paths.append(path)

    # ── 2. ROC Curves ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    if "roc_fpr" in zs_metrics:
        ax.plot(zs_metrics["roc_fpr"], zs_metrics["roc_tpr"],
                label=f"Zero-shot (AUC={zs_metrics['roc_auc']:.3f})", linewidth=2)
    if "roc_fpr" in lora_metrics:
        ax.plot(lora_metrics["roc_fpr"], lora_metrics["roc_tpr"],
                label=f"LoRA (AUC={lora_metrics['roc_auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Zero-shot vs LoRA")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    saved_paths.append(path)

    # ── 3. Precision-Recall Curves ───────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    if "pr_precision" in zs_metrics:
        ax.plot(zs_metrics["pr_recall"], zs_metrics["pr_precision"],
                label=f"Zero-shot (PR-AUC={zs_metrics['pr_auc']:.3f})", linewidth=2)
    if "pr_precision" in lora_metrics:
        ax.plot(lora_metrics["pr_recall"], lora_metrics["pr_precision"],
                label=f"LoRA (PR-AUC={lora_metrics['pr_auc']:.3f})", linewidth=2)
    baseline = labels.sum() / len(labels)
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve: Zero-shot vs LoRA")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "pr_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    saved_paths.append(path)

    # ── 4. Threshold vs Metrics ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metrics, name in [(axes[0], zs_metrics, "Zero-shot"), (axes[1], lora_metrics, "LoRA")]:
        ta = metrics.get("threshold_analysis", [])
        thresholds = [t["threshold"] for t in ta]
        f1s = [t["f1"] for t in ta]
        accs = [t["accuracy"] for t in ta]
        recalls = [t["recall"] for t in ta]
        precs = [t["precision"] for t in ta]
        fprs = [t["fpr"] for t in ta]

        ax.plot(thresholds, f1s, "o-", label="F1", linewidth=2)
        ax.plot(thresholds, accs, "s-", label="Accuracy", linewidth=2)
        ax.plot(thresholds, recalls, "^-", label="Recall", linewidth=2)
        ax.plot(thresholds, precs, "d-", label="Precision", linewidth=2)
        ax.plot(thresholds, fprs, "v-", label="FPR", linewidth=2, color="red")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"{name}: Metrics vs Threshold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(thresholds) * 1.1 if thresholds else 1.0)

    plt.tight_layout()
    path = os.path.join(output_dir, "threshold_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    saved_paths.append(path)

    # ── 5. Box Plot comparison ───────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    data_to_plot = [
        zs_scores[pos_mask], zs_scores[~pos_mask],
        lora_scores[pos_mask], lora_scores[~pos_mask],
    ]
    bp = ax.boxplot(data_to_plot, labels=[
        "ZS-Success", "ZS-Failure", "LoRA-Success", "LoRA-Failure"
    ], patch_artist=True)
    colors = ["lightgreen", "lightcoral", "green", "red"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("p_yes")
    ax.set_title("p_yes Distribution by Model and Label")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "boxplot_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    saved_paths.append(path)

    print(f"[EVAL] 已保存 {len(saved_paths)} 个图表到 {output_dir}/")
    return saved_paths


def generate_report(
    zs_metrics: dict,
    lora_metrics: dict,
    plot_paths: list[str],
    config: EvalConfig,
) -> str:
    """生成 Markdown 报告"""
    report_lines = []
    report_lines.append("# VLM 奖励模型基线评估报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**基础模型**: Qwen3-VL-4B-Instruct (`{config.base_model}`)")
    report_lines.append(f"**LoRA Adapter**: `{config.lora_adapter}`")
    report_lines.append(f"**评估集**: {zs_metrics['n_total']} 条轨迹 ({zs_metrics['n_positive']} 成功 + {zs_metrics['n_negative']} 失败)")
    report_lines.append(f"**任务**: LiftPegUpright-v1")
    report_lines.append(f"**GT 标签**: env_success_at_end (ManiSkill 仿真)")
    report_lines.append(f"**采样帧数**: {config.num_frames}")
    report_lines.append("")

    # ── 核心指标对比 ──
    report_lines.append("## 1. 核心指标对比")
    report_lines.append("")
    report_lines.append("| 指标 | Zero-shot | LoRA Fine-tuned | 改进 |")
    report_lines.append("|------|-----------|-----------------|------|")

    zs_auc = zs_metrics.get("roc_auc", 0)
    lr_auc = lora_metrics.get("roc_auc", 0)
    report_lines.append(f"| ROC-AUC | {zs_auc:.4f} | {lr_auc:.4f} | {lr_auc - zs_auc:+.4f} |")

    zs_prauc = zs_metrics.get("pr_auc", 0)
    lr_prauc = lora_metrics.get("pr_auc", 0)
    report_lines.append(f"| PR-AUC | {zs_prauc:.4f} | {lr_prauc:.4f} | {lr_prauc - zs_prauc:+.4f} |")

    zs_f1 = zs_metrics.get("f1_max", 0)
    lr_f1 = lora_metrics.get("f1_max", 0)
    report_lines.append(f"| Best F1 | {zs_f1:.4f} | {lr_f1:.4f} | {lr_f1 - zs_f1:+.4f} |")

    zs_yt = zs_metrics.get("youden_threshold", 0)
    lr_yt = lora_metrics.get("youden_threshold", 0)
    report_lines.append(f"| Youden's J Threshold | {zs_yt:.4f} | {lr_yt:.4f} | — |")

    zs_f1t = zs_metrics.get("f1_threshold", 0)
    lr_f1t = lora_metrics.get("f1_threshold", 0)
    report_lines.append(f"| F1-max Threshold | {zs_f1t:.4f} | {lr_f1t:.4f} | — |")

    report_lines.append(f"| p_yes range | [{zs_metrics['p_yes_min']:.4f}, {zs_metrics['p_yes_max']:.4f}] | [{lora_metrics['p_yes_min']:.4f}, {lora_metrics['p_yes_max']:.4f}] | — |")
    report_lines.append(f"| p_yes mean (success) | {zs_metrics['p_yes_mean_pos']:.4f}±{zs_metrics['p_yes_std_pos']:.4f} | {lora_metrics['p_yes_mean_pos']:.4f}±{lora_metrics['p_yes_std_pos']:.4f} | — |")
    report_lines.append(f"| p_yes mean (failure) | {zs_metrics['p_yes_mean_neg']:.4f}±{zs_metrics['p_yes_std_neg']:.4f} | {lora_metrics['p_yes_mean_neg']:.4f}±{lora_metrics['p_yes_std_neg']:.4f} | — |")
    report_lines.append("")

    # ── p_yes 分布 ──
    report_lines.append("## 2. p_yes 分布分析")
    report_lines.append("")
    report_lines.append("### Zero-shot")
    report_lines.append(f"- 成功轨迹 p_yes: **{zs_metrics['p_yes_mean_pos']:.4f}** ± {zs_metrics['p_yes_std_pos']:.4f}")
    report_lines.append(f"- 失败轨迹 p_yes: **{zs_metrics['p_yes_mean_neg']:.4f}** ± {zs_metrics['p_yes_std_neg']:.4f}")
    sep = abs(zs_metrics['p_yes_mean_pos'] - zs_metrics['p_yes_mean_neg'])
    report_lines.append(f"- 均值差距: {sep:.4f} ({'有区分' if sep > 0.01 else '几乎无区分'})")
    report_lines.append("")
    report_lines.append("### LoRA Fine-tuned")
    report_lines.append(f"- 成功轨迹 p_yes: **{lora_metrics['p_yes_mean_pos']:.4f}** ± {lora_metrics['p_yes_std_pos']:.4f}")
    report_lines.append(f"- 失败轨迹 p_yes: **{lora_metrics['p_yes_mean_neg']:.4f}** ± {lora_metrics['p_yes_std_neg']:.4f}")
    sep = abs(lora_metrics['p_yes_mean_pos'] - lora_metrics['p_yes_mean_neg'])
    report_lines.append(f"- 均值差距: {sep:.4f} ({'有区分' if sep > 0.01 else '几乎无区分'})")
    report_lines.append("")

    # ── 阈值分析 ──
    report_lines.append("## 3. 阈值分析")
    report_lines.append("")

    for name, metrics in [("Zero-shot", zs_metrics), ("LoRA", lora_metrics)]:
        report_lines.append(f"### {name}")
        report_lines.append("")
        report_lines.append("| 阈值 | TP | FP | TN | FN | Accuracy | Precision | Recall | F1 | FPR |")
        report_lines.append("|------|----|----|----|----|----------|-----------|--------|----|----|")
        for ta in metrics.get("threshold_analysis", []):
            report_lines.append(
                f"| {ta['threshold']:.4f} | {ta['TP']} | {ta['FP']} | {ta['TN']} | {ta['FN']} | "
                f"{ta['accuracy']:.3f} | {ta['precision']:.3f} | {ta['recall']:.3f} | {ta['f1']:.3f} | {ta['fpr']:.3f} |"
            )
        report_lines.append("")

    # ── 最优阈值推荐 ──
    report_lines.append("## 4. 最优阈值推荐")
    report_lines.append("")

    # 找 LoRA 最优
    best_lora = None
    for ta in lora_metrics.get("threshold_analysis", []):
        if best_lora is None or ta["f1"] > best_lora["f1"]:
            best_lora = ta

    best_zs = None
    for ta in zs_metrics.get("threshold_analysis", []):
        if best_zs is None or ta["f1"] > best_zs["f1"]:
            best_zs = ta

    if best_lora:
        report_lines.append(f"### LoRA 最优阈值 (by F1)")
        report_lines.append(f"- **阈值**: {best_lora['threshold']:.4f}")
        report_lines.append(f"- F1: {best_lora['f1']:.3f}, Precision: {best_lora['precision']:.3f}, Recall: {best_lora['recall']:.3f}")
        report_lines.append(f"- FP: {best_lora['FP']}, FN: {best_lora['FN']}")
        report_lines.append(f"- FPR: {best_lora['fpr']:.3f}")
        report_lines.append("")

    if best_zs:
        report_lines.append(f"### Zero-shot 最优阈值 (by F1)")
        report_lines.append(f"- **阈值**: {best_zs['threshold']:.4f}")
        report_lines.append(f"- F1: {best_zs['f1']:.3f}, Precision: {best_zs['precision']:.3f}, Recall: {best_zs['recall']:.3f}")
        report_lines.append(f"- FP: {best_zs['FP']}, FN: {best_zs['FN']}")
        report_lines.append(f"- FPR: {best_zs['fpr']:.3f}")
        report_lines.append("")

    # ── 论文阈值 α=0.8 下的表现 ──
    report_lines.append("## 5. 论文阈值 α=0.8 下的表现")
    report_lines.append("")
    for name, metrics in [("Zero-shot", zs_metrics), ("LoRA", lora_metrics)]:
        ta_08 = None
        for ta in metrics.get("threshold_analysis", []):
            if abs(ta["threshold"] - 0.8) < 0.01:
                ta_08 = ta
                break
        if ta_08:
            report_lines.append(f"### {name} @ α=0.8")
            report_lines.append(f"- TP={ta_08['TP']}, FP={ta_08['FP']}, TN={ta_08['TN']}, FN={ta_08['FN']}")
            report_lines.append(f"- Accuracy: {ta_08['accuracy']:.3f}, F1: {ta_08['f1']:.3f}")
            report_lines.append(f"- **FPR: {ta_08['fpr']:.3f}** (VLAW 论文目标: <20%)")
            report_lines.append("")

    # ── 诊断分析 ──
    report_lines.append("## 6. 诊断分析")
    report_lines.append("")

    # 判断模型是"过于保守"还是"无区分度"
    if lr_auc < 0.55:
        report_lines.append("### 诊断: LoRA 模型**无区分度** (AUC ≈ 0.5)")
        report_lines.append("")
        report_lines.append("- AUC 接近随机，模型无法区分成功/失败轨迹")
        report_lines.append("- 可能原因:")
        report_lines.append("  1. LoRA 微调数据量不足 (当前: 160条)")
        report_lines.append("  2. 训练步数不足 (当前: 200步)")
        report_lines.append("  3. Prompt 设计问题")
        report_lines.append("  4. 视觉信息不足 (ManiSkill 渲染质量)")
    elif lora_metrics["p_yes_max"] < 0.3:
        report_lines.append("### 诊断: LoRA 模型**过于保守** (p_yes_max < 0.3)")
        report_lines.append("")
        report_lines.append(f"- p_yes 最大值仅 {lora_metrics['p_yes_max']:.4f}，远低于论文阈值 α=0.8")
        report_lines.append("- 模型对所有轨迹都倾向回答 'no'")
        report_lines.append("- 但如果降低阈值，可能仍有区分能力")
        if lr_auc > 0.6:
            report_lines.append(f"- **✅ 好消息: AUC={lr_auc:.3f} 表明模型有区分能力，仅需调整阈值**")
        report_lines.append("")
        report_lines.append("### 建议:")
        if best_lora and best_lora["f1"] > 0.5:
            report_lines.append(f"- 使用推荐阈值 **{best_lora['threshold']:.4f}** 替代论文 α=0.8")
            report_lines.append(f"- 该阈值下 F1={best_lora['f1']:.3f}, FPR={best_lora['fpr']:.3f}")
        report_lines.append("- 考虑增加训练数据或步数重新微调")
        report_lines.append("- 考虑使用更大模型 (Qwen3-VL-8B)")
    else:
        report_lines.append("### 诊断: LoRA 模型**表现正常**")
        report_lines.append("")
        report_lines.append(f"- AUC={lr_auc:.3f}, p_yes 范围 [{lora_metrics['p_yes_min']:.4f}, {lora_metrics['p_yes_max']:.4f}]")

    report_lines.append("")

    # ── 可视化 ──
    report_lines.append("## 7. 可视化")
    report_lines.append("")
    for path in plot_paths:
        rel_path = os.path.relpath(path, os.path.dirname(config.output_report))
        name = os.path.basename(path).replace(".png", "").replace("_", " ").title()
        report_lines.append(f"### {name}")
        report_lines.append(f"![{name}]({rel_path})")
        report_lines.append("")

    # ── 结论 ──
    report_lines.append("## 8. 结论与下一步")
    report_lines.append("")
    if lr_auc > zs_auc:
        report_lines.append(f"- LoRA 微调**提升**了 AUC: {zs_auc:.3f} → {lr_auc:.3f} (+{lr_auc - zs_auc:.3f})")
    else:
        report_lines.append(f"- LoRA 微调**未提升** AUC: {zs_auc:.3f} → {lr_auc:.3f} ({lr_auc - zs_auc:+.3f})")

    if best_lora and best_lora["f1"] > 0.5:
        report_lines.append(f"- 推荐阈值: **{best_lora['threshold']:.4f}** (F1={best_lora['f1']:.3f})")
        report_lines.append(f"- 在此阈值下 FPR={best_lora['fpr']:.3f}，{'满足' if best_lora['fpr'] < 0.2 else '不满足'} VLAW <20% 要求")
    else:
        report_lines.append("- ⚠️ 未找到 F1>0.5 的实用阈值")

    report_lines.append("- 对 D_syn 标注: 需使用 LoRA 模型 + 推荐阈值 (非论文 α=0.8)")
    report_lines.append("")

    return "\n".join(report_lines)


def main():
    config = EvalConfig()
    root = Path(__file__).resolve().parent.parent
    # Resolve relative paths
    config.rollout_dir = str(root / config.rollout_dir)
    config.lora_json = str(root / config.lora_json)
    config.base_model = str(root / config.base_model)
    config.lora_adapter = str(root / config.lora_adapter)
    config.output_dir = str(root / config.output_dir)
    config.output_json = str(root / config.output_json)
    config.output_report = str(root / config.output_report)
    config.output_plots_dir = str(root / config.output_plots_dir)

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.output_plots_dir, exist_ok=True)

    print("=" * 60)
    print("VLM 奖励模型基线评估")
    print("=" * 60)

    # Step 1: 加载轨迹
    print("\n[Step 1] 加载轨迹数据...")
    trajectories = load_trajectories(config)
    labels = np.array([t["env_success_at_end"] for t in trajectories], dtype=int)

    # Step 2: Zero-shot 评分
    print("\n[Step 2] Zero-shot 评分...")
    t0 = time.time()
    zs_scores = run_zero_shot_scoring(trajectories, config)
    zs_time = time.time() - t0
    zs_scores = np.array(zs_scores)
    print(f"  Zero-shot 完成 ({zs_time:.1f}s)")
    print(f"  p_yes: mean={zs_scores.mean():.4f}, range=[{zs_scores.min():.4f}, {zs_scores.max():.4f}]")

    # Step 3: LoRA 评分
    print("\n[Step 3] LoRA 评分...")
    # 尝试加载已有分数
    existing_lora = load_existing_lora_scores(config)
    if existing_lora is not None and len(existing_lora) == len(trajectories):
        print(f"  使用已有 LoRA 分数 ({len(existing_lora)} 条)")
        lora_scores = np.array(existing_lora)
        # 同时也跑一遍 fresh LoRA 评分做 double-check
        print("  同时运行 fresh LoRA 评分进行验证...")
        t0 = time.time()
        fresh_lora = run_lora_scoring(trajectories, config)
        lora_time = time.time() - t0
        fresh_lora = np.array(fresh_lora)
        print(f"  Fresh LoRA 完成 ({lora_time:.1f}s)")
        # 对比
        diff = np.abs(lora_scores - fresh_lora)
        print(f"  已有 vs Fresh 差异: mean={diff.mean():.6f}, max={diff.max():.6f}")
        if diff.max() > 0.01:
            print("  ⚠️ 差异较大，使用 fresh 评分")
            lora_scores = fresh_lora
        else:
            print("  ✅ 差异可忽略，使用已有评分")
    else:
        print("  未找到已有 LoRA 分数，运行 fresh 评分...")
        t0 = time.time()
        lora_scores = np.array(run_lora_scoring(trajectories, config))
        lora_time = time.time() - t0
        print(f"  LoRA 完成 ({lora_time:.1f}s)")

    print(f"  p_yes: mean={lora_scores.mean():.4f}, range=[{lora_scores.min():.4f}, {lora_scores.max():.4f}]")

    # Step 4: 计算指标
    print("\n[Step 4] 计算评估指标...")
    zs_metrics = compute_metrics(labels, zs_scores, "Zero-shot")
    lora_metrics = compute_metrics(labels, lora_scores, "LoRA")
    print(f"  Zero-shot: AUC={zs_metrics.get('roc_auc', 0):.4f}, PR-AUC={zs_metrics.get('pr_auc', 0):.4f}")
    print(f"  LoRA:      AUC={lora_metrics.get('roc_auc', 0):.4f}, PR-AUC={lora_metrics.get('pr_auc', 0):.4f}")

    # Step 5: 保存 raw scores JSON
    print("\n[Step 5] 保存分数数据...")
    scores_data = {
        "config": {
            "base_model": config.base_model,
            "lora_adapter": config.lora_adapter,
            "num_frames": config.num_frames,
            "instruction": config.instruction,
            "device": config.device,
        },
        "trajectories": [
            {
                "global_idx": t["global_idx"],
                "file": t["file"],
                "traj_key": t["traj_key"],
                "env_success_at_end": t["env_success_at_end"],
                "zs_p_yes": float(zs_scores[i]),
                "lora_p_yes": float(lora_scores[i]),
            }
            for i, t in enumerate(trajectories)
        ],
        "zero_shot_metrics": {k: v for k, v in zs_metrics.items()
                              if k not in ("roc_fpr", "roc_tpr", "roc_thresholds",
                                           "pr_precision", "pr_recall", "pr_thresholds")},
        "lora_metrics": {k: v for k, v in lora_metrics.items()
                         if k not in ("roc_fpr", "roc_tpr", "roc_thresholds",
                                      "pr_precision", "pr_recall", "pr_thresholds")},
    }
    with open(config.output_json, "w") as f:
        json.dump(scores_data, f, indent=2)
    print(f"  保存到 {config.output_json}")

    # Step 6: 生成图表
    print("\n[Step 6] 生成可视化图表...")
    # 释放 RGB 数据以节省内存
    for t in trajectories:
        del t["rgb_frames"]
    plot_paths = generate_plots(labels, zs_scores, lora_scores, zs_metrics, lora_metrics, config.output_plots_dir)

    # Step 7: 生成报告
    print("\n[Step 7] 生成 Markdown 报告...")
    report = generate_report(zs_metrics, lora_metrics, plot_paths, config)
    with open(config.output_report, "w") as f:
        f.write(report)
    print(f"  保存到 {config.output_report}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    print(f"ROC-AUC:  Zero-shot={zs_metrics.get('roc_auc', 0):.4f}  LoRA={lora_metrics.get('roc_auc', 0):.4f}")
    print(f"PR-AUC:   Zero-shot={zs_metrics.get('pr_auc', 0):.4f}  LoRA={lora_metrics.get('pr_auc', 0):.4f}")
    print(f"Best F1:  Zero-shot={zs_metrics.get('f1_max', 0):.4f}  LoRA={lora_metrics.get('f1_max', 0):.4f}")
    print(f"Youden:   Zero-shot={zs_metrics.get('youden_threshold', 0):.4f}  LoRA={lora_metrics.get('youden_threshold', 0):.4f}")
    print(f"F1-thresh: Zero-shot={zs_metrics.get('f1_threshold', 0):.4f}  LoRA={lora_metrics.get('f1_threshold', 0):.4f}")
    print(f"\n报告: {config.output_report}")
    print(f"分数: {config.output_json}")
    print(f"图表: {config.output_plots_dir}/")


if __name__ == "__main__":
    main()
