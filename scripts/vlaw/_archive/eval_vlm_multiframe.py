#!/usr/bin/env python3
"""VLM 16帧视频序列评估脚本 — 对比单帧(旧基线) vs 16帧视频(论文设定)

目的: 验证从单帧改为16帧等间距采样后, ROC-AUC 是否提升 (旧基线 AUC≈0.59)

用法:
    CUDA_VISIBLE_DEVICES=6 conda run -n rlft_ms3 python scripts/eval_vlm_multiframe.py

输出:
    results/vlaw/vlm_multiframe_eval.json    — 详细指标
    results/vlaw/vlm_multiframe_report.md    — Markdown 报告
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TASK_INSTRUCTIONS: dict[str, str] = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
    "PickCube-v1": "Pick up the cube and place it on the target location.",
    "StackCube-v1": "Stack the red cube on top of the green cube.",
}


# ── 数据加载 ──────────────────────────────────────────────────────────────────


def load_trajectories(
    rollout_dirs: list[str],
    max_frames: int = 16,
) -> list[dict]:
    """从 rollout HDF5 加载轨迹, 包含16帧采样 + 最后1帧 + ground truth."""
    trajs: list[dict] = []
    for rollout_dir in rollout_dirs:
        if not os.path.isdir(rollout_dir):
            print(f"  [SKIP] 目录不存在: {rollout_dir}")
            continue
        for fp in sorted(glob.glob(os.path.join(rollout_dir, "*.h5"))):
            with h5py.File(fp, "r") as f:
                for k in sorted(f.keys()):
                    if not k.startswith("traj"):
                        continue
                    grp = f[k]
                    rgb = grp["rgb_base"][:]  # (T, H, W, 3) uint8
                    success = bool(grp["env_success"][-1])
                    T = rgb.shape[0]

                    # 16帧等间距采样
                    n = min(max_frames, T)
                    uniform_idxs = np.linspace(0, T - 1, n, dtype=int)
                    multi_frames = rgb[uniform_idxs]  # (n, H, W, 3)

                    # 最后1帧 (用于单帧基线对比)
                    last_frame = rgb[-1:]  # (1, H, W, 3)

                    trajs.append({
                        "multi_frames": multi_frames,
                        "last_frame": last_frame,
                        "n_total": T,
                        "n_sampled": n,
                        "success": success,
                        "source": f"{os.path.basename(fp)}:{k}",
                    })
    return trajs


# ── 评估函数 ──────────────────────────────────────────────────────────────────


def eval_with_model(
    model,
    trajs: list[dict],
    instruction: str,
    mode: str = "multi",
) -> list[float]:
    """对所有轨迹评分, 返回 p_yes 列表.

    Args:
        mode: "multi" = 16帧视频; "single" = 仅最后1帧
    """
    p_yes_list: list[float] = []
    t0 = time.time()
    for i, t in enumerate(trajs):
        frames = t["multi_frames"] if mode == "multi" else t["last_frame"]
        result = model.score_trajectory(frames, instruction)
        p_yes_list.append(result["p_yes"])
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{len(trajs)}] p_yes={result['p_yes']:.4f} "
                f"gt={t['success']} nf={result['num_frames']} "
                f"({elapsed:.0f}s)"
            )
    elapsed = time.time() - t0
    print(f"  完成 ({elapsed:.1f}s, {elapsed / len(trajs):.2f}s/traj)")
    return p_yes_list


# ── 指标计算 ──────────────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray,
    p_yes: np.ndarray,
    threshold: float = 0.8,
) -> dict:
    """计算 ROC-AUC, 最优阈值, confusion matrix."""
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

    auc = roc_auc_score(y_true, p_yes)

    fpr, tpr, thresholds = roc_curve(y_true, p_yes)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])

    preds_alpha = (p_yes > threshold).astype(int)
    cm_alpha = confusion_matrix(y_true, preds_alpha, labels=[0, 1])

    preds_best = (p_yes > best_thresh).astype(int)
    cm_best = confusion_matrix(y_true, preds_best, labels=[0, 1])

    # 找 FP < 20% 的最优阈值
    valid = fpr <= 0.20
    if valid.any():
        valid_recalls = tpr[valid]
        best_c_idx = int(np.where(valid)[0][np.argmax(valid_recalls)])
        constrained_thresh = float(thresholds[best_c_idx])
        constrained_recall = float(tpr[best_c_idx])
        constrained_fp = float(fpr[best_c_idx])
    else:
        constrained_thresh = 1.0
        constrained_recall = 0.0
        constrained_fp = 0.0

    return {
        "auc": float(auc),
        "youden_threshold": best_thresh,
        "youden_j": float(j_scores[best_idx]),
        "youden_recall": float(tpr[best_idx]),
        "youden_fp_rate": float(fpr[best_idx]),
        "constrained_threshold": constrained_thresh,
        "constrained_recall": constrained_recall,
        "constrained_fp_rate": constrained_fp,
        "cm_alpha": cm_alpha.tolist(),
        "cm_best": cm_best.tolist(),
        "p_yes_mean_success": float(p_yes[y_true == 1].mean()),
        "p_yes_mean_fail": float(p_yes[y_true == 0].mean()),
        "p_yes_std_success": float(p_yes[y_true == 1].std()),
        "p_yes_std_fail": float(p_yes[y_true == 0].std()),
        "p_yes_min": float(p_yes.min()),
        "p_yes_max": float(p_yes.max()),
    }


def format_cm(cm: list) -> str:
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    return (
        f"|  | Pred Fail | Pred Succ |\n"
        f"|--|-----------|----------|\n"
        f"| GT Fail | {tn} (TN) | {fp} (FP) |\n"
        f"| GT Succ | {fn} (FN) | {tp} (TP) |\n\n"
        f"Acc={acc:.1%}, FP rate={fp_rate:.1%}"
    )


# ── 主函数 ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM 16帧视频 vs 单帧评估"
    )
    parser.add_argument("--task", default="LiftPegUpright-v1")
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument(
        "--model_path",
        default=str(ROOT / "checkpoints/vlaw/reward_model/qwen_vl"),
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    rollout_dirs = [
        str(ROOT / f"data/vlaw/rollouts/iter1/{args.task}"),
        str(ROOT / f"data/vlaw/rollouts/iter1_highsuc/{args.task}"),
    ]

    out_dir = ROOT / "results" / "vlaw"
    out_dir.mkdir(parents=True, exist_ok=True)
    instruction = TASK_INSTRUCTIONS.get(args.task, "Complete the task.")

    # ── 加载轨迹 ──
    print("=" * 60)
    print("=== 加载轨迹 ===")
    print("=" * 60)
    trajs = load_trajectories(rollout_dirs, args.max_frames)
    y_true = np.array([int(t["success"]) for t in trajs])
    n_succ = int(y_true.sum())
    n_fail = len(y_true) - n_succ
    print(f"共 {len(trajs)} 条轨迹: {n_succ} 成功, {n_fail} 失败")
    if len(trajs) == 0:
        print("ERROR: 没有找到轨迹!")
        return

    frame_counts = [t["n_sampled"] for t in trajs]
    print(
        f"帧数统计: min={min(frame_counts)}, max={max(frame_counts)}, "
        f"mean={np.mean(frame_counts):.1f}"
    )

    from rlft.vlaw.reward.reward_model import VLAWRewardConfig, VLAWRewardModel

    all_results: dict[str, dict] = {}
    report_parts: list[str] = []

    # ── 评估配置: (标签, 帧模式, use_video_format) ──
    configs = [
        ("single_frame_images", "single", False),    # 1帧 (旧基线)
        ("16frame_images", "multi", False),           # 16帧多图模式
        ("16frame_video", "multi", True),             # 16帧视频模式 (论文)
    ]

    for label, frame_mode, use_video in configs:
        n_desc = (
            "1帧 (最后一帧)" if frame_mode == "single"
            else f"≤{args.max_frames}帧 ({'video' if use_video else 'images'})"
        )
        print(f"\n{'=' * 60}")
        print(f"=== {label}: {n_desc} ===")
        print(f"{'=' * 60}")

        cfg = VLAWRewardConfig(
            model_path=args.model_path,
            threshold=args.threshold,
            device="cuda:0",
            num_frames=args.max_frames,
            use_video_format=use_video,
            video_fps=2.0,
        )
        model = VLAWRewardModel(cfg)
        model.load_model()

        p_yes = np.array(eval_with_model(model, trajs, instruction, mode=frame_mode))
        metrics = compute_metrics(y_true, p_yes, args.threshold)

        all_results[label] = {**metrics, "p_yes_all": p_yes.tolist()}

        print(f"\n  AUC = {metrics['auc']:.4f}")
        print(
            f"  p_yes (success): {metrics['p_yes_mean_success']:.4f} "
            f"± {metrics['p_yes_std_success']:.4f}"
        )
        print(
            f"  p_yes (fail):    {metrics['p_yes_mean_fail']:.4f} "
            f"± {metrics['p_yes_std_fail']:.4f}"
        )
        print(f"  p_yes range: [{metrics['p_yes_min']:.4f}, {metrics['p_yes_max']:.4f}]")
        print(
            f"  Youden阈值: {metrics['youden_threshold']:.4f} "
            f"(J={metrics['youden_j']:.4f}, recall={metrics['youden_recall']:.1%}, "
            f"FP={metrics['youden_fp_rate']:.1%})"
        )
        print(
            f"  FP<20%阈值: {metrics['constrained_threshold']:.4f} "
            f"(recall={metrics['constrained_recall']:.1%}, "
            f"FP={metrics['constrained_fp_rate']:.1%})"
        )

        report_parts.append(f"""### {label} ({n_desc})
- **ROC-AUC: {metrics['auc']:.4f}**
- p_yes (成功): {metrics['p_yes_mean_success']:.4f} ± {metrics['p_yes_std_success']:.4f}
- p_yes (失败): {metrics['p_yes_mean_fail']:.4f} ± {metrics['p_yes_std_fail']:.4f}
- p_yes 范围: [{metrics['p_yes_min']:.4f}, {metrics['p_yes_max']:.4f}]
- Youden 阈值: {metrics['youden_threshold']:.4f} (recall={metrics['youden_recall']:.1%}, FP={metrics['youden_fp_rate']:.1%})
- FP<20% 阈值: {metrics['constrained_threshold']:.4f} (recall={metrics['constrained_recall']:.1%}, FP={metrics['constrained_fp_rate']:.1%})

**CM @ α={args.threshold}:**
{format_cm(metrics['cm_alpha'])}

**CM @ Youden 阈值={metrics['youden_threshold']:.4f}:**
{format_cm(metrics['cm_best'])}
""")
        model.unload_model()

    # ── 对比摘要 ──
    print(f"\n{'=' * 60}")
    print("=== 对比摘要 ===")
    print(f"{'=' * 60}")
    header = f"{'Config':<25} {'AUC':>8} {'p_yes(+)':>10} {'p_yes(-)':>10} {'Youden_θ':>10} {'Recall@FP20':>12}"
    print(header)
    for key, m in all_results.items():
        print(
            f"{key:<25} {m['auc']:>8.4f} "
            f"{m['p_yes_mean_success']:>10.4f} {m['p_yes_mean_fail']:>10.4f} "
            f"{m['youden_threshold']:>10.4f} "
            f"{m['constrained_recall']:>12.1%}"
        )

    # ── 与旧基线对比 ──
    baseline_auc = 0.5852  # 之前单帧 zero-shot 结果
    best_new = max(
        (m["auc"], k) for k, m in all_results.items() if "16frame" in k
    )
    delta = best_new[0] - baseline_auc
    print(f"\n旧基线 (单帧 zero-shot): AUC = {baseline_auc:.4f}")
    print(f"最佳16帧配置 ({best_new[1]}): AUC = {best_new[0]:.4f}")
    print(f"AUC 提升: {delta:+.4f} ({'改善' if delta > 0 else '退化'})")

    # ── 保存 JSON ──
    json_path = out_dir / "vlm_multiframe_eval.json"
    with open(json_path, "w") as jf:
        json_out = {}
        for mode, m in all_results.items():
            json_out[mode] = {k: v for k, v in m.items()}
        json.dump(json_out, jf, indent=2)
    print(f"\n✅ Metrics → {json_path}")

    # ── 保存 Markdown 报告 ──
    report = f"""# VLM 16帧视频序列评估报告 — {args.task}

## 数据
- 轨迹: {len(trajs)} (成功={n_succ}, 失败={n_fail})
- 来源: {', '.join(rollout_dirs)}
- 指令: "{instruction}"
- 帧数: min={min(frame_counts)}, max={max(frame_counts)}, mean={np.mean(frame_counts):.1f}

## 旧基线
- 单帧 zero-shot AUC = {baseline_auc:.4f}

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

## AUC 提升
- 旧基线: {baseline_auc:.4f}
- 最佳16帧 ({best_new[1]}): {best_new[0]:.4f}
- **ΔAUC = {delta:+.4f}**
"""

    report_path = out_dir / "vlm_multiframe_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"✅ Report → {report_path}")


if __name__ == "__main__":
    main()
