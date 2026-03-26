#!/usr/bin/env python3
"""VLM zero-shot vs LoRA 基线评估脚本 (VLAW P3.1)

从 rollout HDF5 中提取轨迹, 用 VLAWRewardModel 评分, 计算 ROC-AUC 和混淆矩阵.
"""
import json
import os
import sys
import glob
from pathlib import Path

import h5py
import numpy as np

# 添加项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_trajectories(rollout_dir: str) -> list[dict]:
    """从 rollout HDF5 加载所有轨迹: 提取最后一帧 + success 标签."""
    trajs = []
    for fp in sorted(glob.glob(os.path.join(rollout_dir, "*.h5"))):
        with h5py.File(fp, "r") as f:
            for k in sorted(f.keys()):
                if not k.startswith("traj"):
                    continue
                grp = f[k]
                rgb = grp["rgb_base"][-1]  # 最后一帧: (H, W, 3) uint8
                success = bool(grp["env_success"][-1])
                trajs.append({
                    "frame": rgb,
                    "success": success,
                    "source": f"{os.path.basename(fp)}:{k}",
                })
    return trajs


def evaluate_model(model, trajs: list[dict], instruction: str) -> list[float]:
    """对所有轨迹评分, 返回 p_yes 列表."""
    p_yes_list = []
    for i, t in enumerate(trajs):
        # 传入单帧 numpy (1, H, W, 3)
        frame = t["frame"][np.newaxis]  # (1, H, W, 3)
        result = model.score_trajectory(frame, instruction)
        p_yes_list.append(result["p_yes"])
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(trajs)}] p_yes={result['p_yes']:.4f} "
                  f"gt={t['success']} src={t['source']}")
    return p_yes_list


def compute_metrics(y_true: np.ndarray, p_yes: np.ndarray, threshold: float = 0.8):
    """计算 ROC-AUC, 最优阈值, confusion matrix."""
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

    auc = roc_auc_score(y_true, p_yes)

    # 最优阈值 (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, p_yes)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])

    # 论文阈值 α=0.8 的混淆矩阵
    preds_alpha = (p_yes > threshold).astype(int)
    cm_alpha = confusion_matrix(y_true, preds_alpha, labels=[0, 1])

    # 最优阈值的混淆矩阵
    preds_best = (p_yes > best_thresh).astype(int)
    cm_best = confusion_matrix(y_true, preds_best, labels=[0, 1])

    return {
        "auc": auc,
        "best_threshold": best_thresh,
        "best_j": float(j_scores[best_idx]),
        "cm_alpha": cm_alpha.tolist(),  # [[TN, FP], [FN, TP]]
        "cm_best": cm_best.tolist(),
        "p_yes_mean_success": float(p_yes[y_true == 1].mean()),
        "p_yes_mean_fail": float(p_yes[y_true == 0].mean()),
        "p_yes_std_success": float(p_yes[y_true == 1].std()),
        "p_yes_std_fail": float(p_yes[y_true == 0].std()),
    }


def format_cm(cm, labels=("Fail", "Success")) -> str:
    """格式化混淆矩阵为 Markdown 表格."""
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    lines = [
        f"|  | Pred Fail | Pred Success |",
        f"|--|-----------|--------------|",
        f"| GT Fail    | {tn} (TN) | {fp} (FP) |",
        f"| GT Success | {fn} (FN) | {tp} (TP) |",
        f"",
        f"Accuracy: {acc:.1%}, FP Rate: {fp_rate:.1%}",
    ]
    return "\n".join(lines)


def main():
    from rlft.vlaw.reward.reward_model import VLAWRewardModel, VLAWRewardConfig

    rollout_dir = str(ROOT / "data/vlaw/rollouts/iter1_lift_only/LiftPegUpright-v1")
    base_model_path = str(ROOT / "checkpoints/vlaw/reward_model/qwen_vl")
    lora_path = str(ROOT / "checkpoints/vlaw/reward_model/lora_iter1/final")
    out_dir = ROOT / "results/vlaw"
    out_dir.mkdir(parents=True, exist_ok=True)
    instruction = "Pick up the peg and lift it upright."

    # ── 加载轨迹 ──
    print("=== Loading trajectories ===")
    trajs = load_trajectories(rollout_dir)
    y_true = np.array([int(t["success"]) for t in trajs])
    n_success = y_true.sum()
    n_fail = len(y_true) - n_success
    print(f"Loaded {len(trajs)} trajs: {n_success} success, {n_fail} fail")

    results = {}
    report_sections = []

    # ── Zero-shot 评估 ──
    print("\n=== Zero-shot evaluation ===")
    cfg = VLAWRewardConfig(model_path=base_model_path, device="cuda:0")
    model = VLAWRewardModel(cfg)
    model.load_model()
    p_yes_zs = np.array(evaluate_model(model, trajs, instruction))
    metrics_zs = compute_metrics(y_true, p_yes_zs)
    results["zero_shot"] = {**metrics_zs, "p_yes_all": p_yes_zs.tolist()}
    print(f"\n  AUC={metrics_zs['auc']:.4f}, Best θ={metrics_zs['best_threshold']:.4f}")
    print(f"  p_yes (success): {metrics_zs['p_yes_mean_success']:.4f} ± {metrics_zs['p_yes_std_success']:.4f}")
    print(f"  p_yes (fail):    {metrics_zs['p_yes_mean_fail']:.4f} ± {metrics_zs['p_yes_std_fail']:.4f}")
    model.unload_model()

    report_sections.append(f"""### Zero-shot (Qwen3-VL-4B)
- ROC-AUC: **{metrics_zs['auc']:.4f}**
- p_yes (success): {metrics_zs['p_yes_mean_success']:.4f} ± {metrics_zs['p_yes_std_success']:.4f}
- p_yes (fail): {metrics_zs['p_yes_mean_fail']:.4f} ± {metrics_zs['p_yes_std_fail']:.4f}
- Best threshold (Youden's J): {metrics_zs['best_threshold']:.4f}

**Confusion Matrix @ α=0.8:**
{format_cm(metrics_zs['cm_alpha'])}

**Confusion Matrix @ Best threshold={metrics_zs['best_threshold']:.4f}:**
{format_cm(metrics_zs['cm_best'])}
""")

    # ── LoRA 评估 (如果可用) ──
    has_lora = os.path.isfile(os.path.join(lora_path, "adapter_config.json"))
    if has_lora:
        print("\n=== LoRA evaluation ===")
        model2 = VLAWRewardModel(cfg)
        model2.load_model(lora_path=lora_path)
        p_yes_lora = np.array(evaluate_model(model2, trajs, instruction))
        metrics_lora = compute_metrics(y_true, p_yes_lora)
        results["lora"] = {**metrics_lora, "p_yes_all": p_yes_lora.tolist()}
        print(f"\n  AUC={metrics_lora['auc']:.4f}, Best θ={metrics_lora['best_threshold']:.4f}")
        model2.unload_model()

        report_sections.append(f"""### LoRA fine-tuned
- ROC-AUC: **{metrics_lora['auc']:.4f}**
- p_yes (success): {metrics_lora['p_yes_mean_success']:.4f} ± {metrics_lora['p_yes_std_success']:.4f}
- p_yes (fail): {metrics_lora['p_yes_mean_fail']:.4f} ± {metrics_lora['p_yes_std_fail']:.4f}
- Best threshold: {metrics_lora['best_threshold']:.4f}

**Confusion Matrix @ α=0.8:**
{format_cm(metrics_lora['cm_alpha'])}

**Confusion Matrix @ Best threshold={metrics_lora['best_threshold']:.4f}:**
{format_cm(metrics_lora['cm_best'])}
""")
    else:
        report_sections.append("### LoRA fine-tuned\n⚠️ LoRA checkpoint not found. Skipped.\n")
        print("\n⚠️ No LoRA adapter found, skipping LoRA evaluation.")

    # ── 写报告 ──
    report = f"""# VLM Baseline Report — LiftPegUpright-v1

## 数据统计
- 轨迹总数: {len(trajs)} (success={n_success}, fail={n_fail})
- 数据源: `data/vlaw/rollouts/iter1_lift_only/LiftPegUpright-v1/`
- 评估方式: 最后一帧 (single frame)
- Instruction: "{instruction}"

## 结果

{"".join(report_sections)}

## 结论
{'Zero-shot p_yes 极低 (< 0.15)，α=0.8 阈值无法使用。需 LoRA fine-tune 后才能有效标注 D_syn。' if metrics_zs['p_yes_mean_success'] < 0.3 else 'Zero-shot VLM 有一定区分能力。'}
"""
    report_path = out_dir / "vlm_baseline_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n✅ Report saved to {report_path}")

    # 也保存 JSON 方便后续使用
    json_path = out_dir / "vlm_baseline_metrics.json"
    # 去掉 numpy 不可序列化的, cm 已是 list
    json_results = {}
    for mode, m in results.items():
        json_results[mode] = {k: v for k, v in m.items()}
    with open(json_path, "w") as jf:
        json.dump(json_results, jf, indent=2)
    print(f"✅ Metrics saved to {json_path}")


if __name__ == "__main__":
    main()
