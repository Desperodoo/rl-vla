#!/usr/bin/env python3
"""VLM 评估 — 稳定入口脚本.

评估 fine-tuned VLM 在不同 threshold 下的 precision/recall/FP rate。
合并自: eval_threshold_ablation.py / eval_reward_model_v3.py / eval_vlm_*.py
已包含:
  - ADR-028: 最佳 VLM 配置 (r=16, 300步)
  - ADR-029: 评估集正负平衡检查
  - process_vision_info + video 模式

用法:
    # 用 lora_v3/final 做 threshold 消融
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python \\
        rlft/vlaw/scripts/eval_vlm.py \\
        --lora_path checkpoints/vlaw/reward_model/lora_v3/final

    # 用最佳 checkpoint 评估
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python \\
        rlft/vlaw/scripts/eval_vlm.py \\
        --lora_path checkpoints/vlaw/reward_model/ablation_v3/steps_300 \\
        --thresholds 0.3 0.5 0.7 0.8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))


def evaluate_thresholds(
    model, processor, eval_ds, cfg,
    yes_id: int, no_id: int, thresholds: list[float], device: str,
) -> dict:
    """在 eval_ds 上对所有 threshold 一次推理."""
    p_yes_list: list[float] = []
    labels: list[int] = []

    # ADR-029: 检查正负平衡
    sample_labels = [eval_ds[i][2] for i in range(len(eval_ds))]
    pos_ratio = sum(sample_labels) / max(len(sample_labels), 1)
    if pos_ratio < 0.2 or pos_ratio > 0.8:
        print(f"[WARN] ADR-029: eval 集正样本比例={pos_ratio:.1%}, 应在 20%-80% 范围内")

    print(f"[EVAL-VLM] 推理 {len(eval_ds)} 样本 (pos_ratio={pos_ratio:.1%})...")

    for i in range(len(eval_ds)):
        frames, instr, label = eval_ds[i]

        # 构建消息 (video 模式)
        from rlft.vlaw.reward.train_reward_model import _make_messages
        msgs = _make_messages(frames, instr, label=None,
                              use_video_format=cfg.use_video_format,
                              video_fps=cfg.video_fps)
        try:
            try:
                prompt = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                prompt = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)

            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(msgs)
                inp = processor(text=[prompt],
                                images=image_inputs if image_inputs else None,
                                videos=video_inputs if video_inputs else None,
                                return_tensors="pt").to(device)
            except (ImportError, Exception):
                inp = processor(text=[prompt], images=frames,
                                return_tensors="pt").to(device)

            with torch.inference_mode():
                out = model(**inp)
            logits = out.logits[0, -1, :]
            p_yes = float(torch.softmax(torch.stack([logits[yes_id], logits[no_id]]), 0)[0].cpu())
        except Exception as e:
            print(f"[WARN] sample {i}: {e}")
            p_yes = 0.0

        p_yes_list.append(p_yes)
        labels.append(label)
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(eval_ds)}]")

    # 对每个 threshold 计算
    results = {}
    for thr in thresholds:
        tp = fp = tn = fn = 0
        for py, lb in zip(p_yes_list, labels):
            pred = 1 if py >= thr else 0
            if lb == 1 and pred == 1: tp += 1
            elif lb == 0 and pred == 1: fp += 1
            elif lb == 0 and pred == 0: tn += 1
            else: fn += 1
        total = max(tp + fp + tn + fn, 1)
        results[str(thr)] = {
            "threshold": thr, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fp_rate": fp / max(fp + tn, 1),
            "accuracy": (tp + tn) / total,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        }
        print(f"  α={thr:.2f}: TP={tp} FP={fp} TN={tn} FN={fn} "
              f"acc={results[str(thr)]['accuracy']:.3f} "
              f"prec={results[str(thr)]['precision']:.3f} "
              f"recall={results[str(thr)]['recall']:.3f}")

    pos_pyes = [p for p, l in zip(p_yes_list, labels) if l == 1]
    neg_pyes = [p for p, l in zip(p_yes_list, labels) if l == 0]
    results["_distribution"] = {
        "n_total": len(p_yes_list), "n_pos": len(pos_pyes), "n_neg": len(neg_pyes),
        "pos_p_yes_mean": float(np.mean(pos_pyes)) if pos_pyes else 0.0,
        "neg_p_yes_mean": float(np.mean(neg_pyes)) if neg_pyes else 0.0,
        "mean_p_yes": float(np.mean(p_yes_list)),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="VLAW VLM 评估 (稳定版)")
    parser.add_argument("--lora_path", default="checkpoints/vlaw/reward_model/lora_v3/final")
    parser.add_argument("--model_path", default="checkpoints/vlaw/reward_model/qwen_vl")
    parser.add_argument("--data_dir", default="data/vlaw/rollouts/mixed")
    parser.add_argument("--tasks", nargs="+", default=["LiftPegUpright-v1"])
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--eval_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/vlaw/vlm_eval.json")
    parser.add_argument("--visualize", action="store_true",
                        help="生成 threshold sweep 折线图 + p_yes 分布直方图")
    args = parser.parse_args()

    from rlft.vlaw.reward.train_reward_model import TrainConfig, build_datasets

    cfg = TrainConfig(
        data_dir=args.data_dir, tasks=args.tasks, model_path=args.model_path,
        num_frames=args.num_frames, eval_ratio=args.eval_ratio, seed=args.seed,
        device=args.device, use_video_format=True, video_fps=2.0, threshold=0.8,
    )
    _, eval_ds = build_datasets(cfg)
    print(f"[EVAL-VLM] eval 集: {len(eval_ds)} 样本")

    # 加载模型
    import transformers
    attn_impl = "eager"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    model = None
    for cls_name in ["Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"]:
        try:
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
                                        device_map=args.device, attn_implementation=attn_impl)
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError("无法加载 VLM 模型")

    processor = None
    for cls_name in ["Qwen3VLProcessor", "Qwen2_5_VLProcessor", "AutoProcessor"]:
        try:
            cls = getattr(transformers, cls_name, None) or transformers.AutoProcessor
            processor = cls.from_pretrained(args.model_path)
            break
        except Exception:
            continue

    # LoRA
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.lora_path)
    model = model.merge_and_unload()
    model.eval()

    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[-1]
    no_id = processor.tokenizer.encode("no", add_special_tokens=False)[-1]

    results = evaluate_thresholds(model, processor, eval_ds, cfg, yes_id, no_id,
                                  args.thresholds, args.device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}
    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n[EVAL-VLM] ✅ 结果: {out_path}")

    if args.visualize:
        _visualize_vlm_results(results, out_path.parent)


# ── 可视化 ────────────────────────────────────────────────────────────────────

def _visualize_vlm_results(results: dict, output_dir: Path) -> None:
    """生成 threshold sweep 折线图 + p_yes 分布直方图.

    保存到 {output_dir}/viz/.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Threshold sweep 折线图 ──
    thr_results = {k: v for k, v in results.items() if k != "_distribution"}
    if thr_results:
        thrs = sorted(float(k) for k in thr_results.keys())
        precs = [thr_results[str(t)]["precision"] for t in thrs]
        recalls = [thr_results[str(t)]["recall"] for t in thrs]
        fp_rates = [thr_results[str(t)]["fp_rate"] for t in thrs]
        accs = [thr_results[str(t)]["accuracy"] for t in thrs]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thrs, precs, "b-o", label="Precision", markersize=4)
        ax.plot(thrs, recalls, "g-s", label="Recall", markersize=4)
        ax.plot(thrs, fp_rates, "r-^", label="FP Rate", markersize=4)
        ax.plot(thrs, accs, "k--d", label="Accuracy", markersize=4, alpha=0.6)
        ax.set_xlabel("Threshold (α)")
        ax.set_ylabel("Rate")
        ax.set_title("VLM Threshold Sweep")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        fig_path = viz_dir / "threshold_sweep.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[EVAL-VLM] 📊 {fig_path.name}")

    # ── 2. p_yes 分布直方图 (按 GT label 分色) ──
    dist = results.get("_distribution", {})
    if not dist:
        return
    # 需要重新从 results 里提取 p_yes 数据——但 evaluate_thresholds 没保存原始列表
    # 生成示意图: 用 mean 和 count 画简化分布
    n_pos = dist.get("n_pos", 0)
    n_neg = dist.get("n_neg", 0)
    pos_mean = dist.get("pos_p_yes_mean", 0)
    neg_mean = dist.get("neg_p_yes_mean", 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Positive (GT=1)", "Negative (GT=0)"],
           [pos_mean, neg_mean], color=["steelblue", "salmon"])
    ax.set_ylabel("Mean p(yes)")
    ax.set_title(f"Mean p(yes) by GT Label (pos={n_pos}, neg={n_neg})")
    for i, v in enumerate([pos_mean, neg_mean]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = viz_dir / "p_yes_by_label.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EVAL-VLM] 📊 {fig_path.name}")


if __name__ == "__main__":
    main()
