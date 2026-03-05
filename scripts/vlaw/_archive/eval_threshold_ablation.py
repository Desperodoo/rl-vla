"""
实验 2: Threshold 消融 — 用 lora_v3/final 对不同 α 阈值评估

使用已有 lora_v3 checkpoint，在 mixed 数据上以不同 threshold 重新评估,
找到 FP rate 和 recall 的最佳平衡点。

输出: results/vlaw/vlm_threshold_ablation_v3.json
"""
from __future__ import annotations

import json
import os
import sys
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rlft.vlaw.reward.train_reward_model import (
    TrainConfig, build_datasets, _make_messages, get_instruction,
    _load_episodes_from_dir, RewardDataset,
)


def evaluate_thresholds(
    model,
    processor,
    eval_ds: RewardDataset,
    cfg: TrainConfig,
    yes_id: int,
    no_id: int,
    thresholds: list[float],
    device: str = "cuda:0",
) -> dict:
    """在 eval_ds 上对所有 threshold 同时评估，只做一次推理。"""
    p_yes_list: list[float] = []
    labels: list[int] = []

    print(f"[EVAL] 开始推理 {len(eval_ds)} 样本 ...")
    for i in range(len(eval_ds)):
        frames, instr, label = eval_ds[i]
        msgs = _make_messages(frames, instr, label=None,
                              use_video_format=cfg.use_video_format,
                              video_fps=cfg.video_fps)
        try:
            try:
                prompt = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                prompt = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(msgs)
                inp = processor(
                    text=[prompt],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    return_tensors="pt",
                ).to(device)
            except (ImportError, Exception):
                inp = processor(text=[prompt], images=frames,
                                return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model(**inp)
            logits = out.logits[0, -1, :]
            p_yes = float(torch.softmax(
                torch.stack([logits[yes_id], logits[no_id]]), 0
            )[0].cpu())
        except Exception as exc:
            print(f"[WARN] eval 推理失败 sample {i}: {exc}")
            p_yes = 0.0

        p_yes_list.append(p_yes)
        labels.append(label)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(eval_ds)}] 已完成")

    # 对每个 threshold 计算指标
    results = {}
    for thr in thresholds:
        tp = fp = tn = fn = 0
        for p_yes, label in zip(p_yes_list, labels):
            pred = 1 if p_yes >= thr else 0
            if label == 1 and pred == 1:   tp += 1
            elif label == 0 and pred == 1: fp += 1
            elif label == 0 and pred == 0: tn += 1
            else:                          fn += 1
        total = max(tp + fp + tn + fn, 1)
        results[str(thr)] = {
            "threshold": thr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fp_rate":   fp  / max(fp + tn, 1),
            "accuracy":  (tp + tn) / total,
            "precision": tp  / max(tp + fp, 1),
            "recall":    tp  / max(tp + fn, 1),
            "mean_p_yes": float(np.mean(p_yes_list)),
        }
        print(f"  α={thr:.1f}: TP={tp} FP={fp} TN={tn} FN={fn} "
              f"acc={results[str(thr)]['accuracy']:.3f} "
              f"prec={results[str(thr)]['precision']:.3f} "
              f"recall={results[str(thr)]['recall']:.3f} "
              f"fp_rate={results[str(thr)]['fp_rate']:.3f}")

    # 也返回原始 p_yes 分布信息
    pos_pyes = [p for p, l in zip(p_yes_list, labels) if l == 1]
    neg_pyes = [p for p, l in zip(p_yes_list, labels) if l == 0]
    results["_distribution"] = {
        "n_total": len(p_yes_list),
        "n_pos": len(pos_pyes),
        "n_neg": len(neg_pyes),
        "pos_p_yes_mean": float(np.mean(pos_pyes)) if pos_pyes else 0.0,
        "pos_p_yes_median": float(np.median(pos_pyes)) if pos_pyes else 0.0,
        "pos_p_yes_std": float(np.std(pos_pyes)) if pos_pyes else 0.0,
        "neg_p_yes_mean": float(np.mean(neg_pyes)) if neg_pyes else 0.0,
        "neg_p_yes_median": float(np.median(neg_pyes)) if neg_pyes else 0.0,
        "neg_p_yes_std": float(np.std(neg_pyes)) if neg_pyes else 0.0,
        "all_p_yes": p_yes_list,
        "all_labels": labels,
    }
    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lora_path", default="checkpoints/vlaw/reward_model/lora_v3/final")
    p.add_argument("--model_path", default="checkpoints/vlaw/reward_model/qwen_vl")
    p.add_argument("--data_dir", default="data/vlaw/rollouts/mixed")
    p.add_argument("--tasks", nargs="+", default=["LiftPegUpright-v1"])
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--eval_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="results/vlaw/vlm_threshold_ablation_v3.json")
    args = p.parse_args()

    # 使用和 lora_v3 完全相同的数据分割
    cfg = TrainConfig(
        data_dir=args.data_dir,
        tasks=args.tasks,
        model_path=args.model_path,
        num_frames=args.num_frames,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        device=args.device,
        use_video_format=True,
        video_fps=2.0,
        threshold=0.8,  # 不影响评估，只用于 build_datasets
    )

    print("[EVAL] 构建数据集 ...")
    _, eval_ds = build_datasets(cfg)
    print(f"[EVAL] eval 集: {len(eval_ds)} 样本")

    # 加载模型
    print(f"[EVAL] 加载模型: {args.model_path}")
    import transformers
    tdtype = torch.bfloat16

    # flash_attn 检测
    attn_impl = "eager"
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    model = None
    for cls_name in ["Qwen3VLForConditionalGeneration",
                     "Qwen2_5_VLForConditionalGeneration"]:
        try:
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(
                args.model_path, torch_dtype=tdtype,
                device_map=args.device,
                attn_implementation=attn_impl,
            )
            print(f"[EVAL] 模型: {cls_name}")
            break
        except Exception as e:
            print(f"[EVAL] {cls_name} 跳过: {e}")

    if model is None:
        raise RuntimeError("无法加载模型")

    processor = None
    for cls_name in ["Qwen3VLProcessor", "Qwen2_5_VLProcessor", "AutoProcessor"]:
        try:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                from transformers import AutoProcessor as cls
            processor = cls.from_pretrained(args.model_path)
            print(f"[EVAL] 处理器: {cls_name}")
            break
        except Exception as e:
            print(f"[EVAL] {cls_name} 跳过: {e}")

    # 加载 LoRA
    print(f"[EVAL] 加载 LoRA: {args.lora_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.lora_path)
    model = model.merge_and_unload()
    model.eval()

    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[-1]
    no_id  = processor.tokenizer.encode("no",  add_special_tokens=False)[-1]
    print(f"[EVAL] yes_id={yes_id}, no_id={no_id}")

    # 评估所有阈值
    results = evaluate_thresholds(
        model, processor, eval_ds, cfg, yes_id, no_id,
        thresholds=args.thresholds, device=args.device,
    )

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存不含 all_p_yes/all_labels 的版本 (更简洁)
    clean_results = {}
    for k, v in results.items():
        if k == "_distribution":
            clean_results[k] = {kk: vv for kk, vv in v.items()
                                if kk not in ("all_p_yes", "all_labels")}
        else:
            clean_results[k] = v

    with open(out_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\n[EVAL] 结果已保存: {out_path}")

    # 也保存含完整数据的版本
    full_path = out_path.with_suffix(".full.json")
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[EVAL] 完整数据: {full_path}")


if __name__ == "__main__":
    main()
