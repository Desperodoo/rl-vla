#!/usr/bin/env python3
"""Diagnostic: Compare zero-shot, working LoRA, and broken LoRA on one sample.

Tests both multi-image and video format to isolate the issue.
"""
import sys
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))


def load_one_sample(task="LiftPegUpright-v1", success=True, num_frames=16):
    """Load one trajectory from rollout data."""
    dirs = [
        ROOT / f"data/vlaw/rollouts/iter1/{task}",
        ROOT / f"data/vlaw/rollouts/iter1_highsuc/{task}",
    ]
    for d in dirs:
        if not d.is_dir():
            continue
        for fp in sorted(d.glob("*.h5")):
            with h5py.File(str(fp), "r") as f:
                for k in sorted(f.keys()):
                    if not k.startswith("traj"):
                        continue
                    grp = f[k]
                    s = bool(grp["env_success"][-1])
                    if s != success:
                        continue
                    rgb = grp["rgb_base"][:]
                    T = rgb.shape[0]
                    n = min(num_frames, T)
                    idxs = np.linspace(0, T - 1, n, dtype=int)
                    frames = [Image.fromarray(rgb[i]) for i in idxs]
                    return frames, s, f"{fp.name}:{k}"
    raise ValueError(f"No {'success' if success else 'fail'} trajectory found")


def test_model(model, processor, frames, instruction, yes_ids, no_ids, mode="multi-image"):
    """Test a model on frames in specified mode. Returns p_yes (2-logit) and p_yes (multi-variant)."""
    from qwen_vl_utils import process_vision_info

    n = len(frames)
    text = (
        f"These {n} frames show a robot manipulation trajectory. "
        f"Task: '{instruction}'. "
        "Has the robot successfully completed the task? "
        "Answer only 'yes' or 'no'."
    )

    if mode == "video":
        content = [
            {"type": "video", "video": frames, "fps": 2.0},
            {"type": "text", "text": text},
        ]
    else:
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": text})

    msgs = [{"role": "user", "content": content}]

    try:
        text_input = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text_input = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model(**inputs)
        logits = out.logits[0, -1, :].float()

    # Method 1: 2-logit (as in training eval)
    p_yes_2 = float(torch.softmax(
        torch.stack([logits[yes_ids[0]], logits[no_ids[0]]]), 0
    )[0].cpu())

    # Method 2: multi-variant (as in eval_vlm_ablation.py via reward_model.py)
    yes_logits = logits[yes_ids]
    no_logits = logits[no_ids]
    all_logits = torch.cat([yes_logits, no_logits])
    all_probs = torch.softmax(all_logits, dim=0)
    p_yes_mv = float(all_probs[:len(yes_ids)].sum().cpu())

    # Raw logit values for diagnosis
    yes_raw = [float(logits[i].cpu()) for i in yes_ids]
    no_raw = [float(logits[i].cpu()) for i in no_ids]

    return {
        "p_yes_2logit": p_yes_2,
        "p_yes_multivar": p_yes_mv,
        "yes_raw_logits": yes_raw,
        "no_raw_logits": no_raw,
    }


def main():
    import transformers

    model_path = str(ROOT / "checkpoints/vlaw/reward_model/qwen_vl")
    instruction = "Lift the peg and insert it upright into the holder."

    lora_configs = [
        ("WORKING (r16_200step)", str(ROOT / "checkpoints/vlaw/reward_model/lora_iter1_16frame/final")),
        ("BROKEN (100steps)", str(ROOT / "checkpoints/vlaw/reward_model/ablation_100steps/final")),
        ("BROKEN (800steps)", str(ROOT / "checkpoints/vlaw/reward_model/ablation_800steps/step_200")),
        ("BROKEN (r32)", str(ROOT / "checkpoints/vlaw/reward_model/ablation_lora_r32/final")),
    ]

    print("Loading test samples...")
    frames_succ, _, src_succ = load_one_sample(success=True)
    frames_fail, _, src_fail = load_one_sample(success=False)
    print(f"  Success: {src_succ} ({len(frames_succ)} frames)")
    print(f"  Failure: {src_fail} ({len(frames_fail)} frames)")

    print(f"\nLoading base model: {model_path}")
    dtype = torch.bfloat16

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map="cuda:0",
        attn_implementation="eager",
    )
    processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    tok = processor.tokenizer
    yes_ids = list({tok.encode(w, add_special_tokens=False)[0] for w in ["yes", "Yes", "YES", " yes"]})
    no_ids = list({tok.encode(w, add_special_tokens=False)[0] for w in ["no", "No", "NO", " no"]})
    print(f"  yes_ids={yes_ids}, no_ids={no_ids}")

    # 1. Zero-shot test
    print("\n" + "="*60)
    print("  ZERO-SHOT (no LoRA)")
    print("="*60)
    model.eval()
    for mode in ["multi-image", "video"]:
        for label, frames in [("SUCCESS", frames_succ), ("FAILURE", frames_fail)]:
            r = test_model(model, processor, frames, instruction, yes_ids, no_ids, mode)
            print(f"  [{mode:12s}] {label}: p_yes_2={r['p_yes_2logit']:.6f} p_yes_mv={r['p_yes_multivar']:.6f}")

    # 2. Test each LoRA
    for name, lora_path in lora_configs:
        if not os.path.exists(lora_path):
            print(f"\n  SKIP {name}: path not found ({lora_path})")
            continue

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  LoRA: {lora_path}")
        print("="*60)

        # Reload base model (to avoid stacking LoRA)
        del model
        torch.cuda.empty_cache()

        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, device_map="cuda:0",
            attn_implementation="eager",
        )

        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        model.eval()

        for mode in ["multi-image", "video"]:
            for label, frames in [("SUCCESS", frames_succ), ("FAILURE", frames_fail)]:
                r = test_model(model, processor, frames, instruction, yes_ids, no_ids, mode)
                print(f"  [{mode:12s}] {label}: p_yes_2={r['p_yes_2logit']:.6f} p_yes_mv={r['p_yes_multivar']:.6f}")
                if "yes_raw" in r:
                    print(f"                         yes_raw={[f'{v:.2f}' for v in r['yes_raw_logits']]}")
                    print(f"                         no_raw={[f'{v:.2f}' for v in r['no_raw_logits']]}")


if __name__ == "__main__":
    main()
