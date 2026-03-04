#!/usr/bin/env python3
"""Diagnostic: Test if merge_and_unload() is the bug.
Compare PeftModel (no merge) vs merged model for both working and broken LoRA.
"""
import sys
from pathlib import Path
import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))


def load_one_sample(success=True, num_frames=16):
    d = ROOT / "data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1"
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
                return [Image.fromarray(rgb[i]) for i in idxs]
    raise ValueError("Not found")


def eval_single(model, processor, frames, instruction, yes_id, no_id, all_yes_ids, all_no_ids):
    """Eval in multi-image mode (same as training)."""
    n = len(frames)
    text = (
        f"These {n} frames show a robot manipulation trajectory. "
        f"Task: '{instruction}'. "
        "Has the robot successfully completed the task? "
        "Answer only 'yes' or 'no'."
    )
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": text})
    msgs = [{"role": "user", "content": content}]

    try:
        prompt = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    inp = processor(text=[prompt], images=frames, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model(**inp)
        logits = out.logits[0, -1, :].float()

    # Method 1: training-style (single yes/no token)
    p_yes_train = float(torch.softmax(
        torch.stack([logits[yes_id], logits[no_id]]), 0
    )[0].cpu())

    # Method 2: multi-variant
    yes_logits = logits[all_yes_ids]
    no_logits = logits[all_no_ids]
    all_l = torch.cat([yes_logits, no_logits])
    all_p = torch.softmax(all_l, dim=0)
    p_yes_mv = float(all_p[:len(all_yes_ids)].sum().cpu())

    # Raw logits for each variant
    raw = {}
    for i, tid in enumerate(all_yes_ids):
        raw[f"yes[{tid}]"] = float(logits[tid].cpu())
    for i, tid in enumerate(all_no_ids):
        raw[f"no[{tid}]"] = float(logits[tid].cpu())

    return p_yes_train, p_yes_mv, raw


def main():
    import transformers
    from peft import PeftModel

    model_path = str(ROOT / "checkpoints/vlaw/reward_model/qwen_vl")
    instruction = "Lift the peg and insert it upright into the holder."
    dtype = torch.bfloat16

    print("Loading samples...")
    frames_s = load_one_sample(success=True)
    frames_f = load_one_sample(success=False)
    print(f"  Success: {len(frames_s)} frames, Failure: {len(frames_f)} frames")

    lora_paths = [
        ("WORKING", str(ROOT / "checkpoints/vlaw/reward_model/lora_iter1_16frame/final")),
        ("BROKEN_100", str(ROOT / "checkpoints/vlaw/reward_model/ablation_100steps/final")),
    ]

    for lora_name, lora_path in lora_paths:
        print(f"\n{'='*70}")
        print(f"  Testing {lora_name}: {lora_path}")
        print(f"{'='*70}")

        # Load base model
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, device_map="cuda:0",
            attn_implementation="eager",
        )
        processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        tok = processor.tokenizer
        # Training uses encode("yes")[-1] and encode("no")[-1]
        yes_id = tok.encode("yes", add_special_tokens=False)[-1]
        no_id = tok.encode("no", add_special_tokens=False)[-1]
        print(f"  Training yes_id={yes_id}, no_id={no_id}")

        # All variants
        all_yes = list({tok.encode(w, add_special_tokens=False)[0] for w in ["yes", "Yes", "YES", " yes"]})
        all_no = list({tok.encode(w, add_special_tokens=False)[0] for w in ["no", "No", "NO", " no"]})
        print(f"  All yes_ids={all_yes}, no_ids={all_no}")

        # === Test 1: PeftModel (NO merge) ===
        print(f"\n  --- PeftModel (no merge) ---")
        peft_model = PeftModel.from_pretrained(model, lora_path)
        peft_model.eval()

        for label, frames in [("SUCCESS", frames_s), ("FAILURE", frames_f)]:
            pt, pm, raw = eval_single(peft_model, processor, frames, instruction,
                                      yes_id, no_id, all_yes, all_no)
            print(f"    {label}: p_train={pt:.6f}  p_multivar={pm:.6f}")
            # Show top 2 yes and no logits
            yes_items = [(k, v) for k, v in raw.items() if k.startswith("yes")]
            no_items = [(k, v) for k, v in raw.items() if k.startswith("no")]
            yes_items.sort(key=lambda x: -x[1])
            no_items.sort(key=lambda x: -x[1])
            print(f"            yes_logits: {', '.join(f'{k}={v:.2f}' for k, v in yes_items)}")
            print(f"            no_logits:  {', '.join(f'{k}={v:.2f}' for k, v in no_items)}")

        # === Test 2: Merged model ===
        print(f"\n  --- Merged (merge_and_unload) ---")
        merged_model = peft_model.merge_and_unload()
        merged_model.eval()

        for label, frames in [("SUCCESS", frames_s), ("FAILURE", frames_f)]:
            pt, pm, raw = eval_single(merged_model, processor, frames, instruction,
                                      yes_id, no_id, all_yes, all_no)
            print(f"    {label}: p_train={pt:.6f}  p_multivar={pm:.6f}")
            yes_items = [(k, v) for k, v in raw.items() if k.startswith("yes")]
            no_items = [(k, v) for k, v in raw.items() if k.startswith("no")]
            yes_items.sort(key=lambda x: -x[1])
            no_items.sort(key=lambda x: -x[1])
            print(f"            yes_logits: {', '.join(f'{k}={v:.2f}' for k, v in yes_items)}")
            print(f"            no_logits:  {', '.join(f'{k}={v:.2f}' for k, v in no_items)}")

        del peft_model, merged_model, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
