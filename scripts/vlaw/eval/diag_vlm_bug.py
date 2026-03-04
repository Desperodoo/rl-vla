#!/usr/bin/env python3
"""Diagnostic: Load LoRA checkpoints and evaluate with EXACT same method as training-time eval."""
import sys, os, torch, json, h5py, numpy as np, time, glob
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

# Import the training's eval helpers
from rlft.vlaw.reward.train_reward_model import (
    _make_messages, evaluate, RewardDataset, _Episode, _load_episodes_from_dir,
    build_datasets, TrainConfig
)

TASK_INSTRUCTIONS = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
}


def load_model_with_lora(model_path, lora_path, device="cuda:0"):
    """Load model + LoRA exactly as done in evaluation."""
    import transformers
    from peft import PeftModel
    
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="flash_attention_2"
    )
    processor = transformers.AutoProcessor.from_pretrained(model_path)
    
    if lora_path:
        print(f"Loading LoRA from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        # DON'T merge - keep as PEFT model (same as training-time eval)
    
    model.eval()
    
    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[-1]
    no_id = processor.tokenizer.encode("no", add_special_tokens=False)[-1]
    
    return model, processor, yes_id, no_id


def quick_eval(model, processor, yes_id, no_id, device="cuda:0", n_samples=10):
    """Quick eval using the training's _make_messages format."""
    # Build eval dataset using same config as training
    cfg = TrainConfig(
        data_dirs=["data/vlaw/rollouts/iter1", "data/vlaw/rollouts/iter1_highsuc"],
        tasks=["LiftPegUpright-v1"],
        num_frames=16, seed=42, eval_ratio=0.2
    )
    _, eval_ds = build_datasets(cfg)
    
    # Use training's evaluate function
    metrics = evaluate(model, processor, eval_ds, cfg, yes_id, no_id, device=device)
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    model_path = "checkpoints/vlaw/reward_model/qwen_vl"
    checkpoints = {
        "zero-shot": None,
        "200step": "checkpoints/vlaw/reward_model/lora_iter1_16frame/final",
        "100step": "checkpoints/vlaw/reward_model/ablation_100steps/final",
        "400step": "checkpoints/vlaw/reward_model/ablation_400steps/final",
    }
    
    results = {}
    for label, lora_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {label}")
        print(f"{'='*60}")
        
        model, processor, yes_id, no_id = load_model_with_lora(
            model_path, lora_path, args.device
        )
        
        metrics = quick_eval(model, processor, yes_id, no_id, device=args.device)
        results[label] = metrics
        
        print(f"  {label}: acc={metrics['accuracy']:.3f} fp_rate={metrics['fp_rate']:.3f} "
              f"mean_p_yes={metrics['mean_p_yes']:.6f} "
              f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for label, m in results.items():
        print(f"  {label:12s}: mean_p_yes={m['mean_p_yes']:.6f}  acc={m['accuracy']:.3f}  "
              f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    
    with open("results/vlaw/vlm_steps_ablation/diag_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/vlaw/vlm_steps_ablation/diag_results.json")


if __name__ == "__main__":
    main()
