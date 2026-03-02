#!/bin/bash
set -e
cd /home/wjz/rl-vla

export RESULT_FILE="/home/wjz/rl-vla/logs/vlaw/Reward-Agent-result-20260302_143121.md"

echo "========================================" 
echo " T-EXP-VLM-02: VLM LoRA Rank Ablation"
echo " GPUs: 3,4 | Ranks: 8, 32, 64"
echo " Start: $(date)"
echo "========================================" 

for RANK in 8 32 64; do
    echo ""
    echo "────────────────────────────────────────"
    echo " Starting LoRA r=$RANK at $(date)"
    echo "────────────────────────────────────────"
    
    OUT_DIR="checkpoints/vlaw/reward_model/ablation_lora_r${RANK}"
    LOG_FILE="logs/vlaw/vlm_lora_r${RANK}_train.log"
    
    CUDA_VISIBLE_DEVICES=3,4 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run -n vlaw_reward accelerate launch \
        --num_processes 2 --multi_gpu \
        rlft/vlaw/reward/train_reward_model.py \
        --data_dirs data/vlaw/rollouts/iter1 data/vlaw/rollouts/iter1_highsuc \
        --tasks LiftPegUpright-v1 \
        --model_path checkpoints/vlaw/reward_model/qwen_vl \
        --output_dir "$OUT_DIR" \
        --num_frames 16 --train_steps 200 --lora_r $RANK \
        --per_device_batch_size 1 --gradient_accumulation_steps 128 \
        --use_wandb \
        --multi_gpu \
        2>&1 | tee "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        FINAL_LOSS=$(grep -oP 'Step \d+.*loss[=: ]+\K[0-9.]+' "$LOG_FILE" | tail -1)
        FINAL_ACC=$(grep -oP 'acc[uracy]*[=: ]+\K[0-9.]+' "$LOG_FILE" | tail -1)
        FP_RATE=$(grep -oP 'FP[_ ]*[Rr]ate[=: ]+\K[0-9.]+%?' "$LOG_FILE" | tail -1)
        ROC_AUC=$(grep -oP 'ROC.AUC[=: ]+\K[0-9.]+' "$LOG_FILE" | tail -1)
        
        echo "- [x] LoRA r=$RANK: ✅ loss=$FINAL_LOSS acc=$FINAL_ACC FP=$FP_RATE ROC-AUC=$ROC_AUC — $(date +%H:%M)" >> "$RESULT_FILE"
        echo " ✅ r=$RANK completed: loss=$FINAL_LOSS acc=$FINAL_ACC FP=$FP_RATE ROC-AUC=$ROC_AUC"
    else
        echo "- [ ] LoRA r=$RANK: ❌ FAILED (exit=$EXIT_CODE) — $(date +%H:%M)" >> "$RESULT_FILE"
        echo " ❌ r=$RANK FAILED with exit code $EXIT_CODE"
    fi
    
    echo ""
done

echo "" >> "$RESULT_FILE"
echo "## 所有 LoRA rank 消融完成 — $(date)" >> "$RESULT_FILE"
echo ""
echo "========================================"
echo " ALL DONE at $(date)"
echo "========================================"
