#!/bin/bash
# T-EXP-VLM-03: VLM 训练步数消融实验 (100/400/800 步)
# 基线: 200步, acc=0.824, FP=3.7%, AUC=0.808
# GPU: 6 (单卡), conda: vlaw_reward
set -e

export CUDA_VISIBLE_DEVICES=6
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use vlaw_reward python directly
PYTHON=/home/wjz/miniconda3/envs/vlaw_reward/bin/python

RESULT_DIR="/home/wjz/rl-vla/results/vlaw/vlm_steps_ablation"
mkdir -p "$RESULT_DIR"

LOG_FILE="$RESULT_DIR/ablation_log.txt"
echo "=== T-EXP-VLM-03: VLM Steps Ablation ===" | tee "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"

cd /home/wjz/rl-vla

STEPS_LIST=(100 400 800)

for STEPS in "${STEPS_LIST[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "  Training: ${STEPS} steps" | tee -a "$LOG_FILE"
    echo "  Start: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    OUTPUT_DIR="checkpoints/vlaw/reward_model/ablation_${STEPS}steps"

    # 训练
    $PYTHON rlft/vlaw/reward/train_reward_model.py \
        --data_dirs data/vlaw/rollouts/iter1 data/vlaw/rollouts/iter1_highsuc \
        --tasks LiftPegUpright-v1 \
        --model_path checkpoints/vlaw/reward_model/qwen_vl \
        --output_dir "$OUTPUT_DIR" \
        --num_frames 16 \
        --train_steps "$STEPS" \
        --lora_r 16 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 128 \
        --use_wandb --wandb_project vlaw-reward 2>&1 | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
    echo "  Training ${STEPS} steps done: $(date)" | tee -a "$LOG_FILE"

    # 评估
    echo "  Evaluating ${STEPS} steps ..." | tee -a "$LOG_FILE"
    EVAL_JSON="$RESULT_DIR/eval_${STEPS}steps.json"

    $PYTHON scripts/vlaw/eval/eval_vlm_ablation.py \
        --lora_path "$OUTPUT_DIR/final" \
        --num_frames 16 \
        --task LiftPegUpright-v1 \
        --threshold 0.8 \
        --output_json "$EVAL_JSON" 2>&1 | tee -a "$LOG_FILE"

    echo "  Eval ${STEPS} steps done: $(date)" | tee -a "$LOG_FILE"
    echo "  Results saved → $EVAL_JSON" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "  ALL DONE: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 汇总结果
echo "" | tee -a "$LOG_FILE"
echo "=== Summary ===" | tee -a "$LOG_FILE"
for STEPS in "${STEPS_LIST[@]}"; do
    EVAL_JSON="$RESULT_DIR/eval_${STEPS}steps.json"
    if [ -f "$EVAL_JSON" ]; then
        echo "--- ${STEPS} steps ---" | tee -a "$LOG_FILE"
        $PYTHON -c "
import json
with open('$EVAL_JSON') as f:
    m = json.load(f)
print(f'  AUC       = {m[\"auc\"]:.4f}')
print(f'  Acc@α=0.8 = {m[\"acc_alpha\"]:.4f}')
print(f'  FP@α=0.8  = {m[\"fp_rate_alpha\"]:.4f}')
print(f'  Acc@Youden= {m.get(\"acc_youden\", \"N/A\")}')
print(f'  FP@Youden = {m.get(\"fp_rate_youden\", \"N/A\")}')
print(f'  Youden θ  = {m.get(\"youden_threshold\", \"N/A\")}')
print(f'  p_yes(+)  = {m[\"p_yes_mean_success\"]:.4f} ± {m[\"p_yes_std_success\"]:.4f}')
print(f'  p_yes(-)  = {m[\"p_yes_mean_fail\"]:.4f} ± {m[\"p_yes_std_fail\"]:.4f}')
" 2>&1 | tee -a "$LOG_FILE"
    fi
done

# 保存汇总表 (markdown)
$PYTHON -c "
import json, os

result_dir = '$RESULT_DIR'
steps_list = [100, 200, 400, 800]
rows = []

# 200步基线 (已知)
rows.append({
    'steps': 200,
    'auc': 0.808,
    'acc_alpha': 0.824,
    'fp_rate_alpha': 0.037,
    'source': 'baseline (lora_iter1_16frame)',
})

for s in [100, 400, 800]:
    fp = os.path.join(result_dir, f'eval_{s}steps.json')
    if os.path.isfile(fp):
        with open(fp) as f:
            m = json.load(f)
        rows.append({
            'steps': s,
            'auc': m['auc'],
            'acc_alpha': m['acc_alpha'],
            'fp_rate_alpha': m['fp_rate_alpha'],
            'acc_youden': m.get('acc_youden', ''),
            'fp_rate_youden': m.get('fp_rate_youden', ''),
            'youden_threshold': m.get('youden_threshold', ''),
            'source': f'ablation_{s}steps',
        })

rows.sort(key=lambda x: x['steps'])

summary = os.path.join(result_dir, 'summary.md')
with open(summary, 'w') as f:
    f.write('# T-EXP-VLM-03: VLM Training Steps Ablation\n\n')
    f.write('| Steps | ROC-AUC | Acc@α=0.8 | FP@α=0.8 | Source |\n')
    f.write('|-------|---------|-----------|----------|--------|\n')
    for r in rows:
        f.write(f\"| {r['steps']} | {r['auc']:.4f} | {r['acc_alpha']:.4f} | {r['fp_rate_alpha']:.4f} | {r['source']} |\n\")
print(f'Summary saved → {summary}')
" 2>&1 | tee -a "$LOG_FILE"

echo "DONE_FLAG" >> "$LOG_FILE"
