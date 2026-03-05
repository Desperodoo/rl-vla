#!/bin/bash
# T-IMAGINATION-003: 500 条合成轨迹大规模生成
# 在 tmux 中后台运行, GPU 4, num_interact=12

cd /home/wjz/rl-vla

# 激活 conda 环境
source /home/wjz/miniconda3/etc/profile.d/conda.sh
conda activate rlft_ms3

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

# 日志文件
LOG_FILE="/home/wjz/rl-vla/logs/vlaw/imagination_003_500trajs.log"
echo "Starting T-IMAGINATION-003 at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"

python scripts/vlaw/run/run_imagination_iter1.py \
    --num_trajs 500 \
    --num_interact 12 \
    --act_steps 5 \
    --gpu_id 0 \
    --wm_ckpt /home/wjz/rl-vla/checkpoints/vlaw/world_model/iter1/checkpoint-2000.pt \
    --policy_ckpt /home/wjz/rl-vla/checkpoints/il/best_eval_success_once.pt \
    --output_dir /home/wjz/rl-vla/data/vlaw/synthetic/iter1_003_500trajs \
    --task_id LiftPegUpright-v1 \
    --save_every 100 \
    --use_real_policy \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "T-IMAGINATION-003 finished at $(date) with exit code $?" | tee -a "$LOG_FILE"

# 保持 tmux 会话
sleep 3600
