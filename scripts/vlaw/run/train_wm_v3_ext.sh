#!/bin/bash
# VLAW Phase 1b — Ctrl-World WM 扩展训练 (T-WM-V3-EXTENDED)
#
# 背景 (ADR-034):
#   - ckpt-400 eval_WM PSNR=29 有误导性 (单步 GT history, 不反映 Imagination 实际质量)
#   - Imagination 200 条人工审核不可用, 所有下游阻塞
#   - 从 pretrained 全新训练 4000 步, 每 200 步保存 checkpoint
#   - 解除条件: 某 checkpoint Imagination 可视化经人工确认"可用"
#
# 数据: v3 干净数据 (1752 条轨迹, 2 HDF5, frame_skip=4=5Hz)
# GPU:  0-3, DeepSpeed ZeRO-2, 4 GPU
# env:  conda ctrl_world
#
# 用法:
#   cd /home/wjz/rl-vla/ctrl_world && bash ../scripts/vlaw/run/train_wm_v3_ext.sh 2>&1 | tee ../logs/vlaw/train_wm_v3_ext.log

set -e

# ---- GPU ----
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
# 加速提示: 可扩展至 8 GPU (如 0,1,2,3,8,9,...)，此时应将 GRAD_ACCUM 改为 4
# 以保持 eff_batch=32 不变 (batch_size=1 × 8gpu × 4accum = 32)

# ---- Paths (absolute) ----
ROOT="/home/wjz/rl-vla"
CTRL_WORLD="${ROOT}/ctrl_world"
SCRIPTS_DIR="${CTRL_WORLD}/scripts"

# Data
DATASET_ROOT="${ROOT}/data/vlaw/encoded/train"
DATASET_NAMES="LiftPegUpright-v1"
STAT_PATH="${ROOT}/data/vlaw/meta_info/maniskill/stat.json"
META_INFO="${ROOT}/data/vlaw/meta_info"

# Model
SVD_PATH="${ROOT}/checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"
CLIP_PATH="${ROOT}/checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32"
CKPT_PATH="${ROOT}/checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt"

# Output — v3 扩展训练版本
OUTPUT_DIR="${ROOT}/checkpoints/vlaw/world_model/iter1_v3_ext"

# ---- Training hyperparameters ----
MAX_STEPS=4000          # 翻倍: 2000 → 4000 (ADR-034)
LR=1e-5
BATCH_SIZE=1
GRAD_ACCUM=8            # effective batch = 1 × 4gpu × 8 = 32
CKPT_STEPS=200          # save every 200 steps → 20 checkpoints total
VAL_STEPS=200           # validate every 200 steps
LOG_EVERY=10            # log loss every 10 steps
VIDEO_NUM=2             # validation videos per checkpoint

# ---- DeepSpeed config ----
DS_CONFIG="${CTRL_WORLD}/ds_zero2.json"

echo "========================================"
echo "  Phase 1b — WM Extended Training (T-WM-V3-EXTENDED)"
echo "  ADR-034: Imagination 人工审核不可用, 继续训练"
echo "  GPUs:     ${CUDA_VISIBLE_DEVICES}"
echo "  Data:     ${DATASET_ROOT}/${DATASET_NAMES} (v3, 1752 traj)"
echo "  Pretrained: ${CKPT_PATH}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Steps:    ${MAX_STEPS}, LR=${LR}, eff_batch=32"
echo "  Ckpt/Val: every ${CKPT_STEPS} steps"
echo "  Expected checkpoints: 200, 400, ..., 4000 (20 total)"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"
cd "${SCRIPTS_DIR}"

# Use WANDB_MODE=offline to avoid needing external access
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}/wandb"

# Launch with DeepSpeed ZeRO-2
accelerate launch \
    --num_processes ${NUM_GPUS} \
    --mixed_precision fp16 \
    --use_deepspeed \
    --deepspeed_config_file "${DS_CONFIG}" \
    --main_process_port 29502 \
    train_wm.py \
        --svd_model_path "${SVD_PATH}" \
        --clip_model_path "${CLIP_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --dataset_root_path "${DATASET_ROOT}" \
        --dataset_names "${DATASET_NAMES}" \
        --dataset_meta_info_path "${META_INFO}" \
        --data_stat_path "${STAT_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --tag "iter1_v3_ext_4000" \
        --width 192 \
        --height 384 \
        --action_dim 7 \
        --num_frames 5 \
        --num_history 6 \
        --train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --mixed_precision fp16 \
        --learning_rate ${LR} \
        --max_train_steps ${MAX_STEPS} \
        --checkpointing_steps ${CKPT_STEPS} \
        --validation_steps ${VAL_STEPS} \
        --decode_chunk_size 4 \
        --task_type maniskill \
        --freeze_unet_spatial false \
        --log_every_n_steps ${LOG_EVERY} \
        --video_num ${VIDEO_NUM} \
        --num_workers 8

echo ""
echo "✅ WM Extended Training Complete: ${OUTPUT_DIR}"
echo "Expected checkpoints: 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000"
echo ""
echo "下一步: 对关键 checkpoint 运行 Imagination 可视化评估:"
echo "  bash scripts/vlaw/run/eval_wm_imagination_viz.sh <ckpt_path> <gpu_id> [num_trajs]"
