#!/bin/bash
# VLAW Fresh Start Phase 1 — Ctrl-World WM 微调 (v3 数据)
#
# 参数选择依据:
#   ADR-007: 从 pretrained 直接全量微调 (不用 Phase-A)
#   ADR-010: 2000 步已是最佳 (4000步无额外收益)
#   ADR-013: validation_steps=200, 每 200 步保存 checkpoint
#
# 数据: v3 干净数据 (1752 条轨迹, 2 HDF5, frame_skip=4=5Hz, BUG-024 解决)
# GPU:  0-3, DeepSpeed ZeRO-2, 4 GPU
# env:  conda ctrl_world
#
# 用法:
#   bash scripts/vlaw/run/train_wm_v3.sh 2>&1 | tee logs/vlaw/train_wm_v3.log

set -e

# ---- GPU ----
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

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

# Output — v3 版本
OUTPUT_DIR="${ROOT}/checkpoints/vlaw/world_model/iter1_v3"

# ---- Training hyperparameters ----
MAX_STEPS=2000
LR=1e-5
BATCH_SIZE=1
GRAD_ACCUM=8       # effective batch = 1 × 4gpu × 8 = 32
CKPT_STEPS=200     # save every 200 steps
VAL_STEPS=200      # validate every 200 steps
LOG_EVERY=10       # log loss every 10 steps
VIDEO_NUM=2        # validation videos per checkpoint

# ---- DeepSpeed config ----
DS_CONFIG="${CTRL_WORLD}/ds_zero2.json"

echo "========================================"
echo "  Fresh Start Phase 1 — WM Fine-tuning (v3)"
echo "  GPUs:     ${CUDA_VISIBLE_DEVICES}"
echo "  Data:     ${DATASET_ROOT}/${DATASET_NAMES} (v3, 1752 traj)"
echo "  Pretrained: ${CKPT_PATH}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Steps:    ${MAX_STEPS}, LR=${LR}, eff_batch=32"
echo "  Ckpt/Val: every ${CKPT_STEPS} steps"
echo "========================================"

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
        --tag "iter1_v3_fresh" \
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
        --video_num ${VIDEO_NUM}

echo "✅ WM Training Complete: ${OUTPUT_DIR}"
echo "Checkpoints saved at steps: 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000"
