#!/bin/bash
# VLAW Phase 2 — Ctrl-World WM v5 Training (ADR-040: BUG-C VAE Fixed)
#
# All fixes applied in this training run:
#   - BUG-A (ADR-037): Action conditioning = absolute EE pose (not delta)
#   - BUG-B (ADR-037): VAE encoding = independent per-camera latent concat
#   - BUG-C (ADR-040): VAE encoder = SVD AutoencoderKLTemporalDecoder
#     (previously used sd-vae-ft-mse AutoencoderKL, causing latent mismatch)
#
# Data: v5 re-encoded with SVD VAE (1752 trajs, frame_skip=4=5Hz)
# GPU:  2-9, DeepSpeed ZeRO-2, 8 GPU
# env:  conda ctrl_world
#
# Usage:
#   cd /home/wjz/rl-vla/ctrl_world && bash ../scripts/vlaw/run/train_wm_v5.sh 2>&1 | tee ../logs/vlaw/train_wm_v5.log

set -e

# ---- GPU ----
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
NUM_GPUS=8

# ---- Paths (absolute) ----
ROOT="/home/wjz/rl-vla"
CTRL_WORLD="${ROOT}/ctrl_world"
SCRIPTS_DIR="${CTRL_WORLD}/scripts"

# Data — v5 re-encoded with SVD VAE (BUG-C fix)
DATASET_ROOT="${ROOT}/data/vlaw/encoded/train_v5"
DATASET_NAMES="LiftPegUpright-v1"
STAT_PATH="${ROOT}/data/vlaw/meta_info/maniskill/stat.json"
META_INFO="${ROOT}/data/vlaw/meta_info"

# Model — start from pretrained Ctrl-World (DROID, 10K steps)
SVD_PATH="${ROOT}/checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid"
CLIP_PATH="${ROOT}/checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32"
CKPT_PATH="${ROOT}/checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt"

# Output — v5 (BUG-A/B/C all fixed)
OUTPUT_DIR="${ROOT}/checkpoints/vlaw/world_model/iter1_v5"

# ---- Training hyperparameters ----
MAX_STEPS=4000          # 全量训练 4000 步
LR=1e-5
BATCH_SIZE=1
GRAD_ACCUM=4            # effective batch = 1 x 8gpu x 4 = 32
CKPT_STEPS=200          # save every 200 steps → 20 checkpoints total
VAL_STEPS=200           # validate every 200 steps
LOG_EVERY=10            # log loss every 10 steps
VIDEO_NUM=2             # validation videos per checkpoint

# ---- DeepSpeed config ----
DS_CONFIG="${CTRL_WORLD}/ds_zero2_8gpu.json"

# ---- Memory optimizations ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "  Phase 2 — WM v5 Training (ADR-040: BUG-C VAE Fixed)"
echo "  BUG-A: Action conditioning = absolute EE pose"
echo "  BUG-B: VAE encoding = independent per-camera latent concat"
echo "  BUG-C: VAE encoder = SVD AutoencoderKLTemporalDecoder"
echo "  GPUs:     ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} processes)"
echo "  Data:     ${DATASET_ROOT}/${DATASET_NAMES} (v5 SVD VAE)"
echo "  Pretrained: ${CKPT_PATH}"
echo "  Stat.json:  ${STAT_PATH}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Steps:    ${MAX_STEPS}, LR=${LR}, eff_batch=${BATCH_SIZE}x${NUM_GPUS}x${GRAD_ACCUM}=$((BATCH_SIZE*NUM_GPUS*GRAD_ACCUM))"
echo "  Ckpt/Val: every ${CKPT_STEPS} steps"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"
cd "${SCRIPTS_DIR}"

# Wandb offline (no external access needed)
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}/wandb"

# Launch with DeepSpeed ZeRO-2
accelerate launch \
    --num_processes ${NUM_GPUS} \
    --mixed_precision fp16 \
    --use_deepspeed \
    --deepspeed_config_file "${DS_CONFIG}" \
    --main_process_port 29503 \
    train_wm.py \
        --svd_model_path "${SVD_PATH}" \
        --clip_model_path "${CLIP_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --dataset_root_path "${DATASET_ROOT}" \
        --dataset_names "${DATASET_NAMES}" \
        --dataset_meta_info_path "${META_INFO}" \
        --data_stat_path "${STAT_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --tag "iter1_v5_4000" \
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
echo "WM v5 Training Complete: ${OUTPUT_DIR}"
echo "Checkpoints: 200, 400, ..., 4000 (20 total)"
