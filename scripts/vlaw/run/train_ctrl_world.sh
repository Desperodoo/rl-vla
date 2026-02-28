#!/bin/bash
# VLAW P2.2 — Ctrl-World 世界模型训练启动脚本
#
# Phase A (预热): 仅训练 Action Encoder + UNet temporal attention (~10K steps)
# Phase B (全量): 解冻 UNet 全部 (~20K-50K steps)
#
# 用法:
#   bash scripts/train_ctrl_world.sh phase_a   # Phase A
#   bash scripts/train_ctrl_world.sh phase_b   # Phase B

set -e

PHASE=${1:-phase_a}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

# GPU 分配 (WM 使用 GPU 4-7, 共 4 块 4090)
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4

# 路径
CTRL_WORLD="${ROOT}/ctrl_world"
SCRIPTS_DIR="${CTRL_WORLD}/scripts"
DATA_ROOT="${ROOT}/data/vlaw/encoded"
OUTPUT_DIR="${ROOT}/checkpoints/vlaw/world_model"
META_INFO="${ROOT}/data/vlaw/meta_info"

# conda 环境
CONDA_ENV=ctrl_world

echo "========================================"
echo "  Ctrl-World WM Training — ${PHASE}"
echo "  GPUs:     ${CUDA_VISIBLE_DEVICES}"
echo "  Data:     ${DATA_ROOT}"
echo "  Output:   ${OUTPUT_DIR}"
echo "========================================"

cd "${SCRIPTS_DIR}"

if [ "${PHASE}" = "phase_a" ]; then
    echo "[Phase A] 仅训练 Action Encoder + UNet temporal attention"
    CKPT_PATH="${OUTPUT_DIR}/pretrained/Ctrl-World/checkpoint-10000.pt"
    MAX_STEPS=10000
    LR=1e-4
    FREEZE_UNET=True
    TAG="maniskill_wm_phase_a"
elif [ "${PHASE}" = "phase_b" ]; then
    echo "[Phase B] 解冻 UNet 全量微调"
    # Phase B 从 Phase A 最佳 checkpoint 开始
    CKPT_PATH=$(ls -t "${OUTPUT_DIR}/phase_a"/*.pt 2>/dev/null | head -n1 || echo "")
    if [ -z "${CKPT_PATH}" ]; then
        echo "⚠️  未找到 Phase A checkpoint, 使用原版预训练权重"
        CKPT_PATH="${OUTPUT_DIR}/pretrained/Ctrl-World/checkpoint-10000.pt"
    fi
    MAX_STEPS=30000
    LR=1e-5
    FREEZE_UNET=False
    TAG="maniskill_wm_phase_b"
else
    echo "❌ 未知 phase: ${PHASE}  (支持: phase_a / phase_b)"
    exit 1
fi

echo "Checkpoint: ${CKPT_PATH}"
echo "Max steps:  ${MAX_STEPS}"
echo "LR:         ${LR}"
echo "Freeze UNet spatial: ${FREEZE_UNET}"

conda run -n "${CONDA_ENV}" \
    accelerate launch \
        --num_processes "${NUM_GPUS}" \
        --mixed_precision fp16 \
        --main_process_port 29502 \
    "${SCRIPTS_DIR}/train_wm.py" \
        --svd_model_path "${ROOT}/checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid" \
        --clip_model_path "${ROOT}/checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32" \
        --ckpt_path "${CKPT_PATH}" \
        --dataset_root_path "${DATA_ROOT}" \
        --dataset_names "demos" \
        --dataset_meta_info_path "${META_INFO}" \
        --data_stat_path "${META_INFO}/maniskill/stat.json" \
        --output_dir "${OUTPUT_DIR}/${PHASE}" \
        --tag "${TAG}" \
        --width 192 \
        --height 384 \
        --action_dim 7 \
        --num_frames 5 \
        --num_history 4 \
        --train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --mixed_precision fp16 \
        --learning_rate "${LR}" \
        --max_train_steps "${MAX_STEPS}" \
        --checkpointing_steps 2000 \
        --validation_steps 500 \
        --decode_chunk_size 4 \
        --task_type maniskill \
        --freeze_unet_spatial "${FREEZE_UNET}"

echo "✅ 训练完成: ${PHASE}"
