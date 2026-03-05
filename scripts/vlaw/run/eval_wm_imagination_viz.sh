#!/bin/bash
# VLAW — Imagination 快速可视化评估
#
# 用途: 给定 WM checkpoint, 运行少量 Imagination + VAE decode 可视化
#       用于人工审核 Imagination 视觉质量 (ADR-034)
#
# env:  conda rlft_ms3
#
# 用法:
#   # 基础用法 (15 条, GPU 4)
#   bash scripts/vlaw/run/eval_wm_imagination_viz.sh checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-800.pt 4
#
#   # 指定数量
#   bash scripts/vlaw/run/eval_wm_imagination_viz.sh checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-1000.pt 5 20
#
#   # 查看输出
#   ls results/vlaw/imagination_viz_checkpoints/checkpoint-800/viz/

set -e

# ---- Arguments ----
WM_CKPT="${1:?Usage: $0 <wm_ckpt_path> <gpu_id> [num_trajs]}"
GPU_ID="${2:?Usage: $0 <wm_ckpt_path> <gpu_id> [num_trajs]}"
NUM_TRAJS="${3:-15}"

# ---- Paths ----
ROOT="/home/wjz/rl-vla"
POLICY_CKPT="${ROOT}/checkpoints/il/best_eval_success_once.pt"

# Extract checkpoint name for output dir (e.g., "checkpoint-800")
CKPT_BASENAME="$(basename "${WM_CKPT}" .pt)"
OUTPUT_BASE="${ROOT}/results/vlaw/imagination_viz_checkpoints/${CKPT_BASENAME}"
OUTPUT_DIR="${OUTPUT_BASE}/synthetic"

echo "========================================"
echo "  Imagination 可视化评估 (ADR-034)"
echo "  WM Ckpt:    ${WM_CKPT}"
echo "  Policy:     ${POLICY_CKPT}"
echo "  GPU:        ${GPU_ID}"
echo "  Num trajs:  ${NUM_TRAJS}"
echo "  Output:     ${OUTPUT_BASE}"
echo "========================================"

# Validate paths
if [[ ! -f "${WM_CKPT}" ]]; then
    # Try relative path from ROOT
    if [[ -f "${ROOT}/${WM_CKPT}" ]]; then
        WM_CKPT="${ROOT}/${WM_CKPT}"
    else
        echo "❌ WM checkpoint not found: ${WM_CKPT}"
        exit 1
    fi
fi

if [[ ! -f "${POLICY_CKPT}" ]]; then
    echo "❌ Policy checkpoint not found: ${POLICY_CKPT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "[Step 1/2] 生成 ${NUM_TRAJS} 条 Imagination 合成轨迹 + 可视化..."
echo ""

cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u rlft/vlaw/scripts/run_imagination.py \
    --wm_ckpt "${WM_CKPT}" \
    --policy_ckpt "${POLICY_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_trajs "${NUM_TRAJS}" \
    --gpu_id 0 \
    --num_interact 12 \
    --act_steps 5 \
    --visualize \
    --vis_count "${NUM_TRAJS}" \
    --save_every "${NUM_TRAJS}" \
    --seed 42

echo ""
echo "========================================"
echo "✅ Imagination 可视化评估完成"
echo ""
echo "📁 输出目录: ${OUTPUT_BASE}"
echo "   合成轨迹: ${OUTPUT_DIR}/"
echo "   可视化:   ${OUTPUT_DIR}/viz/"
echo ""
echo "📊 人工审核指南:"
echo "   1. 查看 viz/ 下的 strip PNG (首帧/中间帧/末帧)"
echo "   2. 评估标准: 机械臂形态是否清晰? 物体是否可辨认? 运动是否合理?"
echo "   3. 与 GT 环境渲染对比 (data/vlaw/encoded/eval/ 可作参考)"
echo "   4. 记录审核结论: '可用' / '勉强可用' / '不可用'"
echo "========================================"
