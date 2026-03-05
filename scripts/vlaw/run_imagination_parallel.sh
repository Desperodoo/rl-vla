#!/bin/bash
# Imagination 并行生成 — 4 GPU 各生成 50 条, 总计 200 条
# 用法: bash scripts/vlaw/run_imagination_parallel.sh

set -euo pipefail

WM_CKPT="checkpoints/vlaw/world_model/iter1_v3/checkpoint-400.pt"
POLICY_CKPT="checkpoints/il/best_eval_success_once.pt"
OUTPUT_BASE="data/vlaw/synthetic/iter1"
NUM_PER_GPU=50
GPUS=(0 1 2 3)
SEEDS=(42 43 44 45)
CONDA_ENV="rlft_ms3"
PYTHON="/home/wjz/miniconda3/envs/${CONDA_ENV}/bin/python"

echo "============================================================"
echo "[PARALLEL] 启动 ${#GPUS[@]} GPU 并行 Imagination 生成"
echo "  每 GPU: ${NUM_PER_GPU} 条, 总计: $((NUM_PER_GPU * ${#GPUS[@]})) 条"
echo "  GPUs: ${GPUS[*]}"
echo "  WM: ${WM_CKPT}"
echo "  Policy: ${POLICY_CKPT}"
echo "============================================================"

PIDS=()
LOGS=()

for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    SEED=${SEEDS[$i]}
    OUT_DIR="${OUTPUT_BASE}/gpu${i}"
    LOG="/home/wjz/rl-vla/logs/vlaw/imagination_gpu${GPU}_seed${SEED}.log"

    echo "[PARALLEL] GPU ${GPU}: seed=${SEED}, output=${OUT_DIR}, log=${LOG}"

    CUDA_VISIBLE_DEVICES=${GPU} PYTHONUNBUFFERED=1 ${PYTHON} \
        rlft/vlaw/scripts/run_imagination.py \
        --wm_ckpt "${WM_CKPT}" \
        --policy_ckpt "${POLICY_CKPT}" \
        --output_dir "${OUT_DIR}" \
        --num_trajs ${NUM_PER_GPU} \
        --gpu_id 0 \
        --seed ${SEED} \
        --save_every 25 \
        --visualize --vis_count 2 \
        > "${LOG}" 2>&1 &

    PIDS+=($!)
    LOGS+=("${LOG}")
done

echo ""
echo "[PARALLEL] 进程已启动: PIDs=${PIDS[*]}"
echo "[PARALLEL] 日志: ${LOGS[*]}"
echo ""

# 等待所有进程完成
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    GPU=${GPUS[$i]}
    echo "[PARALLEL] 等待 GPU ${GPU} (PID ${PID})..."
    if wait ${PID}; then
        echo "[PARALLEL] ✅ GPU ${GPU} 完成"
    else
        echo "[PARALLEL] ❌ GPU ${GPU} 失败 (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
if [ ${FAILED} -eq 0 ]; then
    echo "[PARALLEL] ✅ 全部 ${#GPUS[@]} GPU 完成"
else
    echo "[PARALLEL] ⚠️ ${FAILED}/${#GPUS[@]} GPU 失败"
fi

# 显示各 GPU 的 summary
for i in "${!GPUS[@]}"; do
    OUT_DIR="${OUTPUT_BASE}/gpu${i}"
    echo ""
    echo "--- GPU ${GPUS[$i]} ---"
    cat "${OUT_DIR}/generation_summary.json" 2>/dev/null || echo "(no summary)"
done

echo ""
echo "[PARALLEL] 完成。请运行 merge 脚本合并结果。"
