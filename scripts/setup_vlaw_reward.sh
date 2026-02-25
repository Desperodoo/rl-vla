#!/usr/bin/env bash
# 安装 vlaw_reward 虚拟环境并下载 Qwen2.5-VL 模型
# 用法: bash scripts/setup_vlaw_reward.sh [--skip-flash-attn] [--skip-download]

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="vlaw_reward"
ENV_YML="${REPO_ROOT}/env_yml/vlaw_reward_env.yml"
PIP_REQS="${REPO_ROOT}/env_yml/vlaw_reward_pip.txt"
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"    # 实际可用模型
MODEL_DIR="${REPO_ROOT}/checkpoints/vlaw/reward_model/qwen_vl"
GPU_ID="${GPU_ID:-6}"

SKIP_FLASH_ATTN=false
SKIP_DOWNLOAD=false

for arg in "$@"; do
  case $arg in
    --skip-flash-attn) SKIP_FLASH_ATTN=true ;;
    --skip-download)   SKIP_DOWNLOAD=true ;;
  esac
done

echo "=========================================="
echo " VLAW Reward Model 环境搭建脚本"
echo "=========================================="
echo "  conda 环境: ${ENV_NAME}"
echo "  模型 ID:    ${MODEL_ID}"
echo "  模型路径:   ${MODEL_DIR}"
echo "  GPU:        ${GPU_ID}"
echo "=========================================="

# ── 1. 创建 conda 基础环境 (Python 3.10 only) ──────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "[SKIP] 环境 '${ENV_NAME}' 已存在，跳过创建"
else
  echo "[1/5] 创建 conda 基础环境 (Python 3.10) ..."
  conda env create -f "${ENV_YML}" -n "${ENV_NAME}"
fi

# ── 2. 安装 pip 依赖 ─────────────────────────────────────────────────────────
echo "[2/5] 安装 pip 依赖 (PyTorch, transformers, peft 等) ..."
# 先尝试从 PyPI 安装稳定版 transformers
conda run -n "${ENV_NAME}" pip install -r "${PIP_REQS}" \
  || {
    echo "[WARN] stable transformers 版本不满足，尝试从 git 安装 ..."
    # 先装除 transformers 外的其他包
    grep -v '^transformers' "${PIP_REQS}" > /tmp/vlaw_reward_pip_notf.txt
    conda run -n "${ENV_NAME}" pip install -r /tmp/vlaw_reward_pip_notf.txt
    # 然后从 git 安装 transformers
    conda run -n "${ENV_NAME}" pip install \
      "git+https://github.com/huggingface/transformers@main"
  }

# ── 3. 安装 Flash Attention ──────────────────────────────────────────────────
if [ "${SKIP_FLASH_ATTN}" = false ]; then
  echo "[3/5] 安装 flash-attn (需要编译，约 10-20 分钟) ..."
  conda run -n "${ENV_NAME}" pip install flash-attn --no-build-isolation \
    || echo "[WARN] flash-attn 安装失败，继续 (推理速度会稍慢)"
else
  echo "[3/5] 跳过 flash-attn 安装"
fi

# ── 4. 下载 Qwen-VL 模型 ────────────────────────────────────────────────────
if [ "${SKIP_DOWNLOAD}" = false ]; then
  echo "[4/5] 下载模型到 ${MODEL_DIR} ..."
  mkdir -p "${MODEL_DIR}"

  # 优先用 hf_transfer 加速下载
  conda run -n "${ENV_NAME}" pip install hf-transfer -q

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  HF_HUB_ENABLE_HF_TRANSFER=1 \
  conda run -n "${ENV_NAME}" python - <<PYEOF
from huggingface_hub import snapshot_download
import os

model_id = "${MODEL_ID}"
local_dir = "${MODEL_DIR}"

print(f"Downloading {model_id} → {local_dir}")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)
print("Download complete.")
PYEOF
else
  echo "[4/5] 跳过模型下载"
fi

# ── 5. 测试推理 ──────────────────────────────────────────────────────────────
echo "[5/5] 验证推理 (GPU ${GPU_ID}) ..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
conda run -n "${ENV_NAME}" python "${REPO_ROOT}/rlft/vlaw/test_reward_model.py" \
  --model_path "${MODEL_DIR}" \
  || echo "[WARN] 推理测试未通过，请检查日志"

echo ""
echo "=========================================="
echo " 完成！激活方式: conda activate ${ENV_NAME}"
echo "=========================================="
