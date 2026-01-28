#!/bin/bash
# =============================================================================
# ManiSkill 环境配置脚本
# 创建名为 'maniskill' 的 conda 环境，并安装所有必要依赖
# =============================================================================

set -e

ENV_NAME="maniskill"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "ManiSkill 环境配置脚本"
echo "=========================================="

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: conda 未安装或未配置到 PATH"
    exit 1
fi

# 初始化 conda
eval "$(conda shell.bash hook)"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重新创建? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有环境..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "保留现有环境，激活并更新..."
        conda activate ${ENV_NAME}
    fi
fi

# 创建新环境
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# 激活环境
conda activate ${ENV_NAME}

echo "当前环境: $(conda info --envs | grep '*')"
echo "Python 版本: $(python --version)"

# 安装 PyTorch (CUDA 12.1)
echo ""
echo "安装 PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装 ManiSkill3
echo ""
echo "安装 ManiSkill3..."
pip install mani-skill

# 安装 rlft 算法库依赖
echo ""
echo "安装 rlft 算法库依赖..."
pip install \
    tyro \
    diffusers \
    wandb \
    tensorboard \
    h5py \
    einops \
    scikit-learn \
    opencv-python \
    tqdm \
    matplotlib

# 验证安装
echo ""
echo "验证安装..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import mani_skill; print(f'ManiSkill 安装成功')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import wandb; print(f'WandB: {wandb.__version__}')"

echo ""
echo "=========================================="
echo "环境配置完成!"
echo "使用以下命令激活环境:"
echo "  conda activate ${ENV_NAME}"
echo "=========================================="
