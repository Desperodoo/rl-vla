#!/bin/bash
# =============================================================================
# ManiSkill Demo 数据下载脚本
# 下载指定任务的原始演示数据
# =============================================================================

set -e

# 默认下载的任务列表
DEFAULT_TASKS=(
    "LiftPegUpright-v1"
)

# 可以通过参数指定任务
TASKS=("${@:-${DEFAULT_TASKS[@]}}")

echo "=========================================="
echo "ManiSkill Demo 数据下载"
echo "=========================================="

# 检查 mani_skill 是否可用
if ! python -c "import mani_skill" &> /dev/null; then
    echo "错误: mani_skill 未安装，请先运行 setup_maniskill_env.sh"
    exit 1
fi

# 下载演示数据
for task in "${TASKS[@]}"; do
    echo ""
    echo "下载任务: ${task}"
    echo "----------------------------------------"
    python -m mani_skill.utils.download_demo "${task}"
done

echo ""
echo "=========================================="
echo "下载完成!"
echo "数据存储在: ~/.maniskill/demos/"
echo ""
echo "下一步: 运行 replay_demos.sh 生成训练数据集"
echo "=========================================="
