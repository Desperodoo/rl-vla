#!/bin/bash
# =============================================================================
# ManiSkill + RLFT 完整流程运行脚本
# 一站式完成: 环境配置 -> 数据下载 -> Replay -> 训练
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}Step $1: $2${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# =============================================================================
# 配置
# =============================================================================

TASK=${TASK:-"LiftPegUpright-v1"}
QUICK_MODE=${QUICK_MODE:-false}
SKIP_ENV_SETUP=${SKIP_ENV_SETUP:-false}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-false}
SKIP_REPLAY=${SKIP_REPLAY:-false}

# =============================================================================
# 解析参数
# =============================================================================

print_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --quick              快速验证模式 (5000 步)"
    echo "  --full               完整训练模式 (1000000 步)"
    echo "  --task TASK_ID       指定任务 (默认: LiftPegUpright-v1)"
    echo "  --skip-env           跳过环境配置"
    echo "  --skip-download      跳过数据下载"
    echo "  --skip-replay        跳过数据 replay"
    echo "  --help               显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 --quick                    # 快速验证全流程"
    echo "  $0 --full --skip-env          # 完整训练，跳过环境配置"
    echo "  $0 --task PickCube-v1         # 指定其他任务"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --full)
            QUICK_MODE=false
            shift
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --skip-env)
            SKIP_ENV_SETUP=true
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-replay)
            SKIP_REPLAY=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            print_usage
            exit 1
            ;;
    esac
done

# =============================================================================
# 主流程
# =============================================================================

echo ""
echo "=============================================="
echo "ManiSkill + RLFT 完整流程"
echo "=============================================="
echo "任务: ${TASK}"
echo "模式: $([ "$QUICK_MODE" = true ] && echo '快速验证' || echo '完整训练')"
echo "跳过环境配置: ${SKIP_ENV_SETUP}"
echo "跳过数据下载: ${SKIP_DOWNLOAD}"
echo "跳过数据 Replay: ${SKIP_REPLAY}"
echo "=============================================="

# Step 1: 环境配置
if [ "${SKIP_ENV_SETUP}" = false ]; then
    print_step 1 "配置 conda 环境"
    
    # 检查环境是否已存在
    if conda env list | grep -q "^maniskill "; then
        print_warning "环境 'maniskill' 已存在，跳过创建"
    else
        bash "${SCRIPT_DIR}/setup_maniskill_env.sh"
    fi
    
    print_success "环境配置完成"
else
    print_step 1 "跳过环境配置"
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate maniskill || {
    print_error "无法激活 maniskill 环境"
    exit 1
}
print_success "已激活 maniskill 环境"

# Step 2: 下载数据
if [ "${SKIP_DOWNLOAD}" = false ]; then
    print_step 2 "下载演示数据"
    bash "${SCRIPT_DIR}/download_demos.sh" "${TASK}"
    print_success "数据下载完成"
else
    print_step 2 "跳过数据下载"
fi

# Step 3: Replay 数据
if [ "${SKIP_REPLAY}" = false ]; then
    print_step 3 "Replay 演示数据 (生成 RGB + State 数据集)"
    bash "${SCRIPT_DIR}/replay_demos.sh" "${TASK}"
    print_success "数据 Replay 完成"
else
    print_step 3 "跳过数据 Replay"
fi

# Step 4: 启动训练
print_step 4 "启动批量训练"

# 构建训练参数
TRAIN_ARGS=""
if [ "${QUICK_MODE}" = true ]; then
    TRAIN_ARGS="--quick"
else
    TRAIN_ARGS="--full"
fi

# 启动训练
bash "${SCRIPT_DIR}/run_all_algorithms.sh" ${TRAIN_ARGS}

# 获取日志目录
LOG_DIR=$(ls -td logs/training_* 2>/dev/null | head -1)

print_success "训练任务已启动"
echo ""
echo "=============================================="
echo "后续操作"
echo "=============================================="
echo ""
echo "1. 监控训练进度:"
echo "   bash scripts/monitor_training.sh ${LOG_DIR}"
echo ""
echo "2. 查看 GPU 使用:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "3. 查看 WandB 仪表板:"
echo "   https://wandb.ai"
echo ""
echo "4. 终止所有训练:"
echo "   pkill -f 'rlft.offline.train_maniskill'"
echo ""
