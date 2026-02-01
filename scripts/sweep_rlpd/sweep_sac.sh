#!/bin/bash
# =============================================================================
# Sweep: SAC (Soft Actor-Critic)
#
# SAC 是无需预训练的 online RL 基线算法
# 使用控制变量法：baseline + 单参数扫描
#
# Baseline 配置:
#   gamma=0.9, tau=0.005, init_temperature=1.0
#   num_qs=10, num_min_qs=2, online_ratio=0.5
#
# 扫描参数:
#   - gamma: 折扣因子 (0.9, 0.95, 0.99)
#   - tau: target network 软更新系数 (0.001, 0.005, 0.01)
#   - init_temperature: 初始熵温度 (0.1, 1.0, 10.0)
#   - num_qs: Q ensemble 数量 (2, 5, 10)
#   - online_ratio: online/offline 数据比例 (0.25, 0.5, 0.75, 1.0)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_rlpd.sh"

ALGORITHM="sac"
LOG_DIR="${SWEEP_ROOT}/${ALGORITHM}"

# =============================================================================
# Baseline 配置
# =============================================================================

# SAC baseline 超参数
BASELINE_GAMMA="0.9"
BASELINE_TAU="0.005"
BASELINE_INIT_TEMP="1.0"
BASELINE_NUM_QS="10"
BASELINE_NUM_MIN_QS="2"
BASELINE_ONLINE_RATIO="0.5"
BASELINE_REWARD_SCALE="1.0"

# 构建 baseline 参数字符串
build_baseline_params() {
    echo "--gamma ${BASELINE_GAMMA} \
--tau ${BASELINE_TAU} \
--init_temperature ${BASELINE_INIT_TEMP} \
--num_qs ${BASELINE_NUM_QS} \
--num_min_qs ${BASELINE_NUM_MIN_QS} \
--online_ratio ${BASELINE_ONLINE_RATIO} \
--reward_scale ${BASELINE_REWARD_SCALE}"
}

# =============================================================================
# 超参数扫描配置（控制变量法）
# =============================================================================

build_sweep_configs() {
    SWEEP_CONFIGS=()
    
    local baseline=$(build_baseline_params)
    
    # [0] Baseline 配置（作为对照组）
    SWEEP_CONFIGS+=("${baseline}")
    
    # === gamma 扫描（固定其他参数为 baseline）===
    # gamma 越大，考虑的未来奖励越多
    SWEEP_CONFIGS+=("--gamma 0.95 --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}")
    SWEEP_CONFIGS+=("--gamma 0.99 --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}")
    
    # === tau 扫描（target network 软更新系数）===
    # tau 越小，target network 更新越慢，训练越稳定
    SWEEP_CONFIGS+=("--gamma ${BASELINE_GAMMA} --tau 0.001 --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}")
    SWEEP_CONFIGS+=("--gamma ${BASELINE_GAMMA} --tau 0.01 --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}")
    
    # === init_temperature 扫描（探索程度）===
    # temperature 越高，动作分布越随机，探索越多
    SWEEP_CONFIGS+=("--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature 0.1 --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}")
    SWEEP_CONFIGS+=("--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature 10.0 --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}")
    
    # === num_qs 扫描（Q ensemble 数量）===
    # 更多 Q 网络可以减少过估计，但计算开销更大
    SWEEP_CONFIGS+=("--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs 2 --num_min_qs 2 --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}")
    SWEEP_CONFIGS+=("--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs 5 --num_min_qs 2 --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}")
    
    # === online_ratio 扫描（online/offline 数据比例）===
    # 1.0 表示纯 online，0.0 表示纯 offline
    SWEEP_CONFIGS+=("--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio 0.25 --reward_scale ${BASELINE_REWARD_SCALE}")
    SWEEP_CONFIGS+=("--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio 0.75 --reward_scale ${BASELINE_REWARD_SCALE}")
    SWEEP_CONFIGS+=("--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio 1.0 --reward_scale ${BASELINE_REWARD_SCALE}")
    
    # === reward_scale 扫描 ===
    # reward_scale 影响 Q 值的数量级
    SWEEP_CONFIGS+=("--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale 0.1")
    SWEEP_CONFIGS+=("--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale 10.0")
}

# =============================================================================
# 参数解析
# =============================================================================

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "SAC 超参数扫描脚本"
    echo ""
    echo "选项:"
    echo "  --dry-run           只显示要运行的实验，不实际运行"
    echo "  --force             强制重新运行（忽略已完成状态）"
    echo "  --serial            串行模式运行（带重试机制）"
    echo "  --list              只列出所有配置"
    echo "  -h, --help          显示帮助"
    echo ""
    echo "环境变量:"
    echo "  TASK                任务 ID (默认: PickCube-v1)"
    echo "  DEMO_PATH           Demo 文件路径"
    echo "  TOTAL_TIMESTEPS     总训练步数 (默认: 500000)"
    echo "  CUDA_VISIBLE_DEVICES  可用 GPU 列表"
    echo ""
    echo "示例:"
    echo "  $0 --dry-run                    # 预览所有实验"
    echo "  $0 --list                       # 列出配置"
    echo "  TASK=StackCube-v1 $0            # 在不同任务上运行"
    echo "  CUDA_VISIBLE_DEVICES=0,1 $0     # 使用指定 GPU"
}

LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            export DRY_RUN=true
            shift
            ;;
        --force)
            rm -f "${LOG_DIR}/best_params_${ALGORITHM}.sh"
            shift
            ;;
        --serial)
            export SERIAL_MODE=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

# =============================================================================
# 主逻辑
# =============================================================================

print_header "SAC 超参数扫描"
check_demo_file

# 检查是否已完成
if check_stage_completed "${LOG_DIR}" "${ALGORITHM}"; then
    print_info "${ALGORITHM} 已完成，跳过 (使用 --force 强制重新运行)"
    echo ""
    echo "最优参数:"
    cat "${LOG_DIR}/best_params_${ALGORITHM}.sh"
    exit 0
fi

# 构建配置
build_sweep_configs

# 只列出配置
if [ "${LIST_ONLY}" = "true" ]; then
    print_sweep_summary "${ALGORITHM}" SWEEP_CONFIGS
    exit 0
fi

# 打印摘要
print_sweep_summary "${ALGORITHM}" SWEEP_CONFIGS
echo ""

# 确认运行
if [ "${DRY_RUN}" != "true" ]; then
    read -p "确认开始 sweep? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "已取消"
        exit 0
    fi
fi

# 运行 sweep
run_rlpd_sweep "${ALGORITHM}" "${LOG_DIR}" SWEEP_CONFIGS

print_info "SAC sweep 完成"
print_info "日志目录: ${LOG_DIR}"

if [ -f "${LOG_DIR}/best_params_${ALGORITHM}.sh" ]; then
    echo ""
    echo "最优参数:"
    cat "${LOG_DIR}/best_params_${ALGORITHM}.sh"
fi
