#!/bin/bash
# =============================================================================
# Sweep: AWSC (Advantage-Weighted ShortCut Flow)
#
# AWSC 支持两种模式：
#   1. 预训练模式：从 ShortCut Flow checkpoint 初始化（推荐）
#   2. From scratch 模式：无预训练，直接训练
#
# 使用控制变量法：baseline + 单参数扫描
#
# Baseline 配置:
#   awsc_beta=100.0, awsc_bc_weight=1.0, awsc_shortcut_weight=0.3
#   awsc_advantage_threshold=-0.5, awsc_num_inference_steps=8
#   online_ratio=0.5, gamma=0.9
#
# 扫描参数:
#   - awsc_beta: advantage 温度 (10, 50, 100, 200)
#   - awsc_bc_weight: flow matching loss 权重 (0.1, 0.5, 1.0, 2.0)
#   - awsc_shortcut_weight: shortcut consistency 权重 (0.0, 0.1, 0.3, 0.5)
#   - awsc_advantage_threshold: 样本过滤阈值 (-1.0, -0.5, 0.0)
#   - online_ratio: online/offline 比例 (0.25, 0.5, 0.75)
#   - with/without pretrain: 预训练 vs from scratch
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_rlpd.sh"

ALGORITHM="awsc"
LOG_DIR="${SWEEP_ROOT}/${ALGORITHM}"

# =============================================================================
# 预训练 checkpoint 路径（必须手动指定）
# =============================================================================

PRETRAIN_PATH="${PRETRAIN_PATH:-}"  # 默认为空，需要用户指定

# =============================================================================
# Baseline 配置
# =============================================================================

# AWSC baseline 超参数（与 IL/offline_rl sweep 结果对齐）
BASELINE_AWSC_BETA="100.0"
BASELINE_AWSC_BC_WEIGHT="1.0"
BASELINE_AWSC_SHORTCUT_WEIGHT="0.3"
BASELINE_AWSC_SELF_CONSISTENCY_K="0.25"  # Updated: match IL/offline_rl
BASELINE_AWSC_ADVANTAGE_THRESHOLD="-0.5"
BASELINE_AWSC_NUM_INFERENCE_STEPS="8"
BASELINE_AWSC_FILTER_POLICY_DATA="False"

# 通用 RL 超参数
BASELINE_GAMMA="0.9"
BASELINE_TAU="0.005"
BASELINE_ONLINE_RATIO="0.5"
BASELINE_REWARD_SCALE="1.0"
BASELINE_NUM_QS="10"
BASELINE_NUM_MIN_QS="2"

# 构建 baseline 参数字符串（不含 pretrain_path）
build_baseline_params() {
    echo "--awsc_beta ${BASELINE_AWSC_BETA} \
--awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} \
--awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} \
--awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} \
--awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} \
--awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} \
--gamma ${BASELINE_GAMMA} \
--tau ${BASELINE_TAU} \
--online_ratio ${BASELINE_ONLINE_RATIO} \
--reward_scale ${BASELINE_REWARD_SCALE} \
--num_qs ${BASELINE_NUM_QS} \
--num_min_qs ${BASELINE_NUM_MIN_QS}"
}

# =============================================================================
# 超参数扫描配置（控制变量法）
# =============================================================================

build_sweep_configs() {
    local use_pretrain=$1  # "pretrain" 或 "scratch"
    
    SWEEP_CONFIGS=()
    
    local baseline=$(build_baseline_params)
    local pretrain_arg=""
    local suffix=""
    
    if [ "${use_pretrain}" = "pretrain" ]; then
        pretrain_arg="--pretrain_path ${PRETRAIN_PATH}"
        suffix="_pretrain"
        LOG_DIR="${SWEEP_ROOT}/${ALGORITHM}_pretrain"
    else
        pretrain_arg=""
        suffix="_scratch"
        LOG_DIR="${SWEEP_ROOT}/${ALGORITHM}_scratch"
    fi
    
    # [0] Baseline 配置（作为对照组）
    SWEEP_CONFIGS+=("${baseline} ${pretrain_arg}")
    
    # === awsc_beta 扫描（advantage 温度）===
    # beta 越大，高 advantage 样本权重越高
    SWEEP_CONFIGS+=("--awsc_beta 10.0 --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    SWEEP_CONFIGS+=("--awsc_beta 50.0 --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    SWEEP_CONFIGS+=("--awsc_beta 200.0 --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    
    # === awsc_bc_weight 扫描（flow matching loss 权重）===
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight 0.1 --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight 0.5 --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight 2.0 --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    
    # === awsc_shortcut_weight 扫描（shortcut consistency 权重）===
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight 0.0 --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight 0.1 --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight 0.5 --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    
    # === awsc_advantage_threshold 扫描（样本过滤阈值）===
    # 只有 advantage > threshold 的 online 样本用于 policy 训练
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold -1.0 --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold 0.0 --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    
    # === online_ratio 扫描（online/offline 数据比例）===
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio 0.25 --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio 0.75 --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    
    # === awsc_num_inference_steps 扫描（推理步数）===
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps 4 --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps 16 --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    
    # === gamma 扫描（折扣因子）===
    # gamma 越大，考虑的未来奖励越多（offline_rl 使用 0.99）
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma 0.95 --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma 0.99 --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    
    # === reward_scale 扫描 ===
    # offline_rl 使用 0.1，在 sweep 中验证是否适合 online
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale 0.1 --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
    SWEEP_CONFIGS+=("--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale 10.0 --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} ${pretrain_arg}")
}

# =============================================================================
# 参数解析
# =============================================================================

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "AWSC 超参数扫描脚本"
    echo ""
    echo "选项:"
    echo "  --pretrain-path PATH  预训练 ShortCut Flow checkpoint 路径"
    echo "  --mode MODE           运行模式: 'pretrain', 'scratch', 'both' (默认: both)"
    echo "  --dry-run             只显示要运行的实验，不实际运行"
    echo "  --force               强制重新运行（忽略已完成状态）"
    echo "  --serial              串行模式运行（带重试机制）"
    echo "  --list                只列出所有配置"
    echo "  -h, --help            显示帮助"
    echo ""
    echo "环境变量:"
    echo "  TASK                  任务 ID (默认: PickCube-v1)"
    echo "  DEMO_PATH             Demo 文件路径"
    echo "  PRETRAIN_PATH         预训练 checkpoint 路径"
    echo "  TOTAL_TIMESTEPS       总训练步数 (默认: 500000)"
    echo "  CUDA_VISIBLE_DEVICES  可用 GPU 列表"
    echo ""
    echo "示例:"
    echo "  # 只跑 from scratch"
    echo "  $0 --mode scratch"
    echo ""
    echo "  # 只跑预训练模式"
    echo "  $0 --mode pretrain --pretrain-path runs/shortcut_flow/best.pt"
    echo ""
    echo "  # 两种模式都跑"
    echo "  $0 --mode both --pretrain-path runs/shortcut_flow/best.pt"
    echo ""
    echo "  # 预览所有实验"
    echo "  $0 --dry-run --pretrain-path runs/shortcut_flow/best.pt"
}

MODE="both"  # 默认两种模式都跑
LIST_ONLY=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrain-path)
            PRETRAIN_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            if [[ ! "$MODE" =~ ^(pretrain|scratch|both)$ ]]; then
                print_error "无效的模式: $MODE (可选: pretrain, scratch, both)"
                exit 1
            fi
            shift 2
            ;;
        --dry-run)
            export DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
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

print_header "AWSC 超参数扫描"
check_demo_file

# 验证预训练模式的 checkpoint
if [[ "$MODE" = "pretrain" || "$MODE" = "both" ]]; then
    if [ -z "${PRETRAIN_PATH}" ]; then
        print_error "预训练模式需要指定 --pretrain-path"
        print_info "示例: $0 --pretrain-path runs/shortcut_flow/best.pt"
        if [ "$MODE" = "both" ]; then
            print_warn "将只运行 from scratch 模式"
            MODE="scratch"
        else
            exit 1
        fi
    elif ! check_pretrain_checkpoint "${PRETRAIN_PATH}"; then
        if [ "$MODE" = "both" ]; then
            print_warn "预训练 checkpoint 不存在，将只运行 from scratch 模式"
            MODE="scratch"
        else
            exit 1
        fi
    fi
fi

echo ""
echo "运行模式: ${MODE}"
echo "预训练路径: ${PRETRAIN_PATH:-无}"
echo ""

# =============================================================================
# 运行 Sweep
# =============================================================================

run_mode() {
    local mode_type=$1  # "pretrain" 或 "scratch"
    
    print_header "AWSC ${mode_type} 模式"
    
    # 设置日志目录
    if [ "${mode_type}" = "pretrain" ]; then
        LOG_DIR="${SWEEP_ROOT}/${ALGORITHM}_pretrain"
    else
        LOG_DIR="${SWEEP_ROOT}/${ALGORITHM}_scratch"
    fi
    
    # 检查是否已完成
    local algo_suffix="${ALGORITHM}_${mode_type}"
    if [ "${FORCE}" != "true" ] && check_stage_completed "${LOG_DIR}" "${algo_suffix}"; then
        print_info "${algo_suffix} 已完成，跳过 (使用 --force 强制重新运行)"
        if [ -f "${LOG_DIR}/best_params_${algo_suffix}.sh" ]; then
            echo ""
            echo "最优参数:"
            cat "${LOG_DIR}/best_params_${algo_suffix}.sh"
        fi
        return 0
    fi
    
    if [ "${FORCE}" = "true" ]; then
        rm -f "${LOG_DIR}/best_params_${algo_suffix}.sh"
    fi
    
    # 构建配置
    build_sweep_configs "${mode_type}"
    
    # 只列出配置
    if [ "${LIST_ONLY}" = "true" ]; then
        print_sweep_summary "${ALGORITHM}_${mode_type}" SWEEP_CONFIGS
        return 0
    fi
    
    # 打印摘要
    print_sweep_summary "${ALGORITHM}_${mode_type}" SWEEP_CONFIGS
    echo ""
    
    # 确认运行（非 dry-run 模式）
    if [ "${DRY_RUN}" != "true" ]; then
        read -p "确认开始 ${mode_type} sweep? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "已跳过 ${mode_type} 模式"
            return 0
        fi
    fi
    
    # 运行 sweep
    run_rlpd_sweep "${ALGORITHM}_${mode_type}" "${LOG_DIR}" SWEEP_CONFIGS
    
    print_info "AWSC ${mode_type} sweep 完成"
    print_info "日志目录: ${LOG_DIR}"
}

# 根据模式运行
case "${MODE}" in
    scratch)
        run_mode "scratch"
        ;;
    pretrain)
        run_mode "pretrain"
        ;;
    both)
        # 先跑 from scratch，再跑 pretrain
        run_mode "scratch"
        echo ""
        run_mode "pretrain"
        ;;
esac

print_header "AWSC Sweep 完成"
echo "结果目录: ${SWEEP_ROOT}"
