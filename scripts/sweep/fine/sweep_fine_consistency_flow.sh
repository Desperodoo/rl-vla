#!/bin/bash
# =============================================================================
# 精细化 Sweep: Consistency Flow
# 
# 上一轮最佳: success_once=0.48, consistency_weight=0.3, cons_delta=0.02
# 
# 精细化方向:
# 1. consistency_weight 在 0.3 附近: [0.2, 0.25, 0.3, 0.35, 0.4]
# 2. cons_delta 在 0.02 附近: [0.01, 0.015, 0.02, 0.025, 0.03]
# 3. 新参数: consistency_loss_type (l2, l1, huber)
# 4. 新参数: cons_target_ema (EMA 目标网络)
# 5. 继承 flow_matching 的最优参数
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_fine.sh"

ALGORITHM="consistency_flow"
LOG_DIR="${FINE_SWEEP_ROOT}/${ALGORITHM}"

# 加载上一轮最优参数
load_previous_best "${ALGORITHM}" "stage2_dependent_il"
# 也加载基础算法参数
load_previous_best "flow_matching" "stage1_base_il"

# =============================================================================
# 精细化超参数配置
# =============================================================================

# 继承 flow_matching 最优参数
BASE_PARAMS="--lr ${BEST_LR:-3e-4} --num_flow_steps ${BEST_NUM_FLOW_STEPS:-20} --obs_horizon 2 --act_horizon 8 --pred_horizon 16"

SWEEP_CONFIGS=(
    # === consistency_weight 精细化 (cons_delta=0.02 固定) ===
    "${BASE_PARAMS} --consistency_weight 0.2 --cons_delta_mode fixed --cons_delta_fixed 0.02"
    "${BASE_PARAMS} --consistency_weight 0.25 --cons_delta_mode fixed --cons_delta_fixed 0.02"
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode fixed --cons_delta_fixed 0.02"  # 基准
    "${BASE_PARAMS} --consistency_weight 0.35 --cons_delta_mode fixed --cons_delta_fixed 0.02"
    "${BASE_PARAMS} --consistency_weight 0.4 --cons_delta_mode fixed --cons_delta_fixed 0.02"
    
    # === cons_delta 精细化 (consistency_weight=0.3 固定) ===
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode fixed --cons_delta_fixed 0.01"
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode fixed --cons_delta_fixed 0.015"
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode fixed --cons_delta_fixed 0.025"
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode fixed --cons_delta_fixed 0.03"
    
    # === consistency_loss_type 探索 ===
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode fixed --cons_delta_fixed 0.02 --consistency_loss_type l2"
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode fixed --cons_delta_fixed 0.02 --consistency_loss_type l1"
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode fixed --cons_delta_fixed 0.02 --consistency_loss_type huber"
    
    # === cons_target_ema 探索 (target 网络 EMA) ===
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode fixed --cons_delta_fixed 0.02 --cons_target_ema 0.99"
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode fixed --cons_delta_fixed 0.02 --cons_target_ema 0.995"
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode fixed --cons_delta_fixed 0.02 --cons_target_ema 0.999"
    
    # === 随机 delta 模式进一步探索 ===
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode random --cons_delta_min 0.01 --cons_delta_max 0.05"
    "${BASE_PARAMS} --consistency_weight 0.3 --cons_delta_mode random --cons_delta_min 0.01 --cons_delta_max 0.1"
    
    # === 最佳组合候选 ===
    "${BASE_PARAMS} --consistency_weight 0.35 --cons_delta_mode fixed --cons_delta_fixed 0.015 --cons_target_ema 0.995"
    "${BASE_PARAMS} --consistency_weight 0.25 --cons_delta_mode fixed --cons_delta_fixed 0.02 --consistency_loss_type huber"
)

# =============================================================================
# 主逻辑
# =============================================================================

print_fine_sweep_header "${ALGORITHM}"
check_demo_file

if check_fine_sweep_completed "${ALGORITHM}"; then
    print_info "${ALGORITHM} 精细化 sweep 已完成，跳过"
    cat "${LOG_DIR}/best_params_${ALGORITHM}.sh"
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            export DRY_RUN=true
            shift
            ;;
        --force)
            rm -rf "${LOG_DIR}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

run_sweep "${ALGORITHM}" "${LOG_DIR}" SWEEP_CONFIGS

print_info "Consistency Flow 精细化 sweep 完成"
print_info "最优参数文件: ${LOG_DIR}/best_params_${ALGORITHM}.sh"
