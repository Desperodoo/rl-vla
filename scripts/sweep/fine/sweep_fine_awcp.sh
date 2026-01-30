#!/bin/bash
# =============================================================================
# 精细化 Sweep: AWCP (Advantage-Weighted Consistency Policy)
# 
# 上一轮最佳: success_once=0.07, beta=10, reward_scale=1.0, consistency_weight=0.3
# 效果较差，需要大幅探索
# 
# 精细化方向:
# 1. beta (advantage 温度) 更广范围: [0.5, 1.0, 2.0, 5.0, 10.0]
# 2. reward_scale 更广范围: [0.01, 0.05, 0.1, 0.5, 1.0]
# 3. consistency_weight: [0.0, 0.1, 0.3, 0.5]
# 4. 新参数: advantage_clip (advantage 裁剪)
# 5. 新参数: policy_reg_weight (策略正则化)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_fine.sh"

ALGORITHM="awcp"
LOG_DIR="${FINE_SWEEP_ROOT}/${ALGORITHM}"

# 加载上一轮最优参数
load_previous_best "${ALGORITHM}" "stage3_offline_rl"
load_previous_best "consistency_flow" "stage2_dependent_il"
load_previous_best "flow_matching" "stage1_base_il"

# =============================================================================
# 精细化超参数配置
# =============================================================================

BASE_PARAMS="--lr ${BEST_LR:-3e-4} --num_flow_steps ${BEST_NUM_FLOW_STEPS:-20} --obs_horizon 2 --act_horizon 8 --pred_horizon 16"
BASE_PARAMS="${BASE_PARAMS} --cons_delta_mode fixed --cons_delta_fixed ${BEST_CONS_DELTA_FIXED:-0.02}"

SWEEP_CONFIGS=(
    # === beta 更广范围探索 ===
    "${BASE_PARAMS} --beta 0.5 --reward_scale 0.1 --consistency_weight 0.3"
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.3"
    "${BASE_PARAMS} --beta 2.0 --reward_scale 0.1 --consistency_weight 0.3"
    "${BASE_PARAMS} --beta 5.0 --reward_scale 0.1 --consistency_weight 0.3"
    "${BASE_PARAMS} --beta 10.0 --reward_scale 0.1 --consistency_weight 0.3"
    
    # === reward_scale 更广范围探索 (beta=1.0) ===
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.01 --consistency_weight 0.3"
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.05 --consistency_weight 0.3"
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.3"
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.5 --consistency_weight 0.3"
    "${BASE_PARAMS} --beta 1.0 --reward_scale 1.0 --consistency_weight 0.3"
    
    # === consistency_weight 探索 (参考 aw_shortcut_flow 成功经验) ===
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.0"
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.1"
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.5"
    
    # === advantage_clip 探索 ===
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.3 --advantage_clip 10"
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.3 --advantage_clip 50"
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.3 --advantage_clip 100"
    
    # === policy_reg_weight 探索 ===
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.3 --policy_reg_weight 0.01"
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.3 --policy_reg_weight 0.1"
    
    # === 最佳组合候选 (参考 aw_shortcut_flow: beta=1.0, reward_scale=0.1 效果好) ===
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.1"
    "${BASE_PARAMS} --beta 2.0 --reward_scale 0.1 --consistency_weight 0.1"
    "${BASE_PARAMS} --beta 1.0 --reward_scale 0.1 --consistency_weight 0.0 --policy_reg_weight 0.01"
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

print_info "AWCP 精细化 sweep 完成"
print_info "最优参数文件: ${LOG_DIR}/best_params_${ALGORITHM}.sh"
