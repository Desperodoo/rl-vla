#!/bin/bash
# =============================================================================
# 精细化 Sweep: CPQL (Conservative Policy Q-Learning)
# 
# 上一轮最佳: success_once=0.22, alpha=0.001, reward_scale=0.1
# 
# 精细化方向:
# 1. alpha (conservative 权重) 在 0.001 附近: [0.0005, 0.001, 0.002, 0.005]
# 2. reward_scale 在 0.1 附近: [0.05, 0.08, 0.1, 0.15, 0.2]
# 3. 新参数: q_lr (Q 网络学习率)
# 4. 新参数: target_update_freq (目标网络更新频率)
# 5. 新参数: num_q_updates (每步 Q 更新次数)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_fine.sh"

ALGORITHM="cpql"
LOG_DIR="${FINE_SWEEP_ROOT}/${ALGORITHM}"

# 加载上一轮最优参数
load_previous_best "${ALGORITHM}" "stage3_offline_rl"
load_previous_best "consistency_flow" "stage2_dependent_il"
load_previous_best "flow_matching" "stage1_base_il"

# =============================================================================
# 精细化超参数配置
# =============================================================================

# 继承 consistency_flow 最优参数
BASE_PARAMS="--lr ${BEST_LR:-3e-4} --num_flow_steps ${BEST_NUM_FLOW_STEPS:-20} --obs_horizon 2 --act_horizon 8 --pred_horizon 16"
BASE_PARAMS="${BASE_PARAMS} --consistency_weight ${BEST_CONSISTENCY_WEIGHT:-0.3} --cons_delta_mode fixed --cons_delta_fixed ${BEST_CONS_DELTA_FIXED:-0.02}"

SWEEP_CONFIGS=(
    # === alpha 精细化 (reward_scale=0.1 固定) ===
    "${BASE_PARAMS} --alpha 0.0005 --reward_scale 0.1"
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.1"  # 基准
    "${BASE_PARAMS} --alpha 0.002 --reward_scale 0.1"
    "${BASE_PARAMS} --alpha 0.005 --reward_scale 0.1"
    
    # === reward_scale 精细化 (alpha=0.001 固定) ===
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.05"
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.08"
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.15"
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.2"
    
    # === q_lr (Q 网络学习率) 探索 ===
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.1 --q_lr 1e-4"
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.1 --q_lr 3e-4"
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.1 --q_lr 1e-3"
    
    # === target_update_freq 探索 ===
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.1 --target_update_freq 1"
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.1 --target_update_freq 2"
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.1 --target_update_freq 5"
    
    # === num_q_updates 探索 ===
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.1 --num_q_updates 1"
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.1 --num_q_updates 2"
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.1 --num_q_updates 4"
    
    # === 最佳组合候选 ===
    "${BASE_PARAMS} --alpha 0.002 --reward_scale 0.08 --q_lr 3e-4"
    "${BASE_PARAMS} --alpha 0.001 --reward_scale 0.1 --q_lr 3e-4 --num_q_updates 2"
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

print_info "CPQL 精细化 sweep 完成"
print_info "最优参数文件: ${LOG_DIR}/best_params_${ALGORITHM}.sh"
