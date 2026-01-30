#!/bin/bash
# =============================================================================
# Sweep: AW-ShortCut Flow (Offline RL 算法)
# 
# 扫描参数: beta, sc_self_consistency_k, reward_scale
# 继承: shortcut_flow 的最优参数
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ALGORITHM="aw_shortcut_flow"
LOG_DIR="${SWEEP_ROOT}/stage3_offline_rl/${ALGORITHM}"
BASE_ALGO="shortcut_flow"
BASE_DIR="${SWEEP_ROOT}/stage2_dependent_il/${BASE_ALGO}"

# =============================================================================
# 超参数配置 (特有参数)
# =============================================================================

SWEEP_SPECIFIC_CONFIGS=(
    # === beta 扫描 (advantage temperature) ===
    "--beta 1.0 --reward_scale 0.1"
    "--beta 5.0 --reward_scale 0.1"
    "--beta 10.0 --reward_scale 0.1"
    "--beta 50.0 --reward_scale 0.1"
    "--beta 100.0 --reward_scale 0.1"
    
    # === reward_scale 扫描 (beta=10) ===
    "--beta 10.0 --reward_scale 0.01"
    "--beta 10.0 --reward_scale 1.0"
    
    # === 组合: 高 beta + 不同 reward_scale ===
    "--beta 50.0 --reward_scale 0.01"
    "--beta 50.0 --reward_scale 1.0"
)

# =============================================================================
# 主逻辑
# =============================================================================

print_header "AW-ShortCut Flow 超参数扫描"
check_demo_file

# 检查是否已完成
if check_stage_completed "${LOG_DIR}" "${ALGORITHM}"; then
    print_info "${ALGORITHM} 已完成，跳过"
    cat "${LOG_DIR}/best_params_${ALGORITHM}.sh"
    exit 0
fi

# 解析参数
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
        *)
            shift
            ;;
    esac
done

# 加载基础算法的最优参数
INHERITED_PARAMS=""
if [ -f "${BASE_DIR}/best_params_${BASE_ALGO}.sh" ]; then
    INHERITED_PARAMS=$(build_inherited_params "${BASE_ALGO}" "${BASE_DIR}")
    print_info "继承 ${BASE_ALGO} 参数: ${INHERITED_PARAMS}"
else
    # 回退到 flow_matching
    BASE_DIR="${SWEEP_ROOT}/stage1_base_il/flow_matching"
    if [ -f "${BASE_DIR}/best_params_flow_matching.sh" ]; then
        INHERITED_PARAMS=$(build_inherited_params "flow_matching" "${BASE_DIR}")
        print_warn "使用 flow_matching 参数: ${INHERITED_PARAMS}"
    else
        print_warn "未找到上游最优参数，使用默认值"
        INHERITED_PARAMS="--lr 3e-4 --obs_horizon 2 --act_horizon 8 --pred_horizon 16"
    fi
    # 添加 shortcut 默认参数
    INHERITED_PARAMS="${INHERITED_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8"
fi

# 构建完整配置
SWEEP_CONFIGS=()
for specific in "${SWEEP_SPECIFIC_CONFIGS[@]}"; do
    SWEEP_CONFIGS+=("${INHERITED_PARAMS} ${specific}")
done

# 运行 sweep
run_sweep "${ALGORITHM}" "${LOG_DIR}" SWEEP_CONFIGS

print_info "AW-ShortCut Flow sweep 完成"
print_info "最优参数文件: ${LOG_DIR}/best_params_${ALGORITHM}.sh"
