#!/bin/bash
# =============================================================================
# Sweep: Reflected Flow (基础 IL 算法)
# 
# 扫描参数: boundary_reg_weight, reflection_mode
# 继承: flow_matching 的最优基础参数
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ALGORITHM="reflected_flow"
LOG_DIR="${SWEEP_ROOT}/stage1_base_il/${ALGORITHM}"
BASE_ALGO="flow_matching"
BASE_DIR="${SWEEP_ROOT}/stage1_base_il/${BASE_ALGO}"

# =============================================================================
# 超参数配置 (特有参数)
# =============================================================================

SWEEP_SPECIFIC_CONFIGS=(
    # === boundary_reg_weight 扫描 (hard mode) ===
    "--reflection_mode hard --boundary_reg_weight 0.001"
    "--reflection_mode hard --boundary_reg_weight 0.01"
    "--reflection_mode hard --boundary_reg_weight 0.1"
    
    # === soft mode ===
    "--reflection_mode soft --boundary_reg_weight 0.01"
    "--reflection_mode soft --boundary_reg_weight 0.1"
)

# =============================================================================
# 主逻辑
# =============================================================================

print_header "Reflected Flow 超参数扫描"
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
    print_warn "未找到 ${BASE_ALGO} 最优参数，使用默认值"
    INHERITED_PARAMS="--lr 3e-4 --num_flow_steps 10 --obs_horizon 2 --act_horizon 8 --pred_horizon 16"
fi

# 构建完整配置
SWEEP_CONFIGS=()
for specific in "${SWEEP_SPECIFIC_CONFIGS[@]}"; do
    SWEEP_CONFIGS+=("${INHERITED_PARAMS} ${specific}")
done

# 运行 sweep
run_sweep "${ALGORITHM}" "${LOG_DIR}" SWEEP_CONFIGS

print_info "Reflected Flow sweep 完成"
print_info "最优参数文件: ${LOG_DIR}/best_params_${ALGORITHM}.sh"
