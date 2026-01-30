#!/bin/bash
# =============================================================================
# Sweep: ShortCut Flow (依赖 IL 算法)
# 
# 扫描参数: sc_self_consistency_k, sc_step_size_mode, sc_num_inference_steps
# 继承: flow_matching 的最优基础参数
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ALGORITHM="shortcut_flow"
LOG_DIR="${SWEEP_ROOT}/stage2_dependent_il/${ALGORITHM}"
BASE_ALGO="flow_matching"
BASE_DIR="${SWEEP_ROOT}/stage1_base_il/${BASE_ALGO}"

# =============================================================================
# 超参数配置 (特有参数)
# =============================================================================

SWEEP_SPECIFIC_CONFIGS=(
    # === sc_self_consistency_k 扫描 ===
    "--sc_self_consistency_k 0.0 --sc_step_size_mode fixed --sc_num_inference_steps 8"
    "--sc_self_consistency_k 0.1 --sc_step_size_mode fixed --sc_num_inference_steps 8"
    "--sc_self_consistency_k 0.25 --sc_step_size_mode fixed --sc_num_inference_steps 8"
    "--sc_self_consistency_k 0.5 --sc_step_size_mode fixed --sc_num_inference_steps 8"
    
    # === sc_num_inference_steps 扫描 (k=0.25) ===
    "--sc_self_consistency_k 0.25 --sc_step_size_mode fixed --sc_num_inference_steps 4"
    "--sc_self_consistency_k 0.25 --sc_step_size_mode fixed --sc_num_inference_steps 16"
    
    # === step_size_mode 扫描 ===
    "--sc_self_consistency_k 0.25 --sc_step_size_mode power2 --sc_num_inference_steps 8"
    "--sc_self_consistency_k 0.25 --sc_step_size_mode uniform --sc_num_inference_steps 8"
)

# =============================================================================
# 主逻辑
# =============================================================================

print_header "ShortCut Flow 超参数扫描"
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

print_info "ShortCut Flow sweep 完成"
print_info "最优参数文件: ${LOG_DIR}/best_params_${ALGORITHM}.sh"
