#!/bin/bash
# =============================================================================
# Sweep: Flow Matching (基础 IL 算法)
# 
# 扫描参数: lr, num_flow_steps, obs_horizon, act_horizon, pred_horizon, unet_dims
# 无依赖，可独立运行
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ALGORITHM="flow_matching"
LOG_DIR="${SWEEP_ROOT}/stage1_base_il/${ALGORITHM}"

# =============================================================================
# 超参数配置
# =============================================================================

SWEEP_CONFIGS=(
    # === 学习率扫描 (基准配置) ===
    "--lr 1e-4 --num_flow_steps 10 --obs_horizon 2 --act_horizon 8 --pred_horizon 16"
    "--lr 3e-4 --num_flow_steps 10 --obs_horizon 2 --act_horizon 8 --pred_horizon 16"
    "--lr 5e-4 --num_flow_steps 10 --obs_horizon 2 --act_horizon 8 --pred_horizon 16"
    "--lr 1e-3 --num_flow_steps 10 --obs_horizon 2 --act_horizon 8 --pred_horizon 16"
    
    # === 推理步数扫描 (lr=3e-4) ===
    "--lr 3e-4 --num_flow_steps 5 --obs_horizon 2 --act_horizon 8 --pred_horizon 16"
    "--lr 3e-4 --num_flow_steps 20 --obs_horizon 2 --act_horizon 8 --pred_horizon 16"
    
    # === Horizon 扫描 (lr=3e-4) ===
    "--lr 3e-4 --num_flow_steps 10 --obs_horizon 1 --act_horizon 4 --pred_horizon 8"
    "--lr 3e-4 --num_flow_steps 10 --obs_horizon 4 --act_horizon 16 --pred_horizon 32"
)

# =============================================================================
# 主逻辑
# =============================================================================

print_header "Flow Matching 超参数扫描"
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

# 运行 sweep
run_sweep "${ALGORITHM}" "${LOG_DIR}" SWEEP_CONFIGS

print_info "Flow Matching sweep 完成"
print_info "最优参数文件: ${LOG_DIR}/best_params_${ALGORITHM}.sh"
