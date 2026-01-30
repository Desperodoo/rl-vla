#!/bin/bash
# =============================================================================
# 精细化 Sweep: Reflected Flow
# 
# 上一轮: 全部崩溃 (CUDA 错误) - 需要单独用串行模式重跑
# 
# 精细化方向:
# 1. reflection_mode: [hard, soft, adaptive]
# 2. boundary_reg_weight: [0.001, 0.01, 0.05, 0.1, 0.2]
# 3. 新参数: boundary_margin (边界距离)
# 4. 新参数: reflection_eps (反射精度)
# 5. 使用串行模式避免 GPU 资源争抢
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_fine.sh"

ALGORITHM="reflected_flow"
LOG_DIR="${FINE_SWEEP_ROOT}/${ALGORITHM}"

# 强制串行模式（避免 GPU 争抢）
export SERIAL_MODE=true
export MAX_RETRIES=2
export RETRY_WAIT=20

# 加载 flow_matching 最优参数
load_previous_best "flow_matching" "stage1_base_il"

# =============================================================================
# 精细化超参数配置
# =============================================================================

BASE_PARAMS="--lr ${BEST_LR:-3e-4} --num_flow_steps ${BEST_NUM_FLOW_STEPS:-20} --obs_horizon 2 --act_horizon 8 --pred_horizon 16"

SWEEP_CONFIGS=(
    # === reflection_mode 探索 ===
    "${BASE_PARAMS} --reflection_mode hard --boundary_reg_weight 0.01"
    "${BASE_PARAMS} --reflection_mode soft --boundary_reg_weight 0.01"
    "${BASE_PARAMS} --reflection_mode adaptive --boundary_reg_weight 0.01"
    
    # === boundary_reg_weight 探索 (soft 模式) ===
    "${BASE_PARAMS} --reflection_mode soft --boundary_reg_weight 0.001"
    "${BASE_PARAMS} --reflection_mode soft --boundary_reg_weight 0.005"
    "${BASE_PARAMS} --reflection_mode soft --boundary_reg_weight 0.01"
    "${BASE_PARAMS} --reflection_mode soft --boundary_reg_weight 0.05"
    "${BASE_PARAMS} --reflection_mode soft --boundary_reg_weight 0.1"
    
    # === boundary_reg_weight 探索 (hard 模式) ===
    "${BASE_PARAMS} --reflection_mode hard --boundary_reg_weight 0.001"
    "${BASE_PARAMS} --reflection_mode hard --boundary_reg_weight 0.01"
    "${BASE_PARAMS} --reflection_mode hard --boundary_reg_weight 0.05"
    
    # === boundary_margin 探索 ===
    "${BASE_PARAMS} --reflection_mode soft --boundary_reg_weight 0.01 --boundary_margin 0.01"
    "${BASE_PARAMS} --reflection_mode soft --boundary_reg_weight 0.01 --boundary_margin 0.05"
    "${BASE_PARAMS} --reflection_mode soft --boundary_reg_weight 0.01 --boundary_margin 0.1"
    
    # === reflection_eps 探索 ===
    "${BASE_PARAMS} --reflection_mode hard --boundary_reg_weight 0.01 --reflection_eps 1e-4"
    "${BASE_PARAMS} --reflection_mode hard --boundary_reg_weight 0.01 --reflection_eps 1e-3"
    "${BASE_PARAMS} --reflection_mode hard --boundary_reg_weight 0.01 --reflection_eps 1e-2"
)

# =============================================================================
# 主逻辑
# =============================================================================

print_fine_sweep_header "${ALGORITHM}"
check_demo_file

print_warn "⚠️ Reflected Flow 使用串行模式运行，避免 GPU 资源争抢"
print_warn "⚠️ 预计耗时较长，每个实验约 45 分钟"

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

print_info "Reflected Flow 精细化 sweep 完成"
print_info "最优参数文件: ${LOG_DIR}/best_params_${ALGORITHM}.sh"
