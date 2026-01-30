#!/bin/bash
# =============================================================================
# 精细化 Sweep: Shortcut Flow
# 
# 上一轮最佳: success_once=0.34, sc_self_consistency_k=0.25, sc_num_inference_steps=8, sc_step_size_mode=fixed
# 
# 精细化方向:
# 1. sc_self_consistency_k 在 0.25 附近: [0.15, 0.2, 0.25, 0.3, 0.35]
# 2. sc_num_inference_steps: [6, 8, 10, 12]
# 3. 新参数: sc_shortcut_size (shortcut 大小)
# 4. 新参数: sc_warmup_steps (预热步数)
# 5. 继承 flow_matching 的最优参数
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_fine.sh"

ALGORITHM="shortcut_flow"
LOG_DIR="${FINE_SWEEP_ROOT}/${ALGORITHM}"

# 加载上一轮最优参数
load_previous_best "${ALGORITHM}" "stage2_dependent_il"
load_previous_best "flow_matching" "stage1_base_il"

# =============================================================================
# 精细化超参数配置
# =============================================================================

BASE_PARAMS="--lr ${BEST_LR:-3e-4} --num_flow_steps ${BEST_NUM_FLOW_STEPS:-20} --obs_horizon 2 --act_horizon 8 --pred_horizon 16"

SWEEP_CONFIGS=(
    # === sc_self_consistency_k 精细化 (steps=8, fixed 模式) ===
    "${BASE_PARAMS} --sc_self_consistency_k 0.15 --sc_num_inference_steps 8 --sc_step_size_mode fixed"
    "${BASE_PARAMS} --sc_self_consistency_k 0.2 --sc_num_inference_steps 8 --sc_step_size_mode fixed"
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8 --sc_step_size_mode fixed"  # 基准
    "${BASE_PARAMS} --sc_self_consistency_k 0.3 --sc_num_inference_steps 8 --sc_step_size_mode fixed"
    "${BASE_PARAMS} --sc_self_consistency_k 0.35 --sc_num_inference_steps 8 --sc_step_size_mode fixed"
    
    # === sc_num_inference_steps 精细化 ===
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 6 --sc_step_size_mode fixed"
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 10 --sc_step_size_mode fixed"
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 12 --sc_step_size_mode fixed"
    
    # === step_size_mode 进一步探索 ===
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8 --sc_step_size_mode power2"
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8 --sc_step_size_mode uniform"
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8 --sc_step_size_mode exponential"
    
    # === sc_shortcut_size 探索 ===
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8 --sc_step_size_mode fixed --sc_shortcut_size 1"
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8 --sc_step_size_mode fixed --sc_shortcut_size 2"
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8 --sc_step_size_mode fixed --sc_shortcut_size 4"
    
    # === sc_warmup_steps 探索 ===
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8 --sc_step_size_mode fixed --sc_warmup_steps 0"
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8 --sc_step_size_mode fixed --sc_warmup_steps 1000"
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8 --sc_step_size_mode fixed --sc_warmup_steps 5000"
    
    # === 最佳组合候选 ===
    "${BASE_PARAMS} --sc_self_consistency_k 0.3 --sc_num_inference_steps 10 --sc_step_size_mode fixed"
    "${BASE_PARAMS} --sc_self_consistency_k 0.25 --sc_num_inference_steps 8 --sc_step_size_mode fixed --sc_shortcut_size 2 --sc_warmup_steps 1000"
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

print_info "Shortcut Flow 精细化 sweep 完成"
print_info "最优参数文件: ${LOG_DIR}/best_params_${ALGORITHM}.sh"
