#!/bin/bash
# =============================================================================
# 精细化 Sweep: Diffusion Policy
# 
# 上一轮最佳: success_once=0.05, lr=1e-4, num_diffusion_iters=100
# 效果较差，需要大幅探索
# 
# 精细化方向:
# 1. lr 更细致搜索: [5e-5, 8e-5, 1e-4, 1.5e-4, 2e-4]
# 2. num_diffusion_iters: [50, 75, 100, 150, 200]
# 3. 新参数: ddpm vs ddim scheduler
# 4. 新参数: beta_schedule (linear, cosine, squaredcos_cap_v2)
# 5. 新参数: prediction_type (epsilon, sample, v_prediction)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_fine.sh"

ALGORITHM="diffusion_policy"
LOG_DIR="${FINE_SWEEP_ROOT}/${ALGORITHM}"

# 加载上一轮最优参数
load_previous_best "${ALGORITHM}" "stage1_base_il"

# =============================================================================
# 精细化超参数配置
# =============================================================================

BASE_PARAMS="--obs_horizon 2 --act_horizon 8 --pred_horizon 16"

SWEEP_CONFIGS=(
    # === lr 精细化 ===
    "${BASE_PARAMS} --lr 5e-5 --num_diffusion_iters 100"
    "${BASE_PARAMS} --lr 8e-5 --num_diffusion_iters 100"
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100"  # 基准
    "${BASE_PARAMS} --lr 1.5e-4 --num_diffusion_iters 100"
    "${BASE_PARAMS} --lr 2e-4 --num_diffusion_iters 100"
    
    # === num_diffusion_iters 精细化 ===
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 50"
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 75"
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 150"
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 200"
    
    # === beta_schedule 探索 ===
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100 --beta_schedule linear"
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100 --beta_schedule cosine"
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100 --beta_schedule squaredcos_cap_v2"
    
    # === prediction_type 探索 ===
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100 --prediction_type epsilon"
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100 --prediction_type sample"
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100 --prediction_type v_prediction"
    
    # === scheduler 类型 ===
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100 --scheduler_type ddpm"
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100 --scheduler_type ddim"
    
    # === EMA 探索 ===
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100 --ema_decay 0.99"
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100 --ema_decay 0.9999"
    
    # === 最佳组合候选 ===
    "${BASE_PARAMS} --lr 8e-5 --num_diffusion_iters 150 --beta_schedule cosine"
    "${BASE_PARAMS} --lr 1e-4 --num_diffusion_iters 100 --beta_schedule cosine --prediction_type v_prediction"
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

print_info "Diffusion Policy 精细化 sweep 完成"
print_info "最优参数文件: ${LOG_DIR}/best_params_${ALGORITHM}.sh"
