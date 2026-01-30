#!/bin/bash
# =============================================================================
# 精细化 Sweep: Flow Matching
# 
# 上一轮最佳: success_once=0.28, lr=3e-4, num_flow_steps=20
# 
# 精细化方向:
# 1. lr 在 3e-4 附近细化: [2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4]
# 2. num_flow_steps 探索更大值: [15, 20, 25, 30, 40]
# 3. 新参数: sigma_min (flow matching 噪声下限)
# 4. 新参数: ema_decay (EMA 衰减率)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_fine.sh"

ALGORITHM="flow_matching"
LOG_DIR="${FINE_SWEEP_ROOT}/${ALGORITHM}"

# 加载上一轮最优参数
load_previous_best "${ALGORITHM}" "stage1_base_il"

# =============================================================================
# 精细化超参数配置
# =============================================================================

# 基础参数 (继承上一轮最优)
BASE_PARAMS="--obs_horizon 2 --act_horizon 8 --pred_horizon 16"

SWEEP_CONFIGS=(
    # === lr 精细化 (num_flow_steps=20 固定) ===
    "${BASE_PARAMS} --lr 2e-4 --num_flow_steps 20"
    "${BASE_PARAMS} --lr 2.5e-4 --num_flow_steps 20"
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 20"  # 基准
    "${BASE_PARAMS} --lr 3.5e-4 --num_flow_steps 20"
    "${BASE_PARAMS} --lr 4e-4 --num_flow_steps 20"
    
    # === num_flow_steps 精细化 (lr=3e-4 固定) ===
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 15"
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 25"
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 30"
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 40"
    
    # === sigma_min 探索 (噪声调度) ===
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 20 --sigma_min 0.001"
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 20 --sigma_min 0.01"
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 20 --sigma_min 0.1"
    
    # === EMA 衰减率探索 ===
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 20 --ema_decay 0.99"
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 20 --ema_decay 0.995"
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 20 --ema_decay 0.9995"
    
    # === 最佳组合候选 ===
    "${BASE_PARAMS} --lr 3e-4 --num_flow_steps 25 --sigma_min 0.01 --ema_decay 0.995"
    "${BASE_PARAMS} --lr 3.5e-4 --num_flow_steps 25 --ema_decay 0.995"
)

# =============================================================================
# 主逻辑
# =============================================================================

print_fine_sweep_header "${ALGORITHM}"
check_demo_file

# 检查是否已完成
if check_fine_sweep_completed "${ALGORITHM}"; then
    print_info "${ALGORITHM} 精细化 sweep 已完成，跳过"
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
            rm -rf "${LOG_DIR}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# 运行 sweep
run_sweep "${ALGORITHM}" "${LOG_DIR}" SWEEP_CONFIGS

print_info "Flow Matching 精细化 sweep 完成"
print_info "最优参数文件: ${LOG_DIR}/best_params_${ALGORITHM}.sh"
