#!/bin/bash
# =============================================================================
# 精细化 Sweep - 通用配置
# 
# 基于第一轮 sweep 的结果，在最优参数附近进行更细致的搜索
# 继承上一轮的最优参数，探索新的超参数维度
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# 精细化 Sweep 配置
export FINE_SWEEP_ROOT="${SWEEP_ROOT}/fine_sweep_$(date +%Y%m%d_%H%M)"
export ORIGINAL_SWEEP_ROOT="${ORIGINAL_SWEEP_ROOT:-logs/cascade_sweep_20260130}"

# =============================================================================
# 工具函数
# =============================================================================

# 加载上一轮 sweep 的最优参数
# 用法: load_previous_best <algorithm> <stage_name>
# 返回: 设置 PREVIOUS_BEST_PARAMS 环境变量
load_previous_best() {
    local algorithm=$1
    local stage_name=$2
    
    local params_file="${ORIGINAL_SWEEP_ROOT}/${stage_name}/${algorithm}/best_params_${algorithm}.sh"
    
    if [ ! -f "${params_file}" ]; then
        print_warn "未找到上一轮最优参数: ${params_file}"
        return 1
    fi
    
    print_info "加载上一轮最优参数: ${params_file}"
    
    # 读取并构建参数字符串
    PREVIOUS_BEST_PARAMS=""
    while IFS='=' read -r key value; do
        # 跳过注释和空行
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # 移除 BEST_ 前缀，转为小写
        local param_name=$(echo "${key#BEST_}" | tr '[:upper:]' '[:lower:]')
        # 移除引号
        value="${value%\"}"
        value="${value#\"}"
        
        # 导出为环境变量供后续使用
        export "BEST_${key#BEST_}=${value}"
        
        PREVIOUS_BEST_PARAMS="${PREVIOUS_BEST_PARAMS} --${param_name} ${value}"
    done < "${params_file}"
    
    export PREVIOUS_BEST_PARAMS
    echo "  参数: ${PREVIOUS_BEST_PARAMS}"
    return 0
}

# 检查精细化 sweep 是否已完成
check_fine_sweep_completed() {
    local algorithm=$1
    local fine_dir="${FINE_SWEEP_ROOT}/${algorithm}"
    
    if [ -f "${fine_dir}/best_params_${algorithm}.sh" ]; then
        return 0
    fi
    return 1
}

print_fine_sweep_header() {
    local algorithm=$1
    print_header "精细化 Sweep: ${algorithm}"
    echo "基于上一轮结果: ${ORIGINAL_SWEEP_ROOT}"
    echo "输出目录: ${FINE_SWEEP_ROOT}/${algorithm}"
}
