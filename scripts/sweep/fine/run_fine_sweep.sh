#!/bin/bash
# =============================================================================
# 精细化 Sweep 主控脚本
# 
# 基于上一轮 sweep 结果，运行更深入的超参数搜索
# 优先级: reflected_flow (修复) > diffusion_policy (改进) > cpql > awcp > 
#         flow_matching > consistency_flow > shortcut_flow
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_fine.sh"

# =============================================================================
# 配置
# =============================================================================

# 所有待精细化的算法（按优先级排序）
# 排除 aw_shortcut_flow (已达 80%)
PRIORITY_ALGOS=(
    "reflected_flow"      # 全崩，需要修复
    "diffusion_policy"    # 5%，效果很差
    "cpql"                # 22%，中等
    "awcp"                # 7%，很差
    "flow_matching"       # 28%，基准
    "consistency_flow"    # 48%，较好
    "shortcut_flow"       # 34%，中等
)

# =============================================================================
# 参数解析
# =============================================================================

DRY_RUN=false
FORCE=false
ALGORITHM=""
LIST_ONLY=false

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --algorithm ALGO    只运行指定算法的精细化 sweep"
    echo "  --dry-run           只显示要运行的实验，不实际运行"
    echo "  --force             强制重新运行已完成的 sweep"
    echo "  --list              列出所有待运行的算法"
    echo "  --original-sweep DIR 指定上一轮 sweep 目录 (默认: logs/cascade_sweep_20260130)"
    echo "  -h, --help          显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 --list                        # 列出所有待精细化的算法"
    echo "  $0 --algorithm reflected_flow    # 只运行 reflected_flow"
    echo "  $0                               # 按优先级运行所有"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        --original-sweep)
            export ORIGINAL_SWEEP_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

# =============================================================================
# 主逻辑
# =============================================================================

print_header "精细化超参数扫描"
echo "原始 Sweep 目录: ${ORIGINAL_SWEEP_ROOT}"
echo "输出目录: ${FINE_SWEEP_ROOT}"

if [ "${LIST_ONLY}" = "true" ]; then
    echo ""
    echo "待精细化算法 (按优先级排序):"
    echo ""
    
    # 读取上一轮结果
    csv_file="${ORIGINAL_SWEEP_ROOT}/sweep_results.csv"
    if [ -f "${csv_file}" ]; then
        echo "算法                    上轮最佳    状态"
        echo "----------------------------------------"
        for algo in "${PRIORITY_ALGOS[@]}"; do
            # 从 CSV 提取该算法的最佳结果
            best=$(grep "${algo}" "${csv_file}" | awk -F',' '{if($17!="") print $17}' | sort -rn | head -1)
            if [ -z "$best" ]; then
                best="N/A (崩溃)"
            fi
            
            # 检查是否已完成精细化
            status="待运行"
            if [ -f "${FINE_SWEEP_ROOT}/${algo}/best_params_${algo}.sh" ]; then
                status="已完成"
            fi
            
            printf "%-24s %-10s %s\n" "${algo}" "${best}" "${status}"
        done
    else
        for algo in "${PRIORITY_ALGOS[@]}"; do
            echo "  - ${algo}"
        done
    fi
    exit 0
fi

# 构建要运行的算法列表
if [ -n "${ALGORITHM}" ]; then
    ALGOS_TO_RUN=("${ALGORITHM}")
else
    ALGOS_TO_RUN=("${PRIORITY_ALGOS[@]}")
fi

# 统计
total=${#ALGOS_TO_RUN[@]}
completed=0
failed=0

echo ""
echo "计划运行: ${total} 个算法"
echo ""

for algo in "${ALGOS_TO_RUN[@]}"; do
    script="${SCRIPT_DIR}/sweep_fine_${algo}.sh"
    
    if [ ! -f "${script}" ]; then
        print_warn "未找到脚本: ${script}，跳过"
        continue
    fi
    
    print_header "精细化 Sweep: ${algo}"
    
    # 构建参数
    args=""
    if [ "${DRY_RUN}" = "true" ]; then
        args="${args} --dry-run"
    fi
    if [ "${FORCE}" = "true" ]; then
        args="${args} --force"
    fi
    
    # 运行脚本
    bash "${script}" ${args}
    exit_code=$?
    
    if [ ${exit_code} -eq 0 ]; then
        completed=$((completed + 1))
        print_info "✅ ${algo} 完成"
    else
        failed=$((failed + 1))
        print_error "❌ ${algo} 失败 (退出码: ${exit_code})"
    fi
done

# =============================================================================
# 汇总
# =============================================================================

print_header "精细化 Sweep 汇总"
echo "总计: ${total} | 完成: ${completed} | 失败: ${failed}"
echo ""
echo "结果目录: ${FINE_SWEEP_ROOT}"

# 运行全局分析
if [ "${DRY_RUN}" != "true" ] && [ ${completed} -gt 0 ]; then
    print_info "运行全局分析..."
    python scripts/analyze_sweep.py \
        --log_dir "${FINE_SWEEP_ROOT}" \
        --recursive \
        --output_dir "${FINE_SWEEP_ROOT}"
    
    if [ -f "${FINE_SWEEP_ROOT}/sweep_report.md" ]; then
        print_info "最终报告: ${FINE_SWEEP_ROOT}/sweep_report.md"
    fi
fi

print_info "✅ 精细化 Sweep 完成"
