#!/bin/bash
# =====================================================================
# DSRL Pipeline 测试运行脚本
#
# 分层测试设计，按依赖关系由低到高执行：
#   L0 - 导入冒烟测试（无 GPU）
#   L1 - 组件单元测试（CPU）
#   L2 - Checkpoint 加载（需要 checkpoint）
#   L3 - Flow Policy 集成（需要 checkpoint + GPU）
#   L4 - 环境集成（需要 checkpoint + GPU + ManiSkill3）
#   L5 - 端到端评估（需要 checkpoint + GPU + ManiSkill3）
#   L6 - 训练冒烟测试（需要 checkpoint + GPU + ManiSkill3，~5-10min）
#
# 使用方法:
#   cd /home/lizh/rl-vla
#   bash rlft/tests/dsrl/run_tests.sh          # 运行全部
#   bash rlft/tests/dsrl/run_tests.sh L0 L1    # 仅运行 L0+L1
#   bash rlft/tests/dsrl/run_tests.sh L5       # 仅运行 L5
# =====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate carm

echo "================================================================"
echo " DSRL Pipeline Test Suite"
echo " Project root: $PROJECT_ROOT"
echo " Python:       $(python --version 2>&1)"
echo " PyTorch:      $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo " CUDA avail:   $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "================================================================"

# 解析要运行的级别
if [ $# -eq 0 ]; then
    LEVELS="L0 L1 L2 L3 L4 L5 L6"
else
    LEVELS="$@"
fi

PASSED=0
FAILED=0
SKIPPED=0

run_level() {
    local level="$1"
    local desc="$2"
    local cmd="$3"

    echo ""
    echo "================================================================"
    echo " [$level] $desc"
    echo "================================================================"

    if eval "$cmd"; then
        echo "  [$level] ✓ PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  [$level] ✗ FAILED"
        FAILED=$((FAILED + 1))
    fi
}

for level in $LEVELS; do
    case $level in
        L0)
            run_level "L0" "导入与冒烟测试" \
                "python -m pytest rlft/tests/dsrl/test_L0_imports.py -v --tb=short"
            ;;
        L1)
            run_level "L1" "组件单元测试" \
                "python -m pytest rlft/tests/dsrl/test_L1_unit.py -v --tb=short"
            ;;
        L2)
            run_level "L2" "Checkpoint 加载测试" \
                "python -m pytest rlft/tests/dsrl/test_L2_checkpoint.py -v -s --tb=short"
            ;;
        L3)
            run_level "L3" "Flow Policy 集成测试" \
                "python -m pytest rlft/tests/dsrl/test_L3_flow.py -v -s --tb=short"
            ;;
        L4)
            run_level "L4" "环境集成测试" \
                "python -m pytest rlft/tests/dsrl/test_L4_env.py -v -s --tb=short"
            ;;
        L5)
            run_level "L5" "端到端评估测试" \
                "python rlft/tests/dsrl/test_L5_eval.py"
            ;;
        L6)
            run_level "L6" "训练冒烟测试" \
                "python rlft/tests/dsrl/test_L6_train.py"
            ;;
        *)
            echo "Unknown level: $level (valid: L0 L1 L2 L3 L4 L5 L6)"
            SKIPPED=$((SKIPPED + 1))
            ;;
    esac
done

# 汇总
echo ""
echo "================================================================"
echo " Test Summary"
echo "================================================================"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo "================================================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
