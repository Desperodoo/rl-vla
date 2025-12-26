#!/bin/bash
# -*- coding: utf-8 -*-
#
# 多进程推理系统启动脚本
# 需要在两个终端中分别启动控制节点和推理节点
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认参数
MODEL="X5"
INTERFACE="can0"
CHECKPOINT=""
DRY_RUN=false
MODE=""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "用法: $0 <mode> [options]"
    echo ""
    echo "模式:"
    echo "  control     启动控制节点 (需要硬件权限)"
    echo "  inference   启动推理节点 (需要 GPU)"
    echo "  test        运行测试"
    echo "  benchmark   运行性能基准测试"
    echo ""
    echo "选项:"
    echo "  -m, --model MODEL        机器人型号 (default: X5)"
    echo "  -i, --interface IFACE    CAN 接口 (default: can0)"
    echo "  -c, --checkpoint PATH    模型检查点路径"
    echo "  -d, --dry-run           模拟模式 (不连接真实硬件)"
    echo "  -h, --help              显示帮助"
    echo ""
    echo "示例:"
    echo "  # 启动控制节点 (终端 1)"
    echo "  $0 control -m X5 -i can0"
    echo ""
    echo "  # 启动推理节点 (终端 2)"
    echo "  $0 inference -c /path/to/checkpoint.pt"
    echo ""
    echo "  # 模拟模式测试"
    echo "  $0 control --dry-run"
    echo "  $0 inference -c /path/to/checkpoint.pt --dry-run"
    echo ""
    echo "  # 运行测试"
    echo "  $0 test"
    echo ""
    echo "  # 运行基准测试"
    echo "  $0 benchmark -c /path/to/checkpoint.pt"
}

# 解析参数
if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

MODE="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -i|--interface)
            INTERFACE="$2"
            shift 2
            ;;
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# 检查环境
check_env() {
    echo -e "${BLUE}检查环境...${NC}"
    
    # 检查 Python
    if ! command -v python &> /dev/null; then
        echo -e "${RED}错误: 未找到 Python${NC}"
        exit 1
    fi
    
    # 检查项目路径
    if [[ ! -f "$PROJECT_ROOT/inference/shared_state.py" ]]; then
        echo -e "${RED}错误: 未找到 inference 模块${NC}"
        echo "请确保从正确的项目目录运行此脚本"
        exit 1
    fi
    
    echo -e "${GREEN}环境检查通过${NC}"
}

# 启动控制节点
start_control() {
    check_env
    
    echo -e "${BLUE}启动控制节点...${NC}"
    echo -e "  模型: ${YELLOW}$MODEL${NC}"
    echo -e "  接口: ${YELLOW}$INTERFACE${NC}"
    echo -e "  模拟模式: ${YELLOW}$DRY_RUN${NC}"
    echo ""
    
    CMD="cd $PROJECT_ROOT && python -m inference.control_node"
    CMD="$CMD --model $MODEL"
    CMD="$CMD --interface $INTERFACE"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        CMD="$CMD --dry-run"
    fi
    
    echo -e "${GREEN}执行: $CMD${NC}"
    echo ""
    eval $CMD
}

# 启动推理节点
start_inference() {
    check_env
    
    if [[ -z "$CHECKPOINT" && "$DRY_RUN" != "true" ]]; then
        echo -e "${RED}错误: 推理模式需要指定检查点路径 (-c/--checkpoint)${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}启动推理节点...${NC}"
    echo -e "  检查点: ${YELLOW}$CHECKPOINT${NC}"
    echo -e "  模拟模式: ${YELLOW}$DRY_RUN${NC}"
    echo ""
    
    CMD="cd $PROJECT_ROOT && python -m inference.control_node"
    
    if [[ -n "$CHECKPOINT" ]]; then
        CMD="$CMD -c $CHECKPOINT"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        CMD="$CMD --dry-run"
    fi
    
    echo -e "${GREEN}执行: $CMD${NC}"
    echo ""
    eval $CMD
}

# 运行测试
run_test() {
    check_env
    
    echo -e "${BLUE}运行多进程架构测试...${NC}"
    echo ""
    
    cd "$PROJECT_ROOT"
    
    echo -e "${YELLOW}[1/3] 共享内存测试${NC}"
    python -m inference.tests.test_multiprocess --test shm
    
    echo ""
    echo -e "${YELLOW}[2/3] 多进程通信测试${NC}"
    python -m inference.tests.test_multiprocess --test mp
    
    echo ""
    echo -e "${YELLOW}[3/3] 安全机制测试${NC}"
    python -m inference.tests.test_multiprocess --test safety
    
    echo ""
    echo -e "${GREEN}所有测试完成!${NC}"
}

# 运行基准测试
run_benchmark() {
    check_env
    
    if [[ -z "$CHECKPOINT" ]]; then
        echo -e "${RED}错误: 基准测试需要指定检查点路径 (-c/--checkpoint)${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}运行基准测试...${NC}"
    echo -e "  检查点: ${YELLOW}$CHECKPOINT${NC}"
    echo ""
    
    cd "$PROJECT_ROOT"
    python -m inference.tests.benchmark -c "$CHECKPOINT" --flow-steps 5,10,15,20 --components
}

# 主逻辑
case "$MODE" in
    control)
        start_control
        ;;
    inference)
        start_inference
        ;;
    test)
        run_test
        ;;
    benchmark)
        run_benchmark
        ;;
    *)
        echo -e "${RED}未知模式: $MODE${NC}"
        usage
        exit 1
        ;;
esac
