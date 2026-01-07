#!/bin/bash
# RTC 推理系统启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 默认参数
MODEL="X5"
INTERFACE="can0"
CHECKPOINT=""
DRY_RUN=""
VERBOSE=""
RTC_MODE=""
USE_CPP="false"

# 帮助信息
show_help() {
    cat << EOF
RTC 推理系统启动脚本

用法: $0 [选项]

选项:
  -c, --checkpoint PATH   模型 checkpoint 路径 (必需)
  -m, --model MODEL       机械臂型号 (默认: X5)
  -i, --interface IFACE   CAN 接口 (默认: can0)
  -r, --rtc               启用 RTC 模式 (Phase B)
  --cpp                   使用 C++ 伺服 (需要编译)
  -d, --dry-run           模拟模式 (不连接真机)
  -v, --verbose           详细输出
  -h, --help              显示帮助

示例:
  # 正常运行 (Python 伺服)
  $0 -c /path/to/checkpoint.pt

  # 模拟模式
  $0 -c /path/to/checkpoint.pt --dry-run

  # 启用 RTC 模式
  $0 -c /path/to/checkpoint.pt --rtc

EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -i|--interface)
            INTERFACE="$2"
            shift 2
            ;;
        -r|--rtc)
            RTC_MODE="--rtc"
            shift
            ;;
        --cpp)
            USE_CPP="true"
            shift
            ;;
        -d|--dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查 checkpoint
if [[ -z "$CHECKPOINT" ]]; then
    print_error "请指定 checkpoint 路径: -c /path/to/checkpoint.pt"
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    print_error "Checkpoint 文件不存在: $CHECKPOINT"
    exit 1
fi

# 清理旧的共享内存
print_info "清理旧的共享内存..."
rm -f /dev/shm/rtc_keyframes 2>/dev/null || true

# 创建 tmux 会话
SESSION_NAME="rtc_inference"

# 检查是否已有会话
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    print_warn "会话 '$SESSION_NAME' 已存在，正在关闭..."
    tmux kill-session -t "$SESSION_NAME"
fi

print_info "创建 tmux 会话: $SESSION_NAME"

# 环境设置命令
ENV_SETUP="conda activate arx-py310 && \
export PYTHONPATH=\$PYTHONPATH:$PROJECT_DIR/arx5-sdk/python:$PROJECT_DIR:$PROJECT_DIR/rlft/diffusion_policy && \
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$PROJECT_DIR/arx5-sdk/lib/x86_64:\$CONDA_PREFIX/lib"

# 创建新会话，第一个窗口运行 Python 推理
HEADLESS_FLAG=""
[[ -n "$DRY_RUN" ]] && HEADLESS_FLAG="--headless"

tmux new-session -d -s "$SESSION_NAME" -n "inference" \
    "cd $PROJECT_DIR && $ENV_SETUP && python -m inference_rtc.python.inference_main \
        -c $CHECKPOINT $DRY_RUN $VERBOSE $HEADLESS_FLAG; \
     echo '按任意键退出...'; read"

# 等待 Python 进程创建共享内存
print_info "等待推理进程启动..."
sleep 3

# 第二个窗口运行伺服
if [[ "$USE_CPP" == "true" ]]; then
    # C++ 伺服 (需要编译)
    SERVO_BIN="$SCRIPT_DIR/build/servo_main"
    if [[ ! -f "$SERVO_BIN" ]]; then
        print_error "C++ 伺服程序未编译。请先运行:"
        print_error "  cd $SCRIPT_DIR && mkdir -p build && cd build && cmake .. && make"
        print_error "或者使用 Python 伺服 (不带 --cpp 选项)"
        tmux kill-session -t "$SESSION_NAME"
        exit 1
    fi
    
    tmux new-window -t "$SESSION_NAME" -n "servo" \
        "$SERVO_BIN --model $MODEL --interface $INTERFACE $RTC_MODE $VERBOSE ${DRY_RUN:+--dry-run}; \
         echo '按任意键退出...'; read"
else
    # Python 伺服 (默认)
    VERBOSE_FLAG=""
    DRY_RUN_FLAG=""
    [[ -n "$VERBOSE" ]] && VERBOSE_FLAG="-v"
    [[ -n "$DRY_RUN" ]] && DRY_RUN_FLAG="-d"
    
    tmux new-window -t "$SESSION_NAME" -n "servo" \
        "cd $PROJECT_DIR && $ENV_SETUP && python -m inference_rtc.python.servo_main \
            -m $MODEL -i $INTERFACE $DRY_RUN_FLAG $VERBOSE_FLAG; \
         echo '按任意键退出...'; read"
fi

# 附加到会话
print_info "启动完成! 正在附加到 tmux 会话..."
print_info "  切换窗口: Ctrl+b n (下一个) / Ctrl+b p (上一个)"
print_info "  退出会话: Ctrl+b d (detach)"
print_info "  关闭会话: tmux kill-session -t $SESSION_NAME"

tmux attach-session -t "$SESSION_NAME"
