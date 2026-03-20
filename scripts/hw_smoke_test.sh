#!/usr/bin/env bash
#
# 真机 Smoke Test 一键启动脚本
#
# 自动执行:
#   1. 预检 (preflight_check.py)
#   2. 启动 roscore (如未运行)
#   3. 启动 realsense 相机
#   4. SDK 级 mock 测试 (test_mock_arm_motion.py)
#   5. 全栈推理测试 (live_inference_test.py)
#   6. 收集日志
#
# Usage:
#   bash scripts/hw_smoke_test.sh --pretrain /path/to/model.pt
#   bash scripts/hw_smoke_test.sh --pretrain /path/to/model.pt --skip-mock
#   bash scripts/hw_smoke_test.sh --pretrain /path/to/model.pt --mock-only
#
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_VLA_ROOT="$(dirname "$SCRIPT_DIR")"
CARM_DEPLOY="$RL_VLA_ROOT/carm_ros_deploy/src/carm_deploy"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RL_VLA_ROOT/logs/hw_test_$TIMESTAMP"

# ── Defaults ──────────────────────────────────────────────────────────────
PRETRAIN=""
ROBOT_IP="10.42.0.101"
SAFETY_CONFIG="$CARM_DEPLOY/safety_config.json"
SKIP_MOCK=false
MOCK_ONLY=false
CONDA_ENV="carm"

# ── Colors ────────────────────────────────────────────────────────────────
RED='\033[91m'
GREEN='\033[92m'
YELLOW='\033[93m'
BLUE='\033[94m'
BOLD='\033[1m'
END='\033[0m'

info()  { echo -e "${BLUE}[INFO]${END} $*"; }
ok()    { echo -e "${GREEN}[OK]${END} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${END} $*"; }
err()   { echo -e "${RED}[ERROR]${END} $*"; }
header(){ echo -e "\n${BOLD}════════════════════════════════════════════════════════════${END}"; echo -e "${BOLD}  $*${END}"; echo -e "${BOLD}════════════════════════════════════════════════════════════${END}"; }

confirm() {
    local msg="$1"
    echo -en "\n${YELLOW}$msg [Y/n]: ${END}"
    read -r ans
    case "$ans" in
        n|N|no|NO) return 1 ;;
        *) return 0 ;;
    esac
}

# ── Parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrain)   PRETRAIN="$2"; shift 2 ;;
        --robot_ip)   ROBOT_IP="$2"; shift 2 ;;
        --safety_config) SAFETY_CONFIG="$2"; shift 2 ;;
        --skip-mock)  SKIP_MOCK=true; shift ;;
        --mock-only)  MOCK_ONLY=true; shift ;;
        --conda_env)  CONDA_ENV="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --pretrain /path/to/model.pt [options]"
            echo ""
            echo "Options:"
            echo "  --pretrain PATH       Policy checkpoint (required for full test)"
            echo "  --robot_ip IP         Robot IP (default: 10.42.0.101)"
            echo "  --safety_config PATH  Safety config JSON"
            echo "  --skip-mock           Skip SDK mock tests, go straight to full stack"
            echo "  --mock-only           Only run SDK mock tests, skip full stack"
            echo "  --conda_env NAME      Conda environment (default: carm)"
            exit 0
            ;;
        *) err "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Setup ─────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
info "日志目录: $LOG_DIR"
info "Robot IP: $ROBOT_IP"
info "Checkpoint: ${PRETRAIN:-'(none)'}"

# Cleanup function for background processes
PIDS_TO_KILL=()
cleanup() {
    info "Cleaning up background processes..."
    for pid in "${PIDS_TO_KILL[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

# ── Phase 1: Preflight ───────────────────────────────────────────────────
header "Phase 1: 预检"

PREFLIGHT_ARGS="--robot_ip $ROBOT_IP"
if [[ -n "$PRETRAIN" ]]; then
    PREFLIGHT_ARGS="$PREFLIGHT_ARGS --pretrain $PRETRAIN"
fi

if conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/preflight_check.py" $PREFLIGHT_ARGS 2>&1 | tee "$LOG_DIR/preflight.log"; then
    ok "预检通过"
else
    err "预检失败，请修复上述问题后重试"
    exit 1
fi

# ── Phase 2: ROS 环境 ────────────────────────────────────────────────────
if [[ "$SKIP_MOCK" == false ]] && [[ "$MOCK_ONLY" == true ]]; then
    info "Mock-only 模式，跳过 ROS 启动"
else
    header "Phase 2: ROS 环境"

    # Check roscore
    if rostopic list &>/dev/null; then
        ok "roscore 已运行"
    else
        info "启动 roscore..."
        roscore &>"$LOG_DIR/roscore.log" &
        PIDS_TO_KILL+=($!)
        sleep 2
        if rostopic list &>/dev/null; then
            ok "roscore 已启动"
        else
            err "roscore 启动失败"
            exit 1
        fi
    fi

    # Check camera
    if rostopic list 2>/dev/null | grep -q '/camera/color/image_raw'; then
        ok "相机 topic 已存在"
    else
        info "启动 realsense 相机..."
        roslaunch realsense2_camera rs_camera.launch &>"$LOG_DIR/camera.log" &
        PIDS_TO_KILL+=($!)
        sleep 5
        if rostopic list 2>/dev/null | grep -q '/camera/color/image_raw'; then
            ok "相机已启动"
        else
            warn "相机 topic 未出现，可能需要手动检查"
        fi
    fi
fi

# ── Phase 3: SDK Mock 测试 ───────────────────────────────────────────────
if [[ "$SKIP_MOCK" == false ]]; then
    header "Phase 3: SDK Mock 测试 (5 项)"

    if ! confirm "机械臂周围无障碍物，准备开始 SDK 测试？"; then
        warn "用户取消 SDK 测试"
    else
        MOCK_ARGS="--robot_ip $ROBOT_IP"
        if conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/test_mock_arm_motion.py" \
            $MOCK_ARGS --log_dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/mock_test.log"; then
            ok "SDK Mock 测试完成"
        else
            err "SDK Mock 测试失败"
            if [[ "$MOCK_ONLY" == true ]]; then
                exit 1
            fi
            if ! confirm "Mock 测试有失败项，是否继续全栈测试？"; then
                exit 1
            fi
        fi
    fi
fi

if [[ "$MOCK_ONLY" == true ]]; then
    header "完成"
    ok "Mock-only 模式，测试结束"
    ok "日志: $LOG_DIR"
    exit 0
fi

# ── Phase 4: 全栈推理测试 ────────────────────────────────────────────────
header "Phase 4: 全栈推理测试"

if [[ -z "$PRETRAIN" ]]; then
    warn "未指定 --pretrain，跳过全栈推理测试"
    warn "用法: $0 --pretrain /path/to/model.pt"
else
    if ! confirm "准备启动推理节点 + 联调测试？"; then
        warn "用户取消全栈测试"
    else
        # 提前做初始化确认（inference_ros.py 后台运行时 input() 无法显示）
        echo -e "\n${YELLOW}[安全确认] 推理节点启动后机械臂将移动到初始位姿:${END}"
        echo -e "  位姿: [0.2475, 0.0014, 0.3251, 0.9996, -0.0034, 0.0255, -0.0074]"
        echo -e "  速度: init_speed=2.0 (慢速)"
        echo -e "  控制模式: PF 模式 (mode=4)"
        if ! confirm "工作空间已清空，确认允许机械臂移动到初始位姿？"; then
            warn "用户取消"
        else
            info "启动推理节点 (后台)..."
            # --skip_init_confirm: 已在上方做了确认，跳过 inference_ros.py 内部的 input() 阻塞
            conda run -n "$CONDA_ENV" python "$CARM_DEPLOY/inference/inference_ros.py" \
                --pretrain "$PRETRAIN" \
                --robot_ip "$ROBOT_IP" \
                --safety_config "$SAFETY_CONFIG" \
                --log_dir "$LOG_DIR/inference_logs" \
                --skip_init_confirm \
                &>"$LOG_DIR/inference_node.log" &
            INFERENCE_PID=$!
            PIDS_TO_KILL+=($INFERENCE_PID)

            info "等待推理节点初始化 (10s)..."
            sleep 10

            if ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
                err "推理节点启动失败，查看日志: $LOG_DIR/inference_node.log"
                tail -20 "$LOG_DIR/inference_node.log"
                exit 1
            fi
            ok "推理节点已启动 (PID: $INFERENCE_PID)"

            # Run live test checklist
            info "启动联调检查表..."
            conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/live_inference_test.py" \
                --pretrain "$PRETRAIN" \
                --safety_config "$SAFETY_CONFIG" \
                2>&1 | tee "$LOG_DIR/live_test.log" || true

            # Stop inference node
            info "停止推理节点..."
            kill "$INFERENCE_PID" 2>/dev/null || true
            wait "$INFERENCE_PID" 2>/dev/null || true
            ok "推理节点已停止"
        fi
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────
header "测试完成"
ok "所有日志保存在: $LOG_DIR"
echo ""
echo "  日志文件:"
ls -1 "$LOG_DIR"/ 2>/dev/null | while read -r f; do
    echo "    $LOG_DIR/$f"
done
