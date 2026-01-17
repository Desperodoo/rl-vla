#!/bin/bash
# CARM 数据记录启动脚本
# 自动设置环境变量并运行 record_data_ros.py

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CARM Data Recording Launcher${NC}"
echo -e "${GREEN}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 设置 SDK 库路径
SDK_LIB_PATH="$WORKSPACE_ROOT/carm_demo/arm_control_sdk/lib"
POCO_LIB_PATH="$WORKSPACE_ROOT/carm_demo/arm_control_sdk/poco/lib"

if [ ! -d "$SDK_LIB_PATH" ]; then
    echo -e "${RED}Error: SDK lib path not found: $SDK_LIB_PATH${NC}"
    exit 1
fi

export LD_LIBRARY_PATH="$SDK_LIB_PATH:$POCO_LIB_PATH:$LD_LIBRARY_PATH"
echo -e "${YELLOW}LD_LIBRARY_PATH set${NC}"

# 解决 conda libffi 与系统库冲突问题
# 优先使用 conda 环境的 libffi
CONDA_LIBFFI="$HOME/miniconda3/envs/carm/lib/libffi.so.8"
if [ -f "$CONDA_LIBFFI" ]; then
    export LD_PRELOAD="$CONDA_LIBFFI"
    echo -e "${YELLOW}LD_PRELOAD set for libffi fix${NC}"
fi

# Source ROS
if [ -f "/opt/ros/noetic/setup.bash" ]; then
    source /opt/ros/noetic/setup.bash
    echo -e "${YELLOW}ROS Noetic sourced${NC}"
fi

# Source catkin workspace
CATKIN_SETUP="$WORKSPACE_ROOT/catkin_ws/devel/setup.bash"
if [ -f "$CATKIN_SETUP" ]; then
    source "$CATKIN_SETUP"
    echo -e "${YELLOW}Catkin workspace sourced${NC}"
fi

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate carm
echo -e "${YELLOW}Conda env 'carm' activated${NC}"

# 默认参数
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_ROOT/recorded_data}"
ROBOT_IP="${ROBOT_IP:-10.42.0.101}"

echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Output dir: $OUTPUT_DIR"
echo "  Robot IP: $ROBOT_IP"
echo ""

# 运行脚本
cd "$SCRIPT_DIR"
python record_data_ros.py \
    --output_dir "$OUTPUT_DIR" \
    --vis \
    --robot_ip "$ROBOT_IP" \
    "$@"
