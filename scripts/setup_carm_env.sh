#!/bin/bash
# CARM 环境设置脚本
# 用法: source scripts/setup_carm_env.sh

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RL_VLA_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 设置 CARM SDK 库路径
CARM_SDK_LIB="$RL_VLA_ROOT/carm_demo/arm_control_sdk/lib"
CARM_POCO_LIB="$RL_VLA_ROOT/carm_demo/arm_control_sdk/poco/lib"

# 添加到 LD_LIBRARY_PATH（避免重复添加）
if [[ ":$LD_LIBRARY_PATH:" != *":$CARM_SDK_LIB:"* ]]; then
    export LD_LIBRARY_PATH="$CARM_SDK_LIB:$LD_LIBRARY_PATH"
fi

if [[ ":$LD_LIBRARY_PATH:" != *":$CARM_POCO_LIB:"* ]]; then
    export LD_LIBRARY_PATH="$CARM_POCO_LIB:$LD_LIBRARY_PATH"
fi

# Source ROS 和 catkin 工作区
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
fi

if [ -f "$RL_VLA_ROOT/catkin_ws/devel/setup.bash" ]; then
    source "$RL_VLA_ROOT/catkin_ws/devel/setup.bash"
fi

echo "CARM 环境已设置:"
echo "  RL_VLA_ROOT: $RL_VLA_ROOT"
echo "  LD_LIBRARY_PATH 已添加:"
echo "    - $CARM_SDK_LIB"
echo "    - $CARM_POCO_LIB"
