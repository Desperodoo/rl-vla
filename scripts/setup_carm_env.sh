#!/bin/bash
# CARM 环境设置脚本
# 用法: source scripts/setup_carm_env.sh
# 注意: 必须使用 source 命令执行，不能直接 ./setup_carm_env.sh

# 检测是否通过 source 执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "❌ 错误: 请使用 source 命令执行此脚本!"
    echo ""
    echo "正确用法:"
    echo "  source scripts/setup_carm_env.sh"
    echo "  # 或"
    echo "  . scripts/setup_carm_env.sh"
    echo ""
    echo "从任意目录执行:"
    echo "  source ~/rl-vla/scripts/setup_carm_env.sh"
    exit 1
fi

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

# 激活 conda 环境
# 首先初始化 conda（如果还没初始化）
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

# 激活 carm 环境
if conda info --envs | grep -q "^carm "; then
    conda activate carm
    CONDA_STATUS="✓ conda 环境 'carm' 已激活"
else
    CONDA_STATUS="⚠ conda 环境 'carm' 不存在，请先创建"
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  CARM 环境设置完成"
echo "════════════════════════════════════════════════════"
echo "  RL_VLA_ROOT: $RL_VLA_ROOT"
echo "  $CONDA_STATUS"
echo "  ✓ ROS/catkin 工作区已加载"
echo "  ✓ LD_LIBRARY_PATH 已添加:"
echo "      - $CARM_SDK_LIB"
echo "      - $CARM_POCO_LIB"
echo "════════════════════════════════════════════════════"
echo ""
