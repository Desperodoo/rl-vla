#!/bin/bash
# ============================================
# CARM 机械臂开发环境配置脚本
# 使用方法: source carm_scripts/setup_carm_env.sh
# ============================================

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  CARM 机械臂开发环境配置"
echo "=========================================="

# 1. 激活 conda 环境
echo "[1/3] 激活 conda 环境 'carm'..."
if command -v conda &> /dev/null; then
    # 确保 conda 命令可用
    eval "$(conda shell.bash hook)"
    conda activate carm
    if [ $? -eq 0 ]; then
        echo "      ✓ conda 环境 'carm' 已激活"
        echo "      Python: $(which python)"
        echo "      版本: $(python --version)"
    else
        echo "      ✗ 激活 conda 环境失败，请先创建环境: conda create -n carm python=3.10"
        return 1
    fi
else
    echo "      ✗ conda 未安装或不可用"
    return 1
fi

# 2. 配置 arm_control_sdk 环境变量
echo "[2/3] 配置 arm_control_sdk 环境变量..."
SDK_SETUP="${PROJECT_ROOT}/carm_demo/arm_control_sdk/setup.bash"
if [ -f "$SDK_SETUP" ]; then
    source "$SDK_SETUP"
    echo "      ✓ arm_control_sdk 环境已配置"
    echo "      SDK_DIR: $arm_control_sdk_DIR"
else
    echo "      ✗ 找不到 setup.bash: $SDK_SETUP"
    return 1
fi

# 3. 配置 ROS1 环境 (可选)
echo "[3/3] 配置 ROS1 环境..."
if [ -f "/opt/ros/noetic/setup.bash" ]; then
    source /opt/ros/noetic/setup.bash
    echo "      ✓ ROS1 noetic 环境已配置"
    
    # 如果 carm_ros 工作空间已编译，也 source 它
    CARM_ROS_SETUP="${PROJECT_ROOT}/carm_demo/carm_ros/devel/setup.bash"
    if [ -f "$CARM_ROS_SETUP" ]; then
        source "$CARM_ROS_SETUP"
        echo "      ✓ carm_ros 工作空间已配置"
    else
        echo "      ! carm_ros 工作空间未编译，跳过"
    fi
else
    echo "      ! ROS1 noetic 未安装，跳过"
fi

echo ""
echo "=========================================="
echo "  环境配置完成！"
echo "=========================================="
echo ""
echo "机械臂连接信息:"
echo "  IP: 10.42.0.101"
echo "  端口: 8090"
echo ""
echo "快速测试命令:"
echo "  python carm_scripts/test_connection.py  # 测试连接"
echo "  python carm_scripts/test_motion.py      # 测试运动"
echo "  python carm_scripts/test_gripper.py     # 测试夹爪"
echo ""
