#!/bin/bash
# =============================================================================
# CARM 环境设置脚本
# =============================================================================
# 用法: source scripts/setup/setup_carm_env.sh
#
# 功能:
#   1. 激活 conda 环境 (carm)
#   2. 加载 ROS Noetic
#   3. 加载 CARM SDK (LD_LIBRARY_PATH + arm_control_sdk_DIR)
#   4. 加载 catkin 工作空间 (如果已编译)
#
# 注意: 必须使用 source 执行，不能直接 ./setup_carm_env.sh
# =============================================================================

# 检测是否通过 source 执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "错误: 请使用 source 命令执行此脚本"
    echo "  source scripts/setup/setup_carm_env.sh"
    exit 1
fi

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export RL_VLA_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# -------------------------
# 1. Conda 环境
# -------------------------
if [[ "$CONDA_DEFAULT_ENV" != "carm" ]]; then
    # 初始化 conda
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        source ~/anaconda3/etc/profile.d/conda.sh
    elif [ -f ~/miniforge-pypy3/etc/profile.d/conda.sh ]; then
        source ~/miniforge-pypy3/etc/profile.d/conda.sh
    elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
        source /opt/conda/etc/profile.d/conda.sh
    fi

    if conda info --envs 2>/dev/null | grep -q "^carm "; then
        conda activate carm
    else
        echo "警告: conda 环境 'carm' 不存在，请先创建: conda create -n carm python=3.10"
    fi
fi

# -------------------------
# 2. ROS Noetic
# -------------------------
if [ -z "$ROS_DISTRO" ]; then
    if [ -f /opt/ros/noetic/setup.bash ]; then
        source /opt/ros/noetic/setup.bash
    fi
fi

# -------------------------
# 3. CARM SDK
# -------------------------
SDK_DIR="$RL_VLA_ROOT/arm_control_sdk"
if [ -f "$SDK_DIR/setup.bash" ]; then
    source "$SDK_DIR/setup.bash"
else
    echo "警告: arm_control_sdk/setup.bash 未找到"
fi

# -------------------------
# 4. Catkin 工作空间
# -------------------------
if [ -f "$RL_VLA_ROOT/carm_ros_deploy/devel/setup.bash" ]; then
    source "$RL_VLA_ROOT/carm_ros_deploy/devel/setup.bash"
fi

# -------------------------
# 输出摘要
# -------------------------
echo ""
echo "═══════════════════════════════════════════"
echo "  CARM 环境就绪"
echo "═══════════════════════════════════════════"
echo "  项目:   $RL_VLA_ROOT"
echo "  Conda:  ${CONDA_DEFAULT_ENV:-未激活}"
echo "  Python: $(python --version 2>/dev/null || echo '未找到')"
echo "  ROS:    ${ROS_DISTRO:-未加载}"
echo "  SDK:    ${arm_control_sdk_DIR:-未加载}"
if [ -f "$RL_VLA_ROOT/carm_ros_deploy/devel/setup.bash" ]; then
    echo "  Catkin: 已加载"
else
    echo "  Catkin: 未编译 (运行 ./scripts/setup/build_catkin.sh)"
fi
echo "═══════════════════════════════════════════"
echo ""
