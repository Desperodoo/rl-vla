#!/bin/bash
# =============================================================================
# CARM catkin 工作空间编译脚本
# =============================================================================
# 用法:
#   ./scripts/setup/build_catkin.sh            # 增量编译
#   ./scripts/setup/build_catkin.sh --clean    # 清理后重新编译
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CATKIN_WS="$PROJECT_ROOT/carm_ros_deploy"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  CARM Catkin 编译脚本${NC}"
echo -e "${BLUE}================================================${NC}"

# -------------------------
# 加载环境 (conda + ROS + SDK)
# -------------------------
# setup_carm_env.sh 需要 source 执行；这里只需要其中的 conda / ROS / SDK
# 但它会检测 BASH_SOURCE == $0，所以我们内联必要步骤

# Conda
if [[ "$CONDA_DEFAULT_ENV" != "carm" ]]; then
    for conda_sh in \
        ~/miniconda3/etc/profile.d/conda.sh \
        ~/anaconda3/etc/profile.d/conda.sh \
        ~/miniforge-pypy3/etc/profile.d/conda.sh \
        /opt/conda/etc/profile.d/conda.sh; do
        if [ -f "$conda_sh" ]; then
            source "$conda_sh"
            break
        fi
    done
    conda activate carm 2>/dev/null || {
        echo -e "${RED}无法激活 carm 环境，请先: conda create -n carm python=3.10${NC}"
        exit 1
    }
fi
echo -e "${GREEN}Conda:  $CONDA_DEFAULT_ENV$(python --version 2>/dev/null | sed 's/Python / (Python /')${GREEN})${NC}"

# ROS
if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/noetic/setup.bash 2>/dev/null || {
        echo -e "${RED}ROS Noetic 未安装${NC}"; exit 1
    }
fi
echo -e "${GREEN}ROS:    $ROS_DISTRO${NC}"

# SDK
SDK_DIR="$PROJECT_ROOT/arm_control_sdk"
if [ -f "$SDK_DIR/setup.bash" ]; then
    source "$SDK_DIR/setup.bash"
else
    echo -e "${RED}arm_control_sdk/setup.bash 未找到${NC}"; exit 1
fi

# -------------------------
# 检查依赖
# -------------------------
pip show empy  >/dev/null 2>&1 || pip install -q empy==3.3.4
pip show catkin_pkg >/dev/null 2>&1 || pip install -q catkin_pkg
pip show rospkg >/dev/null 2>&1 || pip install -q rospkg

# -------------------------
# 检查 ROS 包
# -------------------------
for pkg in realsense-ros carm_deploy; do
    if [ ! -d "$CATKIN_WS/src/$pkg" ]; then
        echo -e "${RED}错误: $pkg 不存在于 carm_ros_deploy/src/${NC}"; exit 1
    fi
done

# -------------------------
# 编译
# -------------------------
cd "$CATKIN_WS"

if [ "$1" == "--clean" ]; then
    echo -e "${YELLOW}清理 build/ devel/ ...${NC}"
    rm -rf build devel
fi

SDK_POCO_LIB="$SDK_DIR/poco/lib"

# 将 SDK 加入 CMAKE_PREFIX_PATH（使用环境变量让 catkin_make 自动处理）
export CMAKE_PREFIX_PATH="$SDK_DIR:${CMAKE_PREFIX_PATH:-}"

catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPYTHON_EXECUTABLE=$(which python) \
  -DCMAKE_EXE_LINKER_FLAGS="-L$SDK_POCO_LIB -Wl,-rpath,$SDK_POCO_LIB"

echo -e ""
echo -e "${GREEN}编译完成!${NC}"
echo -e "加载环境: ${YELLOW}source scripts/setup/setup_carm_env.sh${NC}"
echo -e ""
