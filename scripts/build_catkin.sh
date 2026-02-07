#!/bin/bash
# ============================================================
# CARM 项目 catkin 工作空间统一编译脚本
# 使用 carm conda 环境进行编译，确保 Python 版本一致
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CATKIN_WS="$PROJECT_ROOT/carm_ros_deploy"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  CARM Catkin Workspace Build Script${NC}"
echo -e "${BLUE}================================================${NC}"

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "carm" ]]; then
    echo -e "${YELLOW}当前不在 carm 环境，尝试激活...${NC}"
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate carm
    if [[ "$CONDA_DEFAULT_ENV" != "carm" ]]; then
        echo -e "${RED}无法激活 carm 环境，请手动执行: conda activate carm${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ 当前环境: $CONDA_DEFAULT_ENV${NC}"
echo -e "${GREEN}✓ Python: $(python --version)${NC}"

# 检查 ROS
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}加载 ROS Noetic...${NC}"
    source /opt/ros/noetic/setup.bash
fi
echo -e "${GREEN}✓ ROS: $ROS_DISTRO${NC}"

# 确保 catkin 依赖已安装
echo -e "${BLUE}检查 catkin 依赖...${NC}"
pip show empy >/dev/null 2>&1 || pip install empy==3.3.4
pip show catkin_pkg >/dev/null 2>&1 || pip install catkin_pkg
pip show rospkg >/dev/null 2>&1 || pip install rospkg
echo -e "${GREEN}✓ catkin 依赖已安装${NC}"

# 创建 carm_ros_deploy 目录结构（如果不存在）
if [ ! -d "$CATKIN_WS/src" ]; then
    echo -e "${YELLOW}创建 catkin 工作空间...${NC}"
    mkdir -p "$CATKIN_WS/src"
fi

# 检查 ROS 包是否存在
echo -e "${BLUE}检查 ROS 包...${NC}"
cd "$CATKIN_WS/src"

if [ ! -d "realsense-ros" ]; then
    echo -e "${RED}错误: realsense-ros 不存在于 carm_ros_deploy/src/${NC}"
    exit 1
fi
echo -e "${GREEN}✓ realsense-ros${NC}"

if [ ! -d "carm_deploy" ]; then
    echo -e "${RED}错误: carm_deploy 不存在于 carm_ros_deploy/src/${NC}"
    exit 1
fi
echo -e "${GREEN}✓ carm_deploy${NC}"

if [ -d "carm_api" ]; then
    echo -e "${GREEN}✓ carm_api${NC}"
fi

# 初始化 catkin 工作空间
if [ ! -f "$CATKIN_WS/src/CMakeLists.txt" ]; then
    cd "$CATKIN_WS"
    catkin_make --only-pkg-with-deps=
    echo -e "${GREEN}✓ 初始化 catkin 工作空间${NC}"
fi

# 编译
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  开始编译...${NC}"
echo -e "${BLUE}================================================${NC}"

cd "$CATKIN_WS"

# 清理选项
if [ "$1" == "--clean" ]; then
    echo -e "${YELLOW}清理编译目录...${NC}"
    rm -rf build devel
fi

# 执行 catkin_make
# arm_control_sdk 位于项目根目录，需要指定 CMAKE_PREFIX_PATH 以供 find_package 查找
# SDK 自带 Poco v71 (位于 arm_control_sdk/poco/lib)，需要同时加入链接搜索路径
SDK_DIR="$PROJECT_ROOT/arm_control_sdk"
SDK_POCO_LIB="$SDK_DIR/poco/lib"
export LD_LIBRARY_PATH="$SDK_POCO_LIB:$SDK_DIR/lib:${LD_LIBRARY_PATH:-}"
catkin_make -DPYTHON_EXECUTABLE=$(which python) \
  -DCMAKE_PREFIX_PATH="$SDK_DIR;$CMAKE_PREFIX_PATH" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$SDK_POCO_LIB -Wl,-rpath,$SDK_POCO_LIB"

echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}  编译完成!${NC}"
echo -e "${BLUE}================================================${NC}"

echo -e ""
echo -e "使用以下命令加载环境:"
echo -e "  ${YELLOW}source $CATKIN_WS/devel/setup.bash${NC}"
echo -e ""
