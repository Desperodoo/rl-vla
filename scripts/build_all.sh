#!/bin/bash
# ============================================================================
# rl-vla 统一编译脚本
# ============================================================================
# 用法: ./scripts/build_all.sh [选项]
#
# 选项:
#   --clean       清理所有 build 目录后重新编译
#   --arx5-only   只编译 arx5-sdk
#   --skip-ros    跳过 LeRobot-Anything-U-Arm (ROS) 编译
#
# 示例:
#   ./scripts/build_all.sh                # 完整编译
#   ./scripts/build_all.sh --clean        # 清理后重新编译
#   ./scripts/build_all.sh --arx5-only    # 只编译 arx5-sdk
#
# ============================================================================
# 已知问题与解决方案 (踩坑记录)
# ============================================================================
#
# 【坑1】kdl_parser 依赖问题
#   问题: arx5-sdk 编译时报错 "kdl_parser/kdl_parser.hpp: No such file"
#   原因: arx5-sdk 需要 kdl_parser 库来解析 URDF 进行运动学计算
#   解决: 从 robostack-staging 安装 ros-humble-kdl-parser (不是系统 ROS):
#         mamba install ros-humble-kdl-parser -c robostack-staging -c conda-forge
#   注意: 不要尝试用系统 /opt/ros/noetic 的路径，会导致 glibc 冲突！
#
# 【坑2】pyparsing 版本冲突
#   问题: catkin_make 报错 "TypeError: 'type' object is not subscriptable"
#   原因: conda 的 ros-humble 包依赖新版 pyparsing (3.2.x)，
#         但 ROS Noetic 的 catkin_pkg 不兼容 list[str] 类型注解语法
#   解决: 降级 pyparsing 到兼容版本:
#         pip install 'pyparsing<3.1'
#   验证: python -c "import pyparsing; print(pyparsing.__version__)"  # 应为 3.0.x
#
# 【坑3】conda 与 ROS Noetic 的 Python 环境冲突
#   问题: conda 环境中的包可能与系统 ROS 冲突
#   原因: arx5-sdk 需要 robostack 的 ros-humble C++ 库，
#         LeRobot-Anything-U-Arm 需要系统 ROS Noetic
#   解决: 两者可以共存，关键是 pyparsing 版本要兼容 (见坑2)
#   说明: ros-humble 包只提供 C++ 库 (kdl_parser)，运行时不需要 ROS2 Python
#
# 【坑4】catkin_make 找不到包
#   问题: catkin_make 报错找不到某些 ROS 包
#   解决: 确保正确设置环境变量:
#         source /opt/ros/noetic/setup.bash
#         export CMAKE_PREFIX_PATH=/opt/ros/noetic:$CMAKE_PREFIX_PATH
#
# ============================================================================

set -e

# ========== 获取项目根目录 ==========
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=========================================="
echo "  rl-vla 编译脚本"
echo "=========================================="
echo ""
echo "项目根目录: $ROOT_DIR"
echo ""

# ========== 参数解析 ==========
CLEAN=false
ARX5_ONLY=false
SKIP_ROS=false

for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN=true
            ;;
        --arx5-only)
            ARX5_ONLY=true
            ;;
        --skip-ros)
            SKIP_ROS=true
            ;;
        --help|-h)
            head -50 "$0" | grep "^#" | sed 's/^# //'
            exit 0
            ;;
    esac
done

# ========== 检查 conda 环境 ==========
if [[ "$CONDA_DEFAULT_ENV" != "arx-py310" ]]; then
    echo "[错误] 请先激活 conda 环境:"
    echo "  conda activate arx-py310"
    exit 1
fi

echo "Conda 环境: $CONDA_DEFAULT_ENV ✓"

# ========== 检查关键依赖 ==========
echo "检查关键依赖..."

# 检查 kdl_parser
if ! python -c "import sys; sys.path.insert(0, '$CONDA_PREFIX/lib'); import ctypes; ctypes.CDLL('$CONDA_PREFIX/lib/libkdl_parser.so')" 2>/dev/null; then
    if [[ ! -f "$CONDA_PREFIX/lib/libkdl_parser.so" ]]; then
        echo ""
        echo "[错误] 缺少 kdl_parser 库"
        echo "  请运行: mamba install ros-humble-kdl-parser -c robostack-staging -c conda-forge"
        exit 1
    fi
fi
echo "  kdl_parser ✓"

# 检查 pyparsing 版本
PYPARSING_VERSION=$(python -c "import pyparsing; print(pyparsing.__version__)" 2>/dev/null || echo "未安装")
if [[ "$PYPARSING_VERSION" == "未安装" ]]; then
    echo "[错误] pyparsing 未安装"
    exit 1
elif [[ "$PYPARSING_VERSION" > "3.1" ]]; then
    echo ""
    echo "[警告] pyparsing 版本过高 ($PYPARSING_VERSION)，可能导致 catkin_make 失败"
    echo "  建议运行: pip install 'pyparsing<3.1'"
    echo ""
fi
echo "  pyparsing $PYPARSING_VERSION ✓"
echo ""

# ========== 1. 编译 arx5-sdk ==========
echo "=========================================="
echo "  Step 1: 编译 arx5-sdk"
echo "=========================================="

ARX5_DIR="$ROOT_DIR/arx5-sdk"
ARX5_BUILD="$ARX5_DIR/build"

if $CLEAN; then
    echo "清理 arx5-sdk/build..."
    rm -rf "$ARX5_BUILD"
fi

mkdir -p "$ARX5_BUILD"
cd "$ARX5_BUILD"

echo "运行 cmake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python)

echo "运行 make..."
make -j$(nproc)

echo ""
echo "✓ arx5-sdk 编译完成"
echo "  Python 模块: $ARX5_DIR/python/arx5_interface.cpython*.so"
echo ""

# 测试 Python 模块
echo "测试 Python 模块导入..."
cd "$ROOT_DIR"
if python -c "import sys; sys.path.insert(0, '$ARX5_DIR/python'); import arx5_interface; print('  arx5_interface 导入成功 ✓')"; then
    :
else
    echo "  [警告] arx5_interface 导入失败"
fi
echo ""

# 如果只编译 arx5-sdk，到此结束
if $ARX5_ONLY; then
    echo "=========================================="
    echo "  arx5-sdk 编译完成！"
    echo "=========================================="
    exit 0
fi

# ========== 2. 编译 LeRobot-Anything-U-Arm (ROS) ==========
if ! $SKIP_ROS; then
    echo "=========================================="
    echo "  Step 2: 编译 LeRobot-Anything-U-Arm (ROS)"
    echo "=========================================="

    LEROBOT_DIR="$ROOT_DIR/LeRobot-Anything-U-Arm"

    if [[ -f "/opt/ros/noetic/setup.bash" ]]; then
        source /opt/ros/noetic/setup.bash
        export CMAKE_PREFIX_PATH=/opt/ros/noetic:$CMAKE_PREFIX_PATH
        
        if $CLEAN; then
            echo "清理 LeRobot-Anything-U-Arm/build 和 devel..."
            rm -rf "$LEROBOT_DIR/build" "$LEROBOT_DIR/devel"
        fi
        
        cd "$LEROBOT_DIR"
        
        echo "运行 catkin_make..."
        # 参考: howtoplay.md Method 2
        catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        
        if [[ -f "$LEROBOT_DIR/devel/setup.bash" ]]; then
            echo ""
            echo "✓ LeRobot-Anything-U-Arm 编译完成"
            echo "  验证: source devel/setup.bash && rospack find uarm"
        fi
    else
        echo "[跳过] ROS Noetic 未安装"
        echo "       如果只需要运行推理脚本，不需要编译此包"
    fi
    echo ""
else
    echo "=========================================="
    echo "  [跳过] LeRobot-Anything-U-Arm (--skip-ros)"
    echo "=========================================="
    echo ""
fi

# ========== 3. 安装 rlft ==========
echo "=========================================="
echo "  Step 3: 安装 rlft (pip install -e)"
echo "=========================================="

RLFT_DIR="$ROOT_DIR/rlft"

cd "$RLFT_DIR/diffusion_policy"
echo "安装 diffusion_policy..."
pip install -e . --quiet 2>/dev/null || pip install -e .

if [[ -d "$RLFT_DIR/act" ]]; then
    cd "$RLFT_DIR/act"
    echo "安装 act..."
    pip install -e . --quiet 2>/dev/null || pip install -e .
fi

echo ""
echo "✓ rlft 安装完成"
echo ""

# ========== 完成 ==========
echo "=========================================="
echo "  ✓ 全部编译完成！"
echo "=========================================="
echo ""
echo "已完成:"
echo "  [1] arx5-sdk        - Python 模块: arx5_interface"
echo "  [2] LeRobot (ROS)   - catkin 工作空间"
echo "  [3] rlft            - diffusion_policy, act"
echo ""
echo "下一步:"
echo "  1. 加载环境: source scripts/setup_env.sh"
echo "  2. 测试导入: python -c \"import arx5_interface; print('OK')\""
echo "  3. 运行推理: python inference/arx5_inference.py --help"
echo ""
