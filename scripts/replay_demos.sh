#!/bin/bash
# =============================================================================
# ManiSkill Demo Replay 脚本
# 将原始轨迹数据 replay 成带观测 (RGB/State) 的训练数据集
# 使用 physx_cuda 后端，保存奖励（dense）
# =============================================================================

set -e

# 配置参数
TASK=${1:-"LiftPegUpright-v1"}
CONTROL_MODE=${2:-"pd_ee_delta_pose"}
NUM_ENVS=${3:-64}  # GPU 并行环境数量

# 数据路径
DEMO_DIR="${HOME}/.maniskill/demos/${TASK}/rl"
TRAJ_FILE="${DEMO_DIR}/trajectory.none.pd_ee_delta_pose.physx_cuda.h5"

echo "=========================================="
echo "ManiSkill Demo Replay"
echo "=========================================="
echo "任务: ${TASK}"
echo "控制模式: ${CONTROL_MODE}"
echo "GPU 并行环境数: ${NUM_ENVS}"
echo ""

# 检查原始轨迹文件是否存在
if [ ! -f "${TRAJ_FILE}" ]; then
    echo "错误: 原始轨迹文件不存在: ${TRAJ_FILE}"
    echo "请先运行 download_demos.sh 下载数据"
    exit 1
fi

# 函数: 执行 replay
replay_trajectory() {
    local obs_mode=$1
    local reward_mode=$2
    local output_suffix="${obs_mode}.${CONTROL_MODE}.physx_cuda"
    local output_file="${DEMO_DIR}/trajectory.${output_suffix}.h5"
    
    # 检查输出文件是否已存在
    # if [ -f "${output_file}" ]; then
    #     echo "文件已存在，跳过: ${output_file}"
    #     return 0
    # fi
    
    echo ""
    echo "----------------------------------------"
    echo "Replay: obs_mode=${obs_mode}, reward_mode=${reward_mode}"
    echo "输出: ${output_file}"
    echo "----------------------------------------"
    
    python -m mani_skill.trajectory.replay_trajectory \
        --traj-path "${TRAJ_FILE}" \
        -o "${obs_mode}" \
        -c "${CONTROL_MODE}" \
        -b "physx_cuda" \
        -n "${NUM_ENVS}" \
        --record-rewards \
        --reward-mode "${reward_mode}" \
        --use-first-env-state \
        --save-traj 
}

# Replay RGB 观测 + dense 奖励
echo ""
echo "=========================================="
echo "1. Replay RGB 观测 (dense 奖励)"
echo "=========================================="
replay_trajectory "rgb" "dense"

# Replay State 观测 + dense 奖励
echo ""
echo "=========================================="
echo "2. Replay State 观测 (dense 奖励)"
echo "=========================================="
replay_trajectory "state" "dense"

echo ""
echo "=========================================="
echo "Replay 完成!"
echo ""
echo "生成的数据集文件:"
ls -lh ${DEMO_DIR}/*.h5 2>/dev/null || echo "无文件"
echo ""
echo "下一步: 运行训练脚本"
echo "  bash scripts/run_all_algorithms.sh"
echo "=========================================="
