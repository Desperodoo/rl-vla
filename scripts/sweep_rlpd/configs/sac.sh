#!/bin/bash
# =============================================================================
# SAC (Soft Actor-Critic) - Hyperparameter Sweep Configs
# =============================================================================
# 
# SAC 是无需预训练的 online RL 基线算法
# 使用控制变量法：baseline + 单参数扫描
#
# Baseline 配置:
#   gamma=0.9, tau=0.005, init_temperature=1.0
#   num_qs=10, num_min_qs=2, online_ratio=0.5, reward_scale=1.0
#
# 扫描参数:
#   - gamma: 折扣因子 (0.9, 0.95, 0.99)
#   - tau: target network 软更新系数 (0.001, 0.005, 0.01)
#   - init_temperature: 初始熵温度 (0.1, 1.0, 10.0)
#   - num_qs: Q ensemble 数量 (2, 5, 10)
#   - online_ratio: online/offline 数据比例 (0.25, 0.5, 0.75, 1.0)
#   - reward_scale: 奖励缩放 (0.1, 1.0, 10.0)
#
# Format: "config_name:--param1 value1 --param2 value2"
# =============================================================================

# -----------------------------------------------------------------------------
# Baseline 超参数
# -----------------------------------------------------------------------------
BASELINE_GAMMA="0.9"
BASELINE_TAU="0.005"
BASELINE_INIT_TEMP="1.0"
BASELINE_NUM_QS="10"
BASELINE_NUM_MIN_QS="2"
BASELINE_ONLINE_RATIO="0.5"
BASELINE_REWARD_SCALE="1.0"

# 构建 baseline 参数字符串
_baseline="--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}"

# -----------------------------------------------------------------------------
# Sweep 配置数组
# 格式: "config_name:extra_args"
# -----------------------------------------------------------------------------
SWEEP_CONFIGS=(
    # ===== Baseline 配置（作为对照组）=====
    "baseline:${_baseline}"
    
    # ===== gamma 扫描（折扣因子）=====
    # gamma 越大，考虑的未来奖励越多
    "gamma_0.95:--gamma 0.95 --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}"
    "gamma_0.99:--gamma 0.99 --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}"
    
    # ===== tau 扫描（target network 软更新系数）=====
    # tau 越小，target network 更新越慢，训练越稳定
    "tau_0.001:--gamma ${BASELINE_GAMMA} --tau 0.001 --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}"
    "tau_0.01:--gamma ${BASELINE_GAMMA} --tau 0.01 --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}"
    
    # ===== init_temperature 扫描（探索程度）=====
    # temperature 越高，动作分布越随机，探索越多
    "init_temp_0.1:--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature 0.1 --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}"
    "init_temp_10.0:--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature 10.0 --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}"
    
    # ===== num_qs 扫描（Q ensemble 数量）=====
    # 更多 Q 网络可以减少过估计，但计算开销更大
    "num_qs_2:--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs 2 --num_min_qs 2 --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}"
    "num_qs_5:--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs 5 --num_min_qs 2 --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE}"
    
    # ===== online_ratio 扫描（online/offline 数据比例）=====
    # 1.0 表示纯 online，0.0 表示纯 offline
    "online_ratio_0.25:--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio 0.25 --reward_scale ${BASELINE_REWARD_SCALE}"
    "online_ratio_0.75:--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio 0.75 --reward_scale ${BASELINE_REWARD_SCALE}"
    "online_ratio_1.0:--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio 1.0 --reward_scale ${BASELINE_REWARD_SCALE}"
    
    # ===== reward_scale 扫描 =====
    # reward_scale 影响 Q 值的数量级
    "reward_scale_0.1:--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale 0.1"
    "reward_scale_10.0:--gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --init_temperature ${BASELINE_INIT_TEMP} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale 10.0"
)
