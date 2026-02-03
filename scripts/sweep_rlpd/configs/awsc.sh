#!/bin/bash
# =============================================================================
# AWSC (Advantage-Weighted ShortCut Flow) - Hyperparameter Sweep Configs
# =============================================================================
#
# AWSC 支持两种模式：
#   1. 预训练模式 (pretrain): 从 ShortCut Flow checkpoint 初始化（推荐）
#   2. From scratch 模式 (scratch): 无预训练，直接训练
#
# 使用控制变量法：baseline + 单参数扫描
#
# Baseline 配置:
#   awsc_beta=100.0, awsc_bc_weight=1.0, awsc_shortcut_weight=0.3
#   awsc_self_consistency_k=0.25, awsc_advantage_threshold=-0.5
#   awsc_num_inference_steps=8, gamma=0.9, tau=0.005, online_ratio=0.5
#
# 扫描参数:
#   - awsc_beta: advantage 温度 (10, 50, 100, 200)
#   - awsc_bc_weight: flow matching loss 权重 (0.1, 0.5, 1.0, 2.0)
#   - awsc_shortcut_weight: shortcut consistency 权重 (0.0, 0.1, 0.3, 0.5)
#   - awsc_advantage_threshold: 样本过滤阈值 (-1.0, -0.5, 0.0)
#   - awsc_num_inference_steps: 推理步数 (4, 8, 16)
#   - online_ratio: online/offline 比例 (0.25, 0.5, 0.75)
#   - gamma: 折扣因子 (0.9, 0.95, 0.99)
#   - reward_scale: 奖励缩放 (0.1, 1.0, 10.0)
#
# Format: "config_name:--param1 value1 --param2 value2"
# =============================================================================

# -----------------------------------------------------------------------------
# Baseline 超参数
# -----------------------------------------------------------------------------
# AWSC 特有参数
BASELINE_AWSC_BETA="100.0"
BASELINE_AWSC_BC_WEIGHT="1.0"
BASELINE_AWSC_SHORTCUT_WEIGHT="0.3"
BASELINE_AWSC_SELF_CONSISTENCY_K="0.25"
BASELINE_AWSC_ADVANTAGE_THRESHOLD="-0.5"
BASELINE_AWSC_NUM_INFERENCE_STEPS="8"

# 通用 RL 超参数
BASELINE_GAMMA="0.9"
BASELINE_TAU="0.005"
BASELINE_ONLINE_RATIO="0.5"
BASELINE_REWARD_SCALE="1.0"
BASELINE_NUM_QS="10"
BASELINE_NUM_MIN_QS="2"

# 构建 baseline 参数字符串（不含 pretrain_path，由 sweep.sh 注入）
_awsc_baseline="--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"

# -----------------------------------------------------------------------------
# Sweep 配置数组
# 格式: "config_name:extra_args"
# 注意：pretrain_path 参数由 sweep.sh 根据 --mode 自动注入
# -----------------------------------------------------------------------------
SWEEP_CONFIGS=(
    # ===== Baseline 配置（作为对照组）=====
    "baseline:${_awsc_baseline}"
    
    # ===== awsc_beta 扫描（advantage 温度）=====
    # beta 越大，高 advantage 样本权重越高
    "beta_10:--awsc_beta 10.0 --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    "beta_50:--awsc_beta 50.0 --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    "beta_200:--awsc_beta 200.0 --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    
    # ===== awsc_bc_weight 扫描（flow matching loss 权重）=====
    "bc_weight_0.1:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight 0.1 --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    "bc_weight_0.5:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight 0.5 --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    "bc_weight_2.0:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight 2.0 --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    
    # ===== awsc_shortcut_weight 扫描（shortcut consistency 权重）=====
    "shortcut_0.0:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight 0.0 --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    "shortcut_0.1:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight 0.1 --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    "shortcut_0.5:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight 0.5 --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    
    # ===== awsc_advantage_threshold 扫描（样本过滤阈值）=====
    # 只有 advantage > threshold 的 online 样本用于 policy 训练
    "threshold_-1.0:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold -1.0 --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    "threshold_0.0:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold 0.0 --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    
    # ===== awsc_num_inference_steps 扫描（推理步数）=====
    "steps_4:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps 4 --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    "steps_16:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps 16 --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    
    # ===== online_ratio 扫描（online/offline 数据比例）=====
    "online_ratio_0.25:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio 0.25 --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    "online_ratio_0.75:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio 0.75 --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    
    # ===== gamma 扫描（折扣因子）=====
    # gamma 越大，考虑的未来奖励越多
    "gamma_0.95:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma 0.95 --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    "gamma_0.99:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma 0.99 --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    
    # ===== reward_scale 扫描 =====
    "reward_scale_0.1:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale 0.1 --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
    "reward_scale_10.0:--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_advantage_threshold ${BASELINE_AWSC_ADVANTAGE_THRESHOLD} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale 10.0 --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS}"
)
