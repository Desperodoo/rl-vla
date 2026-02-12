#!/bin/bash
# =============================================================================
# AWSC (Advantage-Weighted ShortCut Flow) - Wave 2 Sweep: Actor Protection
# =============================================================================
#
# 基于 Wave 1 sweep 结果和 catastrophic forgetting 分析（86%→12%），
# Wave 2 聚焦新增的 Actor 保护机制和 Advantage 计算改进。
#
# ===== Wave 1 关键发现 → Wave 2 设计 =====
#
# 1. 主要问题：Actor 从失败 online 数据学习导致灾难性遗忘
#    → A 组：actor_policy_mode=success_only，Actor 只用成功数据
#
# 2. Q-weighting 保护不足（batch-relative advantage 无法区分 per-state 好坏）
#    → B 组：per_state_v advantage，标准 AWAC 方式
#
# 3. V(s) 估计精度影响 per_state_v 效果
#    → C 组：num_v_samples 采样数
#
# 4. 两个机制的独立/组合效果
#    → D 组：正交组合实验
#
# 5. 与 Wave 1 最佳超参组合
#    → E 组：在 Wave 1 最优参数基础上测试新机制
#
# ===== 扫描总览：5 组，14 configs =====
#
# A. Actor 数据源 (actor_policy_mode):       2 configs
# B. Advantage 计算 (advantage_mode):        2 configs
# C. V(s) 采样数 (num_v_samples):            2 configs
# D. 正交组合 (policy_mode × adv_mode):      3 configs
# E. Wave 1 最优 + 新机制:                    4 configs
# + Baseline (Wave 1 baseline, 对照):        1 config
# -------------------------------------------
# Total:                                     14 configs
#
# Format: "config_name:--param1 value1 --param2 value2"
# Note: pretrain_path + load_pretrain_critic 由 sweep.sh 根据 --mode 自动注入
# =============================================================================

# -----------------------------------------------------------------------------
# Baseline 超参数 (继承 Wave 1 baseline)
# -----------------------------------------------------------------------------
# AWSC 核心参数
BASELINE_AWSC_BETA="100.0"
BASELINE_AWSC_BC_WEIGHT="1.0"
BASELINE_AWSC_SHORTCUT_WEIGHT="0.3"
BASELINE_AWSC_SELF_CONSISTENCY_K="0.25"
BASELINE_AWSC_EMA_DECAY="0.9995"
BASELINE_AWSC_WEIGHT_CLIP="200.0"
BASELINE_AWSC_NUM_INFERENCE_STEPS="8"

# RL 通用参数
BASELINE_GAMMA="0.9"
BASELINE_TAU="0.005"
BASELINE_ONLINE_RATIO="0.5"
BASELINE_REWARD_SCALE="1.0"
BASELINE_NUM_QS="10"
BASELINE_NUM_MIN_QS="2"

# 学习率
BASELINE_LR_ACTOR="3e-4"
BASELINE_LR_CRITIC="3e-4"

# 新增参数默认值 (Wave 2 关注)
BASELINE_ACTOR_POLICY_MODE="all"
BASELINE_ADVANTAGE_MODE="batch_mean"
BASELINE_NUM_V_SAMPLES="4"

# 构建 baseline 参数字符串
_awsc_base="--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_ema_decay ${BASELINE_AWSC_EMA_DECAY} --awsc_weight_clip ${BASELINE_AWSC_WEIGHT_CLIP} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --lr_actor ${BASELINE_LR_ACTOR} --lr_critic ${BASELINE_LR_CRITIC} --actor_policy_mode ${BASELINE_ACTOR_POLICY_MODE} --awsc_advantage_mode ${BASELINE_ADVANTAGE_MODE} --awsc_num_v_samples ${BASELINE_NUM_V_SAMPLES}"

# 辅助函数
_make_config() {
    local overrides="$1"
    echo "${_awsc_base} ${overrides}"
}

# -----------------------------------------------------------------------------
# Sweep 配置数组
# 格式: "config_name:extra_args"
# 注意：pretrain_path + load_pretrain_critic 由 sweep.sh 根据 --mode 自动注入
# -----------------------------------------------------------------------------
SWEEP_CONFIGS=(
    # ===========================================================================
    # Baseline 配置（Wave 1 baseline 复现，作为 Wave 2 的对照组）
    # 使用 Wave 1 默认值：actor_policy_mode=all, advantage_mode=batch_mean
    # ===========================================================================
    "baseline:${_awsc_base}"

    # ===========================================================================
    # A 组：Actor 数据源 (actor_policy_mode)
    # ---------------------------------------------------------------------------
    # 核心假设：Actor 只用成功数据（demo + online success），避免从失败
    # rollout 中学习，是解决 catastrophic forgetting (86%→12%) 的关键。
    # Critic 仍用全部数据更新，保证 Q 值估计的全面性。
    # ===========================================================================

    # A1: success_only — Actor 保护核心实验
    # 只有 demo 和成功的在线 rollout 参与 actor 训练
    "policy_success_only:$(_make_config "--actor_policy_mode success_only")"

    # A2: success_only + 低 bc_weight — 测试保护机制下是否可以放松 BC 约束
    # 假设：success_only 提供数据层保护后，BC 权重可以适度降低让 RL 更自由
    "policy_success_bc_0.5:$(_make_config "--actor_policy_mode success_only --awsc_bc_weight 0.5")"

    # ===========================================================================
    # B 组：Advantage 计算模式 (awsc_advantage_mode)
    # ---------------------------------------------------------------------------
    # 核心假设：per_state_v 使用 A(s,a) = Q(s,a) - V(s) 代替
    # batch_mean 的 A(s,a) = Q(s,a) - mean(Q)，能更准确地区分
    # 每个状态下的好坏 action，避免 Q-weighting 的 state-independent 偏差。
    # ===========================================================================

    # B1: per_state_v — 标准 AWAC advantage
    # V(s) = E_{a'~π}[Q(s,a')]，使用 4 个采样
    "adv_per_state_v:$(_make_config "--awsc_advantage_mode per_state_v")"

    # B2: per_state_v + 高 beta — 测试更尖锐的 advantage weighting
    # 更准确的 advantage 允许使用更高 beta 而不崩溃
    "adv_per_state_v_beta200:$(_make_config "--awsc_advantage_mode per_state_v --awsc_beta 200.0")"

    # ===========================================================================
    # C 组：V(s) 采样数 (awsc_num_v_samples)
    # ---------------------------------------------------------------------------
    # per_state_v 模式下，V(s) 通过采样 K 个 action 估计。
    # 更多采样 → 更准确的 V(s)，但计算开销线性增长。
    # ===========================================================================

    # C1: 少量采样 (K=2) — 低开销，但 V(s) 方差大
    "v_samples_2:$(_make_config "--awsc_advantage_mode per_state_v --awsc_num_v_samples 2")"

    # C2: 大量采样 (K=8) — 更准确的 V(s)，开销约 2x
    "v_samples_8:$(_make_config "--awsc_advantage_mode per_state_v --awsc_num_v_samples 8")"

    # ===========================================================================
    # D 组：正交组合 (actor_policy_mode × advantage_mode)
    # ---------------------------------------------------------------------------
    # 两个机制解决不同层面的问题：
    #   - success_only: 数据层面 — 过滤失败数据
    #   - per_state_v:  算法层面 — 改善 advantage 估计
    # 测试独立效果 vs 组合效果
    # ===========================================================================

    # D1: 双重保护 = success_only + per_state_v
    # 最保守策略：数据过滤 + 精确 advantage
    "combined_success_perv:$(_make_config "--actor_policy_mode success_only --awsc_advantage_mode per_state_v")"

    # D2: 双重保护 + 低 beta — 保守策略下进一步降温
    # 假设：双重保护已足够，低 beta 让权重更均匀，避免过拟合少量高 Q 样本
    "combined_success_perv_beta50:$(_make_config "--actor_policy_mode success_only --awsc_advantage_mode per_state_v --awsc_beta 50.0")"

    # D3: 双重保护 + 高 online ratio — 测试保护机制下能否更激进利用 online 数据
    # 假设：有保护后，可以安全地增加 online 数据比例
    "combined_success_perv_or75:$(_make_config "--actor_policy_mode success_only --awsc_advantage_mode per_state_v --online_ratio 0.75")"

    # ===========================================================================
    # E 组：Wave 1 最优参数 + 新机制
    # ---------------------------------------------------------------------------
    # 将新机制与 Wave 1 中可能表现较好的超参组合交叉测试
    # 使用 Wave 1 中几个关键配置 + 新保护机制
    # ===========================================================================

    # E1: 保守策略（Wave 1 F1）+ success_only
    # Wave 1 conservative: low temp + strong BC + small lr + demo-heavy
    "w1_conservative_success:$(_make_config "--awsc_beta 50.0 --awsc_bc_weight 2.0 --lr_actor 1e-4 --lr_critic 1e-4 --online_ratio 0.25 --actor_policy_mode success_only")"

    # E2: 保守策略 + 双重保护
    "w1_conservative_combined:$(_make_config "--awsc_beta 50.0 --awsc_bc_weight 2.0 --lr_actor 1e-4 --lr_critic 1e-4 --online_ratio 0.25 --actor_policy_mode success_only --awsc_advantage_mode per_state_v")"

    # E3: Wave 3 知识 (Wave 1 F2) + success_only
    # Wave 1 wave3: optimal lr + medium temp + longer horizon
    "w1_wave3_success:$(_make_config "--awsc_beta 100.0 --lr_actor 2e-4 --lr_critic 2e-4 --gamma 0.95 --actor_policy_mode success_only")"

    # E4: Wave 3 知识 + 双重保护
    "w1_wave3_combined:$(_make_config "--awsc_beta 100.0 --lr_actor 2e-4 --lr_critic 2e-4 --gamma 0.95 --actor_policy_mode success_only --awsc_advantage_mode per_state_v")"
)
