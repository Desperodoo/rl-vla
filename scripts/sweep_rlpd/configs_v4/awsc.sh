#!/bin/bash
# =============================================================================
# AWSC (Advantage-Weighted ShortCut Flow) - Wave 4 Sweep: Combined Optimals + Extended Training
# =============================================================================
#
# 基于 Wave 3 的 100K vs 250K 综合深度分析，Wave 4 有三个核心目标：
#
# 1. 组合 v3 发现的三个最优个体超参，验证其叠加效果
#    v3 三冠: beta=80 (peak s_end=0.72), or=0.15 (综合冠军), K=2 (隐式正则化)
#    这三个超参分别独立表现最佳，但从未同时组合过
#    → A 组：组合实验
#
# 2. 在 v3 冠军 (or_0.15) 基础上进行精细邻域搜索
#    → B/C/D 组：beta/LR/OR 邻域搜索
#
# 3. 延长训练至 500K 步（建议），验证长训练趋势
#    v3 250K 数据显示 or_0.15 在 250K 时 s_end 仍在上升
#    → 所有配置建议配合 TOTAL_TIMESTEPS=500000 运行
#
# ===== Wave 3 关键发现（指导 Wave 4 设计） =====
#
# 1. 100K 排序被 250K 完全颠覆：lr_2e4 从冠军跌至中游
#    → 训练预算显著影响最优超参，短训练结论不可靠
#
# 2. beta_80 在 250K 达到 s_end=0.72（历史最高），但振荡严重
#    → 高 beta 在 critic 成熟后实现精确 exploitation，但需稳定机制
#
# 3. or_0.15 是唯一在 250K 时 s_end 仍在上升的配置（0.60，综合冠军）
#    → 更重 demo 提供更强的策略锚定
#
# 4. v_samples_2 (K=2): s_end=0.60，隐式正则化使 advantage 权重更均匀
#    → K 小 → Var[V(s)] 大 → 权重更温和 → 更稳定
#
# 5. 消融揭示：success_only 无用甚至有害，per_state_v 提供晚期稳定性
#    - ablation_all_perv (all+perv): final_se=0.56（稳定）
#    - ablation_all_bm  (all+bm):   final_se=0.38（下滑！）
#    - ablation_so_bm   (so+bm):    best_se=0.44（最差）
#    → v4 基线改为 all+perv（去掉 success_only，保留 per_state_v）
#
# 6. LR 最优点随训练预算移动：50K→高LR, 100K→2e-4, 250K→1e-4, 500K→?
#    → 需要探索 5e-5 ~ 1e-4 区间
#
# ===== 扫描总览：4 组，12 configs =====
#
# A. 最优超参组合实验:    4 configs  (核心实验)
# B. Beta 精细搜索:       3 configs
# C. LR 500K 优化:        2 configs
# D. OR 精细搜索:          2 configs
# + Baseline (v3 冠军改良): 1 config
# -------------------------------------------
# Total:                   12 configs
#
# Format: "config_name:--param1 value1 --param2 value2"
# Note: pretrain_path + load_pretrain_critic 由 sweep.sh 根据 --mode 自动注入
# =============================================================================

# -----------------------------------------------------------------------------
# Wave 4 Baseline 超参数
# = v3 冠军 (or_0.15) + 消融洞察 (all+perv 优于 so+perv)
#
# 变化 vs v3 baseline:
#   actor_policy_mode: success_only → all  (消融证明 so 无用/有害)
#   online_ratio:      0.25 → 0.15         (v3 冠军配置)
#   其他参数不变
# -----------------------------------------------------------------------------

# AWSC 核心参数
BASELINE_AWSC_BETA="50.0"
BASELINE_AWSC_BC_WEIGHT="2.0"
BASELINE_AWSC_SHORTCUT_WEIGHT="0.3"
BASELINE_AWSC_SELF_CONSISTENCY_K="0.25"
BASELINE_AWSC_EMA_DECAY="0.9995"
BASELINE_AWSC_WEIGHT_CLIP="200.0"
BASELINE_AWSC_NUM_INFERENCE_STEPS="8"

# RL 通用参数
BASELINE_GAMMA="0.9"
BASELINE_TAU="0.005"
BASELINE_ONLINE_RATIO="0.15"       # ← v3 冠军设定 (was 0.25)
BASELINE_REWARD_SCALE="1.0"
BASELINE_NUM_QS="10"
BASELINE_NUM_MIN_QS="2"

# 学习率
BASELINE_LR_ACTOR="1e-4"
BASELINE_LR_CRITIC="1e-4"

# Actor/Advantage 模式 (基于 v3 消融结论)
BASELINE_ACTOR_POLICY_MODE="all"           # ← 消融证明 all ≥ success_only
BASELINE_ADVANTAGE_MODE="per_state_v"      # ← 消融证明 perv 提供晚期稳定性
BASELINE_NUM_V_SAMPLES="4"

# Success 判定
BASELINE_SUCCESS_CRITERIA="success_once"

# 构建 baseline 参数字符串
_awsc_base="--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_ema_decay ${BASELINE_AWSC_EMA_DECAY} --awsc_weight_clip ${BASELINE_AWSC_WEIGHT_CLIP} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --lr_actor ${BASELINE_LR_ACTOR} --lr_critic ${BASELINE_LR_CRITIC} --actor_policy_mode ${BASELINE_ACTOR_POLICY_MODE} --awsc_advantage_mode ${BASELINE_ADVANTAGE_MODE} --awsc_num_v_samples ${BASELINE_NUM_V_SAMPLES} --success_criteria ${BASELINE_SUCCESS_CRITERIA}"

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
    # Baseline = v3 冠军 (or_0.15) 改良版
    # 改动: actor_policy_mode → all (去掉无效的 success_only)
    # 参数: beta=50, bc=2.0, lr=1e-4, or=0.15, K=4, all+perv
    # ===========================================================================
    "baseline:${_awsc_base}"

    # ===========================================================================
    # A 组：最优超参组合实验 — THE HEADLINE EXPERIMENT
    # ---------------------------------------------------------------------------
    # v3 三冠: beta=80 (peak), or=0.15 (稳定), K=2 (正则化)
    # 这三个超参各自独立最优，从未同时组合。
    #
    # 核心假设：
    #   beta=80 提供精确 exploitation (critic 成熟后利刃效应)
    #   or=0.15 提供策略锚定 (85% demo 防止遗忘)
    #   K=2 提供隐式权重正则化 (Var[V(s)] 大 → 权重更均匀)
    #   三者机制互补，可能获得 s_end>0.72 且 s_once>0.80
    # ===========================================================================

    # A1: 三冠合一 — beta=80 + K=2 (or=0.15 已在 baseline)
    # 预期: 最高 s_end (>0.72?) + 比纯 beta_80 更稳定
    "combo_full:$(_make_config "--awsc_beta 80.0 --awsc_num_v_samples 2")"

    # A2: 只加 beta=80 — 测试 beta_80 在 or=0.15 baseline 上的效果
    # v3 的 beta_80 运行在 or=0.25 上; 在 or=0.15 上是否更稳定?
    "combo_beta80:$(_make_config "--awsc_beta 80.0")"

    # A3: 只加 K=2 — 测试 K=2 在 or=0.15 baseline 上的效果
    # v3 的 v_samples_2 运行在 or=0.25 上; 在 or=0.15 上是否更好?
    "combo_k2:$(_make_config "--awsc_num_v_samples 2")"

    # A4: 三冠合一但 or=0.25 — 消融 or: 组合中 0.15 vs 0.25 的差异
    # 如果 ≈ combo_full → or 在组合中不关键
    # 如果 << combo_full → or=0.15 的策略锚定在组合中也至关重要
    "combo_or25:$(_make_config "--awsc_beta 80.0 --awsc_num_v_samples 2 --online_ratio 0.25")"

    # ===========================================================================
    # B 组：Beta 精细搜索
    # ---------------------------------------------------------------------------
    # v3 发现: beta_80 > beta_50 > beta_30 (250K peak)
    # 但 beta_80 振荡严重 (0.72 → 0.56)。
    # 在 80 两侧探索: 是否有 peak 更高且更稳定的 beta?
    # Base: or=0.15, lr=1e-4, bc=2.0, K=4, all+perv
    # ===========================================================================

    # B1: beta=60 — 50 和 80 之间，可能兼顾稳定性和 exploitation
    "beta_60:$(_make_config "--awsc_beta 60.0")"

    # B2: beta=100 — 超过 80，更极端的 exploitation
    # v2 使用 beta=100 作默认 (表现差)，但 v2 用 or=0.5 + lr=3e-4
    # 在保守 baseline (or=0.15, lr=1e-4) 下效果可能完全不同
    "beta_100:$(_make_config "--awsc_beta 100.0")"

    # B3: beta=120 — 极限探索
    # 如果 beta=100 > beta=80 → beta 最优点可能更高
    # 如果 beta=100 < beta=80 → 确认 80 附近是最优区间
    "beta_120:$(_make_config "--awsc_beta 120.0")"

    # ===========================================================================
    # C 组：LR 优化 (面向 500K 训练)
    # ---------------------------------------------------------------------------
    # v3 发现: LR 最优点随训练预算移动
    #   100K → lr=2e-4 最优
    #   250K → lr=1e-4 最优
    #   500K → lr=5e-5 ~ 7e-5 可能最优?
    # 在 [5e-5, 1e-4] 区间精细搜索
    # Base: or=0.15, beta=50, bc=2.0, K=4, all+perv
    # ===========================================================================

    # C1: lr=7e-5 — 1e-4 和 5e-5 之间
    # 可能是 500K 的最优 LR: 足够低避免过冲，足够高保持学习速度
    "lr_7e5:$(_make_config "--lr_actor 7e-5 --lr_critic 7e-5")"

    # C2: lr=5e-5 — v3 中 s_once 最高 (0.96)，250K 时 s_end 仍在上升
    # 500K 训练中最保守 LR 可能最终胜出
    "lr_5e5:$(_make_config "--lr_actor 5e-5 --lr_critic 5e-5")"

    # ===========================================================================
    # D 组：Online Ratio 精细搜索
    # ---------------------------------------------------------------------------
    # v3 发现: or=0.15 > 0.25 > 0.35 (综合排名)
    # 在 0.15 两侧探索: 更极端的 demo-heavy 是否更好?
    # Base: lr=1e-4, beta=50, bc=2.0, K=4, all+perv
    # ===========================================================================

    # D1: or=0.10 — 90% demo, 10% online
    # 极端保守: 是否能防止 所有 s_once 下滑?
    # 风险: online 数据太少，critic 可能无法学到新技能
    "or_0.10:$(_make_config "--online_ratio 0.10")"

    # D2: or=0.20 — 0.15 和 0.25 之间
    # 如果 or_0.20 ≈ or_0.15 → 0.15 附近是平台
    # 如果 or_0.20 > or_0.15 → 0.15 可能过于保守
    "or_0.20:$(_make_config "--online_ratio 0.20")"
)
