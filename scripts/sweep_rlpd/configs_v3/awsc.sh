#!/bin/bash
# =============================================================================
# AWSC (Advantage-Weighted ShortCut Flow) - Wave 3 Sweep: Fixed Success + Refined Conservative
# =============================================================================
#
# 基于 Wave 2 sweep 深度分析结果，Wave 3 有两个核心目标：
#
# 1. 验证修复后的 success_only 机制的真实效果
#    Wave 2 发现 success_only 完全失效（在线 success_at_end=0%，
#    _success_indices 始终为空，fallback 到标准采样）。
#    修复：success_criteria 从 success_at_end 改为 success_once。
#    冠军配置 (w1_conservative_combined) 的成功是否来自 success_only？
#    → A 组：控制变量消融实验
#
# 2. 在冠军配置基础上精细搜索最优超参
#    Wave 2 发现：lr=1e-4, online_ratio=0.25, bc_weight=2.0 的
#    "三重保守"组合是性能的主要驱动力。
#    → B-E 组：各维度的邻域搜索
#
# ===== Wave 2 关键发现（指导 Wave 3 设计） =====
#
# 1. 冠军：w1_conservative_combined (s_once=0.90, s_end=0.28, return=72.8)
#    参数：beta=50, bc=2.0, lr=1e-4, or=0.25, success_only, per_state_v
#    但 success_only 实际无效 → 需要消融验证
#
# 2. 保守超参是核心：lr=1e-4 >> 3e-4，or=0.25 >> 0.5
#    防止灾难性遗忘（min s_once: 0.72 vs 0.14）
#
# 3. per_state_v 在稳定训练下有小幅增益 (+0.08 s_once, +0.08 s_end)
#    但在高 gamma (0.95) 下有害
#
# 4. K=2 (V(s) samples) 因隐式正则化意外最优
#
# 5. s_once 和 s_end 存在矛盾：保护越强 → s_once 越稳 → 新技能学习越慢
#
# ===== 扫描总览：6 组，14 configs =====
#
# A. 消融实验 (success_only + per_state_v):     3 configs
# B. 学习率邻域搜索:                             2 configs
# C. Online Ratio 邻域搜索:                      2 configs
# D. BC Weight 邻域搜索:                          2 configs
# E. 温度 + V(s) 采样微调:                       3 configs
# F. 默认参数 + success_only（对照）:             1 config
# + Baseline（v2 冠军 + success_once fix）:       1 config
# -------------------------------------------
# Total:                                         14 configs
#
# Format: "config_name:--param1 value1 --param2 value2"
# Note: pretrain_path + load_pretrain_critic 由 sweep.sh 根据 --mode 自动注入
# =============================================================================

# -----------------------------------------------------------------------------
# Baseline 超参数 (= Wave 2 冠军 w1_conservative_combined)
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
BASELINE_ONLINE_RATIO="0.25"
BASELINE_REWARD_SCALE="1.0"
BASELINE_NUM_QS="10"
BASELINE_NUM_MIN_QS="2"

# 学习率 (保守设定)
BASELINE_LR_ACTOR="1e-4"
BASELINE_LR_CRITIC="1e-4"

# Wave 2 新增参数 (冠军配置)
BASELINE_ACTOR_POLICY_MODE="success_only"
BASELINE_ADVANTAGE_MODE="per_state_v"
BASELINE_NUM_V_SAMPLES="4"

# Wave 3 新增参数
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
    # Baseline 配置 = Wave 2 冠军 + success_once fix
    # 参数: beta=50, bc=2.0, lr=1e-4, or=0.25, success_only, per_state_v
    # 对比 v2 的 w1_conservative_combined（success_only 当时无效）
    # ===========================================================================
    "baseline:${_awsc_base}"

    # ===========================================================================
    # A 组：消融实验 — 验证 success_only 和 per_state_v 的真实贡献
    # ---------------------------------------------------------------------------
    # 核心问题：v2 冠军的 success_only 当时完全失效，其 0.90 的 s_once
    # 是否完全来自保守超参？success_once fix 后是否有额外增益？
    # 所有配置共享保守超参 (lr=1e-4, or=0.25, bc=2.0, beta=50)
    # ===========================================================================

    # A1: 无 success_only, 无 per_state_v（纯保守超参基线）
    # 如果 = v2 冠军性能 → 两个机制无用
    # 如果 < v2 冠军 → v2 中虽然 success_only 无效，per_state_v 有贡献
    "ablation_all_bm:$(_make_config "--actor_policy_mode all --awsc_advantage_mode batch_mean")"

    # A2: success_only ON, per_state_v OFF
    # 隔离 success_only（修复后）的独立效果
    "ablation_so_bm:$(_make_config "--awsc_advantage_mode batch_mean")"

    # A3: success_only OFF, per_state_v ON
    # 隔离 per_state_v 的独立效果
    "ablation_all_perv:$(_make_config "--actor_policy_mode all")"

    # ===========================================================================
    # B 组：学习率邻域搜索
    # ---------------------------------------------------------------------------
    # v2 发现 lr 是最关键因素：1e-4 全程稳定，3e-4 剧烈波动。
    # 在 1e-4 附近做更细致的探索。
    # Base: success_only + per_state_v + bc=2.0 + or=0.25 + beta=50
    # ===========================================================================

    # B1: 更保守的学习率 — 是否能进一步稳定训练？
    "lr_5e5:$(_make_config "--lr_actor 5e-5 --lr_critic 5e-5")"

    # B2: 稍高学习率 — 在有 success_only 保护下能否放松？
    "lr_2e4:$(_make_config "--lr_actor 2e-4 --lr_critic 2e-4")"

    # ===========================================================================
    # C 组：Online Ratio 邻域搜索
    # ---------------------------------------------------------------------------
    # v2 发现 or=0.25 >> 0.5 >> 0.75。
    # 在 0.25 附近探索最优平衡点。
    # Base: success_only + per_state_v + lr=1e-4 + bc=2.0 + beta=50
    # ===========================================================================

    # C1: 更重 demo — 是否能更稳定？
    "or_0.15:$(_make_config "--online_ratio 0.15")"

    # C2: 稍多 online — success_only fix 后能否安全增加？
    # （success_only 现在能过滤失败数据，理论上可以容忍更高 OR）
    "or_0.35:$(_make_config "--online_ratio 0.35")"

    # ===========================================================================
    # D 组：BC Weight 邻域搜索
    # ---------------------------------------------------------------------------
    # v2 发现 bc=2.0 最佳，bc=0.5 崩 → 恢复但s_end高。
    # 在 bc=2.0 两侧探索。
    # Base: success_only + per_state_v + lr=1e-4 + or=0.25 + beta=50
    # ===========================================================================

    # D1: 降低 BC — success_only 保护下是否可以放松 BC 约束？
    "bc_1.0:$(_make_config "--awsc_bc_weight 1.0")"

    # D2: 更强 BC — 是否进一步稳定？
    "bc_3.0:$(_make_config "--awsc_bc_weight 3.0")"

    # ===========================================================================
    # E 组：温度和 V(s) 采样微调
    # ---------------------------------------------------------------------------
    # v2 发现 beta=50 > 100 > 200；K=2 意外最优（隐式正则化）
    # Base: success_only + per_state_v + lr=1e-4 + or=0.25 + bc=2.0
    # ===========================================================================

    # E1: 更低温度 — 权重更均匀，更强正则化
    "beta_30:$(_make_config "--awsc_beta 30.0")"

    # E2: 中等温度 — 在 50 和 100 之间
    "beta_80:$(_make_config "--awsc_beta 80.0")"

    # E3: K=2 — v2 中隐式正则化效果好（Var[V(s)] 更大 → 权重更均匀）
    "v_samples_2:$(_make_config "--awsc_num_v_samples 2")"

    # ===========================================================================
    # F 组：默认参数 + success_only（对照实验）
    # ---------------------------------------------------------------------------
    # 测试在非保守超参下，修复后的 success_only 是否也能防止遗忘。
    # v2 中 policy_success_only (默认参数) 达到 s_end=0.30（全场并列最高），
    # 但 success_only 当时无效，说明是训练随机性。修复后效果如何？
    # ===========================================================================

    # F1: 默认超参 + success_only + success_once
    "default_so:$(_make_config "--awsc_beta 100.0 --awsc_bc_weight 1.0 --lr_actor 3e-4 --lr_critic 3e-4 --online_ratio 0.5 --actor_policy_mode success_only --awsc_advantage_mode batch_mean")"
)
