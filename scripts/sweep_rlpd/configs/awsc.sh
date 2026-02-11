#!/bin/bash
# =============================================================================
# AWSC (Advantage-Weighted ShortCut Flow) - Pretrain-Focused Sweep v2
# =============================================================================
#
# 基于 Wave 3 深度分析结果重新设计，聚焦 pretrain 模式（从 IL/offline RL
# checkpoint 初始化再 online fine-tuning）。
#
# ===== Wave 3 关键发现 → 扫描设计映射 =====
#
# 1. 有效温度 (beta × reward_scale) 是最关键旋钮
#    → A 组：5 个配置，网格探索 beta×rs 组合
#
# 2. lr 对 AW 类方法极度敏感 (lr_5e-4 导致 AW-SC 灾难性崩溃)
#    → D 组：探索 fine-tuning 适合的小 lr
#
# 3. bc_weight 过低会崩溃 (consistency_flow bc=0.5 → 0.27)
#    → B 组：谨慎探索，pretrain 模式下 bc_weight 控制对预训练策略的锚定
#
# 4. 已固化为默认值（不再扫描）：
#    - shortcut_weight=0.3 (Wave 3 最优)
#    - self_consistency_k=0.25 (稳定最优)
#    - ema_decay=0.9995 (Wave 3 最优)
#    - weight_clip=200.0 (Wave 3 最优)
#    - num_qs=10, num_min_qs=2 (Wave 3 无效应)
#
# ===== 扫描总览：6 组，17 configs =====
#
# A. 有效温度 (beta × reward_scale):    5 configs
# B. 预训练正则化 (bc_weight):          2 configs
# C. 数据混合 (online_ratio):           2 configs
# D. 学习率 (lr_actor/lr_critic):       2 configs
# E. 折扣因子 (gamma):                  2 configs
# F. 组合假设 (保守/平衡v3/激进):       3 configs
# + Baseline:                            1 config
# -------------------------------------------
# Total:                                17 configs
#
# Format: "config_name:--param1 value1 --param2 value2"
# Note: pretrain_path 由 sweep.sh 根据 --mode 自动注入
# =============================================================================

# -----------------------------------------------------------------------------
# Baseline 超参数 (Wave 3 更新版)
# -----------------------------------------------------------------------------
# AWSC 核心参数
BASELINE_AWSC_BETA="100.0"             # Online RL 高温，不同于 offline 的 10.0
BASELINE_AWSC_BC_WEIGHT="1.0"          # Flow matching 正则化权重
BASELINE_AWSC_SHORTCUT_WEIGHT="0.3"    # Wave 3 最优 (fixed)
BASELINE_AWSC_SELF_CONSISTENCY_K="0.25" # Wave 3 最优 (fixed)
BASELINE_AWSC_EMA_DECAY="0.9995"       # Wave 3 最优 (updated from 0.999)
BASELINE_AWSC_WEIGHT_CLIP="200.0"      # Wave 3 最优 (updated from 100.0)
BASELINE_AWSC_NUM_INFERENCE_STEPS="8"

# RL 通用参数
BASELINE_GAMMA="0.9"
BASELINE_TAU="0.005"
BASELINE_ONLINE_RATIO="0.5"
BASELINE_REWARD_SCALE="1.0"
BASELINE_NUM_QS="10"
BASELINE_NUM_MIN_QS="2"

# 学习率 (Wave 3: lr 对 AW 方法最敏感)
BASELINE_LR_ACTOR="3e-4"
BASELINE_LR_CRITIC="3e-4"

# 构建 baseline 参数字符串（不含 pretrain_path，由 sweep.sh 注入）
_awsc_base="--awsc_beta ${BASELINE_AWSC_BETA} --awsc_bc_weight ${BASELINE_AWSC_BC_WEIGHT} --awsc_shortcut_weight ${BASELINE_AWSC_SHORTCUT_WEIGHT} --awsc_self_consistency_k ${BASELINE_AWSC_SELF_CONSISTENCY_K} --awsc_ema_decay ${BASELINE_AWSC_EMA_DECAY} --awsc_weight_clip ${BASELINE_AWSC_WEIGHT_CLIP} --awsc_num_inference_steps ${BASELINE_AWSC_NUM_INFERENCE_STEPS} --gamma ${BASELINE_GAMMA} --tau ${BASELINE_TAU} --online_ratio ${BASELINE_ONLINE_RATIO} --reward_scale ${BASELINE_REWARD_SCALE} --num_qs ${BASELINE_NUM_QS} --num_min_qs ${BASELINE_NUM_MIN_QS} --lr_actor ${BASELINE_LR_ACTOR} --lr_critic ${BASELINE_LR_CRITIC}"

# 辅助函数：生成单参数变体（baseline + 覆盖参数；tyro 中后出现的参数覆盖先出现的）
_make_config() {
    local overrides="$1"
    echo "${_awsc_base} ${overrides}"
}

# -----------------------------------------------------------------------------
# Sweep 配置数组
# 格式: "config_name:extra_args"
# 注意：pretrain_path 参数由 sweep.sh 根据 --mode 自动注入
# -----------------------------------------------------------------------------
SWEEP_CONFIGS=(
    # ===========================================================================
    # Baseline 配置（对照组）
    # ===========================================================================
    "baseline:${_awsc_base}"

    # ===========================================================================
    # A 组：有效温度网格 (beta × reward_scale)
    # ---------------------------------------------------------------------------
    # Wave 3 核心发现：有效温度 = beta × reward_scale × reward_range
    #   - 过高 → 少数高 advantage 样本主导，flow matching 覆盖性受损
    #   - 过低 → 权重退化为均匀，Q 信号浪费
    # Online RL 的 reward 尺度不同于 offline，需要重新探索
    # ===========================================================================

    # A1: 低有效温度 (beta↓, rs↓)
    "eff_temp_low:$(_make_config "--awsc_beta 50.0 --reward_scale 0.5")"

    # A2: 中低有效温度 (beta↓，保持 rs)
    "eff_temp_mid_low:$(_make_config "--awsc_beta 50.0")"

    # A3: 中高有效温度 (beta↑, rs↓) — 与 baseline 乘积相近但分布不同
    "eff_temp_mid_high:$(_make_config "--awsc_beta 200.0 --reward_scale 0.5")"

    # A4: 高有效温度 (beta↑)
    "eff_temp_high:$(_make_config "--awsc_beta 200.0")"

    # A5: 极低有效温度 (类似 offline 最优 regime)
    "eff_temp_offline:$(_make_config "--awsc_beta 100.0 --reward_scale 0.1")"

    # ===========================================================================
    # B 组：预训练正则化 (bc_weight)
    # ---------------------------------------------------------------------------
    # Wave 3: bc_weight 过低导致灾难性崩溃
    # Pretrain 模式下，bc_weight 控制策略对预训练 checkpoint 的锚定强度：
    #   - 低 bc_weight → 更激进的 RL 优化（风险：遗忘预训练策略）
    #   - 高 bc_weight → 保守微调（风险：RL 信号被压制）
    # ===========================================================================

    # B1: 适度降低（谨慎，Wave 3 显示 0.5 已在边缘）
    "bc_weight_0.5:$(_make_config "--awsc_bc_weight 0.5")"

    # B2: 强锚定（保守保护预训练质量）
    "bc_weight_2.0:$(_make_config "--awsc_bc_weight 2.0")"

    # ===========================================================================
    # C 组：数据混合比例 (online_ratio)
    # ---------------------------------------------------------------------------
    # online_ratio 控制 mini-batch 中 online vs demo replay 的比例
    # Pretrain 模式下，可能需要更多 demo 数据来稳定 fine-tuning
    # ===========================================================================

    # C1: demo 数据主导（稳定，适合 pretrain 初期）
    "online_ratio_0.25:$(_make_config "--online_ratio 0.25")"

    # C2: online 数据主导（激进，测试 online experience 价值）
    "online_ratio_0.75:$(_make_config "--online_ratio 0.75")"

    # ===========================================================================
    # D 组：学习率 (lr_actor + lr_critic)
    # ---------------------------------------------------------------------------
    # Wave 3 最关键发现之一：lr 对 AW 方法影响巨大
    #   - lr=2e-4 最优 (offline AW-SC, AWCP)
    #   - lr=5e-4 灾难性崩溃 (AW-SC offline: success_once=0.06)
    # 原因：Q-weighted 梯度方差大，小 lr 能抑制方差放大
    # Pretrain fine-tuning 可能需要更小的 lr 避免遗忘
    # ===========================================================================

    # D1: Wave 3 最优 lr（推荐用于 pretrain fine-tuning）
    "lr_2e-4:$(_make_config "--lr_actor 2e-4 --lr_critic 2e-4")"

    # D2: 极保守 lr（专为 pretrain fine-tuning 设计）
    "lr_1e-4:$(_make_config "--lr_actor 1e-4 --lr_critic 1e-4")"

    # ===========================================================================
    # E 组：折扣因子 (gamma)
    # ---------------------------------------------------------------------------
    # gamma 决定 critic 关注的时间尺度
    #   - 低 gamma (0.9) → 近视，适合 dense reward
    #   - 高 gamma (0.99) → 远视，适合 sparse reward
    # ManiSkill 任务的 reward 结构决定最优 gamma
    # ===========================================================================

    # E1: 中等视野
    "gamma_0.95:$(_make_config "--gamma 0.95")"

    # E2: 长视野
    "gamma_0.99:$(_make_config "--gamma 0.99")"

    # ===========================================================================
    # F 组：组合假设（基于 Wave 3 洞察的端到端策略）
    # ---------------------------------------------------------------------------
    # 不是单参数扫描，而是综合 Wave 3 的多个发现，设计整体"方案"
    # ===========================================================================

    # F1: 保守预训练策略
    # 低有效温度 + 强 BC 锚定 + 小 lr + demo 主导
    # 假设：pretrain checkpoint 已经很好，只需小幅 RL 优化
    "combined_conservative:$(_make_config "--awsc_beta 50.0 --awsc_bc_weight 2.0 --lr_actor 1e-4 --lr_critic 1e-4 --online_ratio 0.25")"

    # F2: Wave 3 知识整合策略
    # Wave 3 最优 lr + 中等温度 + 长视野（pretrain enables longer horizon）
    # 假设：pretrain 的好初始化允许用更长 gamma 和更稳定的 lr
    "combined_wave3:$(_make_config "--awsc_beta 100.0 --lr_actor 2e-4 --lr_critic 2e-4 --gamma 0.95")"

    # F3: 激进探索策略
    # 高有效温度 + 弱 BC + 高 online ratio
    # 假设：大胆使用 RL 信号，让 pretrain 策略快速适应
    "combined_aggressive:$(_make_config "--awsc_beta 200.0 --awsc_bc_weight 0.5 --online_ratio 0.75")"
)
