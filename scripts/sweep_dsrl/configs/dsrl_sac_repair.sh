#!/bin/bash
# =============================================================================
# DSRL-SAC Completed Configs (从日志提取的22个已完成配置)
# =============================================================================
# 这个文件包含从当前扫描中已经完成的所有配置，可以用于重新运行或验证这些配置。
#
# 已完成配置列表:
#   - baseline: 1个
#   - Action Magnitude: 5个 (am_0.5, am_1.0, am_1.5, am_2.5, am_3.0)
#   - UTD Ratio: 5个 (utd_5, utd_10, utd_20, utd_60, utd_80)
#   - Learning Rate: 3个 (lr_1e-4, lr_1e-3, lr_3e-3)
#   - Gamma: 3个 (gamma_0.9, gamma_0.95, gamma_0.999)
#   - Temperature: 3个 (init_temp_0.1, init_temp_0.5, init_temp_5.0)
#   - Target Entropy: 2个 (target_ent_-7.0, target_ent_-3.5)
#
# 总计: 22个已完成配置
# =============================================================================

# Helper: create a config entry with a single parameter override
_make_config() {
    local name=$1
    local arg=$2
    local value=$3
    echo "${name}:--${arg} ${value}"
}

# =============================================================================
# Baseline (new defaults: am=2.0, utd=40, 3×512, layer_norm=True)
# =============================================================================
BASELINE="baseline:"

# =============================================================================
# Group 1: Action Magnitude (5个已完成配置)
# =============================================================================
AM_CONFIGS=(
    "$(_make_config "am_0.5"  action_magnitude 0.5)"
    "$(_make_config "am_1.0"  action_magnitude 1.0)"
    "$(_make_config "am_1.5"  action_magnitude 1.5)"
    "$(_make_config "am_2.5"  action_magnitude 2.5)"
    "$(_make_config "am_3.0"  action_magnitude 3.0)"
)

# =============================================================================
# Group 2: UTD Ratio (5个已完成配置)
# =============================================================================
UTD_CONFIGS=(
    "$(_make_config "utd_5"   utd_ratio 5)"
    "$(_make_config "utd_10"  utd_ratio 10)"
    "$(_make_config "utd_20"  utd_ratio 20)"
    "$(_make_config "utd_60"  utd_ratio 60)"
    "$(_make_config "utd_80"  utd_ratio 80)"
)

# =============================================================================
# Group 3: Learning Rate (3个已完成配置)
# =============================================================================
LR_CONFIGS=(
    "$(_make_config "lr_1e-4" learning_rate 0.0001)"
    "$(_make_config "lr_1e-3" learning_rate 0.001)"
    "$(_make_config "lr_3e-3" learning_rate 0.003)"
)

# =============================================================================
# Group 4: Discount Factor (gamma) (3个已完成配置)
# =============================================================================
GAMMA_CONFIGS=(
    "$(_make_config "gamma_0.9"   gamma 0.9)"
    "$(_make_config "gamma_0.95"  gamma 0.95)"
    "$(_make_config "gamma_0.999" gamma 0.999)"
)

# =============================================================================
# Group 5: Temperature (3个已完成配置)
# =============================================================================
TEMP_CONFIGS=(
    "$(_make_config "init_temp_0.1"  init_temperature 0.1)"
    "$(_make_config "init_temp_0.5"  init_temperature 0.5)"
    "$(_make_config "init_temp_5.0"  init_temperature 5.0)"
)

# =============================================================================
# Group 6: Target Entropy (2个已完成配置)
# =============================================================================
ENTROPY_CONFIGS=(
    "$(_make_config "target_ent_-7.0"  target_entropy -7.0)"
    "$(_make_config "target_ent_-3.5"  target_entropy -3.5)"
)

# =============================================================================
# Assemble All Completed Configs
# =============================================================================
SWEEP_CONFIGS=(
    "$BASELINE"
    "${AM_CONFIGS[@]}"
    "${UTD_CONFIGS[@]}"
    "${LR_CONFIGS[@]}"
    "${GAMMA_CONFIGS[@]}"
    "${TEMP_CONFIGS[@]}"
    "${ENTROPY_CONFIGS[@]}"
)

echo "Loaded ${#SWEEP_CONFIGS[@]} DSRL-SAC completed configs"
