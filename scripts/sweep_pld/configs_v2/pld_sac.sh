#!/bin/bash
# =============================================================================
# PLD-SAC Sweep Configurations — v2: SAC Core Hyperparameters
# =============================================================================
# Sweeps over SAC backbone hyperparameters inherited from DSRL.
# These are general RL hyperparameters, not PLD-specific.
#
# DEFAULTS (from DSRL sweep best):
#   utd_ratio            = 60
#   learning_rate        = 1e-3
#   gamma                = 0.95
#   init_temperature     = 0.5
#   target_entropy       = -3.5
#   log_std_init         = -5.0
#   tau                  = 0.005
#   num_layers           = 3
#   layer_size           = 2048
#   use_layer_norm       = True
#   num_qs               = 10
#   batch_size           = 256
#   online_buffer_size   = 500000
#   offline_buffer_size  = 200000
#
# Total configs: 42  (baseline + 5 + 3 + 3 + 3 + 3 + 3 + 3 + 5 + 1 + 3 + 3 + 3 + 3)
#
# Format: "config_name:--arg1 val1 --arg2 val2 ..."
# =============================================================================

# Helper: create a config entry with a single parameter override
_make_config() {
    local name=$1
    local arg=$2
    local value=$3
    echo "${name}:--${arg} ${value}"
}

# =============================================================================
# Baseline (all defaults)
# =============================================================================
BASELINE="baseline:"

# =============================================================================
# Group 1: UTD Ratio (update-to-data ratio)
# DSRL finding: 40-80 optimal range, 60 chosen as PLD default
# =============================================================================
UTD_CONFIGS=(
    "$(_make_config "utd_10"  utd_ratio 10)"
    "$(_make_config "utd_20"  utd_ratio 20)"
    "$(_make_config "utd_40"  utd_ratio 40)"
    # baseline uses 60
    "$(_make_config "utd_80"  utd_ratio 80)"
    "$(_make_config "utd_100" utd_ratio 100)"
)

# =============================================================================
# Group 2: Learning Rate
# =============================================================================
LR_CONFIGS=(
    "$(_make_config "lr_1e-4" learning_rate 0.0001)"
    "$(_make_config "lr_3e-4" learning_rate 0.0003)"
    # baseline uses 1e-3
    "$(_make_config "lr_3e-3" learning_rate 0.003)"
)

# =============================================================================
# Group 3: Discount Factor (gamma)
# =============================================================================
GAMMA_CONFIGS=(
    "$(_make_config "gamma_0.9"   gamma 0.9)"
    # baseline uses 0.95
    "$(_make_config "gamma_0.99"  gamma 0.99)"
    "$(_make_config "gamma_0.999" gamma 0.999)"
)

# =============================================================================
# Group 4: Temperature
# =============================================================================
TEMP_CONFIGS=(
    "$(_make_config "init_temp_0.1"  init_temperature 0.1)"
    # baseline uses 0.5
    "$(_make_config "init_temp_1.0"  init_temperature 1.0)"
    "$(_make_config "init_temp_5.0"  init_temperature 5.0)"
)

# =============================================================================
# Group 5: Target Entropy
# =============================================================================
ENTROPY_CONFIGS=(
    "$(_make_config "target_ent_-7.0" target_entropy -7.0)"
    # baseline uses -3.5
    "$(_make_config "target_ent_0.0"  target_entropy 0.0)"
    "$(_make_config "target_ent_-1.0" target_entropy -1.0)"
)

# =============================================================================
# Group 6: Log Std Init (initial exploration noise)
# =============================================================================
LOG_STD_CONFIGS=(
    "$(_make_config "log_std_-8.0" log_std_init -8.0)"
    # baseline uses -5.0
    "$(_make_config "log_std_-3.0" log_std_init -3.0)"
    "$(_make_config "log_std_-1.0" log_std_init -1.0)"
)

# =============================================================================
# Group 7: Tau (target network update rate)
# =============================================================================
TAU_CONFIGS=(
    "$(_make_config "tau_0.001" tau 0.001)"
    # baseline uses 0.005
    "$(_make_config "tau_0.01"  tau 0.01)"
    "$(_make_config "tau_0.02"  tau 0.02)"
)

# =============================================================================
# Group 8: Network Architecture
# =============================================================================
ARCH_CONFIGS=(
    "arch_2x512:--num_layers 2 --layer_size 512"
    "arch_3x512:--num_layers 3 --layer_size 512"
    "arch_3x1024:--num_layers 3 --layer_size 1024"
    # baseline uses num_layers=3, layer_size=2048
    "arch_4x2048:--num_layers 4 --layer_size 2048"
    "arch_3x4096:--num_layers 3 --layer_size 4096"
)

# =============================================================================
# Group 9: Layer Norm
# =============================================================================
LN_CONFIGS=(
    "ln_off:--no-use_layer_norm"
)

# =============================================================================
# Group 10: Num Q-functions (min-Q ensemble size)
# =============================================================================
NUM_QS_CONFIGS=(
    "$(_make_config "num_qs_2"  num_qs 2)"
    "$(_make_config "num_qs_5"  num_qs 5)"
    # baseline uses 10
    "$(_make_config "num_qs_15" num_qs 15)"
)

# =============================================================================
# Group 11: Batch Size
# =============================================================================
BATCH_CONFIGS=(
    "$(_make_config "batch_128"  batch_size 128)"
    # baseline uses 256
    "$(_make_config "batch_512"  batch_size 512)"
    "$(_make_config "batch_1024" batch_size 1024)"
)

# =============================================================================
# Group 12: Buffer Sizes (online / offline)
# =============================================================================
ONLINE_BUFFER_CONFIGS=(
    "$(_make_config "obuf_100k"  online_buffer_size 100000)"
    "$(_make_config "obuf_250k"  online_buffer_size 250000)"
    # baseline uses 500000
    "$(_make_config "obuf_1m"    online_buffer_size 1000000)"
)

OFFLINE_BUFFER_CONFIGS=(
    "$(_make_config "ofbuf_50k"  offline_buffer_size 50000)"
    "$(_make_config "ofbuf_100k" offline_buffer_size 100000)"
    # baseline uses 200000
    "$(_make_config "ofbuf_500k" offline_buffer_size 500000)"
)

# =============================================================================
# Assemble All Configs
# =============================================================================
SWEEP_CONFIGS=(
    "$BASELINE"
    "${UTD_CONFIGS[@]}"
    "${LR_CONFIGS[@]}"
    "${GAMMA_CONFIGS[@]}"
    "${TEMP_CONFIGS[@]}"
    "${ENTROPY_CONFIGS[@]}"
    "${LOG_STD_CONFIGS[@]}"
    "${TAU_CONFIGS[@]}"
    "${ARCH_CONFIGS[@]}"
    "${LN_CONFIGS[@]}"
    "${NUM_QS_CONFIGS[@]}"
    "${BATCH_CONFIGS[@]}"
    "${ONLINE_BUFFER_CONFIGS[@]}"
    "${OFFLINE_BUFFER_CONFIGS[@]}"
)

echo "Loaded ${#SWEEP_CONFIGS[@]} PLD-SAC v2 sweep configs (SAC core parameters)"
