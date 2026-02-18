#!/bin/bash
# =============================================================================
# DSRL-SAC Sweep Configurations (aligned with dsrl_official findings)
# =============================================================================
# Hyperparameter sweep for DSRL-SAC (SAC in noise space of ShortCut Flow).
#
# Pipeline fixes applied (v2):
#   - Training envs now use FrameStack (matching eval), fixing cross-episode
#     obs_history contamination.
#   - num_seed_steps = 5000 (was 1000), matching dsrl_official's 5001.
#   - obs_history safely cleared on env auto-reset.
#
# NEW DEFAULTS (from dsrl_official best config: arch-medium, utd=40):
#   action_magnitude = 2.0    (was 1.5, sweep showed 2.0 slightly better)
#   utd_ratio        = 40     (was 20,  sweep v2 baseline, significantly better)
#   layer_size       = 512    (was 2048, 3×512 was #1 in sweep)
#   use_layer_norm   = True   (was False, official DSRL recommends True)
#   num_seed_steps   = 5000   (was 1000, aligned with dsrl_official 5001)
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
# Baseline (new defaults: am=2.0, utd=40, 3×512, layer_norm=True)
# =============================================================================
BASELINE="baseline:"

# =============================================================================
# Group 1: Action Magnitude (noise scale — most critical for DSRL)
# dsrl_official finding: am=2.0 (236.74) > am=1.5 (baseline) > am=1.0 (207.90)
# am=2.5 (236.19) nearly as good as 2.0
# =============================================================================
AM_CONFIGS=(
    "$(_make_config "am_0.5"  action_magnitude 0.5)"
    "$(_make_config "am_1.0"  action_magnitude 1.0)"
    "$(_make_config "am_1.5"  action_magnitude 1.5)"
    # baseline uses 2.0
    "$(_make_config "am_2.5"  action_magnitude 2.5)"
    "$(_make_config "am_3.0"  action_magnitude 3.0)"
)

# =============================================================================
# Group 2: UTD Ratio (update-to-data ratio)
# dsrl_official finding: utd=40 (baseline) > utd=60 (229) > utd=80 (226) > utd=100 (219)
# Higher UTD has diminishing returns; go lower to check sensitivity
# =============================================================================
UTD_CONFIGS=(
    "$(_make_config "utd_5"   utd_ratio 5)"
    "$(_make_config "utd_10"  utd_ratio 10)"
    "$(_make_config "utd_20"  utd_ratio 20)"
    # baseline uses 40
    "$(_make_config "utd_60"  utd_ratio 60)"
    "$(_make_config "utd_80"  utd_ratio 80)"
)

# =============================================================================
# Group 3: Learning Rate
# =============================================================================
LR_CONFIGS=(
    "$(_make_config "lr_1e-4" learning_rate 0.0001)"
    # baseline uses 3e-4
    "$(_make_config "lr_1e-3" learning_rate 0.001)"
    "$(_make_config "lr_3e-3" learning_rate 0.003)"
)

# =============================================================================
# Group 4: Discount Factor (gamma)
# =============================================================================
GAMMA_CONFIGS=(
    "$(_make_config "gamma_0.9"   gamma 0.9)"
    "$(_make_config "gamma_0.95"  gamma 0.95)"
    # baseline uses 0.99
    "$(_make_config "gamma_0.999" gamma 0.999)"
)

# =============================================================================
# Group 5: Temperature & Entropy
# =============================================================================
TEMP_CONFIGS=(
    "$(_make_config "init_temp_0.1"  init_temperature 0.1)"
    "$(_make_config "init_temp_0.5"  init_temperature 0.5)"
    # baseline uses 1.0
    "$(_make_config "init_temp_5.0"  init_temperature 5.0)"
)

ENTROPY_CONFIGS=(
    "$(_make_config "target_ent_-7.0"  target_entropy -7.0)"
    "$(_make_config "target_ent_-3.5"  target_entropy -3.5)"
    # baseline uses 0.0
)

# =============================================================================
# Group 6: Log Std Init (initial exploration noise)
# =============================================================================
LOG_STD_CONFIGS=(
    "$(_make_config "log_std_-5.0" log_std_init -5.0)"
    # baseline uses -3.0
    "$(_make_config "log_std_-1.0" log_std_init -1.0)"
    "$(_make_config "log_std_0.0"  log_std_init 0.0)"
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
# dsrl_official finding (all with utd=40):
#   3×512 (medium) = 242.63 ← BEST
#   2×512 (small)  = 238.01
#   3×1024 (large) = 233.00
#   4×1024 (xlarge)= 236.73
# Conclusion: 512 width is optimal; deeper/wider networks hurt
# Baseline is now 3×512, so we sweep around that
# =============================================================================
ARCH_CONFIGS=(
    "arch_2x256:--num_layers 2 --layer_size 256"
    "arch_2x512:--num_layers 2 --layer_size 512"
    # baseline uses num_layers=3, layer_size=512
    "arch_3x1024:--num_layers 3 --layer_size 1024"
    "arch_4x512:--num_layers 4 --layer_size 512"
    "arch_4x1024:--num_layers 4 --layer_size 1024"
    "arch_3x2048:--num_layers 3 --layer_size 2048"
)

# =============================================================================
# Group 9: Layer Norm
# dsrl_official train_dsrl.py recommends use_layer_norm=True
# Baseline now True; check if False is competitive
# =============================================================================
LN_CONFIGS=(
    "ln_off:--no-use_layer_norm"
)

# =============================================================================
# Group 10: Num Q-functions
# =============================================================================
NUM_QS_CONFIGS=(
    # baseline uses 2
    "$(_make_config "num_qs_5"  num_qs 5)"
    "$(_make_config "num_qs_10" num_qs 10)"
)

# =============================================================================
# Group 11: Seed Steps (warmup)
# dsrl_official train_dsrl_sac.py used 5001 warmup steps.
# Baseline is now 5000 (aligned with dsrl_official).
# =============================================================================
SEED_CONFIGS=(
    "$(_make_config "seed_steps_0"     num_seed_steps 0)"
    "$(_make_config "seed_steps_1000"  num_seed_steps 1000)"
    "$(_make_config "seed_steps_2000"  num_seed_steps 2000)"
    # baseline uses 5000
    "$(_make_config "seed_steps_10000" num_seed_steps 10000)"
)

# =============================================================================
# Group 12: Batch Size
# =============================================================================
BATCH_CONFIGS=(
    "$(_make_config "batch_128"  batch_size 128)"
    # baseline uses 256
    "$(_make_config "batch_512"  batch_size 512)"
    "$(_make_config "batch_1024" batch_size 1024)"
)

# =============================================================================
# Group 13: Buffer Size
# dsrl_official finding: buffer_500k (232.09) OK, buffer_100k (167.75) bad
# =============================================================================
BUFFER_CONFIGS=(
    "$(_make_config "buffer_100k" buffer_size 100000)"
    "$(_make_config "buffer_500k" buffer_size 500000)"
    # baseline uses 1M
    "$(_make_config "buffer_2m"   buffer_size 2000000)"
)

# =============================================================================
# Group 14: Combined (promising combinations, post-pipeline-fix)
# =============================================================================
COMBINED_CONFIGS=(
    "combined_dsrl_official_best:--num_layers 3 --layer_size 512 --utd_ratio 40 --action_magnitude 2.0 --use_layer_norm"
    "combined_conservative:--action_magnitude 1.0 --log_std_init -5.0 --init_temperature 0.1"
    "combined_high_utd_large_batch:--utd_ratio 60 --batch_size 512 --num_qs 5"
    "combined_small_net_high_am:--num_layers 2 --layer_size 512 --action_magnitude 2.5"
    "combined_long_warmup_conservative:--num_seed_steps 10000 --init_temperature 0.5 --log_std_init -5.0"
    "combined_neg_target_ent:--target_entropy -7.0 --action_magnitude 1.5 --num_seed_steps 10000"
)

# =============================================================================
# Assemble All Configs
# =============================================================================
SWEEP_CONFIGS=(
    "$BASELINE"
    "${AM_CONFIGS[@]}"
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
    "${SEED_CONFIGS[@]}"
    "${BATCH_CONFIGS[@]}"
    "${BUFFER_CONFIGS[@]}"
    "${COMBINED_CONFIGS[@]}"
)

echo "Loaded ${#SWEEP_CONFIGS[@]} DSRL-SAC sweep configs"
