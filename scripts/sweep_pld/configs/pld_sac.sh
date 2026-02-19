#!/bin/bash
# =============================================================================
# PLD-SAC Sweep Configurations — v1: PLD-Specific Parameters
# =============================================================================
# Sweeps over hyperparameters *unique to PLD* or particularly relevant to
# the residual-action / offline-pretraining / probing mechanism.
#
# DEFAULTS (from config.sh / training code):
#   action_scale         = 0.25
#   calql_pretrain_steps = 2000
#   calql_alpha          = 5.0
#   probe_steps          = 5
#   probing_alpha        = 0.6
#   online_ratio         = 0.5
#   offline_demo_episodes= 200
#
# Total configs: 42  (baseline + 6 + 6 + 4 + 4 + 4 + 5 + 4 + 8)
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
# Group 1: Action Scale ξ (residual action bounds, unique to PLD)
# Smaller ξ → more conservative fine-tuning, larger ξ → more freedom.
# =============================================================================
ACTION_SCALE_CONFIGS=(
    "$(_make_config "as_0.1"  action_scale 0.1)"
    "$(_make_config "as_0.15" action_scale 0.15)"
    "$(_make_config "as_0.2"  action_scale 0.2)"
    # baseline uses 0.25
    "$(_make_config "as_0.3"  action_scale 0.3)"
    "$(_make_config "as_0.4"  action_scale 0.4)"
    "$(_make_config "as_0.5"  action_scale 0.5)"
)

# =============================================================================
# Group 2: Cal-QL Pretraining Steps (sweep below and above default)
# baseline uses 2000; test less pretraining (0–1000) and more (3000–5000).
# =============================================================================
CALQL_STEPS_CONFIGS=(
    "$(_make_config "calql_0"    calql_pretrain_steps 0)"
    "$(_make_config "calql_500"  calql_pretrain_steps 500)"
    "$(_make_config "calql_1000" calql_pretrain_steps 1000)"
    # baseline uses 2000
    "$(_make_config "calql_3000" calql_pretrain_steps 3000)"
    "$(_make_config "calql_4000" calql_pretrain_steps 4000)"
    "$(_make_config "calql_5000" calql_pretrain_steps 5000)"
)

# =============================================================================
# Group 3: Cal-QL Alpha (conservative loss weight)
# =============================================================================
CALQL_ALPHA_CONFIGS=(
    "$(_make_config "calql_alpha_0.0"  calql_alpha 0.0)"
    "$(_make_config "calql_alpha_1.0"  calql_alpha 1.0)"
    # baseline uses 5.0
    "$(_make_config "calql_alpha_10.0" calql_alpha 10.0)"
    "$(_make_config "calql_alpha_20.0" calql_alpha 20.0)"
)

# =============================================================================
# Group 4: Base Policy Probing (unique to PLD)
# probe_steps: number of RL steps at episode start using base policy only
# =============================================================================
PROBE_STEPS_CONFIGS=(
    "$(_make_config "probe_0"  probe_steps 0)"
    "$(_make_config "probe_3"  probe_steps 3)"
    # baseline uses 5
    "$(_make_config "probe_8"  probe_steps 8)"
    "$(_make_config "probe_12" probe_steps 12)"
)

# =============================================================================
# Group 5: Probing Alpha (probability of probing at episode start)
# =============================================================================
PROBING_ALPHA_CONFIGS=(
    "$(_make_config "palpha_0.0" probing_alpha 0.0)"
    "$(_make_config "palpha_0.3" probing_alpha 0.3)"
    # baseline uses 0.6
    "$(_make_config "palpha_0.8" probing_alpha 0.8)"
    "$(_make_config "palpha_1.0" probing_alpha 1.0)"
)

# =============================================================================
# Group 6: Online Ratio (online/offline buffer mixing)
# =============================================================================
ONLINE_RATIO_CONFIGS=(
    "$(_make_config "or_0.2" online_ratio 0.2)"
    "$(_make_config "or_0.3" online_ratio 0.3)"
    # baseline uses 0.5
    "$(_make_config "or_0.7" online_ratio 0.7)"
    "$(_make_config "or_0.9" online_ratio 0.9)"
    "$(_make_config "or_1.0" online_ratio 1.0)"
)

# =============================================================================
# Group 7: Offline Demo Episodes (data from base policy)
# =============================================================================
OFFLINE_DEMO_CONFIGS=(
    "$(_make_config "demos_50"   offline_demo_episodes 50)"
    "$(_make_config "demos_100"  offline_demo_episodes 100)"
    # baseline uses 200
    "$(_make_config "demos_500"  offline_demo_episodes 500)"
    "$(_make_config "demos_1000" offline_demo_episodes 1000)"
)

# =============================================================================
# Group 8: Combined (PLD-specific combinations)
# =============================================================================
COMBINED_CONFIGS=(
    # No pretraining, no probing — pure DSRL-like online learning
    "combined_no_pld_features:--calql_pretrain_steps 0 --probe_steps 0 --probing_alpha 0.0 --online_ratio 1.0"
    # Aggressive residual: wider action range + more demos + max pretraining
    "combined_aggressive_residual:--action_scale 0.5 --offline_demo_episodes 500 --calql_pretrain_steps 2000"
    # Conservative: small residual + strong pretraining + heavy probing
    "combined_conservative:--action_scale 0.1 --calql_pretrain_steps 2000 --calql_alpha 10.0 --probe_steps 8 --probing_alpha 0.8"
    # Probing ablation: max probe, no Cal-QL
    "combined_probe_only:--calql_pretrain_steps 0 --probe_steps 12 --probing_alpha 1.0"
    # Cal-QL ablation: max pretraining, no probing
    "combined_calql_only:--calql_pretrain_steps 2000 --calql_alpha 10.0 --probe_steps 0 --probing_alpha 0.0"
    # Offline-heavy mixing: mostly offline data + strong pretraining
    "combined_offline_heavy:--online_ratio 0.2 --offline_demo_episodes 500 --calql_pretrain_steps 2000"
    # High UTD + large batch: maximize sample efficiency
    "combined_high_utd_large_batch:--utd_ratio 100 --batch_size 512 --num_qs 15"
    # Small net + low UTD: lightweight baseline
    "combined_small_net:--num_layers 2 --layer_size 512 --utd_ratio 20 --num_qs 5"
)

# =============================================================================
# Assemble All Configs
# =============================================================================
SWEEP_CONFIGS=(
    "$BASELINE"
    "${ACTION_SCALE_CONFIGS[@]}"
    "${CALQL_STEPS_CONFIGS[@]}"
    "${CALQL_ALPHA_CONFIGS[@]}"
    "${PROBE_STEPS_CONFIGS[@]}"
    "${PROBING_ALPHA_CONFIGS[@]}"
    "${ONLINE_RATIO_CONFIGS[@]}"
    "${OFFLINE_DEMO_CONFIGS[@]}"
    "${COMBINED_CONFIGS[@]}"
)

echo "Loaded ${#SWEEP_CONFIGS[@]} PLD-SAC v1 sweep configs (PLD-specific parameters)"
