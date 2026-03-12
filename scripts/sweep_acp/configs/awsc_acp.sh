#!/bin/bash
# =============================================================================
# AWSC + ACP Sweep Configurations (v2 — data-driven redesign)
# =============================================================================
# Based on comparative analysis of awsc_acp_mirror (wa52z9ce) vs sim baseline (ewh7z7lx):
#
# ACP Mirror baseline (seed 42): best_s_end=66%, best_s_once=90%
# Sim baseline (seed 42):        best_s_end=72%, best_s_once=92%
#
# Key diagnostic findings:
#   1. online_cum_reward=0.05 vs offline=4.34 → 87x gap, critic dominated by demo
#   2. success_once degrades 0.82→0.60 after 200K (insufficient BC anchoring)
#   3. advantage_mean≈0.8 is high (positive bias), weight_max spikes to 41
#   4. Q-value saturates at ~3.9 with gamma=0.9 (ACP signal drowns in demo Q)
#   5. Late phase (300K+): ACP s_once=0.60 vs SIM s_once=0.76 → ACP causes drift
#
# Strategy: Focus on 3 axes:
#   A. Amplify ACP signal (higher reward_scale, higher online_ratio)
#   B. Prevent forgetting (higher bc_weight)
#   C. Fix credit assignment (lower gamma for ACP, lower beta for weighting)
#
# Total: 16 configs (1 baseline + 15 sweep)
# =============================================================================

_make_config() {
    local name=$1; shift
    echo "${name}:$*"
}

# =============================================================================
# Baseline (identical to acp_mirror AWSC run wa52z9ce)
# =============================================================================
BASELINE="baseline:"

# =============================================================================
# Group 1: ACP Reward Scale (HIGHEST PRIORITY)
# online_cum_reward=0.05 with scale=100 → actual per-step ACP reward ≈ 0.0005
# Need >> 100 to make ACP signal compete with demo Q-values (~4.3)
# =============================================================================
SCALE_CONFIGS=(
    "$(_make_config "scale_500"   --acp_reward_scale 500.0)"
    "$(_make_config "scale_1000"  --acp_reward_scale 1000.0)"
    "$(_make_config "scale_2000"  --acp_reward_scale 2000.0)"
)

# =============================================================================
# Group 2: BC Weight (CRITICAL for preventing s_once degradation)
# mirror: bc_weight=2.0 → s_once drops 0.82→0.60
# Stronger BC anchoring should prevent policy drift while ACP refines behavior
# =============================================================================
BC_WEIGHT_CONFIGS=(
    "$(_make_config "bc_4.0"  --awsc_bc_weight 4.0)"
    "$(_make_config "bc_8.0"  --awsc_bc_weight 8.0)"
)

# =============================================================================
# Group 3: Online Ratio (increase ACP signal weight in training)
# 0.15 means only 15% online data → ACP reward signal is diluted
# Higher online_ratio amplifies ACP signal relative to demo
# =============================================================================
ONLINE_RATIO_CONFIGS=(
    "$(_make_config "or_0.3"  --online_ratio 0.3)"
    "$(_make_config "or_0.5"  --online_ratio 0.5)"
)

# =============================================================================
# Group 4: Gamma (credit assignment with ACP TD reward)
# gamma=0.9 → Q saturates at ~3.9, demo-dominated
# Lower gamma → shorter credit horizon, less demo Q accumulation
# =============================================================================
GAMMA_CONFIGS=(
    "$(_make_config "gamma_0.7"  --gamma 0.7)"
    "$(_make_config "gamma_0.5"  --gamma 0.5)"
)

# =============================================================================
# Group 5: Combined (multi-parameter configs based on analysis)
# =============================================================================
COMBINED_CONFIGS=(
    # C1: Amplify ACP + anchor BC — address both main issues
    "combined_amp_anchor:--acp_reward_scale 1000.0 --awsc_bc_weight 4.0"
    # C2: Full rebalance — high scale + high BC + more online data
    "combined_rebalance:--acp_reward_scale 1000.0 --awsc_bc_weight 4.0 --online_ratio 0.3"
    # C3: Short horizon + high scale — fast credit assignment
    "combined_short_fast:--acp_reward_scale 1000.0 --gamma 0.7 --awsc_bc_weight 4.0"
    # C4: Maximum ACP signal — scale 2000 + online 0.5 + short gamma
    "combined_max_acp:--acp_reward_scale 2000.0 --online_ratio 0.5 --gamma 0.7"
    # C5: Best guess — balanced tuning on all axes
    "combined_balanced:--acp_reward_scale 1000.0 --awsc_bc_weight 4.0 --online_ratio 0.3 --gamma 0.7"
)

# =============================================================================
# Assemble
# =============================================================================
SWEEP_CONFIGS=(
    "$BASELINE"
    "${SCALE_CONFIGS[@]}"
    "${BC_WEIGHT_CONFIGS[@]}"
    "${ONLINE_RATIO_CONFIGS[@]}"
    "${GAMMA_CONFIGS[@]}"
    "${COMBINED_CONFIGS[@]}"
)

echo "Loaded ${#SWEEP_CONFIGS[@]} AWSC+ACP sweep configs"
