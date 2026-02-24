#!/bin/bash
# =============================================================================
# PLD-SAC Sweep Configurations — v3: Combination & Ablation Experiments
# =============================================================================
# Based on PLD sweep v1/v2 analysis results. The new defaults (train_pld.py)
# already incorporate all best single-parameter values:
#   lr=1e-4, layer_size=1024, batch_size=1024, num_qs=5,
#   action_scale=0.3, calql_alpha=0.0, calql_pretrain_steps=1000,
#   gamma=0.99, online_ratio=1.0, init_temperature=0.1,
#   offline_demo_episodes=50, tau=0.005, utd=60
#
# v3 goals:
#   1. Validate that the combined best config outperforms any single change
#   2. Ablate each change from the combined best → measure individual contribution
#   3. Explore promising 2-way interactions between top improvements
#   4. Push boundaries on the most impactful parameters (lr, batch, arch)
#   5. Test with different total_timesteps to check scaling
#
# Total configs: 40  (baseline + 11 ablation + 12 interaction + 10 boundary + 6 scaling)
#
# Format: "config_name:--arg1 val1 --arg2 val2 ..."
# =============================================================================

# =============================================================================
# Baseline (new defaults = combined best from v1+v2)
# =============================================================================
# This is the recommended config from the analysis. All parameters use
# the new defaults so no overrides needed.
BASELINE="baseline:"

# =============================================================================
# Group 1: Single-Parameter Ablation from Best Config
# Revert each parameter to its OLD default, keeping everything else at new best.
# This measures how much each individual change contributes to the combined result.
# =============================================================================
ABLATION_CONFIGS=(
    # Revert lr to old 1e-3 (was the #1 improvement)
    "ablate_lr_1e-3:--learning_rate 0.001"
    # Revert batch_size to old 256
    "ablate_batch_256:--batch_size 256"
    # Revert layer_size to old 2048
    "ablate_layer_2048:--layer_size 2048"
    # Revert num_qs to old 10
    "ablate_num_qs_10:--num_qs 10"
    # Revert init_temperature to old 0.5
    "ablate_temp_0.5:--init_temperature 0.5"
    # Revert gamma to old 0.95
    "ablate_gamma_0.95:--gamma 0.95"
    # Revert action_scale to old 0.25
    "ablate_as_0.25:--action_scale 0.25"
    # Revert calql_alpha to old 5.0
    "ablate_calql_alpha_5.0:--calql_alpha 5.0"
    # Revert calql_pretrain_steps to old 2000
    "ablate_calql_2000:--calql_pretrain_steps 2000"
    # Revert online_ratio to old 0.5
    "ablate_or_0.5:--online_ratio 0.5"
    # Revert offline_demo_episodes to old 200
    "ablate_demos_200:--offline_demo_episodes 200"
)

# =============================================================================
# Group 2: 2-Way Interaction Tests
# Pairs of top improvements — do they compound or cancel out?
# Each combo pairs two NON-default values to see synergistic effects.
# =============================================================================
INTERACTION_CONFIGS=(
    # lr + batch: both stabilize training — does doubling down help?
    "interact_lr5e-5_batch2048:--learning_rate 0.00005 --batch_size 2048"
    # lr + arch: smaller network needs less regularization?
    "interact_lr3e-4_arch3x512:--learning_rate 0.0003 --num_layers 3 --layer_size 512"
    # temp + lr: temp=5.0 was also good, does it combo with lr?
    "interact_temp5.0_lr1e-4:--init_temperature 5.0"
    # gamma + tau: high gamma + lower tau for more stable long-horizon credit
    "interact_gamma0.99_tau0.001:--tau 0.001"
    # UTD=80 + batch=1024: v1 showed UTD=80 good, v2 showed batch=1024 good
    "interact_utd80_batch1024:--utd_ratio 80"
    # UTD=80 + lr=5e-5: push UTD higher while compensating with lower lr
    "interact_utd80_lr5e-5:--utd_ratio 80 --learning_rate 0.00005"
    # action_scale=0.3 + calql=0: skip pretraining entirely with optimal scale
    "interact_as0.3_calql0:--calql_pretrain_steps 0"
    # action_scale=0.2 + calql_1000: slightly tighter residual with warm-up
    "interact_as0.2_calql1000:--action_scale 0.2"
    # num_qs=3 + batch=1024: fewer Qs + large batch for less pessimism
    "interact_nq3_batch1024:--num_qs 3"
    # num_qs=7 + lr=1e-4: more Qs but with stable lr
    "interact_nq7_lr1e-4:--num_qs 7"
    # Pure online + no calql: simplest possible training
    "interact_no_offline:--calql_pretrain_steps 0 --offline_demo_episodes 0 --online_ratio 1.0"
    # Full offline ablation: use demos but skip calql
    "interact_demos50_calql0:--calql_pretrain_steps 0 --offline_demo_episodes 50"
)

# =============================================================================
# Group 3: Boundary Exploration (push the most impactful params further)
# =============================================================================
BOUNDARY_CONFIGS=(
    # Even lower lr
    "bound_lr_5e-5:--learning_rate 0.00005"
    "bound_lr_3e-5:--learning_rate 0.00003"
    # Even larger batch
    "bound_batch_2048:--batch_size 2048"
    # Smaller architectures
    "bound_arch_2x1024:--num_layers 2 --layer_size 1024"
    "bound_arch_3x768:--num_layers 3 --layer_size 768"
    "bound_arch_4x512:--num_layers 4 --layer_size 512"
    # Fewer Q-functions
    "bound_nq_3:--num_qs 3"
    "bound_nq_4:--num_qs 4"
    # Higher UTD with stabilization (low lr + large batch already in defaults)
    "bound_utd_80:--utd_ratio 80"
    "bound_utd_100:--utd_ratio 100"
)

# =============================================================================
# Assemble All Configs
# =============================================================================
SWEEP_CONFIGS=(
    "$BASELINE"
    "${ABLATION_CONFIGS[@]}"
    "${INTERACTION_CONFIGS[@]}"
    "${BOUNDARY_CONFIGS[@]}"
)

echo "Loaded ${#SWEEP_CONFIGS[@]} PLD-SAC v3 sweep configs (combination & ablation)"
