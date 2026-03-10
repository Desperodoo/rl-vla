#!/bin/bash
# =============================================================================
# Fair Comparison Configs — Best Parameters per Algorithm
# =============================================================================
# Each array defines configs for one algorithm.
# Format: "config_name:extra_args"
#
# Config names intentionally encode the key hyperparameters for traceability.
# extra_args override the base args built in utils.sh::build_train_command().
#
# 【来源】
#   AWSC  — RLPD sweep v3/v4 最优: online_ratio=0.15, beta=50, bc_weight=2.0
#   PLD   — PLD sweep v1/v2 最优:  action_scale=0.3, utd=60, gamma=0.99
#   DSRL  — DSRL sweep 最优:       action_mag=2.5, utd=60, gamma=0.95
# =============================================================================

# ── AWSC (RLPD) ──────────────────────────────────────────────────────────────
# Uses: python -m rlft.online.train_rlpd --algorithm awsc
# Note: online_ratio=0.15 mixes 15% online + 85% demo replay
AWSC_CONFIGS=(
    "best:--online_ratio 0.15 --utd_ratio 20 --lr_actor 1e-4 --lr_critic 1e-4 --num_qs 10 --num_min_qs 2 --awsc_beta 50.0 --awsc_bc_weight 2.0 --awsc_advantage_mode per_state_v --awsc_num_inference_steps 8"
)

# ── PLD-SAC ───────────────────────────────────────────────────────────────────
# Uses: python -m rlft.online.train_pld
# Note: pure online (online_ratio=1.0), offline demos only for Cal-QL pretrain
PLD_CONFIGS=(
    "best:--action_scale 0.3 --utd_ratio 60 --gamma 0.99 --target_entropy -3.5 --init_temperature 0.1 --learning_rate 1e-4 --num_layers 3 --layer_size 1024 --num_qs 5 --calql_pretrain_steps 1000 --calql_alpha 0.0 --online_ratio 1.0 --offline_demo_episodes 50"
)

# ── DSRL-SAC ──────────────────────────────────────────────────────────────────
# Uses: python -m rlft.online.train_dsrl
# Note: no seed steps (pretrained policy provides initial exploration)
DSRL_CONFIGS=(
    "best:--action_magnitude 2.5 --utd_ratio 60 --gamma 0.95 --target_entropy -3.5 --log_std_init -5.0 --learning_rate 3e-4 --num_layers 3 --layer_size 2048 --num_qs 10 --num_seed_steps 0"
)
