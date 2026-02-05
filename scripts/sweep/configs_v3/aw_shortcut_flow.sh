#!/bin/bash
# =============================================================================
# AW-ShortCut-Flow - Wave 3 Hyperparameter Sweep Configs
# =============================================================================
# Focus: joint stability of advantage weighting + shortcut consistency
# Single-seed sweep (budget-aware)
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # ---------------------------------------------------------------------
    # 1. Beta x Reward scale (strong coupling)
    # ---------------------------------------------------------------------
    "beta5_rs0.1:--beta 5.0 --reward_scale 0.1"
    "beta8_rs0.1:--beta 8.0 --reward_scale 0.1"
    "beta10_rs0.1:--beta 10.0 --reward_scale 0.1"
    "beta10_rs0.05:--beta 10.0 --reward_scale 0.05"
    "beta10_rs0.2:--beta 10.0 --reward_scale 0.2"
    "beta15_rs0.1:--beta 15.0 --reward_scale 0.1"
    
    # ---------------------------------------------------------------------
    # 2. Consistency weight x Fixed step size (shortcut bias vs stability)
    # ---------------------------------------------------------------------
    "cw0.1_step0.1:--consistency_weight 0.1 --sc_fixed_step_size 0.1"
    "cw0.2_step0.1:--consistency_weight 0.2 --sc_fixed_step_size 0.1"
    "cw0.3_step0.125:--consistency_weight 0.3 --sc_fixed_step_size 0.125"
    "cw0.5_step0.125:--consistency_weight 0.5 --sc_fixed_step_size 0.125"
    "cw0.3_step0.15:--consistency_weight 0.3 --sc_fixed_step_size 0.15"
    "cw0.5_step0.15:--consistency_weight 0.5 --sc_fixed_step_size 0.15"
    
    # ---------------------------------------------------------------------
    # 3. Weight clip (advantage outliers)
    # ---------------------------------------------------------------------
    "wclip50:--weight_clip 50.0"
    "wclip100:--weight_clip 100.0"
    "wclip200:--weight_clip 200.0"
    
    # ---------------------------------------------------------------------
    # 4. Q ensemble size (stability vs compute)
    # ---------------------------------------------------------------------
    "num_qs_2:--num_qs 2 --num_min_qs 1"
    "num_qs_5:--num_qs 5 --num_min_qs 2"
    "num_qs_10:--num_qs 10 --num_min_qs 2"
    
    # ---------------------------------------------------------------------
    # 5. Learning rate (minor tuning)
    # ---------------------------------------------------------------------
    "lr_2e-4:--lr 2e-4"
    "lr_3e-4:--lr 3e-4"
    "lr_5e-4:--lr 5e-4"
)
