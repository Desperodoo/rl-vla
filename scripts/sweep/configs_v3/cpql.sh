#!/bin/bash
# =============================================================================
# CPQL - Wave 3 Hyperparameter Sweep Configs
# =============================================================================
# Focus: reduce Q-driven instability (single-seed)
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # ---------------------------------------------------------------------
    # 1. Alpha x BC weight (Q push vs BC anchor)
    # ---------------------------------------------------------------------
    "a0.0005_bc0.5:--alpha 0.0005 --bc_weight 0.5"
    "a0.001_bc0.5:--alpha 0.001 --bc_weight 0.5"
    "a0.002_bc0.3:--alpha 0.002 --bc_weight 0.3"
    "a0.001_bc0.8:--alpha 0.001 --bc_weight 0.8"
    
    # ---------------------------------------------------------------------
    # 2. Consistency weight (faster inference vs accuracy)
    # ---------------------------------------------------------------------
    "cons_w0.5:--consistency_weight 0.5"
    "cons_w1.0:--consistency_weight 1.0"
    
    # ---------------------------------------------------------------------
    # 3. Reward scale (Q-value stability)
    # ---------------------------------------------------------------------
    "rs0.05:--reward_scale 0.05"
    "rs0.1:--reward_scale 0.1"
    "rs0.2:--reward_scale 0.2"
    
    # ---------------------------------------------------------------------
    # 4. Q target clipping
    # ---------------------------------------------------------------------
    "qclip50:--q_target_clip 50.0"
    "qclip100:--q_target_clip 100.0"
    "qclip200:--q_target_clip 200.0"
)
