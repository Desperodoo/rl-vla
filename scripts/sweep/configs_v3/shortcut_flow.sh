#!/bin/bash
# =============================================================================
# ShortCut Flow - Wave 3 Hyperparameter Sweep Configs
# =============================================================================
# Focus: shortcut consistency balance (single-seed)
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # ---------------------------------------------------------------------
    # 1. Loss weight balance (best was 1.0 / 1.0)
    # ---------------------------------------------------------------------
    "weights_1.0_0.5:--bc_weight 1.0 --consistency_weight 0.5"
    "weights_1.0_1.0:--bc_weight 1.0 --consistency_weight 1.0"
    "weights_1.5_1.0:--bc_weight 1.5 --consistency_weight 1.0"
    
    # ---------------------------------------------------------------------
    # 2. Fixed step size around best
    # ---------------------------------------------------------------------
    "step_0.1:--sc_fixed_step_size 0.1"
    "step_0.125:--sc_fixed_step_size 0.125"
    "step_0.15:--sc_fixed_step_size 0.15"
    
    # ---------------------------------------------------------------------
    # 3. Self-consistency fraction
    # ---------------------------------------------------------------------
    "sc_k_0.25:--sc_self_consistency_k 0.25"
    "sc_k_0.5:--sc_self_consistency_k 0.5"
    
    # ---------------------------------------------------------------------
    # 4. Teacher steps
    # ---------------------------------------------------------------------
    "teacher_1step:--sc_teacher_steps 1"
    "teacher_2step:--sc_teacher_steps 2"
)
