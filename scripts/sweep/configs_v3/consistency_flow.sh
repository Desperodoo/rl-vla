#!/bin/bash
# =============================================================================
# Consistency Flow - Wave 3 Hyperparameter Sweep Configs
# =============================================================================
# Focus: fixed-delta consistency and teacher stability (single-seed)
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # ---------------------------------------------------------------------
    # 1. Delta around best fixed value
    # ---------------------------------------------------------------------
    "delta_fixed_0.03:--cons_delta_mode fixed --cons_delta_fixed 0.03"
    "delta_fixed_0.04:--cons_delta_mode fixed --cons_delta_fixed 0.04"
    "delta_fixed_0.05:--cons_delta_mode fixed --cons_delta_fixed 0.05"
    
    # ---------------------------------------------------------------------
    # 2. Teacher steps (stability vs speed)
    # ---------------------------------------------------------------------
    "teacher_1step:--cons_teacher_steps 1"
    "teacher_2step:--cons_teacher_steps 2"
    "teacher_3step:--cons_teacher_steps 3"
    
    # ---------------------------------------------------------------------
    # 3. EMA decay (teacher stability)
    # ---------------------------------------------------------------------
    "ema_0.995:--ema_decay 0.995"
    "ema_0.999:--ema_decay 0.999"
    "ema_0.9995:--ema_decay 0.9995"
    
    # ---------------------------------------------------------------------
    # 4. Loss weight balance
    # ---------------------------------------------------------------------
    "weights_1.0_1.0:--bc_weight 1.0 --consistency_weight 1.0"
    "weights_1.0_0.5:--bc_weight 1.0 --consistency_weight 0.5"
    "weights_0.5_1.0:--bc_weight 0.5 --consistency_weight 1.0"
)
