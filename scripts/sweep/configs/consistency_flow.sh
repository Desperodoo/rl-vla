#!/bin/bash
# =============================================================================
# Consistency Flow - Hyperparameter Sweep Configs
# =============================================================================
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # Learning rate sweep
    "lr_1e-4:--lr 1e-4"
    "lr_3e-4:--lr 3e-4"
    "lr_1e-3:--lr 1e-3"
    
    # Consistency delta mode sweep
    "delta_fixed_0.01:--cons_delta_mode fixed --cons_delta_fixed 0.01"
    "delta_fixed_0.02:--cons_delta_mode fixed --cons_delta_fixed 0.02"
    "delta_fixed_0.05:--cons_delta_mode fixed --cons_delta_fixed 0.05"
    "delta_random:--cons_delta_mode random"
    
    # Flow steps sweep
    "flow_steps_10:--num_flow_steps 10"
    "flow_steps_20:--num_flow_steps 20"
)
