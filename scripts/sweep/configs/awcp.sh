#!/bin/bash
# =============================================================================
# AWCP (Advantage Weighted Consistency Policy) - Hyperparameter Sweep Configs
# =============================================================================
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # Learning rate sweep
    "lr_1e-4:--lr 1e-4"
    "lr_3e-4:--lr 3e-4"
    "lr_1e-3:--lr 1e-3"
    
    # Beta (advantage weighting temperature) sweep
    "beta_0.1:--beta 0.1"
    "beta_1.0:--beta 1.0"
    "beta_10.0:--beta 10.0"
    
    # Consistency delta sweep
    "delta_fixed_0.01:--cons_delta_mode fixed --cons_delta_fixed 0.01"
    "delta_fixed_0.02:--cons_delta_mode fixed --cons_delta_fixed 0.02"
    "delta_fixed_0.05:--cons_delta_mode fixed --cons_delta_fixed 0.05"
    
    # Flow steps sweep
    "flow_steps_10:--num_flow_steps 10"
    "flow_steps_20:--num_flow_steps 20"
)
