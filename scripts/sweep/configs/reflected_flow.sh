#!/bin/bash
# =============================================================================
# Reflected Flow - Hyperparameter Sweep Configs
# =============================================================================
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # Learning rate sweep
    "lr_1e-4:--lr 1e-4"
    "lr_3e-4:--lr 3e-4"
    "lr_1e-3:--lr 1e-3"
    
    # Flow steps sweep
    "flow_steps_10:--num_flow_steps 10"
    "flow_steps_20:--num_flow_steps 20"
    "flow_steps_50:--num_flow_steps 50"
    
    # Combined
    "lr_3e-4_steps_20:--lr 3e-4 --num_flow_steps 20"
)
