#!/bin/bash
# =============================================================================
# CPQL (Conservative Policy Q-Learning) - Hyperparameter Sweep Configs
# =============================================================================
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # Learning rate sweep
    "lr_1e-4:--lr 1e-4"
    "lr_3e-4:--lr 3e-4"
    "lr_1e-3:--lr 1e-3"
    
    # Alpha (conservative regularization) sweep
    "alpha_0.0001:--alpha 0.0001"
    "alpha_0.001:--alpha 0.001"
    "alpha_0.01:--alpha 0.01"
    "alpha_0.1:--alpha 0.1"
    
    # Flow steps sweep
    "flow_steps_10:--num_flow_steps 10"
    "flow_steps_20:--num_flow_steps 20"
)
