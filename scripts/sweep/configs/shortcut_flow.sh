#!/bin/bash
# =============================================================================
# Shortcut Flow - Hyperparameter Sweep Configs
# =============================================================================
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # Learning rate sweep
    "lr_1e-4:--lr 1e-4"
    "lr_3e-4:--lr 3e-4"
    "lr_1e-3:--lr 1e-3"
    
    # Fixed step size sweep (1-step inference)
    "step_size_0.0625:--fixed_step_size 0.0625"
    "step_size_0.125:--fixed_step_size 0.125"
    "step_size_0.25:--fixed_step_size 0.25"
    "step_size_0.5:--fixed_step_size 0.5"
    "step_size_1.0:--fixed_step_size 1.0"
    
    # Flow steps sweep
    "flow_steps_10:--num_flow_steps 10"
    "flow_steps_20:--num_flow_steps 20"
)
