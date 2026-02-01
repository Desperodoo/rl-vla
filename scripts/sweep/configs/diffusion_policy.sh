#!/bin/bash
# =============================================================================
# Diffusion Policy - Hyperparameter Sweep Configs
# =============================================================================
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # Learning rate sweep
    "lr_1e-4:--lr 1e-4"
    "lr_3e-4:--lr 3e-4"
    "lr_1e-3:--lr 1e-3"
    
    # Diffusion steps sweep
    "ddpm_steps_50:--num_diffusion_steps 50"
    "ddpm_steps_100:--num_diffusion_steps 100"
    "ddpm_steps_200:--num_diffusion_steps 200"
)
