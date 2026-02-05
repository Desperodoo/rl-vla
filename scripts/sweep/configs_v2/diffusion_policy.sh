#!/bin/bash
# =============================================================================
# Diffusion Policy - Wave 2 Hyperparameter Sweep Configs
# =============================================================================
# Wave 1 covered: lr (1e-4, 3e-4, 1e-3), num_diffusion_iters (50, 100, 200)
# Wave 2 focuses on: finer lr grid, DDPM scheduler settings, architecture
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # ---------------------------------------------------------------------
    # 1. Finer learning rate grid
    # ---------------------------------------------------------------------
    "lr_2e-4:--lr 2e-4"
    "lr_5e-4:--lr 5e-4"
    "lr_7e-4:--lr 7e-4"
    
    # ---------------------------------------------------------------------
    # 2. EMA decay (more important for diffusion models)
    # Diffusion models are sensitive to EMA for stable inference
    # ---------------------------------------------------------------------
    "ema_0.99:--ema_decay 0.99"
    "ema_0.995:--ema_decay 0.995"
    "ema_0.9999:--ema_decay 0.9999"
    
    # ---------------------------------------------------------------------
    # 3. Prediction horizon variations
    # ---------------------------------------------------------------------
    "pred_horizon_8:--pred_horizon 8"
    "pred_horizon_32:--pred_horizon 32"
    
    # ---------------------------------------------------------------------
    # 4. Observation horizon variations
    # ---------------------------------------------------------------------
    "obs_horizon_1:--obs_horizon 1"
    "obs_horizon_4:--obs_horizon 4"
    
    # ---------------------------------------------------------------------
    # 5. Action horizon (execution chunk size)
    # ---------------------------------------------------------------------
    "act_horizon_4:--act_horizon 4"
    "act_horizon_16:--act_horizon 16"
    
    # ---------------------------------------------------------------------
    # 6. Network architecture variations
    # ---------------------------------------------------------------------
    "unet_small:--unet_dims 32 64 128"
    "unet_large:--unet_dims 128 256 512"
    
    # ---------------------------------------------------------------------
    # 7. Batch size
    # ---------------------------------------------------------------------
    "batch_128:--batch_size 128"
    "batch_512:--batch_size 512"
    
    # ---------------------------------------------------------------------
    # 8. Diffusion steps finer grid (around 100)
    # ---------------------------------------------------------------------
    "ddpm_steps_75:--num_diffusion_iters 75"
    "ddpm_steps_150:--num_diffusion_iters 150"
    
    # ---------------------------------------------------------------------
    # 9. Combined: Smaller steps + smaller network (efficiency focus)
    # ---------------------------------------------------------------------
    "efficient_small:--num_diffusion_iters 50 --unet_dims 32 64 128"
    "efficient_medium:--num_diffusion_iters 75 --unet_dims 64 128 256"
)
