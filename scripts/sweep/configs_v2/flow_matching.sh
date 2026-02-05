#!/bin/bash
# =============================================================================
# Flow Matching - Wave 2 Hyperparameter Sweep Configs
# =============================================================================
# Wave 1 covered: lr (1e-4, 3e-4, 1e-3), num_flow_steps (10, 20, 50)
# Wave 2 focuses on: finer lr grid, EMA decay, network architecture
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # ---------------------------------------------------------------------
    # 1. Finer learning rate grid (around best lr=3e-4)
    # ---------------------------------------------------------------------
    "lr_2e-4:--lr 2e-4"
    "lr_5e-4:--lr 5e-4"
    "lr_7e-4:--lr 7e-4"
    
    # ---------------------------------------------------------------------
    # 2. EMA decay variations
    # Higher decay = slower update = more stable but slower adaptation
    # ---------------------------------------------------------------------
    "ema_0.99:--ema_decay 0.99"
    "ema_0.995:--ema_decay 0.995"
    "ema_0.9999:--ema_decay 0.9999"
    
    # ---------------------------------------------------------------------
    # 3. Prediction horizon (action sequence length)
    # Longer horizon = more future planning but harder to train
    # ---------------------------------------------------------------------
    "pred_horizon_8:--pred_horizon 8"
    "pred_horizon_32:--pred_horizon 32"
    
    # ---------------------------------------------------------------------
    # 4. Observation horizon (context length)
    # More history = more context but more computation
    # ---------------------------------------------------------------------
    "obs_horizon_1:--obs_horizon 1"
    "obs_horizon_4:--obs_horizon 4"
    
    # ---------------------------------------------------------------------
    # 5. Action horizon (execution length)
    # Shorter = more reactive, Longer = more consistent
    # ---------------------------------------------------------------------
    "act_horizon_4:--act_horizon 4"
    "act_horizon_16:--act_horizon 16"
    
    # ---------------------------------------------------------------------
    # 6. Network architecture: U-Net dimensions
    # Smaller = faster but less expressive
    # ---------------------------------------------------------------------
    "unet_small:--unet_dims 32 64 128"
    "unet_large:--unet_dims 128 256 512"
    
    # ---------------------------------------------------------------------
    # 7. Batch size variations
    # Larger batch = more stable gradients but more memory
    # ---------------------------------------------------------------------
    "batch_128:--batch_size 128"
    "batch_512:--batch_size 512"
    
    # ---------------------------------------------------------------------
    # 8. Combined: Best lr + different flow steps (fine-grained)
    # ---------------------------------------------------------------------
    "lr_3e-4_steps_15:--lr 3e-4 --num_flow_steps 15"
    "lr_3e-4_steps_30:--lr 3e-4 --num_flow_steps 30"
)
