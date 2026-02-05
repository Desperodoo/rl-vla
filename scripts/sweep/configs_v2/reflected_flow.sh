#!/bin/bash
# =============================================================================
# Reflected Flow - Wave 2 Hyperparameter Sweep Configs
# =============================================================================
# Wave 1 covered: lr, num_flow_steps
# Wave 2 focuses on: reflection mode, boundary regularization, architecture
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # ---------------------------------------------------------------------
    # 1. Finer learning rate grid
    # ---------------------------------------------------------------------
    "lr_2e-4:--lr 2e-4"
    "lr_5e-4:--lr 5e-4"
    
    # ---------------------------------------------------------------------
    # 2. Reflection mode
    # Hard = trajectory reflection, Soft = regularization only
    # ---------------------------------------------------------------------
    "reflection_soft:--reflection_mode soft"
    
    # ---------------------------------------------------------------------
    # 3. Boundary regularization weight (for soft mode)
    # Higher = stronger penalty for boundary violations
    # ---------------------------------------------------------------------
    "boundary_reg_0.001:--boundary_reg_weight 0.001"
    "boundary_reg_0.005:--boundary_reg_weight 0.005"
    "boundary_reg_0.02:--boundary_reg_weight 0.02"
    "boundary_reg_0.05:--boundary_reg_weight 0.05"
    "boundary_reg_0.1:--boundary_reg_weight 0.1"
    
    # ---------------------------------------------------------------------
    # 4. Soft reflection with different regularization
    # ---------------------------------------------------------------------
    "soft_reg_0.01:--reflection_mode soft --boundary_reg_weight 0.01"
    "soft_reg_0.05:--reflection_mode soft --boundary_reg_weight 0.05"
    "soft_reg_0.1:--reflection_mode soft --boundary_reg_weight 0.1"
    
    # ---------------------------------------------------------------------
    # 5. EMA decay
    # ---------------------------------------------------------------------
    "ema_0.99:--ema_decay 0.99"
    "ema_0.995:--ema_decay 0.995"
    
    # ---------------------------------------------------------------------
    # 6. Flow steps finer grid (around best 20)
    # ---------------------------------------------------------------------
    "flow_steps_15:--num_flow_steps 15"
    "flow_steps_25:--num_flow_steps 25"
    "flow_steps_30:--num_flow_steps 30"
    
    # ---------------------------------------------------------------------
    # 7. Prediction/Action horizon
    # ---------------------------------------------------------------------
    "pred_horizon_8:--pred_horizon 8"
    "pred_horizon_32:--pred_horizon 32"
    "act_horizon_4:--act_horizon 4"
    "act_horizon_16:--act_horizon 16"
    
    # ---------------------------------------------------------------------
    # 8. Network architecture
    # ---------------------------------------------------------------------
    "unet_small:--unet_dims 32 64 128"
    "unet_large:--unet_dims 128 256 512"
    
    # ---------------------------------------------------------------------
    # 9. Combined: Hard reflection with different flow steps
    # ---------------------------------------------------------------------
    "hard_steps_15:--reflection_mode hard --num_flow_steps 15"
    "hard_steps_25:--reflection_mode hard --num_flow_steps 25"
)
