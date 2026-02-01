#!/bin/bash
# =============================================================================
# AWCP - Wave 2 Hyperparameter Sweep Configs
# =============================================================================
# Wave 1 covered: lr, beta, num_flow_steps
# Wave 2 focuses on: loss weights, advantage weighting, reward scale, critic settings
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # ---------------------------------------------------------------------
    # 1. Finer learning rate grid
    # ---------------------------------------------------------------------
    "lr_2e-4:--lr 2e-4"
    "lr_5e-4:--lr 5e-4"
    "lr_critic_1e-4:--lr_critic 1e-4"
    "lr_critic_5e-4:--lr_critic 5e-4"
    
    # ---------------------------------------------------------------------
    # 2. BC weight (flow matching loss weight)
    # ---------------------------------------------------------------------
    "bc_weight_0.5:--bc_weight 0.5"
    "bc_weight_2.0:--bc_weight 2.0"
    
    # ---------------------------------------------------------------------
    # 3. Consistency weight
    # ---------------------------------------------------------------------
    "cons_weight_0.1:--consistency_weight 0.1"
    "cons_weight_0.5:--consistency_weight 0.5"
    "cons_weight_1.0:--consistency_weight 1.0"
    
    # ---------------------------------------------------------------------
    # 4. Beta finer grid (advantage temperature)
    # Higher = more aggressive weighting (sharper preference)
    # ---------------------------------------------------------------------
    "beta_0.5:--beta 0.5"
    "beta_2.0:--beta 2.0"
    "beta_5.0:--beta 5.0"
    "beta_20.0:--beta 20.0"
    "beta_50.0:--beta 50.0"
    
    # ---------------------------------------------------------------------
    # 5. Weight clipping (prevents outlier dominance)
    # ---------------------------------------------------------------------
    "weight_clip_10:--weight_clip 10.0"
    "weight_clip_50:--weight_clip 50.0"
    "weight_clip_200:--weight_clip 200.0"
    
    # ---------------------------------------------------------------------
    # 6. Reward scale
    # ---------------------------------------------------------------------
    "reward_scale_0.05:--reward_scale 0.05"
    "reward_scale_0.2:--reward_scale 0.2"
    "reward_scale_0.5:--reward_scale 0.5"
    
    # ---------------------------------------------------------------------
    # 7. Q-learning hyperparameters
    # ---------------------------------------------------------------------
    "gamma_0.95:--gamma 0.95"
    "gamma_0.999:--gamma 0.999"
    "tau_0.001:--tau 0.001"
    "tau_0.01:--tau 0.01"
    
    # ---------------------------------------------------------------------
    # 8. Q target clipping
    # ---------------------------------------------------------------------
    "q_clip_50:--q_target_clip 50.0"
    "q_clip_200:--q_target_clip 200.0"
    
    # ---------------------------------------------------------------------
    # 9. Ensemble Q settings
    # ---------------------------------------------------------------------
    "num_qs_5:--num_qs 5 --num_min_qs 2"
    "num_qs_20:--num_qs 20 --num_min_qs 2"
    
    # ---------------------------------------------------------------------
    # 10. EMA decay
    # ---------------------------------------------------------------------
    "ema_0.99:--ema_decay 0.99"
    "ema_0.995:--ema_decay 0.995"
    
    # ---------------------------------------------------------------------
    # 11. Combined configurations
    # ---------------------------------------------------------------------
    "conservative:--beta 5.0 --bc_weight 2.0 --weight_clip 50.0"
    "aggressive:--beta 20.0 --bc_weight 0.5 --weight_clip 200.0"
)
