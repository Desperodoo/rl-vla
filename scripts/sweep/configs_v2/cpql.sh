#!/bin/bash
# =============================================================================
# CPQL - Wave 2 Hyperparameter Sweep Configs
# =============================================================================
# Wave 1 covered: lr, alpha, num_flow_steps
# Wave 2 focuses on: loss weights, Q-learning settings, gradient modes, critic architecture
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
    # Higher = more BC-like, Lower = more Q-driven
    # ---------------------------------------------------------------------
    "bc_weight_0.5:--bc_weight 0.5"
    "bc_weight_2.0:--bc_weight 2.0"
    "bc_weight_5.0:--bc_weight 5.0"
    
    # ---------------------------------------------------------------------
    # 3. Consistency weight variations
    # Higher = faster inference but potentially less accurate
    # ---------------------------------------------------------------------
    "cons_weight_0.1:--consistency_weight 0.1"
    "cons_weight_0.5:--consistency_weight 0.5"
    "cons_weight_1.0:--consistency_weight 1.0"
    
    # ---------------------------------------------------------------------
    # 4. Alpha finer grid (around best 0.001)
    # Critical for offline RL stability
    # ---------------------------------------------------------------------
    "alpha_0.0005:--alpha 0.0005"
    "alpha_0.002:--alpha 0.002"
    "alpha_0.005:--alpha 0.005"
    
    # ---------------------------------------------------------------------
    # 5. Reward scale (prevents Q-value explosion)
    # ---------------------------------------------------------------------
    "reward_scale_0.05:--reward_scale 0.05"
    "reward_scale_0.2:--reward_scale 0.2"
    "reward_scale_0.5:--reward_scale 0.5"
    
    # ---------------------------------------------------------------------
    # 6. Q-learning hyperparameters
    # ---------------------------------------------------------------------
    "gamma_0.95:--gamma 0.95"
    "gamma_0.999:--gamma 0.999"
    "tau_0.001:--tau 0.001"
    "tau_0.01:--tau 0.01"
    
    # ---------------------------------------------------------------------
    # 7. Q target clipping
    # ---------------------------------------------------------------------
    "q_clip_50:--q_target_clip 50.0"
    "q_clip_200:--q_target_clip 200.0"
    
    # ---------------------------------------------------------------------
    # 8. Ensemble Q settings
    # More Qs = more conservative but more stable
    # ---------------------------------------------------------------------
    "num_qs_5:--num_qs 5 --num_min_qs 2"
    "num_qs_20:--num_qs 20 --num_min_qs 2"
    "num_min_qs_1:--num_qs 10 --num_min_qs 1"
    "num_min_qs_5:--num_qs 10 --num_min_qs 5"
    
    # ---------------------------------------------------------------------
    # 9. EMA decay for policy
    # ---------------------------------------------------------------------
    "ema_0.99:--ema_decay 0.99"
    "ema_0.995:--ema_decay 0.995"
    
    # ---------------------------------------------------------------------
    # 10. Combined: Conservative settings (for stability)
    # ---------------------------------------------------------------------
    "conservative:--alpha 0.0005 --bc_weight 2.0 --reward_scale 0.05"
    "aggressive:--alpha 0.005 --bc_weight 0.5 --reward_scale 0.2"
)
