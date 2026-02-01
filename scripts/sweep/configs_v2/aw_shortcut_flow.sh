#!/bin/bash
# =============================================================================
# AW-ShortCut-Flow - Wave 2 Hyperparameter Sweep Configs
# =============================================================================
# Wave 1 covered: lr, beta, self_consistency_k, fixed_step_size, num_flow_steps
# Wave 2 focuses on: loss weights, step modes, reward scale, critic settings
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
    # 3. Shortcut weight (consistency loss weight)
    # Typically lower than bc_weight for stability
    # ---------------------------------------------------------------------
    "shortcut_weight_0.1:--consistency_weight 0.1"
    "shortcut_weight_0.5:--consistency_weight 0.5"
    "shortcut_weight_1.0:--consistency_weight 1.0"
    
    # ---------------------------------------------------------------------
    # 4. Beta finer grid (advantage temperature)
    # Lower beta is more stable for offline RL
    # ---------------------------------------------------------------------
    "beta_0.5:--beta 0.5"
    "beta_2.0:--beta 2.0"
    "beta_5.0:--beta 5.0"
    "beta_10.0:--beta 10.0"
    
    # ---------------------------------------------------------------------
    # 5. Weight clipping
    # ---------------------------------------------------------------------
    "weight_clip_10:--weight_clip 10.0"
    "weight_clip_50:--weight_clip 50.0"
    "weight_clip_200:--weight_clip 200.0"
    
    # ---------------------------------------------------------------------
    # 6. Reward scale
    # Critical for Q-value stability
    # ---------------------------------------------------------------------
    "reward_scale_0.05:--reward_scale 0.05"
    "reward_scale_0.2:--reward_scale 0.2"
    "reward_scale_0.5:--reward_scale 0.5"
    
    # ---------------------------------------------------------------------
    # 7. Step size mode variations
    # ---------------------------------------------------------------------
    "step_uniform:--sc_step_size_mode uniform --sc_min_step_size 0.0625 --sc_max_step_size 0.25"
    "step_power2:--sc_step_size_mode power2"
    
    # ---------------------------------------------------------------------
    # 8. Target mode
    # ---------------------------------------------------------------------
    "target_endpoint:--sc_target_mode endpoint"
    
    # ---------------------------------------------------------------------
    # 9. Inference settings
    # ---------------------------------------------------------------------
    "inference_4:--sc_num_inference_steps 4"
    "inference_16:--sc_num_inference_steps 16"
    "inference_adaptive:--sc_inference_mode adaptive"
    
    # ---------------------------------------------------------------------
    # 10. Teacher configuration
    # ---------------------------------------------------------------------
    "teacher_2step:--sc_teacher_steps 2"
    "teacher_no_ema:--sc_use_ema_teacher False"
    
    # ---------------------------------------------------------------------
    # 11. Q-learning hyperparameters
    # ---------------------------------------------------------------------
    "gamma_0.95:--gamma 0.95"
    "gamma_0.999:--gamma 0.999"
    "tau_0.001:--tau 0.001"
    "tau_0.01:--tau 0.01"
    
    # ---------------------------------------------------------------------
    # 12. Ensemble Q settings
    # ---------------------------------------------------------------------
    "num_qs_5:--num_qs 5 --num_min_qs 2"
    "num_qs_20:--num_qs 20 --num_min_qs 2"
    
    # ---------------------------------------------------------------------
    # 13. EMA decay
    # ---------------------------------------------------------------------
    "ema_0.99:--ema_decay 0.99"
    "ema_0.995:--ema_decay 0.995"
    
    # ---------------------------------------------------------------------
    # 14. Fixed step size finer grid (around best 0.125)
    # ---------------------------------------------------------------------
    "step_0.1:--sc_fixed_step_size 0.1"
    "step_0.15:--sc_fixed_step_size 0.15"
    "step_0.2:--sc_fixed_step_size 0.2"
    
    # ---------------------------------------------------------------------
    # 15. Combined: Stability-focused
    # ---------------------------------------------------------------------
    "stable:--beta 1.0 --bc_weight 1.0 --consistency_weight 0.3 --reward_scale 0.1 --sc_fixed_step_size 0.125"
    "aggressive:--beta 5.0 --bc_weight 0.5 --consistency_weight 0.5 --reward_scale 0.2 --sc_fixed_step_size 0.25"
)
