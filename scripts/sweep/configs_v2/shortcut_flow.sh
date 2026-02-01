#!/bin/bash
# =============================================================================
# ShortCut Flow - Wave 2 Hyperparameter Sweep Configs
# =============================================================================
# Wave 1 covered: lr, fixed_step_size, num_flow_steps
# Wave 2 focuses on: loss weights, step modes, teacher config, inference settings
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # ---------------------------------------------------------------------
    # 1. Finer learning rate grid
    # ---------------------------------------------------------------------
    "lr_2e-4:--lr 2e-4"
    "lr_5e-4:--lr 5e-4"
    
    # ---------------------------------------------------------------------
    # 2. Flow weight vs Shortcut weight balance
    # flow_weight > shortcut_weight recommended for stability
    # ---------------------------------------------------------------------
    "weights_1.0_0.1:--bc_weight 1.0 --consistency_weight 0.1"
    "weights_1.0_0.5:--bc_weight 1.0 --consistency_weight 0.5"
    "weights_1.0_1.0:--bc_weight 1.0 --consistency_weight 1.0"
    "weights_2.0_0.3:--bc_weight 2.0 --consistency_weight 0.3"
    
    # ---------------------------------------------------------------------
    # 3. Self-consistency fraction (portion of batch for shortcut training)
    # Higher = more shortcut training but slower flow matching
    # ---------------------------------------------------------------------
    "sc_k_0.1:--sc_self_consistency_k 0.1"
    "sc_k_0.5:--sc_self_consistency_k 0.5"
    "sc_k_0.75:--sc_self_consistency_k 0.75"
    
    # ---------------------------------------------------------------------
    # 4. Step size mode variations
    # Fixed is most stable, but uniform/power2 may help generalization
    # ---------------------------------------------------------------------
    "step_uniform_small:--sc_step_size_mode uniform --sc_min_step_size 0.0625 --sc_max_step_size 0.125"
    "step_uniform_wide:--sc_step_size_mode uniform --sc_min_step_size 0.0625 --sc_max_step_size 0.25"
    "step_power2:--sc_step_size_mode power2"
    
    # ---------------------------------------------------------------------
    # 5. Target mode: velocity vs endpoint
    # Velocity has better gradient flow (recommended)
    # ---------------------------------------------------------------------
    "target_endpoint:--sc_target_mode endpoint"
    
    # ---------------------------------------------------------------------
    # 6. Teacher configuration
    # ---------------------------------------------------------------------
    "teacher_2step:--sc_teacher_steps 2"
    "teacher_no_ema:--sc_use_ema_teacher False"
    
    # ---------------------------------------------------------------------
    # 7. Inference mode and steps
    # Uniform is more stable than adaptive
    # ---------------------------------------------------------------------
    "inference_4:--sc_num_inference_steps 4"
    "inference_16:--sc_num_inference_steps 16"
    "inference_adaptive:--sc_inference_mode adaptive"
    
    # ---------------------------------------------------------------------
    # 8. EMA decay
    # ---------------------------------------------------------------------
    "ema_0.99:--ema_decay 0.99"
    "ema_0.995:--ema_decay 0.995"
    
    # ---------------------------------------------------------------------
    # 9. Time sampling variations
    # ---------------------------------------------------------------------
    "t_truncated:--sc_t_sampling_mode truncated --sc_t_min 0.05 --sc_t_max 0.95"
    
    # ---------------------------------------------------------------------
    # 10. Fixed step size finer grid (around best 0.125)
    # ---------------------------------------------------------------------
    "step_0.1:--sc_fixed_step_size 0.1"
    "step_0.15:--sc_fixed_step_size 0.15"
    "step_0.2:--sc_fixed_step_size 0.2"
    
    # ---------------------------------------------------------------------
    # 11. Combined configs for ablation
    # ---------------------------------------------------------------------
    "stable_1step:--sc_fixed_step_size 0.125 --sc_self_consistency_k 0.25 --sc_teacher_steps 1"
    "aggressive_2step:--sc_fixed_step_size 0.25 --sc_self_consistency_k 0.5 --sc_teacher_steps 2"
)
