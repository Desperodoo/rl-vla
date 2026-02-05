#!/bin/bash
# =============================================================================
# Consistency Flow - Wave 2 Hyperparameter Sweep Configs
# =============================================================================
# Wave 1 covered: lr, cons_delta_mode, cons_delta_fixed, num_flow_steps
# Wave 2 focuses on: loss weights, teacher config, t range, ema_decay
# Format: "config_name:--param1 value1 --param2 value2"

SWEEP_CONFIGS=(
    # ---------------------------------------------------------------------
    # 1. Finer learning rate grid (around best lr=3e-4)
    # ---------------------------------------------------------------------
    "lr_2e-4:--lr 2e-4"
    "lr_5e-4:--lr 5e-4"
    
    # ---------------------------------------------------------------------
    # 2. Flow weight vs Consistency weight balance
    # Higher flow_weight = more BC-like, Higher consistency = faster inference
    # ---------------------------------------------------------------------
    "weights_flow_heavy:--bc_weight 2.0 --consistency_weight 0.5"
    "weights_cons_heavy:--bc_weight 0.5 --consistency_weight 2.0"
    "weights_equal:--bc_weight 1.0 --consistency_weight 1.0"
    "weights_flow_only:--bc_weight 1.0 --consistency_weight 0.0"
    
    # ---------------------------------------------------------------------
    # 3. EMA decay for teacher network
    # Faster decay = teacher adapts faster but less stable
    # ---------------------------------------------------------------------
    "ema_0.99:--ema_decay 0.99"
    "ema_0.995:--ema_decay 0.995"
    "ema_0.9999:--ema_decay 0.9999"
    
    # ---------------------------------------------------------------------
    # 4. Consistency t range (avoids boundary instability)
    # Wider range = more temporal diversity
    # ---------------------------------------------------------------------
    "t_range_tight:--cons_t_min 0.1 --cons_t_max 0.9"
    "t_range_wide:--cons_t_min 0.02 --cons_t_max 0.98"
    "t_range_full:--cons_full_t_range True"
    
    # ---------------------------------------------------------------------
    # 5. Teacher rollout steps
    # More steps = more accurate but slower
    # ---------------------------------------------------------------------
    "teacher_1step:--cons_teacher_steps 1"
    "teacher_3step:--cons_teacher_steps 3"
    "teacher_4step:--cons_teacher_steps 4"
    
    # ---------------------------------------------------------------------
    # 6. Consistency loss space
    # Velocity = better gradient flow, Endpoint = direct target matching
    # ---------------------------------------------------------------------
    "loss_endpoint:--cons_loss_space endpoint"
    
    # ---------------------------------------------------------------------
    # 7. Delta fine-grained (fixed mode, around best 0.02)
    # ---------------------------------------------------------------------
    "delta_fixed_0.01:--cons_delta_mode fixed --cons_delta_fixed 0.01"
    "delta_fixed_0.03:--cons_delta_mode fixed --cons_delta_fixed 0.03"
    "delta_fixed_0.04:--cons_delta_mode fixed --cons_delta_fixed 0.04"
    
    # ---------------------------------------------------------------------
    # 8. Random delta range exploration
    # ---------------------------------------------------------------------
    "delta_random_narrow:--cons_delta_mode random --cons_delta_min 0.01 --cons_delta_max 0.05"
    "delta_random_wide:--cons_delta_mode random --cons_delta_min 0.01 --cons_delta_max 0.2"
    
    # ---------------------------------------------------------------------
    # 9. Student/Teacher point variations
    # ---------------------------------------------------------------------
    "student_t_cons:--cons_student_point t_cons"
    "teacher_from_t_cons:--cons_teacher_from t_cons"
    
    # ---------------------------------------------------------------------
    # 10. Combined best practices
    # ---------------------------------------------------------------------
    "combined_stable:--cons_delta_mode fixed --cons_delta_fixed 0.02 --cons_teacher_steps 2 --ema_decay 0.999"
    "combined_fast:--cons_delta_mode fixed --cons_delta_fixed 0.05 --cons_teacher_steps 1 --ema_decay 0.995"
)
