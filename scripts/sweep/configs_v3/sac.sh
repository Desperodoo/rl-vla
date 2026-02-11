#!/bin/bash
# =============================================================================
# SAC (Offline) — Wave 3: Hyperparameter Fine-Tuning
# =============================================================================
# Run AFTER v2 (regularization sweep) identifies working method(s).
# These configs assume you've picked a regularization method from v2 and
# now want to fine-tune learning rates, ensemble, discount, etc.
#
# The default regularization (td3bc, bc_weight=2.0) is used unless overridden.
# After v2, update the defaults here if a different method works better.
#
# Format: "config_name:--param1 value1 --param2 value2"
# =============================================================================

SWEEP_CONFIGS=(
    # =====================================================================
    # 0. Baseline (all defaults — same as v2 baseline for reference)
    # =====================================================================
    "baseline:"

    # =====================================================================
    # 1. Learning Rate — Actor (lr controls actor optimizer)
    # =====================================================================
    "lr_1e-4:--lr 1e-4"
    "lr_2e-4:--lr 2e-4"
    "lr_5e-4:--lr 5e-4"
    "lr_1e-3:--lr 1e-3"

    # =====================================================================
    # 2. Learning Rate — Critic (lr_critic controls critic optimizer)
    # =====================================================================
    "lr_critic_1e-4:--lr_critic 1e-4"
    "lr_critic_5e-4:--lr_critic 5e-4"
    "lr_critic_1e-3:--lr_critic 1e-3"

    # =====================================================================
    # 3. Learning Rate — Asymmetric (critic faster / slower than actor)
    # =====================================================================
    "lr_asym_slow_actor:--lr 1e-4 --lr_critic 3e-4"
    "lr_asym_fast_critic:--lr 3e-4 --lr_critic 1e-3"
    "lr_asym_fast_actor:--lr 1e-3 --lr_critic 3e-4"

    # =====================================================================
    # 4. Initial Temperature
    # =====================================================================
    "init_temp_0.1:--init_temperature 0.1"
    "init_temp_0.3:--init_temperature 0.3"
    "init_temp_0.5:--init_temperature 0.5"
    "init_temp_3.0:--init_temperature 3.0"

    # =====================================================================
    # 5. Target Entropy
    #    Default: -action_dim * act_horizon = -7*8 = -56
    # =====================================================================
    "target_ent_-28:--target_entropy -28.0"
    "target_ent_-42:--target_entropy -42.0"
    "target_ent_-70:--target_entropy -70.0"
    "target_ent_-84:--target_entropy -84.0"

    # =====================================================================
    # 6. Backup Entropy
    # =====================================================================
    "backup_entropy:--backup_entropy"
    "backup_entropy_low_temp:--backup_entropy --init_temperature 0.3"
    "backup_entropy_hi_temp:--backup_entropy --init_temperature 3.0"

    # =====================================================================
    # 7. Actor Q Aggregation Mode
    # =====================================================================
    "actor_q_mean:--actor_q_mode mean"
    "actor_q_mean_backup:--actor_q_mode mean --backup_entropy"

    # =====================================================================
    # 8. Ensemble Q Size
    # =====================================================================
    "num_qs_2:--num_qs 2 --num_min_qs 2"
    "num_qs_5:--num_qs 5 --num_min_qs 2"
    "num_qs_20:--num_qs 20 --num_min_qs 2"

    # =====================================================================
    # 9. Ensemble Q — min_qs (subsample size for pessimism)
    # =====================================================================
    "num_min_qs_1:--num_qs 10 --num_min_qs 1"
    "num_min_qs_3:--num_qs 10 --num_min_qs 3"
    "num_min_qs_5:--num_qs 10 --num_min_qs 5"

    # =====================================================================
    # 10. Discount Factor (gamma)
    # =====================================================================
    "gamma_0.95:--gamma 0.95"
    "gamma_0.98:--gamma 0.98"
    "gamma_0.999:--gamma 0.999"

    # =====================================================================
    # 11. Soft Update Rate (tau)
    # =====================================================================
    "tau_0.001:--tau 0.001"
    "tau_0.01:--tau 0.01"
    "tau_0.02:--tau 0.02"

    # =====================================================================
    # 12. Reward Scale
    # =====================================================================
    "reward_scale_0.01:--sac_reward_scale 0.01"
    "reward_scale_0.1:--sac_reward_scale 0.1"
    "reward_scale_0.5:--sac_reward_scale 0.5"
    "reward_scale_5.0:--sac_reward_scale 5.0"

    # =====================================================================
    # 13. Q Target Clip
    # =====================================================================
    "q_clip_20:--q_target_clip 20.0"
    "q_clip_50:--q_target_clip 50.0"
    "q_clip_200:--q_target_clip 200.0"
    "q_clip_none:--q_target_clip 1000.0"

    # =====================================================================
    # 14. Batch Size
    # =====================================================================
    "batch_128:--batch_size 128"
    "batch_512:--batch_size 512"
    "batch_1024:--batch_size 1024"

    # =====================================================================
    # 15. Combined — Conservative preset
    # =====================================================================
    "combined_conservative:--num_qs 20 --num_min_qs 2 --init_temperature 0.3 --target_entropy -70.0 --backup_entropy --tau 0.001"

    # =====================================================================
    # 16. Combined — Aggressive preset (fast learning)
    # =====================================================================
    "combined_aggressive:--num_qs 5 --num_min_qs 2 --init_temperature 3.0 --tau 0.01 --sac_reward_scale 5.0 --lr 5e-4 --lr_critic 1e-3"

    # =====================================================================
    # 17. Combined — SAC-N style (many Qs, few min)
    # =====================================================================
    "combined_sacn_style:--num_qs 20 --num_min_qs 1 --sac_reward_scale 1.0 --tau 0.005"

    # =====================================================================
    # 18. Combined — Low LR + Large batch (stability focused)
    # =====================================================================
    "combined_stable:--lr 1e-4 --lr_critic 1e-4 --batch_size 512 --tau 0.001 --num_qs 10 --num_min_qs 2"

    # =====================================================================
    # 19. Reward scale x Temperature interaction
    # =====================================================================
    "rs0.01_temp0.1:--sac_reward_scale 0.01 --init_temperature 0.1"
    "rs0.1_temp0.3:--sac_reward_scale 0.1 --init_temperature 0.3"
    "rs0.5_temp1.0:--sac_reward_scale 0.5 --init_temperature 1.0"
    "rs1.0_temp3.0:--sac_reward_scale 1.0 --init_temperature 3.0"

    # =====================================================================
    # 20. Learning Rate — Temperature optimizer
    # =====================================================================
    "lr_temp_1e-5:--lr_temperature 1e-5"
    "lr_temp_5e-5:--lr_temperature 5e-5"
    "lr_temp_3e-4:--lr_temperature 3e-4"
    "lr_temp_1e-3:--lr_temperature 1e-3"
)
