#!/bin/bash
# =============================================================================
# SAC (Offline) - Comprehensive Hyperparameter Sweep
# =============================================================================
# First sweep for Offline SAC — Gaussian policy baseline for offline RL.
# Since this is the only algorithm being swept, configs are comprehensive.
#
# Architecture: DiagGaussianActor + EnsembleQNetwork + LearnableTemperature
# Key insight: Ensemble Q (subsample + min) provides implicit conservatism;
#              no CQL penalty needed (similar to SAC-N / EDAC approach).
#
# Hypotheses:
#   H1: Ensemble pessimism (num_qs × num_min_qs) is the most critical axis
#   H2: Lower entropy target helps for offline (fixed data → less exploration)
#   H3: backup_entropy=True stabilizes Q-values (entropy in target)
#   H4: Critic should learn faster than actor (lr_critic > lr ratio)
#   H5: Reward scale strongly coupled with temperature tuning
#   H6: Q target clip prevents divergence from offline distribution shift
#
# Format: "config_name:--param1 value1 --param2 value2"
# =============================================================================

SWEEP_CONFIGS=(
    # =====================================================================
    # 0. Baseline (all defaults)
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
    #    Hypothesis: critic should see stable targets before actor adapts
    # =====================================================================
    "lr_asym_slow_actor:--lr 1e-4 --lr_critic 3e-4"
    "lr_asym_fast_critic:--lr 3e-4 --lr_critic 1e-3"
    "lr_asym_fast_actor:--lr 1e-3 --lr_critic 3e-4"

    # =====================================================================
    # 4. Initial Temperature
    #    Controls initial entropy regularization strength
    # =====================================================================
    "init_temp_0.1:--init_temperature 0.1"
    "init_temp_0.3:--init_temperature 0.3"
    "init_temp_0.5:--init_temperature 0.5"
    "init_temp_3.0:--init_temperature 3.0"

    # =====================================================================
    # 5. Target Entropy
    #    Default: -action_dim * act_horizon = -7*8 = -56
    #    Lower = more deterministic policy; Higher = more stochastic
    # =====================================================================
    "target_ent_-28:--target_entropy -28.0"
    "target_ent_-42:--target_entropy -42.0"
    "target_ent_-70:--target_entropy -70.0"
    "target_ent_-84:--target_entropy -84.0"

    # =====================================================================
    # 6. Backup Entropy (entropy in Q-target)
    #    True: Q-target includes -alpha*logpi → more conservative
    # =====================================================================
    "backup_entropy:--backup_entropy"
    "backup_entropy_low_temp:--backup_entropy --init_temperature 0.3"
    "backup_entropy_hi_temp:--backup_entropy --init_temperature 3.0"

    # =====================================================================
    # 7. Actor Q Aggregation Mode
    #    min = pessimistic (SB3 default), mean = optimistic
    # =====================================================================
    "actor_q_mean:--actor_q_mode mean"
    "actor_q_mean_backup:--actor_q_mode mean --backup_entropy"

    # =====================================================================
    # 8. Ensemble Q Size (H1: key axis for conservatism)
    #    More Qs + fewer min_qs = stronger pessimism
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
    #    Lower = more myopic, higher = longer-horizon credit assignment
    # =====================================================================
    "gamma_0.95:--gamma 0.95"
    "gamma_0.98:--gamma 0.98"
    "gamma_0.999:--gamma 0.999"

    # =====================================================================
    # 11. Soft Update Rate (tau)
    #     Lower = more stable but slower target updates
    # =====================================================================
    "tau_0.001:--tau 0.001"
    "tau_0.01:--tau 0.01"
    "tau_0.02:--tau 0.02"

    # =====================================================================
    # 12. Reward Scale (H5: coupled with temperature)
    #     Scales rewards before TD computation; affects Q magnitude
    # =====================================================================
    "reward_scale_0.01:--reward_scale 0.01"
    "reward_scale_0.1:--reward_scale 0.1"
    "reward_scale_0.5:--reward_scale 0.5"
    "reward_scale_1.0:--reward_scale 1.0"

    # =====================================================================
    # 13. Q Target Clip (H6: prevents divergence)
    # =====================================================================
    "q_clip_20:--q_target_clip 20.0"
    "q_clip_50:--q_target_clip 50.0"
    "q_clip_200:--q_target_clip 200.0"
    "q_clip_none:--q_target_clip 1000.0"

    # =====================================================================
    # 14. Batch Size
    #     Larger batch = more stable gradients, especially for ensemble Q
    # =====================================================================
    "batch_128:--batch_size 128"
    "batch_512:--batch_size 512"
    "batch_1024:--batch_size 1024"

    # =====================================================================
    # 15. Combined — Conservative preset
    #     Large ensemble, low entropy, stable updates, backup entropy
    # =====================================================================
    "combined_conservative:--num_qs 20 --num_min_qs 2 --init_temperature 0.3 --target_entropy -70.0 --backup_entropy --tau 0.001 --reward_scale 0.05"

    # =====================================================================
    # 16. Combined — Aggressive preset
    #     Smaller ensemble, high entropy, fast updates
    # =====================================================================
    "combined_aggressive:--num_qs 5 --num_min_qs 2 --init_temperature 3.0 --tau 0.01 --reward_scale 0.5 --lr 5e-4 --lr_critic 1e-3"

    # =====================================================================
    # 17. Combined — SAC-N style (many Qs, few min)
    #     Maximum pessimism through ensemble diversification
    # =====================================================================
    "combined_sacn_style:--num_qs 20 --num_min_qs 1 --reward_scale 0.1 --tau 0.005"

    # =====================================================================
    # 18. Combined — EDAC style (moderate ensemble + backup entropy)
    # =====================================================================
    "combined_edac_style:--num_qs 10 --num_min_qs 2 --backup_entropy --init_temperature 0.5 --reward_scale 0.1"

    # =====================================================================
    # 19. Combined — Matched AWCP-best settings (translate from flow agent)
    #     Reward scale and Q settings from best AWCP config
    # =====================================================================
    "combined_match_awcp:--reward_scale 0.05 --gamma 0.99 --tau 0.005 --num_qs 10 --num_min_qs 2 --q_target_clip 100.0"

    # =====================================================================
    # 20. Combined — Low LR + Large batch (stability focused)
    # =====================================================================
    "combined_stable:--lr 1e-4 --lr_critic 1e-4 --batch_size 512 --tau 0.001 --num_qs 10 --num_min_qs 2"

    # =====================================================================
    # 21. Combined — Reward scale x Temperature interaction
    # =====================================================================
    "rs0.01_temp0.1:--reward_scale 0.01 --init_temperature 0.1"
    "rs0.1_temp0.3:--reward_scale 0.1 --init_temperature 0.3"
    "rs0.5_temp1.0:--reward_scale 0.5 --init_temperature 1.0"
    "rs1.0_temp3.0:--reward_scale 1.0 --init_temperature 3.0"
)
