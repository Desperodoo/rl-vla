#!/bin/bash
# =============================================================================
# SAC (Offline) — Wave 2: Regularization Method Sweep (PRIORITY)
# =============================================================================
# Goal: Find which offline regularization method enables stable SAC training.
# This is the FIRST sweep — once we identify working method(s), fine-tune
# other hyperparams (lr, ensemble, gamma, tau, etc.) in v3.
#
# Axes swept (ordered by priority):
#   1. Actor loss type: td3bc / awr / iql / sac (pure)
#   2. Per-method params: bc_weight, awr_temperature, iql_expectile
#   3. CQL critic penalty (orthogonal to actor regularization)
#   4. Key combined presets
#
# Architecture: DiagGaussianActor + EnsembleQNetwork + LearnableTemperature
# Default: td3bc with bc_weight=2.0, sac_reward_scale=1.0, num_qs=10
#
# Format: "config_name:--param1 value1 --param2 value2"
# =============================================================================

SWEEP_CONFIGS=(
    # =====================================================================
    # 0. Baseline (td3bc with all defaults, bc_weight=2.0)
    # =====================================================================
    "baseline:"

    # =====================================================================
    # 1. Actor Loss Type — Core Method Comparison
    #    These define fundamentally different offline RL strategies.
    #    td3bc: reparam Q-max + MSE BC (default)
    #    awr:   advantage-weighted log-prob on data actions (no OOD Q query)
    #    iql:   expectile V + advantage-weighted actor (fully in-sample)
    #    sac:   pure reparam Q-max (no constraint, likely diverges)
    # =====================================================================
    "type_sac:--actor_loss_type sac"
    "type_td3bc:--actor_loss_type td3bc"
    "type_awr:--actor_loss_type awr"
    "type_iql:--actor_loss_type iql"

    # =====================================================================
    # 2. TD3+BC — BC Weight (lam = 1/(1+w))
    #    w=0 -> pure SAC, w=0.5 -> 67%RL, w=1 -> 50/50, w=2 -> 33%RL (default),
    #    w=5 -> 17%RL, w=10 -> 9%RL
    # =====================================================================
    "td3bc_w0:--actor_loss_type td3bc --actor_bc_weight 0.0"
    "td3bc_w0.5:--actor_loss_type td3bc --actor_bc_weight 0.5"
    "td3bc_w1:--actor_loss_type td3bc --actor_bc_weight 1.0"
    "td3bc_w5:--actor_loss_type td3bc --actor_bc_weight 5.0"
    "td3bc_w10:--actor_loss_type td3bc --actor_bc_weight 10.0"

    # =====================================================================
    # 3. AWR — Temperature beta
    #    Controls selectivity: lower beta = more selective (only high-advantage
    #    data actions get high weight), higher beta = more uniform.
    #    Default: 1.0
    # =====================================================================
    "awr_beta0.1:--actor_loss_type awr --awr_temperature 0.1"
    "awr_beta0.3:--actor_loss_type awr --awr_temperature 0.3"
    "awr_beta0.5:--actor_loss_type awr --awr_temperature 0.5"
    "awr_beta1.0:--actor_loss_type awr --awr_temperature 1.0"
    "awr_beta3.0:--actor_loss_type awr --awr_temperature 3.0"
    "awr_beta10.0:--actor_loss_type awr --awr_temperature 10.0"

    # =====================================================================
    # 4. IQL — Expectile tau
    #    Controls optimism of V: tau=0.5 (mean, conservative), tau=0.7 (default),
    #    tau=0.9 (aggressive, approaches max Q). Higher = better policy
    #    improvement but less stable.
    # =====================================================================
    "iql_tau0.5:--actor_loss_type iql --iql_expectile 0.5"
    "iql_tau0.7:--actor_loss_type iql --iql_expectile 0.7"
    "iql_tau0.8:--actor_loss_type iql --iql_expectile 0.8"
    "iql_tau0.9:--actor_loss_type iql --iql_expectile 0.9"
    "iql_tau0.95:--actor_loss_type iql --iql_expectile 0.95"

    # =====================================================================
    # 5. IQL — AWR Temperature (advantage weighting beta for IQL actor)
    #    IQL uses the same exp(A/beta) weighting as AWR, but with learned V.
    # =====================================================================
    "iql_beta0.1:--actor_loss_type iql --awr_temperature 0.1"
    "iql_beta0.3:--actor_loss_type iql --awr_temperature 0.3"
    "iql_beta1.0:--actor_loss_type iql --awr_temperature 1.0"
    "iql_beta3.0:--actor_loss_type iql --awr_temperature 3.0"

    # =====================================================================
    # 6. CQL Critic Penalty (orthogonal — can combine with any actor type)
    #    Penalizes Q for policy-sampled actions vs data actions.
    #    Provides critic conservatism independent of actor regularization.
    # =====================================================================
    "cql_0.1:--cql_alpha 0.1"
    "cql_0.5:--cql_alpha 0.5"
    "cql_1.0:--cql_alpha 1.0"
    "cql_5.0:--cql_alpha 5.0"

    # =====================================================================
    # 7. CQL + Actor Type Combinations (top 2 CQL alphas x 3 actor types)
    # =====================================================================
    "cql1_td3bc:--cql_alpha 1.0 --actor_loss_type td3bc"
    "cql1_awr:--cql_alpha 1.0 --actor_loss_type awr"
    "cql1_iql:--cql_alpha 1.0 --actor_loss_type iql"
    "cql5_td3bc:--cql_alpha 5.0 --actor_loss_type td3bc"
    "cql5_awr:--cql_alpha 5.0 --actor_loss_type awr"
    "cql5_iql:--cql_alpha 5.0 --actor_loss_type iql"

    # =====================================================================
    # 8. Combined Presets — Best-guess configurations for each method
    # =====================================================================

    # Pure TD3+BC (no entropy, strong BC — simplest offline method)
    "preset_td3bc_strong:--actor_loss_type td3bc --actor_bc_weight 5.0 --init_temperature 0.01"

    # TD3+BC + backup entropy (moderate BC, SAC entropy helps exploration)
    "preset_td3bc_entropy:--actor_loss_type td3bc --actor_bc_weight 2.0 --backup_entropy --init_temperature 0.3"

    # AWR conservative (low beta = selective weighting, backup entropy)
    "preset_awr_conservative:--actor_loss_type awr --awr_temperature 0.3 --backup_entropy --init_temperature 0.3"

    # AWR aggressive (higher beta, more data utilization)
    "preset_awr_aggressive:--actor_loss_type awr --awr_temperature 3.0 --init_temperature 1.0"

    # IQL standard (tau=0.7, beta=1.0 — balanced IQL)
    "preset_iql_standard:--actor_loss_type iql --iql_expectile 0.7 --awr_temperature 1.0"

    # IQL optimistic (tau=0.9, beta=0.3 — aggressive policy improvement)
    "preset_iql_optimistic:--actor_loss_type iql --iql_expectile 0.9 --awr_temperature 0.3"

    # IQL conservative (tau=0.5, beta=3.0 — safe, mean V-function)
    "preset_iql_conservative:--actor_loss_type iql --iql_expectile 0.5 --awr_temperature 3.0"

    # CQL-SAC (CQL penalty + pure SAC actor — CQL provides conservatism)
    "preset_cql_sac:--actor_loss_type sac --cql_alpha 5.0"

    # CQL + TD3+BC (double conservatism: CQL on critic + BC on actor)
    "preset_cql_td3bc:--actor_loss_type td3bc --actor_bc_weight 2.0 --cql_alpha 1.0"

    # CQL + IQL (CQL on critic + IQL actor — maximum conservatism)
    "preset_cql_iql:--actor_loss_type iql --iql_expectile 0.7 --cql_alpha 1.0"

    # Full kitchen sink — everything conservative
    "preset_max_conservative:--actor_loss_type iql --iql_expectile 0.7 --cql_alpha 5.0 --backup_entropy --init_temperature 0.3 --num_qs 20 --num_min_qs 2"
)
