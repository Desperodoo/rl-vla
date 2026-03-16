# Training Internals Diagnosis Report

> Generated: 2026-03-16 14:04
> Experiments: 6
> Algorithms: awsc, dsrl, pld

---

## Five-Dimension Scorecard

| Experiment | Algo | Critic | Actor | Exploration | Reward | Advantage | Overall |
|---|---|---|---|---|---|---|---|
| dsrl_acp_v3_sae_s42 | DSRL | F | A | A | B | N/A | **B** |
| pld_acp_v3_sae_s42 | PLD | F | B | B | A | N/A | **B** |
| awsc_acp_v3_so_s42 | AWSC | A | C | N/A | D | C | **C** |
| awsc_acp_v3_sae_s42 | AWSC | A | C | N/A | C | C | **B** |
| dsrl_acp_v3_so_s42 | DSRL | F | A | A | A | N/A | **B** |
| pld_acp_v3_so_s42 | PLD | F | B | B | A | N/A | **B** |

---

## Detailed Findings

### dsrl_acp_v3_sae_s42 (DSRL)

**Critic** (F, score=10):
- Q-value range 85.7 >> 50: critic oscillating severely
- Critic loss final 20%=79.8: not converging
- TD target std=15.0: value estimation unstable
- Evidence: q_mean_avg=57.33, q_range=85.7, critic_loss_final=79.77, td_target_std=14.991

**Actor** (A, score=100):
- Healthy
- Evidence: entropy_min=-9.86, entropy_final=-1.85

**Exploration** (A, score=100):
- Healthy
- Evidence: temperature_avg=0.2565, temperature_final=0.1866, entropy_min=-9.86

**Reward** (B, score=80):
- Q-value avg=57.3: ACP signal likely drowned by sim reward
- Evidence: q_mean_avg=57.33

### pld_acp_v3_sae_s42 (PLD)

**Critic** (F, score=25):
- Q-value range 140.4 >> 50: critic oscillating severely
- Critic loss final 20%=24.55: slow convergence
- TD target std=14.7: value estimation unstable
- Evidence: q_mean_avg=23.45, q_range=140.4, critic_loss_final=24.55, td_target_std=14.662

**Actor** (B, score=70):
- Entropy min=-78: policy collapsed at some point
- Evidence: entropy_min=-77.95, entropy_final=-3.49

**Exploration** (B, score=70):
- Entropy min=-78: historical policy collapse
- Evidence: temperature_avg=0.1458, temperature_final=0.1655, entropy_min=-77.95

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=23.45

### awsc_acp_v3_so_s42 (AWSC)

**Critic** (A, score=100):
- Healthy
- Evidence: q_mean_avg=3.83, q_range=3.9, critic_loss_final=0.17, td_target_std=0.038

**Actor** (C, score=60):
- Flow loss ↓80% but SO declined 81.83%→58.15%: overfitting demo
- Evidence: flow_loss_first=0.1223, flow_loss_last=0.025, flow_loss_ratio=0.204, so_decline=81.83%→58.15%

**Reward** (D, score=45):
- Online/offline reward gap 350x: critic dominated by offline
- ACP step reward avg=0.0001: signal nearly dead
- Evidence: online_cum_reward_avg=0.0124, offline_cum_reward_avg=4.3366, reward_gap_ratio=349.5, acp_step_mean=0.0001

**Advantage** (C, score=65):
- Advantage mean=0.98: moderate positive bias
- Weight max peak=23: few samples over-amplified
- Evidence: advantage_mean_avg=0.98, weight_max_peak=23.2

### awsc_acp_v3_sae_s42 (AWSC)

**Critic** (A, score=100):
- Healthy
- Evidence: q_mean_avg=3.83, q_range=3.9, critic_loss_final=0.13, td_target_std=0.038

**Actor** (C, score=60):
- Flow loss ↓79% but SO declined 81.00%→58.31%: overfitting demo
- Evidence: flow_loss_first=0.1184, flow_loss_last=0.0254, flow_loss_ratio=0.215, so_decline=81.00%→58.31%

**Reward** (C, score=65):
- Online/offline reward gap 90x: significant imbalance
- ACP step reward avg=0.0002: signal nearly dead
- Evidence: online_cum_reward_avg=0.0483, offline_cum_reward_avg=4.3366, reward_gap_ratio=89.8, acp_step_mean=0.0002

**Advantage** (C, score=65):
- Advantage mean=0.94: moderate positive bias
- Weight max peak=33: few samples over-amplified
- Evidence: advantage_mean_avg=0.938, weight_max_peak=32.8

### dsrl_acp_v3_so_s42 (DSRL)

**Critic** (F, score=10):
- Q-value range 76.0 >> 50: critic oscillating severely
- Critic loss final 20%=132.1: not converging
- TD target std=11.8: value estimation unstable
- Evidence: q_mean_avg=26.94, q_range=76.0, critic_loss_final=132.1, td_target_std=11.761

**Actor** (A, score=100):
- Healthy
- Evidence: entropy_min=-23.62, entropy_final=0.74

**Exploration** (A, score=100):
- Healthy
- Evidence: temperature_avg=0.3217, temperature_final=0.3104, entropy_min=-23.62

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=26.94

### pld_acp_v3_so_s42 (PLD)

**Critic** (F, score=10):
- Q-value range 114.3 >> 50: critic oscillating severely
- Critic loss final 20%=59.3: not converging
- TD target std=10.8: value estimation unstable
- Evidence: q_mean_avg=11.36, q_range=114.3, critic_loss_final=59.32, td_target_std=10.849

**Actor** (B, score=70):
- Entropy min=-105: policy collapsed at some point
- Evidence: entropy_min=-105.03, entropy_final=-3.1

**Exploration** (B, score=70):
- Entropy min=-105: historical policy collapse
- Evidence: temperature_avg=0.2492, temperature_final=0.2532, entropy_min=-105.03

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=11.36

---

## Auto-Generated Prescriptions

- **Lower gamma**: Reduce discount factor to shrink Q-value scale and improve critic stability.
- **Increase BC weight**: Raise awsc_bc_weight (e.g., 4-8) to resist policy drift from pretrained distribution.
- **Enable early stopping**: Use --early_stop to halt training when SO degrades while flow_loss drops.
- **Increase ACP reward scale**: Raise --acp_reward_scale (e.g., 500-2000) to strengthen online ACP signal relative to offline demo reward.
- **Increase online_ratio**: Raise --online_ratio (e.g., 0.3-0.5) to give critic more diverse training data.

---

*Report generated by `scripts/analyze_training_internals.py`*