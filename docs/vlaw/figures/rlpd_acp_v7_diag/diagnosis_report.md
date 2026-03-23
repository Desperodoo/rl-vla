# Training Internals Diagnosis Report

> Generated: 2026-03-22 15:52
> Experiments: 6
> Algorithms: awsc, dsrl, pld

---

## Controlled Comparison Summary

| Algo | Reward | Runs | Best SO | Best SAE | Final SAE | SAE Retention | SO-SAE Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| AWSC | acp | 1 | 0.900 | 0.700 | 0.520 | 0.778 | 0.200 |
| AWSC | sim | 1 | 0.900 | 0.760 | 0.740 | 0.844 | 0.140 |
| DSRL | acp | 1 | 0.920 | 0.140 | 0.020 | 0.152 | 0.780 |
| DSRL | sim | 1 | 0.920 | 0.080 | 0.000 | 0.087 | 0.840 |
| PLD | acp | 1 | 0.820 | 0.060 | 0.000 | 0.073 | 0.760 |
| PLD | sim | 1 | 0.800 | 0.040 | 0.000 | 0.050 | 0.760 |

---

## Five-Dimension Scorecard

| Experiment | Algo | Reward | Critic | Actor | Exploration | Reward | Advantage | Overall |
|---|---|---|---|---|---|---|---|---|
| dsrl_v7_diag_sim_s42 | DSRL | sim | C | A | B | A | N/A | **B** |
| pld_v7_diag_sim_s42 | PLD | sim | B | B | D | A | N/A | **B** |
| awsc_v7_diag_sim_s42 | AWSC | sim | A | A | N/A | A | C | **A** |
| awsc_v7_diag_acp_s42 | AWSC | acp | A | A | N/A | D | D | **B** |
| dsrl_v7_diag_acp_s42 | DSRL | acp | C | A | A | A | N/A | **A** |
| pld_v7_diag_acp_s42 | PLD | acp | C | B | B | A | N/A | **B** |

---

## Detailed Findings

### dsrl_v7_diag_sim_s42 (DSRL, sim)

**Critic** (C, score=65):
- Q-value range 13.8 > 10: mild instability
- Critic loss final 20%=1.86: slow convergence
- Evidence: q_mean_avg=16.74, q_range=13.8, critic_loss_final=1.86, td_target_std=0.292

**Actor** (A, score=100):
- Healthy
- Evidence: entropy_min=-32.89, entropy_final=-0.93

**Exploration** (B, score=75):
- Temperature final=0.0103: exploration over-compressed
- Evidence: temperature_avg=0.0134, temperature_final=0.0103, entropy_min=-32.89

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=16.74

### pld_v7_diag_sim_s42 (PLD, sim)

**Critic** (B, score=70):
- Q-value range 39.3 > 10: mild instability
- TD target std=1.11: moderate instability
- Evidence: q_mean_avg=16.4, q_range=39.3, critic_loss_final=0.31, td_target_std=1.106

**Actor** (B, score=70):
- Entropy min=-55: policy collapsed at some point
- Evidence: entropy_min=-54.55, entropy_final=-3.67

**Exploration** (D, score=45):
- Temperature final=0.0032: exploration over-compressed
- Entropy min=-55: historical policy collapse
- Evidence: temperature_avg=0.0278, temperature_final=0.0032, entropy_min=-54.55

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=16.4

### awsc_v7_diag_sim_s42 (AWSC, sim)

**Critic** (A, score=100):
- Healthy
- Evidence: q_mean_avg=3.96, q_range=4.1, critic_loss_final=0.07, td_target_std=0.043

**Actor** (A, score=100):
- Healthy
- Evidence: flow_loss_first=0.1307, flow_loss_last=0.0422, flow_loss_ratio=0.323

**Reward** (A, score=100):
- Healthy
- Evidence: online_cum_reward_avg=0.6698, offline_cum_reward_avg=4.3366, reward_gap_ratio=6.5

**Advantage** (C, score=65):
- Advantage mean=0.54: moderate positive bias
- Weight max peak=61: few samples over-amplified
- Evidence: advantage_mean_avg=0.537, weight_max_peak=61.2

### awsc_v7_diag_acp_s42 (AWSC, acp)

**Critic** (A, score=100):
- Healthy
- Evidence: q_mean_avg=3.81, q_range=3.8, critic_loss_final=0.13, td_target_std=0.036

**Actor** (A, score=100):
- Healthy
- Evidence: flow_loss_first=0.1398, flow_loss_last=0.0331, flow_loss_ratio=0.237

**Reward** (D, score=45):
- Online/offline reward gap 1049x: critic dominated by offline
- ACP step reward avg=0.0001: signal nearly dead
- Evidence: online_cum_reward_avg=0.0041, offline_cum_reward_avg=4.3349, reward_gap_ratio=1048.9, acp_step_mean=0.0001

**Advantage** (D, score=50):
- Advantage mean=1.03: critic unable to discriminate good/bad actions
- Weight max peak=22: few samples over-amplified
- Evidence: advantage_mean_avg=1.025, weight_max_peak=21.9

### dsrl_v7_diag_acp_s42 (DSRL, acp)

**Critic** (C, score=55):
- Q-value range 13.6 > 10: mild instability
- Critic loss final 20%=16.33: slow convergence
- TD target std=1.19: moderate instability
- Evidence: q_mean_avg=10.6, q_range=13.6, critic_loss_final=16.33, td_target_std=1.192

**Actor** (A, score=100):
- Healthy
- Evidence: entropy_min=-8.56, entropy_final=-1.9

**Exploration** (A, score=100):
- Healthy
- Evidence: temperature_avg=0.1411, temperature_final=0.1329, entropy_min=-8.56

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=10.6, acp_base_mean=0.0344, acp_grasp_bonus_mean=0.7778, is_grasping_rate=0.7778

### pld_v7_diag_acp_s42 (PLD, acp)

**Critic** (C, score=55):
- Q-value range 41.8 > 10: mild instability
- Critic loss final 20%=7.24: slow convergence
- TD target std=1.33: moderate instability
- Evidence: q_mean_avg=9.17, q_range=41.8, critic_loss_final=7.24, td_target_std=1.327

**Actor** (B, score=70):
- Entropy min=-56: policy collapsed at some point
- Evidence: entropy_min=-55.53, entropy_final=-3.86

**Exploration** (B, score=70):
- Entropy min=-56: historical policy collapse
- Evidence: temperature_avg=0.1353, temperature_final=0.1286, entropy_min=-55.53

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=9.17, acp_base_mean=0.1099, acp_grasp_bonus_mean=0.627, is_grasping_rate=0.627

---

## Auto-Generated Prescriptions

- **Increase online_ratio**: Raise --online_ratio (e.g., 0.3-0.5) to give critic more diverse training data.
- **Increase ACP reward scale**: Raise --acp_reward_scale (e.g., 500-2000) to strengthen online ACP signal relative to offline demo reward.

---

*Report generated by `scripts/analyze_training_internals.py`*