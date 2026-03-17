# Training Internals Diagnosis Report

> Generated: 2026-03-17 10:48
> Experiments: 15
> Algorithms: awsc, dsrl, pld

---

## Five-Dimension Scorecard

| Experiment | Algo | Critic | Actor | Exploration | Reward | Advantage | Overall |
|---|---|---|---|---|---|---|---|
| pld_v5_baseline_g07_s42 | PLD | F | B | B | A | N/A | **B** |
| awsc_v5_td_sae_s42 | AWSC | A | A | N/A | C | C | **B** |
| awsc_v5_v_reward_sae_s42 | AWSC | A | A | N/A | A | C | **A** |
| awsc_v5_v4repro_s42 | AWSC | A | A | N/A | C | D | **B** |
| dsrl_v5_stable_g03_s42 | DSRL | A | A | A | A | N/A | **A** |
| pld_v5_stable_g05_s42 | PLD | C | B | B | A | N/A | **B** |
| dsrl_v5_baseline_g07_s42 | DSRL | C | A | A | A | N/A | **A** |
| dsrl_v5_stable_g05_s42 | DSRL | C | A | A | A | N/A | **A** |
| pld_v5_v_reward_sae_s42 | PLD | B | B | D | A | N/A | **B** |
| awsc_v5_td_clip_s42 | AWSC | A | A | N/A | D | D | **B** |
| dsrl_v5_v_reward_sae_s42 | DSRL | A | A | B | A | N/A | **A** |
| pld_v5_v_reward_g05_s42 | PLD | B | B | D | A | N/A | **B** |
| dsrl_v5_v_reward_g05_s42 | DSRL | A | A | A | A | N/A | **A** |
| pld_v5_stable_g03_s42 | PLD | C | B | B | A | N/A | **B** |
| awsc_v5_v_reward_s42 | AWSC | A | A | N/A | A | C | **A** |

---

## Detailed Findings

### pld_v5_baseline_g07_s42 (PLD)

**Critic** (F, score=35):
- Q-value range 51.6 >> 50: critic oscillating severely
- Critic loss final 20%=17.49: slow convergence
- TD target std=2.67: moderate instability
- Evidence: q_mean_avg=8.45, q_range=51.6, critic_loss_final=17.49, td_target_std=2.669

**Actor** (B, score=70):
- Entropy min=-55: policy collapsed at some point
- Evidence: entropy_min=-54.98, entropy_final=-2.19

**Exploration** (B, score=70):
- Entropy min=-55: historical policy collapse
- Evidence: temperature_avg=0.1798, temperature_final=0.1795, entropy_min=-54.98

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=8.45

### awsc_v5_td_sae_s42 (AWSC)

**Critic** (A, score=100):
- Healthy
- Evidence: q_mean_avg=3.81, q_range=3.9, critic_loss_final=0.22, td_target_std=0.045

**Actor** (A, score=100):
- Healthy
- Evidence: flow_loss_first=0.1599, flow_loss_last=0.0335, flow_loss_ratio=0.209

**Reward** (C, score=65):
- Online/offline reward gap 52x: significant imbalance
- ACP step reward avg=0.0004: signal nearly dead
- Evidence: online_cum_reward_avg=0.0828, offline_cum_reward_avg=4.3356, reward_gap_ratio=52.4, acp_step_mean=0.0004

**Advantage** (C, score=65):
- Advantage mean=0.90: moderate positive bias
- Weight max peak=26: few samples over-amplified
- Evidence: advantage_mean_avg=0.904, weight_max_peak=25.9

### awsc_v5_v_reward_sae_s42 (AWSC)

**Critic** (A, score=100):
- Healthy
- Evidence: q_mean_avg=3.47, q_range=3.7, critic_loss_final=0.54, td_target_std=0.394

**Actor** (A, score=100):
- Healthy
- Evidence: flow_loss_first=0.1071, flow_loss_last=0.0317, flow_loss_ratio=0.296

**Reward** (A, score=100):
- Healthy
- Evidence: online_cum_reward_avg=-1.9071, offline_cum_reward_avg=4.3356, reward_gap_ratio=2.3, acp_step_mean=-0.0067

**Advantage** (C, score=60):
- Advantage mean=2.29: critic unable to discriminate good/bad actions
- Weight max peak=7.9: moderate amplification
- Evidence: advantage_mean_avg=2.293, weight_max_peak=7.9

### awsc_v5_v4repro_s42 (AWSC)

**Critic** (A, score=85):
- Critic loss final 20%=3.98: slow convergence
- Evidence: q_mean_avg=3.83, q_range=4.1, critic_loss_final=3.98, td_target_std=0.091

**Actor** (A, score=100):
- Healthy
- Evidence: flow_loss_first=0.1517, flow_loss_last=0.0348, flow_loss_ratio=0.229

**Reward** (C, score=65):
- Online/offline reward gap 44x: significant imbalance
- ACP step reward avg=0.0007: signal nearly dead
- Evidence: online_cum_reward_avg=0.0979, offline_cum_reward_avg=4.3344, reward_gap_ratio=44.3, acp_step_mean=0.0007

**Advantage** (D, score=50):
- Advantage mean=1.09: critic unable to discriminate good/bad actions
- Weight max peak=23: few samples over-amplified
- Evidence: advantage_mean_avg=1.091, weight_max_peak=22.7

### dsrl_v5_stable_g03_s42 (DSRL)

**Critic** (A, score=85):
- Critic loss final 20%=9.41: slow convergence
- Evidence: q_mean_avg=2.9, q_range=6.4, critic_loss_final=9.41, td_target_std=0.81

**Actor** (A, score=100):
- Healthy
- Evidence: entropy_min=-12.07, entropy_final=-3.67

**Exploration** (A, score=100):
- Healthy
- Evidence: temperature_avg=0.1994, temperature_final=0.1981, entropy_min=-12.07

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=2.9

### pld_v5_stable_g05_s42 (PLD)

**Critic** (C, score=55):
- Q-value range 44.1 > 10: mild instability
- Critic loss final 20%=6.16: slow convergence
- TD target std=2.55: moderate instability
- Evidence: q_mean_avg=5.01, q_range=44.1, critic_loss_final=6.16, td_target_std=2.55

**Actor** (B, score=70):
- Entropy min=-55: policy collapsed at some point
- Evidence: entropy_min=-55.09, entropy_final=-2.76

**Exploration** (B, score=70):
- Entropy min=-55: historical policy collapse
- Evidence: temperature_avg=0.148, temperature_final=0.1362, entropy_min=-55.09

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=5.01

### dsrl_v5_baseline_g07_s42 (DSRL)

**Critic** (C, score=55):
- Q-value range 15.8 > 10: mild instability
- Critic loss final 20%=37.29: slow convergence
- TD target std=2.33: moderate instability
- Evidence: q_mean_avg=8.15, q_range=15.8, critic_loss_final=37.29, td_target_std=2.326

**Actor** (A, score=100):
- Healthy
- Evidence: entropy_min=-12.53, entropy_final=0.93

**Exploration** (A, score=100):
- Healthy
- Evidence: temperature_avg=0.1874, temperature_final=0.1979, entropy_min=-12.53

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=8.15

### dsrl_v5_stable_g05_s42 (DSRL)

**Critic** (C, score=55):
- Q-value range 12.5 > 10: mild instability
- Critic loss final 20%=14.27: slow convergence
- TD target std=1.51: moderate instability
- Evidence: q_mean_avg=4.55, q_range=12.5, critic_loss_final=14.27, td_target_std=1.512

**Actor** (A, score=100):
- Healthy
- Evidence: entropy_min=-14.21, entropy_final=-3.69

**Exploration** (A, score=100):
- Healthy
- Evidence: temperature_avg=0.1788, temperature_final=0.1784, entropy_min=-14.21

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=4.55

### pld_v5_v_reward_sae_s42 (PLD)

**Critic** (B, score=80):
- Q-value range 18.4 > 10: mild instability
- Evidence: q_mean_avg=-18.62, q_range=18.4, critic_loss_final=0.58, td_target_std=0.352

**Actor** (B, score=70):
- Entropy min=-54: policy collapsed at some point
- Evidence: entropy_min=-53.55, entropy_final=-1.65

**Exploration** (D, score=45):
- Temperature final=0.0055: exploration over-compressed
- Entropy min=-54: historical policy collapse
- Evidence: temperature_avg=0.0649, temperature_final=0.0055, entropy_min=-53.55

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=-18.62

### awsc_v5_td_clip_s42 (AWSC)

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

### dsrl_v5_v_reward_sae_s42 (DSRL)

**Critic** (A, score=85):
- Critic loss final 20%=4.36: slow convergence
- Evidence: q_mean_avg=-17.7, q_range=10.0, critic_loss_final=4.36, td_target_std=0.347

**Actor** (A, score=100):
- Healthy
- Evidence: entropy_min=-46.54, entropy_final=1.84

**Exploration** (B, score=75):
- Temperature final=0.0440: exploration over-compressed
- Evidence: temperature_avg=0.0755, temperature_final=0.044, entropy_min=-46.54

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=-17.7

### pld_v5_v_reward_g05_s42 (PLD)

**Critic** (B, score=70):
- Q-value range 24.9 > 10: mild instability
- TD target std=1.17: moderate instability
- Evidence: q_mean_avg=-16.58, q_range=24.9, critic_loss_final=0.67, td_target_std=1.17

**Actor** (B, score=70):
- Entropy min=-55: policy collapsed at some point
- Evidence: entropy_min=-54.59, entropy_final=-3.72

**Exploration** (D, score=45):
- Temperature final=0.0272: exploration over-compressed
- Entropy min=-55: historical policy collapse
- Evidence: temperature_avg=0.0779, temperature_final=0.0272, entropy_min=-54.59

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=-16.58

### dsrl_v5_v_reward_g05_s42 (DSRL)

**Critic** (A, score=85):
- Critic loss final 20%=5.72: slow convergence
- Evidence: q_mean_avg=-13.51, q_range=9.8, critic_loss_final=5.72, td_target_std=0.839

**Actor** (A, score=100):
- Healthy
- Evidence: entropy_min=-16.95, entropy_final=-5.72

**Exploration** (A, score=100):
- Healthy
- Evidence: temperature_avg=0.0995, temperature_final=0.0693, entropy_min=-16.95

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=-13.51

### pld_v5_stable_g03_s42 (PLD)

**Critic** (C, score=55):
- Q-value range 20.5 > 10: mild instability
- Critic loss final 20%=3.09: slow convergence
- TD target std=1.22: moderate instability
- Evidence: q_mean_avg=3.03, q_range=20.5, critic_loss_final=3.09, td_target_std=1.217

**Actor** (B, score=70):
- Entropy min=-51: policy collapsed at some point
- Evidence: entropy_min=-51.37, entropy_final=-2.51

**Exploration** (B, score=70):
- Entropy min=-51: historical policy collapse
- Evidence: temperature_avg=0.1502, temperature_final=0.1386, entropy_min=-51.37

**Reward** (A, score=100):
- Healthy
- Evidence: q_mean_avg=3.03

### awsc_v5_v_reward_s42 (AWSC)

**Critic** (A, score=100):
- Healthy
- Evidence: q_mean_avg=3.48, q_range=3.7, critic_loss_final=0.45, td_target_std=0.382

**Actor** (A, score=100):
- Healthy
- Evidence: flow_loss_first=0.1056, flow_loss_last=0.0317, flow_loss_ratio=0.3

**Reward** (A, score=100):
- Healthy
- Evidence: online_cum_reward_avg=-1.844, offline_cum_reward_avg=4.3356, reward_gap_ratio=2.4, acp_step_mean=-0.0064

**Advantage** (C, score=60):
- Advantage mean=2.14: critic unable to discriminate good/bad actions
- Weight max peak=7.9: moderate amplification
- Evidence: advantage_mean_avg=2.136, weight_max_peak=7.9

---

## Auto-Generated Prescriptions

- **Lower gamma**: Reduce discount factor to shrink Q-value scale and improve critic stability.
- **Increase ACP reward scale**: Raise --acp_reward_scale (e.g., 500-2000) to strengthen online ACP signal relative to offline demo reward.
- **Increase online_ratio**: Raise --online_ratio (e.g., 0.3-0.5) to give critic more diverse training data.

---

*Report generated by `scripts/analyze_training_internals.py`*