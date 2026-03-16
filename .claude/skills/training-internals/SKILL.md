# training-internals — RLPD/ACP Training Internals Diagnosis

When the user invokes `/training-internals`, diagnose the internal health of RLPD online training experiments. Analyzes loss, Q-value, entropy, reward signal, and advantage weighting from WandB data using a five-dimension diagnostic framework.

**Parameters**: `/training-internals [wandb_project] [run_ids...]`
- If no project specified, reads current experiment state from CLAUDE.md
- If no run_ids specified, analyzes all runs in the project

---

## Five-Dimension Diagnostic Framework

Each experiment is diagnosed on 5 dimensions with A/B/C/D/F grading:

### Dimension 1: Critic Health

**Data sources**: `train/critic/q_mean`, `train/critic/q_std`, `train/critic/td_target_mean`, `train/critic/critic_loss`

| Indicator | Healthy | Warning | Danger |
|-----------|---------|---------|--------|
| Q-value range | < 10 | 10-50 | > 50 |
| Critic loss (final 20%) | < 1.0 | 1-50 | > 50 |
| TD target std | < 1.0 | 1-10 | > 10 |
| Q-value trend | Stable/gradual rise | Oscillating | Diverging |

### Dimension 2: Actor Drift

**Data sources (AWSC)**: `train/actor/flow_loss`, `train/actor/shortcut_loss`
**Data sources (SAC/PLD/DSRL)**: `train/actor/actor_loss`, `train/actor/actor_entropy`

| Indicator | Healthy | Warning | Danger |
|-----------|---------|---------|--------|
| Flow loss trend (AWSC) | Gradual decrease | Drop >50% | Drop >80% AND SO drops |
| Actor entropy (SAC) | [-10, 10] | [-50, -10] | < -50 |

### Dimension 3: Exploration (Entropy & Temperature)

**Data sources**: `train/temp/temperature`, `train/temp/entropy`
*N/A for AWSC (no SAC entropy mechanism).*

| Indicator | Healthy | Warning | Danger |
|-----------|---------|---------|--------|
| Temperature range | 0.1-0.5 | 0.05-0.1 | < 0.05 or > 1.0 |
| Entropy min | > -20 | [-50, -20] | < -50 |

### Dimension 4: Reward Signal

**Data sources (AWSC)**: `train/smdp/online_cum_reward_mean`, `train/smdp/offline_cum_reward_mean`, `train/reward/acp_step_mean`
**Data sources (PLD/DSRL)**: Inferred from `train/critic/q_mean` scale

| Indicator | Healthy | Warning | Danger |
|-----------|---------|---------|--------|
| Online/Offline reward gap | < 10x | 10-100x | > 100x |
| ACP step reward | > 0.01 | 0.001-0.01 | < 0.001 |

### Dimension 5: Advantage Weighting (AWSC only)

**Data sources**: `train/actor/advantage_mean`, `train/actor/advantage_std`, `train/actor/weight_max`
*N/A for PLD/DSRL.*

| Indicator | Healthy | Warning | Danger |
|-----------|---------|---------|--------|
| Advantage mean | [-0.5, 0.5] | [0.5, 1.0] | > 1.0 |
| Weight max | < 5.0 | 5-20 | > 20 |

---

## Execution Steps

### Step 1: Fetch WandB Data

```bash
http_proxy=http://10.20.93.149:7890 https_proxy=http://10.20.93.149:7890 \
conda run -n rlft_ms3 --no-capture-output \
env PYTHONPATH=/home/wjz/rl-vla \
python scripts/sweep_acp/fetch_wandb.py \
    --project {WANDB_PROJECT} \
    --run_ids {RUN_IDS} \
    --output_dir logs/vlaw/wandb_analysis/{PROJECT} \
    --save_csv
```

### Step 2: Run Automated Diagnosis

```bash
PYTHONPATH=/home/wjz/rl-vla python scripts/analyze_training_internals.py \
    --project {WANDB_PROJECT} \
    --data_dir logs/vlaw/wandb_analysis/{PROJECT} \
    --output_dir docs/vlaw/figures/{PROJECT}_internals \
    --no_fetch_wandb
```

This automatically:
1. Loads all CSV data from the data directory
2. Auto-detects algorithm type per run (AWSC/PLD/DSRL)
3. Grades each run on all 5 dimensions (A/B/C/D/F)
4. Generates algorithm-appropriate diagnostic figures
5. Produces markdown report with scorecard, findings, and prescriptions

### Step 3: Review and Augment

Review the auto-generated report:
`docs/vlaw/figures/{PROJECT}_internals/diagnosis_report.md`

Add manual observations where automated grading has limitations:
- Cross-experiment comparison insights
- Prescription prioritization based on project context
- Visual quality assessment of generated figures

---

## Common Pathology Quick Reference

| Symptom | Likely Cause | Evidence Metric | Prescription |
|---------|-------------|-----------------|-------------|
| SO degrades while flow_loss drops | Policy overfitting demo | flow_loss ratio < 0.3 | Increase bc_weight or early stop |
| SAE near 0% but SO high | Reward signal drowned | ACP/Q ratio < 5% | Increase acp_reward_scale |
| SO catastrophic collapse | Critic-driven policy drift | Q blow-up + entropy crash | Add BC loss or lower gamma |
| Q-value oscillation | gamma too high + reward unstable | Q range > 50 | Lower gamma |
| Advantage mean near 1.0 | Critic cannot discriminate | advantage_std too low | Increase online_ratio |
| Online reward near 0 | ACP reward scale insufficient | reward gap > 100x | Increase scale |

---

## References

- Analysis script: `scripts/analyze_training_internals.py`
- WandB data fetcher: `scripts/sweep_acp/fetch_wandb.py`
- Previous diagnosis reports:
  - `docs/vlaw/acp_v3_rlpd_internals_report.md` — ACP v3 internal medicine
  - `logs/vlaw/wandb_analysis/awsc_acp_mirror/analysis_report.md` — ACP mirror AWSC
  - `logs/vlaw/wandb_analysis/rlpd_acp_v3/analysis_report.md` — ACP v3 comparative
