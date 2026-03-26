# ACP hold diagnostics

## Corrected v7 comparison

Using the corrected qclip0 comparison runs:
- DSRL sim: `dsrl_v7_reg_qclip0_sim_s42__1774197495`
- DSRL acp: `dsrl_v7_qclip0_acp_mirror_s42__1774237641`
- PLD sim: `pld_v7_reg_qclip0_sim_s42__1774197493`
- PLD acp: `pld_v7_qclip0_acp_mirror_s42__1774237641`

### Headline numbers
- DSRL sim: SO=0.94, SAE=0.66, retention=0.702
- DSRL acp: SO=0.94, SAE=0.06, retention=0.064
- PLD sim: SO=0.98, SAE=0.82, retention=0.837
- PLD acp: SO=0.82, SAE=0.02, retention=0.024

## What the deeper analysis supports

1. **The previous v7 sim baseline used in the first pass should be replaced by these corrected runs.**
   Under the corrected qclip0-controlled comparison, sim is much stronger than the earlier `diag_sim` runs.

2. **ACP is not simply “worse than sim” in a scalar sense; it changes what gets reinforced.**
   The ACP runs expose grasp/progress-related reward components (`is_grasping_rate`, `acp_grasp_bonus_mean`, `acp_base_mean`), but those signals still do not translate into strong `success_at_end`.

3. **This gets closer to the root cause than SO/SAE alone:**
   - if grasping-related signals rise but SAE stays near zero,
   - then the problem is not “the agent never reaches/grips”,
   - but rather **the reward is not sufficiently discriminating stable hold vs imminent drop**.

4. **v3 mismatch evidence remains important but not sufficient.**
   `v3_sae` improves value semantics on mismatch trajectories, yet PLD/DSRL still fail to retain success at the policy level.

## Generated files
- `fig_corrected_v7_retention_matrix.png`
- `fig_corrected_v7_training_dynamics.png`
- `fig_acp_reward_components.png`
- `fig_v3_mismatch_evidence.png`
- `corrected_v7_summary.json`
