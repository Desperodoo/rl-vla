# ACP retention diagnosis summary

## Key observations

1. **The dominant failure mode for PLD/DSRL is retention failure, not pure progress failure.**
   - In v7 controlled comparison, AWSC retains a high SAE/SO ratio under both rewards.
   - DSRL and PLD show very large SO-SAE gaps under both `sim` and `acp`.

2. **This is not ACP-only.**
   - AWSC sim retention: 0.844
   - AWSC acp retention: 0.778
   - DSRL sim retention: 0.087
   - DSRL acp retention: 0.152
   - PLD sim retention: 0.050
   - PLD acp retention: 0.073

3. **ACP still does not provide a strong enough hold-sensitive signal.**
   - v3 mismatch rate (`SO=True, SAE=False`) is 15.4% over 1250 trajectories.
   - v3_sae improves value prediction accuracy over v3_so, but downstream retention remains poor for PLD/DSRL in v5/v6/v7.
   - This supports: label semantics matter, but algorithmic retention capacity is still the deeper bottleneck.

## Generated files
- `fig_v7_retention_matrix.png`
- `fig_v7_four_quadrants.png`
- `fig_cross_version_best_metrics.png`
- `fig_v3_mismatch_bridge.png`
- `retention_summary.json`
