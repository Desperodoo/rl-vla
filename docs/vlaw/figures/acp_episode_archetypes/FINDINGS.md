# ACP episode archetype findings

## What was extracted

Corrected qclip0 runs were rolled out with deterministic policies and episode-level step traces were collected.

Generated files:
- `dsrl_sim_best_sae_archetypes.png`
- `dsrl_acp_best_so_archetypes.png`
- `pld_sim_best_sae_archetypes.png`
- corresponding `.h5` trace files
- `archetype_summary.json`

## Summary

### 1. DSRL sim
A stable-hold archetype and a drop-after-success archetype were both found.
This is important because it shows the policy family can express both behaviors under the same overall recipe.

### 2. DSRL acp
A stable-hold archetype and a drop-after-success archetype were also found in the sampled episodes, but the selected checkpoint is still the `best_so` ACP checkpoint and aggregate metrics remain poor on SAE.
The key question is therefore not whether stable hold is impossible, but whether the ACP reward sufficiently favors it over transient success/grasp behavior.

### 3. PLD sim
A stable-hold archetype and a drop-after-success archetype were both found.
This reinforces the corrected-run conclusion that PLD itself is capable of hold under sim reward in the corrected qclip0 setup.

### 4. PLD acp
No stable/drop pair was found in the first 12 sampled episodes from the chosen `best_so` checkpoint. This is itself informative: the sampled support is dominated enough that a balanced A/B pair did not emerge quickly.

## Interpretation update

The deeper diagnosis should now be phrased as:

- In the corrected qclip0 comparison, **sim reward clearly supports hold-to-end for both DSRL and PLD**.
- Under ACP reward, policies can still sometimes pass through successful states, but aggregate SAE collapses.
- Episode-level traces are therefore consistent with a reward-credit problem rather than a pure policy-capacity problem.
- More specifically, the current ACP shaping appears to overweight grasp/progress proxies and under-discriminate stable hold vs imminent drop.

## Remaining caveat

The current archetype script selects representative episodes from a small sample (`12` episodes per checkpoint) and uses deterministic rollout from one chosen checkpoint per case. This is enough for qualitative evidence, but not yet a fully exhaustive per-checkpoint statistical study.
