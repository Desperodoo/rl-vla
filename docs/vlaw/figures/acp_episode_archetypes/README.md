# ACP episode archetype diagnostics

This directory is used for the next-layer diagnosis beyond scalar SO/SAE.

## Goal

Construct episode archetypes for corrected qclip0 runs:
- A: `SO=True, SAE=True` (stable hold)
- B: `SO=True, SAE=False` (drop after success)

and compare, for PLD/DSRL under sim vs ACP:
- key frames,
- success timeline,
- grasp timeline,
- simulator reward,
- ACP base / bonus / total reward,
- cumulative ACP reward.

## Expected interpretation

If ACP runs show:
- high grasping,
- visible transient success,
- but near-zero or misleading reward signal during post-success instability,

then the evidence becomes much stronger that the current ACP reward is overweighting grasp/progress and under-discriminating stable hold vs imminent drop.
