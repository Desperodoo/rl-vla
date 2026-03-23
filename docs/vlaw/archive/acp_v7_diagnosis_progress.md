# ACP v7 Diagnosis Index

## Current one-line conclusion
- `q_target_clip=20` 是共享 drift，压坏了 PLD/DSRL 的历史 sim baseline；在切回 `qclip0` 后，PLD/DSRL sim baseline 恢复，但 ACP mirror 仍显著失败，因此当前问题已收敛为 **ACP reward semantics / hold-signal insufficiency**。

## Canonical docs
- Final algorithm-level explanation:
  - `docs/vlaw/archive/acp_v7_failure_analysis.md`
- Final ACP-specific conclusion after removing shared drift:
  - `docs/vlaw/archive/acp_v7_qclip0_acp_analysis.md`

## Archived intermediate docs
- Fair replay stage:
  - `docs/vlaw/archive/acp_v7_fair_replay_analysis.md`
- Drift regression stage:
  - `docs/vlaw/archive/acp_v7_drift_regression_analysis.md`
- Commit/code-path audit stage:
  - `docs/vlaw/archive/acp_v7_codepath_drift_audit.md`

## Canonical report dirs
- 6-cell overview with AWSC comparison:
  - `docs/vlaw/figures/rlpd_acp_v7_diag/`
- Final qclip0-controlled ACP diagnosis:
  - `docs/vlaw/figures/rlpd_acp_v7_qclip0_acp/`

## Archived report dirs
- Historical exact-replay evidence:
  - `docs/vlaw/figures/rlpd_acp_v7_fair_replay/`
- Shared-drift regression evidence:
  - `docs/vlaw/figures/rlpd_acp_v7_drift_reg/`

## Canonical experiment surface
- Scheduler entrypoint:
  - `scripts/acp_v7_scheduler.py`
- Canonical retained experiment:
  - `AWSC / PLD / DSRL × sim/acp mirror`
- Canonical analyzer:
  - `scripts/analyze_training_internals.py`

## Final factual summary
- `diag_core` established the full 6-cell comparison and showed AWSC is the only algorithm with strong retention under both reward sources.
- `fair_replay` showed the historical high-SAE `PLD/DSRL + sim_reward` baseline no longer reproduced under the current code path.
- `drift_regression` showed the main shared drift was `q_target_clip=20`.
- `qclip0_acp_diag` then cleanly separated the two layers:
  - PLD sim `0.82 / 0.82` vs ACP `0.02 / 0.00`
  - DSRL sim `0.66 / 0.66` vs ACP `0.06 / 0.00`
- Therefore ACP failure is no longer confounded by sim-baseline reproduction drift.

## Scope constraints
- No sim-reward blending / warmup switching in ACP tracks.
- Keep this track as diagnosis + minimal reproducible comparison, not a broad sweep.
