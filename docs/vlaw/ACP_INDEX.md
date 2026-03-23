# ACP Documentation Index

## Active entrypoints

### Canonical docs
- ACP module / pipeline overview:
  - `docs/vlaw/acp_pipeline.md`
- Current ACP v7 diagnosis index:
  - `docs/vlaw/archive/acp_v7_diagnosis_progress.md`
- Final algorithm-level v7 explanation:
  - `docs/vlaw/archive/acp_v7_failure_analysis.md`
- Final qclip0-controlled ACP conclusion:
  - `docs/vlaw/archive/acp_v7_qclip0_acp_analysis.md`

### Canonical reports
- 6-cell overview with AWSC / PLD / DSRL × sim/acp:
  - `docs/vlaw/figures/rlpd_acp_v7_diag/diagnosis_report.md`
- Final qclip0-controlled ACP diagnosis:
  - `docs/vlaw/figures/rlpd_acp_v7_qclip0_acp/diagnosis_report.md`

### Canonical scripts
- Minimal retained scheduler:
  - `scripts/acp_v7_scheduler.py`
- Unified analyzer:
  - `scripts/analyze_training_internals.py`

## Current project consensus
- Shared drift was `q_target_clip=20`, which had suppressed the historical PLD/DSRL sim baselines.
- Under `qclip0`, PLD/DSRL sim baselines recover, but ACP mirrors still fail badly.
- Therefore the current ACP problem is no longer baseline-reproduction drift, but ACP reward semantics / hold-signal insufficiency.
- AWSC + ACP is not a counterexample to this diagnosis; AWSC succeeds with a stronger BC/flow anchor and does not rely on ACP to invent hold behavior from scratch.

## Archive structure

### Archived docs
- Historical ACP reports and stage-by-stage diagnosis notes:
  - `docs/vlaw/archive/`

### Archived scripts
- Historical ACP launchers, sweep scripts, and one-off analyzers:
  - `scripts/archive/`

## Scope note
- The active ACP surface is intentionally minimal now.
- Older v3/v4/v5/v6 reports and legacy scripts are preserved for traceability, but are no longer the default entrypoints.
