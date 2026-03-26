# ACP Documentation Index

## Active entrypoints

### Canonical docs
- ACP module / pipeline overview:
  - `docs/vlaw/acp_pipeline.md`
- Current ACP diagnosis index:
  - `docs/vlaw/ACP_INDEX.md`
- Retention / hold diagnosis report:
  - `docs/vlaw/figures/acp_retention_diagnosis/diagnosis_report.md`
  - `docs/vlaw/figures/acp_hold_diagnostics/hold_diagnostics_report.md`
- Episode archetype summaries:
  - `docs/vlaw/figures/acp_episode_archetypes/REPORT_ZH.md`
  - `docs/vlaw/figures/acp_episode_archetypes/FINDINGS.md`
- Historical v7 diagnosis chain:
  - `docs/vlaw/archive/acp_v7_diagnosis_progress.md`
  - `docs/vlaw/archive/acp_v7_failure_analysis.md`
  - `docs/vlaw/archive/acp_v7_qclip0_acp_analysis.md`
- Historical v3 SO-vs-SAE report:
  - `docs/vlaw/archive/acp_v3_so_vs_sae_report.md`

### Canonical reports
- 6-cell overview with AWSC / PLD / DSRL × sim/acp:
  - `docs/vlaw/figures/rlpd_acp_v7_diag/diagnosis_report.md`
- Final qclip0-controlled ACP diagnosis:
  - `docs/vlaw/figures/rlpd_acp_v7_qclip0_acp/diagnosis_report.md`
- Retention-focused aggregate comparison:
  - `docs/vlaw/figures/acp_retention_diagnosis/diagnosis_report.md`
- Hold-signal / corrected-v7 diagnostics:
  - `docs/vlaw/figures/acp_hold_diagnostics/hold_diagnostics_report.md`

### Canonical scripts
- Minimal retained scheduler:
  - `scripts/acp_v7_scheduler.py`
- Unified analyzer:
  - `scripts/analyze_training_internals.py`
- Retention diagnosis:
  - `scripts/analysis/analyze_acp_retention.py`
- Hold / reward diagnostics:
  - `scripts/analysis/analyze_acp_hold_diagnostics.py`
- Episode archetype extraction:
  - `scripts/analysis/extract_acp_episode_archetypes.py`
- Robo-Dopamine export bridge:
  - `scripts/convert_acp_to_robo_dopamine.py`

## Current project consensus
- Shared drift was `q_target_clip=20`, which had suppressed the historical PLD/DSRL sim baselines.
- Under `qclip0`, PLD/DSRL sim baselines recover, but ACP mirrors still fail badly.
- Therefore the current ACP problem is no longer baseline-reproduction drift, but ACP reward semantics / hold-signal insufficiency.
- AWSC + ACP is not a counterexample to this diagnosis; AWSC succeeds with a stronger BC/flow anchor and does not rely on ACP to invent hold behavior from scratch.
- The current active question is how retention / hold semantics are lost when ACP value predictions are transformed into online reward / actor updates.

## Archive structure

### Archived docs
- Historical ACP reports and stage-by-stage diagnosis notes:
  - `docs/vlaw/archive/`
- Legacy non-ACP planning / session notes were also moved under `docs/vlaw/archive/` to keep `docs/vlaw/` focused on current entrypoints.

### Archived scripts
- Historical ACP launchers, sweep scripts, and one-off analyzers:
  - `scripts/archive/`
- Current retained analysis entrypoints live in `scripts/analysis/`.

## Scope note
- The active ACP surface is intentionally minimal now.
- Older v3/v4/v5/v6/v7 reports and legacy scripts are preserved for traceability, but are no longer the default entrypoints.
