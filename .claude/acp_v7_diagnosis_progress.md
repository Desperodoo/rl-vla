# ACP v7 Diagnosis Track Progress

## Status
- Started diagnosis-track implementation.
- Approved plan saved at `.claude/plans/goofy-sauteeing-babbage.md`.
- Current focus: convert v7 from ACP-only mechanism sweep into a controlled comparison entrypoint, then add reward diagnostics and unified analysis outputs.

## Completed
- Reviewed current v7 scheduler: `scripts/acp_v7_scheduler.py`
- Confirmed baseline references:
  - `docs/vlaw/acp_v6_rlpd_report.md`
  - `.github/vlaw-status.md`
  - `.github/VLAW_NEXT_STEPS.md`
- Confirmed ACP integration points:
  - `rlft/envs/acp_reward_wrapper.py`
  - `rlft/online/train_pld.py`
  - `rlft/online/train_dsrl.py`
  - `rlft/online/train_rlpd.py`
- Confirmed existing analyzer to extend:
  - `scripts/analyze_training_internals.py`
- Added `diag_core` mode to `scripts/acp_v7_scheduler.py` for the 6-cell controlled comparison.
- Added ACP reward diagnostics to `rlft/envs/acp_reward_wrapper.py`:
  - current / previous value
  - td delta
  - raw reward and clipped reward
  - clip mask
  - reset mask
  - grasp bonus and is_grasping remain exposed
- Extended `scripts/analyze_training_internals.py` with:
  - reward-source inference (`acp` vs `sim`)
  - machine-readable per-run summary builder
  - algorithm × reward-source controlled comparison aggregation
  - `fig_controlled_comparison.png`
  - report section for controlled comparison summary
- Smoke checks passed:
  - `python -m py_compile scripts/acp_v7_scheduler.py scripts/analyze_training_internals.py rlft/envs/acp_reward_wrapper.py`
  - `python scripts/acp_v7_scheduler.py --mode diag_core --dry_run`

## In Progress
- `diag_core` 6-cell controlled comparison completed.
- WandB data fetched for all 6 diagnostic runs under `rlpd-acp-v7-diag`.
- Generated internals diagnosis artifacts:
  - `docs/vlaw/figures/rlpd_acp_v7_diag/diagnosis_report.md`
  - `docs/vlaw/figures/rlpd_acp_v7_diag/diagnosis_summary.json`
  - `docs/vlaw/figures/rlpd_acp_v7_diag/fig_controlled_comparison.png`
- Wrote a deeper algorithm-level root cause report:
  - `.claude/acp_v7_failure_analysis.md`
- `fair_replay` 4-run exact-recipe mirror bundle completed and analyzed:
  - `pld_v7_fair_sim_s42`
  - `pld_v7_fair_acp_s42`
  - `dsrl_v7_fair_sim_s42`
  - `dsrl_v7_fair_acp_s42`
- Generated fair replay analysis artifacts:
  - `docs/vlaw/figures/rlpd_acp_v7_fair_replay/diagnosis_report.md`
  - `docs/vlaw/figures/rlpd_acp_v7_fair_replay/diagnosis_summary.json`
  - `docs/vlaw/figures/rlpd_acp_v7_fair_replay/fig_controlled_comparison.png`
- Current focus: interpret the replay bundle against historical `runs/fair_comparison` evidence and prepare a staged progress snapshot.

## Findings
- PLD and DSRL ACP training paths both wrap train envs with `DualCameraRewardWrapper`, but eval envs remain standard sim-eval envs. This is desirable for task evaluation, but must be treated as “ACP training / sim evaluation”, not “ACP evaluation env”.
- AWSC/RLPD uses `reward_mode=acp` in the train env path and also evaluates on standard task envs. This is consistent with comparing final task success, but reward-path parity still needs one explicit audit pass in `train_rlpd.py`.
- Existing `docs/vlaw/acp_v3_rlpd_internals_report.md` recorded an old concern about mixed reward semantics; current code path needs to be interpreted against the updated wrapper-based ACP implementation, not the old report alone.
- `diag_core` controlled comparison is now complete and shows:
  - AWSC remains the only algorithm with strong SAE retention under both reward sources.
  - AWSC sim: best SAE 0.76, final SAE 0.74, retention 0.84.
  - AWSC acp: best SAE 0.70, final SAE 0.52, retention 0.78.
  - PLD sim/acp are both poor: best SAE 0.04 / 0.06, final SAE both 0.00.
  - DSRL sim/acp are both poor at end: best SAE 0.08 / 0.14, final SAE 0.00 / 0.02.
- This means the main pathology is **not** “ACP uniquely breaks PLD/DSRL while sim is healthy” under the current controlled setup. Instead, PLD/DSRL are already failing the hold/retention problem even with sim reward in this matched setting.
- AWSC-specific ACP weakness is real but narrower: reward signal is extremely small relative to offline reward (`reward_gap_ratio ≈ 1049x`, `acp_step_mean ≈ 1e-4`), which hurts advantage discrimination (`advantage_mean ≈ 1.03`, `weight_max_peak ≈ 21.9`) and lowers final SAE from 0.74 → 0.52.
- PLD pathology is primarily retention/exploration, not ACP starvation:
  - sim: temperature final 0.0032, entropy min -54.55, final SAE 0.00.
  - acp: temperature final 0.1286 but entropy still reached -55.53 historically, final SAE 0.00.
  - Both reward sources have very large SO-SAE gap (~0.76), meaning the policy can reach but cannot hold.
- DSRL pathology is primarily retention plus critic weakness, not ACP starvation:
  - sim: best SAE 0.08, final 0.00, SO-SAE gap 0.84.
  - acp: best SAE 0.14, final 0.02, SO-SAE gap 0.78.
  - ACP slightly improves peak SAE versus sim, but does not fix end-state retention.
- For PLD/DSRL, the likely root cause is the absence of a strong behavior-cloning / hold-preserving anchor during online updates, combined with objective mismatch toward “touch/grasp/reach” rather than “maintain upright until episode end”.
- The most urgent unresolved issue is now baseline reproducibility: current `diag_core` sim controls did **not** reproduce the historical high-SAE `PLD/DSRL + sim_reward` behavior already captured in `runs/fair_comparison`.
- `fair_replay` 的结果表明：即便切回 `runs/fair_comparison` 的 71K exact recipe，当前代码路径下也**没有复现历史高 SAE sim baseline**。
- replay 正式报告 `docs/vlaw/figures/rlpd_acp_v7_fair_replay/diagnosis_report.md` 给出的 4-run 对照为：
  - PLD sim: best SAE 0.06, final SAE 0.00, best SO 0.80
  - PLD acp: best SAE 0.06, final SAE 0.00, best SO 0.92
  - DSRL sim: best SAE 0.02, final SAE 0.00, best SO 0.88
  - DSRL acp: best SAE 0.06, final SAE 0.00, best SO 0.94
- 这说明“当前问题”的优先级已进一步收敛为：
  1. **historical fair-comparison baseline no longer reproduces under current code path**；
  2. 在当前代码下，ACP 不是主要导致 PLD/DSRL 失败的独立因素；
  3. 主要 pathology 仍然是高 SO、极低 SAE、最终 retention 归零。
- Deeper drift audit now treats two fronts in parallel:
  1. commit-level code audit (`54faf40` for PLD, later stabilization commits for DSRL);
  2. minimal regression experiments to test whether post-v5/v6 stabilization logic is itself causing the reproduction drift.
- Added a new scheduler mode `drift_regression` to `scripts/acp_v7_scheduler.py`.
- `drift_regression` 6-run bundle is now complete and formally analyzed:
  - `docs/vlaw/figures/rlpd_acp_v7_drift_reg/diagnosis_report.md`
  - `docs/vlaw/figures/rlpd_acp_v7_drift_reg/diagnosis_summary.json`
  - `.claude/acp_v7_drift_regression_analysis.md`
- Strong new conclusion: **`q_target_clip=20` is the primary code-path drift that destroyed the historical sim baseline for both PLD and DSRL.**
- Evidence from regressions:
  - `pld_v7_reg_qclip0_sim_s42`: best/final SAE = 0.82 / 0.82
  - `pld_v7_reg_54faf40_sim_s42`: best/final SAE = 0.80 / 0.80
  - `dsrl_v7_reg_qclip0_sim_s42`: best/final SAE = 0.66 / 0.66
  - `dsrl_v7_reg_pre592df92_sim_s42`: best/final SAE = 0.04 / 0.02
- This sharply narrows the diagnosis:
  1. sim baseline failure was mainly caused by post-v5/v6 critic stabilization (`q_target_clip=20`), not by an irrecoverable algorithm collapse;
  2. PLD/DSRL current tuned regimes can still work on sim reward once q clipping is removed;
  3. ACP remains a second independent problem, because ACP mirrors did **not** recover after removing q clipping.

## Next
1. Audit whether the controlled sim baselines are weaker than historical v6 due to exact hyperparameter mismatch (especially DSRL long-train recipe vs current diag_core recipe).
2. Compare v6 best DSRL run against `dsrl_v7_diag_acp_s42` on entropy/temperature/Q/SAE-retention trajectory.
3. Inspect whether PLD/DSRL actor losses correlate with the persistent SO-SAE gap more strongly than reward source.
4. If confirmed, design next ablation around “hold-preservation” rather than only ACP reward strength.

## Constraints
- No sim-reward blending / warmup switching in ACP tracks.
- Keep scope on diagnosis, not broad new sweeps.
