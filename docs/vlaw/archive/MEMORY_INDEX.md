# MEMORY 索引

> task_id → 状态、日志、产物的追溯映射。

| task_id | owner | result_md | 关键产物 |
|---------|-------|-----------|---------|
| T-DATA-LIFT-001 | Data | `logs/vlaw/Data-Agent-result-20260226_174404.md` | `rollouts/iter1_highsuc/` + encoded |
| T-DATA-LIFT-002 | Data | `logs/vlaw/Data-Agent-result-20260226_202856.md` | `rollouts/iter1_lift_inc20/` + encoded |
| T-REWARD-REAL-001 | Reward | `logs/vlaw/Reward-Agent-result-20260226_205054.md` | `labeled/iter1_lift_only/` (n=160, vlm_succ=0%) |
| T-WM-COMP-001 | WM | `logs/vlaw/WM-Agent-result-20260226_165343.md` | `logs/vlaw/wm_comparison_frames/` |
| T-EVAL-BASELINE-001 | Eval | `logs/vlaw/Eval-Agent-result-20260226_172239.md` | `results/vlaw/pld_eval_baseline_20ep.json` |
| T-AUDIT-001 | Eval | `logs/vlaw/Eval-Agent-result-20260226_221642.md` | `logs/vlaw/data_audit_report.{md,json}` |
| T-WM-BASELINE-001 | WM | `logs/vlaw/WM-Agent-result-20260226_223747.md` | `results/vlaw/wm_baseline_report.md` |
| T-VLM-BASELINE-001 | Reward | `logs/vlaw/Reward-Agent-result-20260226_225815.md` | `results/vlaw/vlm_baseline_report.md` |

**规则**: 新任务先分配 task_id → 完成后补 result_md + artifact → vlaw-status.md 只保留摘要。
