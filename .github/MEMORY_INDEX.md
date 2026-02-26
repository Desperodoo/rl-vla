# MEMORY 索引（Memory Index）

> 目的：建立 task_id 到状态、日志、产物、知识条目的可追溯映射。

## 字段约定
- task_id: 任务唯一ID
- owner_agent: 负责Agent
- status_ref: `vlaw-status.md` 对应条目
- result_md_ref: `logs/vlaw/*-result-*.md`
- result_json_ref: `logs/vlaw/*-result-*.json`
- artifact_refs: 关键产物路径（数据/ckpt/图表）
- knowledge_refs: 关联知识文档（bugs/decisions/interfaces）

## 当前索引

| task_id | owner_agent | status_ref | result_md_ref | result_json_ref | artifact_refs | knowledge_refs |
|---|---|---|---|---|---|---|
| T-DATA-LIFT-001 | Data-Agent | `.github/vlaw-status.md` Task 2 系列 | `logs/vlaw/Data-Agent-result-20260226_174404.md` | 待补（Phase A 起执行） | `data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1/LiftPegUpright-v1_real_1772098799.h5`; `data/vlaw/encoded/rollouts/iter1_highsuc/LiftPegUpright-v1/LiftPegUpright-v1_real_1772098799.h5` | `.github/knowledge/maniskill-envs.md`; `.github/knowledge/bugs-and-fixes.md` |
| T-WM-COMP-001 | WM-Agent | `.github/vlaw-status.md` Task 1 | `logs/vlaw/WM-Agent-result-20260226_165343.md` | 待补（Phase A 起执行） | `logs/vlaw/wm_comparison_frames/` | `.github/knowledge/decisions.md` |
| T-EVAL-BASELINE-001 | Eval-Agent | `.github/vlaw-status.md` Task 3 | `logs/vlaw/Eval-Agent-result-20260226_172239.md` | 待补（Phase A 起执行） | `results/vlaw/pld_eval_baseline_20ep.json` | `.github/knowledge/interfaces.md` |

## 维护规则
1. 新任务先分配 `task_id`，再派发。
2. 每个任务完成后补齐 `result_md_ref/result_json_ref/artifact_refs`。
3. `vlaw-status.md` 只保留摘要，详细过程通过本索引跳转。
