# rl-vla Codex Workspace

本目录是从仓库内 `./.claude` 体系完整迁移并重构后的 Codex 工作区说明。

迁移目标：
- 不依赖“引用 `.claude`”的方式保留上下文
- 将原先分散在多个文件里的重复描述合并
- 保留当前仓库的 VLAW 主线、CARM 真机主线、`ctrl_world` 外部代码约束、以及常用工作流

当前 `.codex` 的组织方式：

```text
.codex/
  README.md                               ← 总入口与迁移说明
  WORKSPACE.md                            ← 项目全局背景、架构、环境、操作约束
  CTRL_WORLD.md                           ← 外部 Ctrl-World 代码修改规范
  CARM_PIPELINE.md                        ← CARM 真机当前状态、审计结论、下一阶段计划与 baseline
  CARM_ACTION_SEMANTICS_AND_HITL.md       ← CARM action 语义、teleop uplift、HITL live 约束
  CARM_INFERENCE_ADMISSION_POLICY.md      ← inference episode 准入策略
  PI05_FINETUNING_PITFALLS_2026-04-28.md  ← PI05 微调踩坑、测试门槛、task/subtask 语义
  PI05_PROGRESS_2026-04-22.md             ← PI05 微调阶段性进度与远端训练记录
  VLAW_RESEARCH_NOTES.md                  ← VLAW/ACP/WM 的长期研究结论
  SCRIPT_POLICY_PLATFORM_PLAN.md          ← script runtime 平台设计与当前施工范围
  skills/
    project-operations/SKILL.md           ← 状态检查、记忆更新、进度同步
    training-internals/SKILL.md           ← RLPD/ACP 训练内科诊断
    vlaw-execution/SKILL.md               ← VLAW Algorithm 1 执行与各 Agent 职责
```

使用原则：
- 默认先读 [WORKSPACE.md](/home/amax/rl-vla/.codex/WORKSPACE.md)，理解仓库主线与约束。
- 涉及 `ctrl_world/` 改动时，额外遵守 [CTRL_WORLD.md](/home/amax/rl-vla/.codex/CTRL_WORLD.md)。
- 涉及 CARM 真机 inference 回流、staging、second-stage 训练时，额外遵守 [CARM_PIPELINE.md](/home/amax/rl-vla/.codex/CARM_PIPELINE.md) 和 [CARM_INFERENCE_ADMISSION_POLICY.md](/home/amax/rl-vla/.codex/CARM_INFERENCE_ADMISSION_POLICY.md)。
- 涉及 CARM 的 action 语义、teleop uplift、HITL live owner/source 设计时，优先阅读 [CARM_ACTION_SEMANTICS_AND_HITL.md](/home/amax/rl-vla/.codex/CARM_ACTION_SEMANTICS_AND_HITL.md)。
- 涉及 PI05 微调、dense full-ft、DeepSpeed ZeRO checkpoint、task/subtask 语义时，优先阅读 [PI05_FINETUNING_PITFALLS_2026-04-28.md](/home/wjz/rl-vla/.codex/PI05_FINETUNING_PITFALLS_2026-04-28.md)；需要时间线时再读 [PI05_PROGRESS_2026-04-22.md](/home/wjz/rl-vla/.codex/PI05_PROGRESS_2026-04-22.md)。
- 涉及 ACP、retention/hold 诊断、WM 时间尺度与 imagination 设计时，优先阅读 [VLAW_RESEARCH_NOTES.md](/home/amax/rl-vla/.codex/VLAW_RESEARCH_NOTES.md)。
- 涉及真机 script policy 执行层、skill runtime、任务树和 learned adapter 设计时，优先阅读 [SCRIPT_POLICY_PLATFORM_PLAN.md](/home/amax/rl-vla/.codex/SCRIPT_POLICY_PLATFORM_PLAN.md)。
- 需要使用“技能式”工作流时，读 `skills/` 下对应 `SKILL.md`。

重构说明：
- 原 `.claude/CLAUDE.md` 与多个补充计划文件中重复的“项目状态、阶段、资产、门控、环境、命令”已合并到 [WORKSPACE.md](/home/amax/rl-vla/.codex/WORKSPACE.md) 和 [CARM_PIPELINE.md](/home/amax/rl-vla/.codex/CARM_PIPELINE.md)。
- 原 `.claude/skills/_archive/` 中按单一 Agent 分散的执行说明已按职责域合并到 [vlaw-execution/SKILL.md](/home/amax/rl-vla/.codex/skills/vlaw-execution/SKILL.md)，避免同一套阶段定义和门控规则重复维护。
- 原 `check-status`、`update-memory`、`progress-agent` 的重叠“状态/记忆管理”内容已合并到 [project-operations/SKILL.md](/home/amax/rl-vla/.codex/skills/project-operations/SKILL.md)。
- 原 `carm_current_status_plan_progress.md`、`carm_next_stage_plan.md`、`carm_phase1_phase2_audit.md`、`carm_second_stage_baseline_matrix.md` 的重叠背景、结论、下一步和实验矩阵已整合到 [CARM_PIPELINE.md](/home/amax/rl-vla/.codex/CARM_PIPELINE.md)。
- `docs/` 中关于 CARM action 语义、teleop uplift、HITL owner/source、human chunk proposal、canonical logging 的长期结论已提炼到 [CARM_ACTION_SEMANTICS_AND_HITL.md](/home/amax/rl-vla/.codex/CARM_ACTION_SEMANTICS_AND_HITL.md)。
- `docs/vlaw/` 中关于 ACP 共识、retention/hold 诊断、WM 时间尺度、autoregressive history buffer 的长期研究结论已提炼到 [VLAW_RESEARCH_NOTES.md](/home/amax/rl-vla/.codex/VLAW_RESEARCH_NOTES.md)。

维护约定：
- 以后优先更新 `.codex`，不再把新增上下文继续写回 `.claude`。
- 若新增流程文档，优先按“一个主题一个完整文件”的方式扩展，避免交叉引用式碎片化。
