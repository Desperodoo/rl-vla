# Agent 系统轻量化重构计划（Final v1.0）

> 目标：在不改变 VLAW 业务流程的前提下，降低 Agent 系统维护成本、减少提示词膨胀、提升截断恢复和状态同步的稳定性。

## 1) 重构范围

- Plan 层：将“长期蓝图”与“迭代执行”分离
- Memory 层：建立统一索引与去重规则
- Logs 层：建立可机读摘要，降低 Coordinator 汇总成本
- Agent 协议层：去重通用 RESULT_FILE 规则，保留最小硬约束

## 2) 现状问题（摘要）

1. 通用协议在多个 agent 文件重复，维护成本高，容易漂移。
2. `vlaw-status.md` 同时承担状态表、事件流、复盘，体积和复杂度持续上升。
3. logs 以 markdown 自由文本为主，自动聚合与检索成本高。
4. knowledge/work-log/status 之间缺少统一任务 ID，追溯链路依赖人工。

## 3) 目标架构（轻量版）

### 3.1 Plan 双层

- L0（静态）：`VLAW_REPRODUCTION_PLAN.md`
- L0.5（执行看板，新增）：`VLAW_EXECUTION_BOARD.md`
  - 仅保留：当前迭代任务、依赖、负责人、验收指标、状态

### 3.2 Memory 四层（保留）+ 索引层（新增）

- 保留：L0/L1/L2/L3（Plan/Status/Work-log/Knowledge）
- 新增：`MEMORY_INDEX.md`
  - 映射：`task_id -> status条目 -> result日志 -> 产物路径 -> 知识条目`

### 3.3 Logs 双轨

- 人类可读：`logs/vlaw/*-result-*.md`
- 机读摘要（新增）：`logs/vlaw/*-result-*.json`
  - 固定字段：
    - `task_id`, `agent`, `start_time`, `end_time`, `status`
    - `steps_completed`, `metrics`, `artifacts`, `blockers`

### 3.4 Agent 协议去重

- 当前：每个 worker 都嵌入完整 RESULT_FILE 模板
- 目标：提炼为单一共享规范文档（例如 `.github/agents/RESULT_FILE_PROTOCOL.md`）
- worker 只保留：职责、验收标准、GPU 约束、失败回退

## 4) 迁移策略（低风险）

### Phase A（先观测，不破坏）

1. 引入 `task_id` 命名约定（不改变现有流程）
2. 保留 markdown 结果文件，同时新增 json 摘要文件
3. Coordinator 优先读 json，md 作为兜底

### Phase B（减负）

1. 将 `vlaw-status.md` 历史事件下沉至 work-log
2. `vlaw-status.md` 仅保留“当前事实表 + 最近关键变更 + 链接”

### Phase C（协议收敛）

1. 抽离共享 RESULT_FILE 协议
2. worker agent 文件删去重复模板，仅保留引用与最小强制项

## 5) 验收指标

- Coordinator 平均派发 prompt 长度下降 ≥ 40%
- `vlaw-status.md` 行数稳定（按周增长显著下降）
- 截断恢复平均耗时下降 ≥ 30%
- 任意任务追溯（状态→日志→产物）在 30 秒内完成

## 6) 风险与回退

- 风险：共享协议文件未被 subagent 自动读取导致约束失效
- 风险：json 摘要与 md 摘要不一致
- 回退：保留现有“dispatch 末尾粘贴完整协议”机制，直到新方案验证通过

---

## 7) 待官方文档核验点（本文件后续补充）

1. VS Code/Copilot 对 subagent 上下文注入机制是否支持“共享片段自动继承”
2. prompt 文件/agent 文件引用外部文档时，是否保证自动读取
3. 是否有官方推荐的 structured output 或结果摘要格式
4. memory 能力在当前版本中的作用边界（项目文件记忆 vs 平台记忆）

---

## 8) 官方文档核验结论（2026-02-26）

### 8.1 subagent 上下文行为（关键）

- 结论：subagent **默认不继承**主 agent 的指令与对话历史，只接收传入子任务 prompt。
- 含义：仅把 RESULT_FILE 协议抽到共享文档并“在主 agent 中引用”并不可靠；若 dispatch prompt 未显式携带关键约束，subagent 可能看不到。

### 8.2 自定义规则加载行为

- `.github/copilot-instructions.md` 与 `*.instructions.md` 会被自动注入，但多文件并存时顺序不保证。
- 通过 Markdown 链接引用的说明文件，是否自动带入取决于设置（`chat.includeReferencedInstructions`，默认可为 `false`）。
- 含义：共享协议“只靠链接引用”存在不确定性。

### 8.3 prompt/custom-agent/tool 优先级

- Prompt 文件里的 `tools` 优先级高于 agent 默认工具。
- 可通过 `agents` 白名单限制可被调用的 subagent，避免误选。

### 8.4 可用新机制

- 可使用 Agent Hooks（Preview）在 `SubagentStart/Stop`、`PreToolUse/PostToolUse` 注入或校验约束。
- 可用 Agent Skills（`.github/skills`）承载可复用流程和资源，按需加载，减少上下文膨胀。

## 9) 重构细节敲定（最终决策）

### 9.1 RESULT_FILE 协议收敛方案（确定）

采用“**共享文档 + dispatch 最小硬约束**”双层机制：

1. 共享文档：`.github/agents/RESULT_FILE_PROTOCOL.md`（完整规范，单点维护）。
2. dispatch prompt 强制携带最小硬约束（不可省略）：
   - 日志目录固定为 `logs/vlaw/`
   - 禁止写入 `/tmp/`
   - 第一条命令创建 `RESULT_FILE`
   - 最终消息必须包含：文件路径 + 步骤摘要 + 状态符号（✅/⚠️/❌）
3. worker agent 文件仅保留一句“遵循 RESULT_FILE_PROTOCOL + 若冲突以 dispatch 硬约束为准”。

> 解释：这样既实现去重，又避免“引用文件没被加载”导致协议失效。

### 9.2 Plan 层落地

- 新增 `VLAW_EXECUTION_BOARD.md` 作为“当前迭代看板”（仅近期任务，不记历史流水）。
- `VLAW_REPRODUCTION_PLAN.md` 保持静态蓝图角色，不再承载日常执行细节。

### 9.3 Memory 层落地

- 新增 `MEMORY_INDEX.md`，固定字段：
  - `task_id`, `owner_agent`, `status_ref`, `result_md_ref`, `result_json_ref`, `artifact_refs`, `knowledge_refs`
- 规则：所有新任务必须先分配 `task_id`，再写 status/log/knowledge，避免孤儿记录。

### 9.4 Logs 层落地

- 保留现有 md 报告；新增同名 json 摘要。
- json 字段统一：
  - `task_id`, `agent`, `start_time`, `end_time`, `status`, `steps`, `metrics`, `artifacts`, `errors`, `next_action`
- Coordinator 汇总优先读 json，md 用于人工核查与截断恢复。

## 10) 实施顺序（冻结）

1. 定义 `task_id` 与 json schema（不改现有执行逻辑）。
2. 引入 `VLAW_EXECUTION_BOARD.md` + `MEMORY_INDEX.md`。
3. 增加 result json 产出与 Coordinator 读取优先级。
4. 抽离 RESULT_FILE 协议并瘦身各 worker agent 文件。
5. 最后再做 `vlaw-status.md` 减负（历史下沉 work-log）。

## 11) 验证清单（必须通过）

- 随机抽 3 个 subagent 任务，均能在不读共享文档正文的情况下执行最小硬约束。
- 截断后 30 秒内可通过 `task_id` 定位 status→result→artifact 全链路。
- Coordinator 汇总同一批任务时，优先读取 json 成功率达到 100%。

## 12) 回退策略（最终）

- 若任一验证失败，立即回退到“dispatch 末尾粘贴完整 RESULT_FILE 规范”的当前策略。
- 回退后保留 `task_id` 与 json 日志，不回退这两项（它们与协议抽离解耦，收益稳定）。
