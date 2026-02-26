# VLAW 项目记忆体系指南

> 这是整个记忆体系的**导航入口**。所有 Agent 开始工作前应先读此文件。

## 记忆层级

| 层级 | 文件 | 更新频率 | 内容 |
|------|------|---------|------|
| L0 Plan | `VLAW_REPRODUCTION_PLAN.md` | 不修改 | 完整技术方案、接口定义、超参数 |
| L1 Status | `vlaw-status.md` | 每个阶段完成时 | 阶段状态表、关键指标、GPU 分配 |
| L2 Work Log | `work-logs/YYYY-MM-DD.md` | 每次工作结束时追加 | 详细操作记录、命令、输出、遇到的问题 |
| L3 Knowledge | `knowledge/*.md` | 发现新知识时 | 架构决策、接口契约、Bug 库、环境配置 |

## 推荐阅读顺序

### 新 Agent 上手
1. `MEMORY_GUIDE.md`（本文件）
2. `vlaw-status.md` — 当前进度快照
3. `knowledge/interfaces.md` — 了解你负责模块的接口
4. `knowledge/env-setup.md` — 确认环境可用

### 每次工作前
1. `vlaw-status.md` — 确认前置依赖完成
2. `work-logs/最新日期.md` — 了解上次做了什么、有无未解决问题

### 每次工作后（必须）
1. 追加 `work-logs/今天日期.md` — 记录本次操作
2. 更新 `vlaw-status.md` — 更新阶段状态
3. 如有新 Bug/决策/接口变化 → 更新对应的 `knowledge/*.md`

## 写作规范

### work-logs 格式
```markdown
## HH:MM Agent名称 — 任务简述

**目标**: 一句话说明要做什么
**操作**:
- 步骤1: 命令/代码变更
- 步骤2: ...
**结果**: 成功/失败 + 关键输出数据
**产出**: 新建/修改的文件列表
**遗留**: 未完成的事项或注意事项（无则省略）
```

### knowledge 维护原则
- **decisions.md**: 每条决策说明"做了什么"+"为什么这样做"+"放弃了什么替代方案"
- **bugs-and-fixes.md**: 每条 bug 说明"症状"+"根因"+"修复方案"+"预防"
- **不要重复** L1/L2 里已有的信息，只记录 L1/L2 无法表达的深层知识
