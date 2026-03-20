# update-memory

当用户调用 `/update-memory` 时，执行结构化记忆更新。

**目标**：将本次会话的关键进展同步到三个持久化文件，防止跨会话遗忘。

---

## 执行步骤

### Step 1：读取当前状态

```
读取以下文件：
- /home/lizh/rl-vla/.github/vlaw-status.md
- /home/lizh/rl-vla/.github/VLAW_NEXT_STEPS.md
- /home/lizh/.claude/projects/-home-lizh-rl-vla/memory/MEMORY.md
```

### Step 2：从对话历史推断本次进展

提取以下类型的信息：
- 完成的实验/任务（含关键数值结论）
- 新发现的 Bug 或修复（含根因）
- 路径/配置变更
- 新的阻塞点或决策（ADR 级别）
- GPU 分配变化

如果无法从历史推断，直接询问用户：
> "本次会话完成了什么？有哪些需要记录的关键发现？"

### Step 3：更新 MEMORY.md

路径：`/home/lizh/.claude/projects/-home-lizh-rl-vla/memory/MEMORY.md`

规则：
- **更新已有 section**：如果进展属于已有 section（如 VLAW 阻塞、ACP 状态），直接修改对应行
- **新增 section**：如果是全新主题，在文件末尾追加新 section
- **不重复**：检查是否已有相同内容，避免冗余
- **保持简洁**：每条记录 1-2 行，指向详细报告文件

### Step 4：更新 vlaw-status.md

路径：`/home/lizh/rl-vla/.github/vlaw-status.md`

只更新有变化的部分：
- 阶段状态（✅/🔄/⛔）
- 当前阻塞描述
- GPU 分配表

在文件顶部更新 `最后更新` 日期。

### Step 5：更新 VLAW_NEXT_STEPS.md

路径：`/home/lizh/rl-vla/.github/VLAW_NEXT_STEPS.md`

- 将已完成的任务标记为 ✅ 或删除
- 添加本次产生的新待办项
- 调整优先级（如有变化）

---

## 输出格式

完成后输出简短摘要：

```
## 记忆更新完成

**MEMORY.md**：[新增/更新了哪些内容]
**vlaw-status.md**：[更新了哪些状态]
**VLAW_NEXT_STEPS.md**：[勾掉/新增了哪些任务]
```

**不修改** CLAUDE.md（由用户手动维护）。
**不执行**任何训练/推理命令。
