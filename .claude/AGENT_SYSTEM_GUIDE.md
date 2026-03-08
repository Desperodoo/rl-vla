# Claude Agent + VS Code Copilot Agent 配置攻略

> 更新时间：2026-03-06
> 适用版本：Claude Code 2.x，github.copilot-chat 0.38.x
> 本文基于官方文档 + 本地扩展源码 + 实测调试日志整理

---

## 目录
1. [两套系统的关系](#1-两套系统的关系)
2. [配置文件完整地图](#2-配置文件完整地图)
3. [CLAUDE.md — Claude Code 记忆文件](#3-claudemd--claude-code-记忆文件)
4. [settings.json — 权限 / Hooks / MCP](#4-settingsjson--权限--hooks--mcp)
5. [VS Code Copilot Agent 配置](#5-vs-code-copilot-agent-配置)
6. [Multi-Agent 系统设计模式](#6-multi-agent-系统设计模式)
7. [本项目 Agent System 快速上手](#7-本项目-agent-system-快速上手)
8. [常见问题与最佳实践](#8-常见问题与最佳实践)

---

## 1. 两套系统的关系

```
┌─────────────────────────────────────────────────────────────┐
│                     你的项目目录                             │
│                                                             │
│  .claude/                   .github/                        │
│  ├── CLAUDE.md              ├── copilot-instructions.md     │
│  ├── settings.json          ├── agents/*.agent.md           │
│  └── skills/<name>/         ├── instructions/*.md           │
│       SKILL.md              ├── prompts/*.prompt.md         │
│                             ├── hooks/*.json                │
│                             └── skills/<name>/SKILL.md      │
│                                                             │
│  Claude Code (CLI)          VS Code Copilot Chat            │
│  ↓ 读取                     ↓ 读取                          │
│  CLAUDE.md (必读)           copilot-instructions.md (必读)  │
│  settings.json              *.agent.md / *.instructions.md  │
│  .claude/skills/            .github/skills/ + .claude/skills/│
└─────────────────────────────────────────────────────────────┘
```

**核心区别**：

| 维度 | Claude Code (CLI) | VS Code Copilot + Claude Agent |
|------|-------------------|-------------------------------|
| 启动方式 | 终端 `claude` 命令 | VS Code 侧边栏 Chat |
| 记忆文件 | `CLAUDE.md` | `copilot-instructions.md` |
| Agent 定义 | Task tool (内置子 Agent) | `.github/agents/*.agent.md` |
| 配置入口 | `.claude/settings.json` | `settings.json` (VS Code) |
| Hooks | `.claude/settings.json` | `.github/hooks/*.json` |
| Skills | `.claude/skills/` | `.github/skills/` **或** `.claude/skills/` (共享!) |
| 模型选择 | 调用时指定 | agent frontmatter `model:` 字段 |

**重要**：`.claude/skills/` 目录被 VS Code Copilot 和 Claude Code **同时读取**，是两套系统的共享区域。

---

## 2. 配置文件完整地图

### 优先级（高 → 低）

```
/etc/claude-code/.claude/settings.json   ← 企业管控 (只读)
~/.claude/settings.json                  ← 用户全局
.claude/settings.json                    ← 项目共享 (提交到 git)
.claude/settings.local.json              ← 本地覆盖 (gitignore!)
```

### 文件索引

```
项目根/
├── CLAUDE.md                    # (可选) 与 .claude/CLAUDE.md 等效
├── .claude/
│   ├── CLAUDE.md                # Claude Code 主记忆文件 ← 本项目已创建
│   ├── settings.json            # 项目级权限/hooks/MCP 配置
│   ├── settings.local.json      # 本地敏感配置 (gitignored)
│   └── skills/
│       └── <skill-name>/
│           └── SKILL.md         # Claude Code + Copilot 共享 skill
│
└── .github/
    ├── copilot-instructions.md  # Copilot 全局 workspace 指令
    ├── agents/
    │   ├── *.agent.md           # 自定义 Agent 定义
    │   └── RESULT_FILE_PROTOCOL.md  # (本项目) 协议文档
    ├── instructions/
    │   └── *.instructions.md   # 文件类型/场景专项指令
    ├── prompts/
    │   └── *.prompt.md          # 可复用提示词 (slash commands)
    └── hooks/
        └── *.json               # Copilot hooks (可选)
```

---

## 3. CLAUDE.md — Claude Code 记忆文件

### 什么是 CLAUDE.md

Claude Code 启动时**自动加载**注入为 system prompt，无需手动引用。加载范围：

1. `~/.claude/CLAUDE.md` — 用户全局，永远加载
2. `.claude/CLAUDE.md` 或 `CLAUDE.md`（项目根）— 进入项目时加载
3. 子目录中的 `CLAUDE.md` — Claude 浏览到该目录时动态加载

### 内容建议

```markdown
# 项目简介（1-3句）

## 常用命令
\`\`\`bash
# 激活环境
conda activate rlft_ms3
# 运行训练
python rlft/online/train_policy.py
\`\`\`

## 目录结构（关键路径）
- rlft/vlaw/        ← 核心模块
- checkpoints/vlaw/ ← 模型权重
- data/vlaw/        ← 数据

## 编码规范
- Python 3.10+，类型提示必须
- 配置用 tyro dataclass
- 日志用 wandb

## 环境约束
- GPU: 10 × RTX 4090，用 CUDA_VISIBLE_DEVICES 分配
- 代理: export http_proxy=http://10.20.93.149:7890
```

### 最佳实践

- **简洁**：只写每次任务都会用到的信息；细节写在对应模块的 `instructions.md`
- **命令优先**：把 build/test/run 命令放在最前面，Claude 最常需要这些
- **不要重复**：`copilot-instructions.md` 和 `CLAUDE.md` 内容可以互相指向，不要完整复制
- `.github/copilot-instructions.md` 对 Copilot，`.claude/CLAUDE.md` 对 Claude Code，各司其职

---

## 4. settings.json — 权限 / Hooks / MCP

### 完整 Schema

```jsonc
// .claude/settings.json
{
  // ── 工具权限 ──────────────────────────────────────────────
  "allowedTools": ["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
  "disallowedTools": ["Browser"],

  // ── MCP 服务器 ────────────────────────────────────────────
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/path/to/server.js"],
      "env": { "API_KEY": "..." }
    },
    "python-tools": {
      "command": "python",
      "args": ["-m", "my_mcp_server"],
      "cwd": "/home/wjz/rl-vla"
    }
  },

  // ── Hooks ────────────────────────────────────────────────
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",        // 空字符串 = 匹配所有
        "hooks": [
          {
            "type": "command",
            "command": "echo '=== Session Started ===' >> /tmp/claude-sessions.log",
            "timeout": 5
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            // stdin 收到 JSON: { "tool_name": "Bash", "tool_input": { "command": "..." } }
            // 返回非零退出码 2 可以拒绝工具调用
            "command": "cat > /dev/null",  // 此例仅透传
            "timeout": 5
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write|NotebookEdit",
        "hooks": [
          {
            "type": "command",
            "command": "input=$(cat); fp=$(echo \"$input\" | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))\"); [ -n \"$fp\" ] && python3 -m black \"$fp\" 2>/dev/null; exit 0",
            "timeout": 30,
            "statusMessage": "Formatting with black..."
          }
        ]
      }
    ],
    "SubagentStart": [...],   // 子 Agent 启动时
    "SubagentStop": [...],    // 子 Agent 结束时
    "Stop": [...]             // 会话结束时
  }
}
```

### Hook 事件详解

| 事件 | 触发时机 | stdin JSON 字段 |
|------|---------|----------------|
| `SessionStart` | 新会话第一个 prompt | `{ "session_id": "..." }` |
| `UserPromptSubmit` | 每次用户提交 prompt | `{ "prompt": "..." }` |
| `PreToolUse` | 工具调用前 | `{ "tool_name": "...", "tool_input": {...} }` |
| `PostToolUse` | 工具调用成功后 | `{ "tool_name": "...", "tool_input": {...}, "tool_response": {...} }` |
| `SubagentStart` | Task tool 派遣子 Agent | `{ "subagent_type": "general-purpose", ... }` |
| `SubagentStop` | 子 Agent 完成返回 | `{ "subagent_type": "...", "result": "..." }` |
| `PreCompact` | 上下文压缩前 | `{}` |
| `Stop` | 会话结束 | `{ "session_id": "..." }` |

**Hook 返回规则**：
- 退出码 `0`：允许继续
- 退出码 `2`：**拒绝**该操作（PreToolUse 可用于安全门控）
- stdout 的 JSON `{ "decision": "block", "reason": "..." }` 同样可拒绝

### MCP (Model Context Protocol)

MCP 是 Claude Code 的工具扩展协议，允许挂载外部服务作为工具：

```
.claude/settings.json 中配置 mcpServers
→ Claude Code 启动时连接
→ 工具以 <server>/<tool> 形式出现在 Claude 的工具列表中
```

常见 MCP 用途：数据库查询、API 调用、自定义文件格式解析、CI/CD 集成。

---

## 5. VS Code Copilot Agent 配置

### 5.1 启用 Claude Agent 模式

在 VS Code `settings.json`（用户或工作区级别）：

```json
{
  "github.copilot.chat.claudeAgent.enabled": true,
  "github.copilot.chat.claudeAgent.allowDangerouslySkipPermissions": false
}
```

`claudeAgent.enabled = true` 后，Copilot Chat 面板中会出现 **Claude Agent** 会话入口。这是 Anthropic Claude Agent SDK 直接在 VS Code 内运行，使用你的 Copilot 订阅额度，原生支持 `.github/agents/*.agent.md`。

### 5.2 `.agent.md` 文件格式（完整 Frontmatter）

```yaml
---
name: "Agent Display Name"         # 可选，默认用文件名
description: "简短描述，用于 Agent 选择器和 subagent 发现"  # 必填

# 工具权限
tools:
  - read                           # 读文件
  - edit                           # 编辑文件
  - search                         # 搜索文件/文本
  - execute                        # 执行 shell 命令 (= runCommands)
  - agent                          # 调用其他 agent 作为 subagent
  - web                            # 网页抓取
  - todo                           # 任务列表
  - fetch                          # HTTP 请求

# 可以调用的 subagent 白名单（不填 = 不限制）
agents:
  - Worker-Agent-A
  - Worker-Agent-B

# 模型选择（支持 fallback 列表）
model:
  - claude-sonnet-4.6 (copilot)    # 首选
  - gpt-4o (copilot)               # fallback

# 是否在 Agent 选择器中显示（默认 true）
user-invocable: true

# 防止该 agent 再次调用 subagent（用于叶节点 worker）
disable-model-invocation: false

# 输入提示（显示在 chat 输入框 placeholder）
argument-hint: "描述你要执行的任务..."

# 结构化交接按钮
handoffs:
  - label: "下一步: 训练模型"       # 按钮文字
    agent: WM-Agent                # 目标 agent
    prompt: "开始训练..."           # 预填文本
    send: false                    # false=按钮, true=自动发送
---

# Agent 系统 Prompt (Markdown 正文)

你是 XXX Agent，负责...

## 职责
...

## 禁止操作
...
```

### 5.3 `.instructions.md` 文件格式

```yaml
---
name: "Python Standards"
description: "写 Python 代码时使用。涵盖类型提示、wandb 日志、tyro 配置。"
applyTo: "**/*.py"    # 匹配文件时自动附加（可选）
---

# Python 编码规范
...
```

- **有 `applyTo`**：创建/编辑匹配文件时自动注入
- **无 `applyTo`**：按 description 关键词语义匹配，按需加载

### 5.4 `.prompt.md` 文件格式

```yaml
---
name: vlaw-iteration
description: "执行完整 VLAW 一轮迭代 (Step 1-8)"
agent: VLAW-Coordinator
tools: [agent, runCommands, read, edit, search]
---

# 执行 VLAW 迭代

当前轮次: ${input:iter_num:1}
GPU 分配: ${input:gpu_ids:0-9}

请先读取 .github/vlaw-status.md，然后执行完整迭代流程。
```

- 出现在 Copilot Chat 的 `/` 命令列表
- `${input:varname:default}` 语法支持运行时填参

### 5.5 Skills 格式

```
.github/skills/<skill-name>/SKILL.md
.claude/skills/<skill-name>/SKILL.md   ← 两套系统共享!
```

每个 skill 是一个 `SKILL.md` 文件（类似 agent 但更轻量，无 frontmatter agent 功能）。在 Claude Code 中通过 `/skill-name` 调用。

---

## 6. Multi-Agent 系统设计模式

### 6.1 Coordinator-Worker 模式

```
用户
 │
 ▼
Coordinator Agent           ← 不执行业务代码，只调度
 ├──► Worker-A (并行)       ← 有限工具集，领域专家
 ├──► Worker-B (并行)
 └──► Worker-C (串行，依赖 A/B)
```

**Coordinator 约束**：
- 只能使用 `agent`, `read`, `edit`, `search` 工具
- 禁止执行 `runCommands`（具体命令交给 Worker）
- 通过 `agents: [...]` frontmatter 白名单限制可调用的 Worker

**Worker 约束**：
- `disable-model-invocation: false` → 但通常不需要再下派子 Agent
- 任务结束后必须输出结构化结果

### 6.2 并行调度规则

在同一个 Agent 响应中，不插入文字地连续发出多个 Task 调用 = 并行执行：

```python
# Coordinator 消息体（概念示意）：
# [Task: Worker-A "执行步骤3"] ← 同时发出
# [Task: Worker-B "执行步骤4"] ← 同时发出
# ↑ 两者并行运行，节省时间
```

**不能并行的情况**：B 依赖 A 的输出结果时，必须等待 A 完成再发出 B。

### 6.3 RESULT_FILE 防截断协议（本项目标准）

子 Agent Token 预算耗尽时会截断，父 Agent 收到空响应。解决方案：**双写**

```bash
# Worker Agent 第一行必须执行：
mkdir -p /home/wjz/rl-vla/logs/vlaw
export RESULT_FILE="/home/wjz/rl-vla/logs/vlaw/AGENT_NAME-result-$(date +%Y%m%d_%H%M%S).md"
echo "# AGENT_NAME 任务报告" > "$RESULT_FILE"

# 每完成一步：
echo "- [x] Step N: 描述 ($(date +%H:%M))" >> "$RESULT_FILE"  # 写文件
# 同时在消息中输出该步骤摘要                                    # 写消息
```

**截断恢复流程**（Coordinator）：

```
1. 检测截断: 响应为空 或 缺少 ✅/❌
2. 读取结果文件: ls -lt logs/vlaw/*-result*.md | head -5
3. 更新 vlaw-status.md: 标记 ⚠️ 截断
4. 重新派遣: "继续 Step N，跳过已完成步骤"
    ↑ 绝不自己接管 Worker 的业务任务
```

完整协议见：`.github/agents/RESULT_FILE_PROTOCOL.md`

### 6.4 上下文传递最佳实践

| 方式 | 适用场景 |
|------|---------|
| 共享状态文件（如 `vlaw-status.md`）| Agent 间异步状态同步 |
| Dispatch prompt 直接携带参数 | 少量结构化参数传递 |
| RESULT_FILE 路径在消息中返回 | 大量输出结果传递 |
| 共享 HDF5/safetensors 文件路径 | 模型/数据产物传递 |

**不要**：把大段文本直接嵌入 dispatch prompt，让 Worker 自己读状态文件。

---

## 7. 本项目 Agent System 快速上手

### 现有 Agent 一览

| Agent | 文件 | 职责 |
|-------|------|------|
| VLAW-Coordinator | `vlaw-coordinator.agent.md` | 总调度，管理迭代循环 |
| Data-Agent | `data-agent.agent.md` | 数据收集、VAE 编码 |
| WM-Agent | `wm-agent.agent.md` | Ctrl-World 训练 |
| Reward-Agent | `reward-agent.agent.md` | VLM 奖励标注 |
| Imagination-Agent | `imagination-agent.agent.md` | 想象引擎 |
| Policy-Agent | `policy-agent.agent.md` | ShortCut Flow 策略更新 |
| Eval-Agent | `eval-agent.agent.md` | 评估与指标统计 |
| Progress-Agent | `progress-agent.agent.md` | 进度汇总、状态文件更新 |

### 典型使用流程

**在 VS Code Copilot Chat 中：**

```
1. 打开 Chat 面板 (Ctrl+Shift+I)
2. 点击 Agent 选择器 → 选择 "VLAW-Coordinator"
3. 输入: "开始第 1 轮迭代。先读 vlaw-status.md 确认当前状态。"
4. Coordinator 自动:
   - 读 .github/vlaw-status.md
   - 并行派遣 Reward-Agent + WM-Agent
   - 等待结果 → 派遣 Imagination-Agent
   - ...
   - 结束前派遣 Progress-Agent 更新状态
```

**单独调用 Worker（调试用）：**

```
Agent: Data-Agent
Prompt: "执行 P0.2 验证: obs_mode=rgbd 输出格式确认。GPU: 0"
```

### 需要创建的 Claude Code 配置

本项目已有完整的 Copilot agent 系统，还缺少 Claude Code 侧配置：

1. **`.claude/CLAUDE.md`** ← 已创建（见本目录）
2. **`.claude/settings.json`** ← 可选，添加 hooks 和 MCP

参考配置：

```jsonc
// .claude/settings.json （按需添加）
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "input=$(cat); fp=$(echo \"$input\" | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))\"); [ -f \"$fp\" ] && [[ \"$fp\" == *.py ]] && python3 -m black \"$fp\" 2>/dev/null; exit 0",
            "timeout": 15,
            "statusMessage": "Formatting Python..."
          }
        ]
      }
    ]
  }
}
```

---

## 8. 常见问题与最佳实践

### Q: CLAUDE.md 和 copilot-instructions.md 内容要一致吗？

不需要完全一致，但核心信息应该同步。建议：
- `copilot-instructions.md`：面向 VS Code Copilot，强调 agent 调度规则、状态文件路径
- `CLAUDE.md`：面向 Claude Code CLI，强调常用命令、GPU 分配、conda 环境

两者都短而精。细节分散到 `instructions/*.instructions.md`。

### Q: 什么时候用 `.github/hooks/` vs `.claude/settings.json` 的 hooks？

- `.claude/settings.json`（Claude Code hooks）：控制 Claude Code CLI 行为，如代码格式化、危险命令拦截
- `.github/hooks/`（Copilot hooks）：控制 VS Code Copilot agent 行为

两套 hook 系统格式相同，但作用域不同。大多数情况下，在 `.claude/settings.json` 中配置 Claude Code hooks 就够了。

### Q: model 字段写什么？

```yaml
model: ['claude-sonnet-4.6 (copilot)']   # 通过 Copilot 订阅调用 Claude
model: ['claude-opus-4-6']               # 直接 API（需要 Anthropic key）
model: ['gpt-4o (copilot)']              # OpenAI via Copilot
```

`(copilot)` 后缀表示通过 GitHub Copilot 订阅计费，不需要单独的 Anthropic API Key。

### Q: Worker Agent 被截断了怎么办？

遵循 `§T 截断恢复三步法`（详见 `vlaw-coordinator.agent.md`）：
1. 读 `logs/vlaw/*-result*.md` 找进度
2. 更新 `vlaw-status.md` 标记截断
3. 重新派遣，不自己接管

### Q: 如何让 Coordinator 严格不执行业务代码？

在 `vlaw-coordinator.agent.md` 的 frontmatter 中不包含 `execute`/`runCommands` 工具：

```yaml
tools: ['agent', 'edit', 'search', 'read']  # 无 execute!
```

这从根本上阻止 Coordinator 运行 shell 命令。

### Q: 如何调试 Claude Code hooks？

```bash
# 查看当前会话的调试日志
ls -lt ~/.claude/debug/ | head -5
tail -f ~/.claude/debug/<session-id>.txt

# 日志中搜索 hook 执行情况
grep -i "hook" ~/.claude/debug/<session-id>.txt
```

### Q: 多个 Agent 并行时如何避免 GPU 竞争？

在 Coordinator dispatch prompt 中明确分配 GPU：

```
Step 3 → Reward-Agent: CUDA_VISIBLE_DEVICES=6,7
Step 4 → WM-Agent: CUDA_VISIBLE_DEVICES=0,1,2,3
```

见 `vlaw-coordinator.agent.md` 的 GPU 分配表。

---

## 参考资源

当网络可访问时（走代理 `http://10.20.93.149:7890`），参考官方文档：

- Claude Code 概述：`https://docs.anthropic.com/en/docs/claude-code/overview`
- CLAUDE.md 记忆：`https://docs.anthropic.com/en/docs/claude-code/memory`
- Hooks 参考：`https://docs.anthropic.com/en/docs/claude-code/hooks`
- MCP 集成：`https://docs.anthropic.com/en/docs/claude-code/mcp`
- Sub-agents：`https://docs.anthropic.com/en/docs/claude-code/sub-agents`
- VS Code Copilot 自定义 agents：`https://code.visualstudio.com/docs/copilot/customization/custom-agents`
- VS Code Copilot instructions：`https://code.visualstudio.com/docs/copilot/customization/custom-instructions`
- VS Code Copilot hooks：`https://code.visualstudio.com/docs/copilot/customization/hooks`

本地参考（无需网络）：
- `~/.vscode-server/extensions/github.copilot-chat-0.38.1/assets/prompts/skills/agent-customization/references/`
  - `agents.md` — agent frontmatter 完整 schema
  - `instructions.md` — instructions 文件规范
  - `hooks.md` — hooks 事件和格式
  - `skills.md` — skills 搜索路径和格式
