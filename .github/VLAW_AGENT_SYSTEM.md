# VLAW Multi-Agent System 设计文档

> **创建时间**: 2026-02-24
> **最后更新**: 2026-02-24
> **目的**: 基于 VS Code Copilot 的 Custom Agent 特性，搭建多 Agent 协同系统来并行推进 VLAW 复现

---

## 一、系统总览

### 1.1 设计理念

VLAW 复现项目包含 8 个阶段 (P0-P7)，模块间有明确的依赖关系但也存在并行空间。我们利用 VS Code Copilot 的 **Custom Agent + Subagent + Prompt File + Instruction File** 四层机制，构建一个协调器-工人 (Coordinator-Worker) 模式的多 Agent 系统。

**核心思路**:
- **1 个 Coordinator Agent** 管理完整迭代循环
- **6 个 Worker Agent** 各负责一个专业领域
- **共享记忆** 通过 `.github/vlaw-status.md` + `copilot-instructions.md` 实现
- **Prompt Files** 封装常见可重复任务
- **Instruction Files** 自动注入编码规范

### 1.2 架构图

```
┌─────────────────────────────────────────────────────┐
│                  用户 (多窗口)                        │
│   Chat 1: Coordinator  │  Chat 2: WM  │  Chat 3: ..│
└────────┬────────────────┴──────┬───────┴────────────┘
         │                       │
    ┌────▼────────────────────────▼────────────────────┐
    │          VS Code Copilot Agent System            │
    │                                                   │
    │  ┌─────────────────────────────────────────────┐ │
    │  │   Global Instructions (copilot-instructions) │ │
    │  │   + VLAW_REPRODUCTION_PLAN.md (只读参考)     │ │
    │  └─────────────────────────────────────────────┘ │
    │                                                   │
    │  ┌───────────────┐    ┌──────────────────────┐  │
    │  │  VLAW-        │───▶│  Worker Agents       │  │
    │  │  Coordinator  │    │                      │  │
    │  │  (orchestrate)│    │  WM-Agent            │  │
    │  │               │◀───│  Reward-Agent        │  │
    │  │  agents:      │    │  Data-Agent          │  │
    │  │  [all workers]│    │  Imagination-Agent   │  │
    │  │               │    │  Policy-Agent        │  │
    │  │  handoffs →   │    │  Eval-Agent          │  │
    │  └───────────────┘    └──────────────────────┘  │
    │                                                   │
    │  ┌───────────────┐    ┌──────────────────────┐  │
    │  │ Prompt Files  │    │ Instruction Files    │  │
    │  │ /vlaw-iter... │    │ /vlaw-module.inst..  │  │
    │  │ /collect-...  │    │ /ctrl-world.inst..   │  │
    │  │ /train-wm...  │    │ /python-std.inst..   │  │
    │  │ /run-imag...  │    │                      │  │
    │  │ /status-...   │    │                      │  │
    │  └───────────────┘    └──────────────────────┘  │
    │                                                   │
    │  ┌──────────────────────────────────────────┐   │
    │  │        Shared Memory Layer                │   │
    │  │  .github/vlaw-status.md (读写)            │   │
    │  │  .github/VLAW_REPRODUCTION_PLAN.md (只读) │   │
    │  │  checkpoints/vlaw/ (模型权重)             │   │
    │  │  data/vlaw/ (数据)                        │   │
    │  └──────────────────────────────────────────┘   │
    └──────────────────────────────────────────────────┘
```

---

## 二、文件清单

### 2.1 Agent 文件 (`.github/agents/`)

| 文件 | 名称 | 角色 | 可用工具 | 子Agent |
|------|------|------|---------|---------|
| `vlaw-coordinator.agent.md` | VLAW-Coordinator | 总协调器 | agent,edit,search,read,run_in_terminal,fetch | 全部 Worker |
| `wm-agent.agent.md` | WM-Agent | 世界模型专家 | edit,search,read,run_in_terminal,fetch,agent | — |
| `reward-agent.agent.md` | Reward-Agent | VLM 奖励模型专家 | edit,search,read,run_in_terminal,fetch | — |
| `data-agent.agent.md` | Data-Agent | 数据管线专家 | edit,search,read,run_in_terminal | — |
| `imagination-agent.agent.md` | Imagination-Agent | Imagination 引擎专家 | edit,search,read,run_in_terminal | — |
| `policy-agent.agent.md` | Policy-Agent | 策略更新专家 | edit,search,read,run_in_terminal | — |
| `eval-agent.agent.md` | Eval-Agent | 评估专家 | edit,search,read,run_in_terminal | — |

### 2.2 Prompt 文件 (`.github/prompts/`)

| 文件 | 斜杠命令 | 用途 | 绑定 Agent |
|------|---------|------|-----------|
| `vlaw-status-check.prompt.md` | `/vlaw-status-check` | 检查项目进度 | agent |
| `collect-rollouts.prompt.md` | `/collect-rollouts` | 收集 ManiSkill rollout | Data-Agent |
| `train-world-model.prompt.md` | `/train-world-model` | 训练 Ctrl-World | WM-Agent |
| `run-imagination.prompt.md` | `/run-imagination` | 运行 Imagination 引擎 | Imagination-Agent |
| `vlaw-iteration.prompt.md` | `/vlaw-iteration` | 执行一轮完整迭代 | VLAW-Coordinator |

### 2.3 Instruction 文件 (`.github/instructions/`)

| 文件 | 适用范围 | 用途 |
|------|---------|------|
| `vlaw-module.instructions.md` | `rlft/vlaw/**/*.py` | VLAW 模块编码规范 |
| `ctrl-world.instructions.md` | `ctrl_world/**/*.py` | Ctrl-World 修改规范 |
| `python-standards.instructions.md` | `**/*.py` | 全局 Python 规范 |

### 2.4 全局指令

| 文件 | 用途 |
|------|------|
| `.github/copilot-instructions.md` | 自动注入所有 chat — 项目结构、技术栈、协作约定 |

### 2.5 共享记忆

| 文件 | 用途 | 读写权限 |
|------|------|---------|
| `.github/vlaw-status.md` | 实时状态跟踪 (阶段、checkpoint、数据、GPU) | 所有 Agent 读写 |
| `.github/VLAW_REPRODUCTION_PLAN.md` | 完整技术方案 (Phase/接口/参数) | 只读参考 |

---

## 三、使用方式

### 3.1 方式一：通过 Agent 下拉菜单选择 (推荐)

在 VS Code Chat 中，从 Agent 下拉菜单选择对应 Agent：

1. **VLAW-Coordinator**: 适用于整体推进、迭代管理、跨模块协调
2. **WM-Agent**: 专攻 Ctrl-World 相关任务
3. **Reward-Agent**: 专攻 VLM 奖励模型
4. **Data-Agent**: 专攻数据收集和处理
5. **Imagination-Agent**: 专攻 Imagination 引擎
6. **Policy-Agent**: 专攻策略更新
7. **Eval-Agent**: 专攻评估

### 3.2 方式二：通过 Prompt File 斜杠命令

在 chat 中输入 `/` 触发斜杠命令：
- `/vlaw-status-check` — 快速查看项目进度
- `/collect-rollouts` — 启动数据收集
- `/train-world-model` — 启动 WM 训练
- `/run-imagination` — 启动 Imagination
- `/vlaw-iteration` — 执行完整迭代

### 3.3 方式三：多窗口并行

打开多个 VS Code Chat 窗口，各选择不同 Agent，实现并行工作：

```
窗口 1: VLAW-Coordinator (协调全局)
窗口 2: Data-Agent (P1 数据收集 — GPU 4-5)
窗口 3: WM-Agent (P2 WM 训练 — GPU 0-3)
窗口 4: Reward-Agent (P3 VLM — GPU 6-7)
```

### 3.4 方式四：Coordinator 自动委派 Subagent

选择 VLAW-Coordinator 后，它会通过 `runSubagent` 工具自动将子任务委派给专业 Agent。也可以通过 Handoff 按钮手动切换。

---

## 四、Agent 交互协议

### 4.1 共享记忆协议

所有 Agent 通过 `.github/vlaw-status.md` 进行异步通信：

```
Agent A 完成任务 → 更新 vlaw-status.md 状态
Agent B 开始新任务 → 先读取 vlaw-status.md 检查前置依赖
```

**状态值**:
- ⬜ 未开始
- 🔄 进行中
- ✅ 已完成
- ❌ 阻塞
- ⚠️ 需要修复

### 4.2 数据传递协议

Agent 间通过文件系统传递数据：

```
Data-Agent → data/vlaw/rollouts/ → WM-Agent (读取训练数据)
Data-Agent → data/vlaw/encoded/ → WM-Agent (读取 VAE latent)
WM-Agent → checkpoints/vlaw/world_model/ → Imagination-Agent (加载 WM)
Imagination-Agent → data/vlaw/synthetic/ → Reward-Agent (标注)
Reward-Agent → data/vlaw/synthetic/ (标注后) → Policy-Agent (训练)
Policy-Agent → checkpoints/vlaw/policy/ → Eval-Agent (评估)
```

### 4.3 Handoff 工作流

Coordinator 提供预定义的 Handoff 按钮，支持以下流转：

```
Coordinator → Data-Agent:      "Start Data Collection"
Coordinator → WM-Agent:        "Train World Model"
Coordinator → Reward-Agent:    "Build Reward Model"
Coordinator → Imagination-Agent: "Build Imagination Engine"
Coordinator → Policy-Agent:    "Update Policy"
Coordinator → Eval-Agent:      "Run Evaluation"

WM-Agent → Imagination-Agent:  "Start Imagination"
WM-Agent → Eval-Agent:         "Verify WM Quality"
Data-Agent → WM-Agent:         "Train World Model"
Data-Agent → Reward-Agent:     "Label with VLM"
Imagination-Agent → Reward-Agent: "Label Synthetic Data"
Imagination-Agent → Policy-Agent: "Update Policy"
Policy-Agent → Eval-Agent:     "Evaluate Policy"
Reward-Agent → Imagination-Agent: "Start Imagination"
```

---

## 五、并行策略

### 5.1 可并行的阶段

```
P0: 三个子任务可完全并行
  P0.1 (WM 环境) ← WM-Agent (GPU 0)
  P0.2 (ManiSkill 验证) ← Data-Agent (GPU 4)
  P0.3 (VLM 获取) ← Reward-Agent (GPU 6)

P1 + P3.1: 可并行
  P1 (数据管线) ← Data-Agent (GPU 4-5)
  P3.1 (奖励模型实现) ← Reward-Agent (GPU 6-7, 代码实现不需要 GPU)

迭代中: 部分并行
  Step 1-2 (Rollout + VAE) → Step 3 (VLM 标注) 可在 Step 4 (WM 训练) 开始前并行进行
```

### 5.2 严格串行的阶段

```
P2 → P4: WM 必须训练完才能 Imagination
P4 → P5: 合成数据必须生成完才能更新策略
P5 → P6/P7: 策略必须更新完才能评估
```

---

## 六、故障恢复

### 6.1 Agent 上下文丢失

每个 Agent 的第一步都是读取 `.github/vlaw-status.md`，因此即使 chat 窗口关闭/重开，Agent 也能恢复上下文。

### 6.2 训练中断

- Checkpoint 定期保存 (每 5K steps)
- 训练脚本支持 `--resume` 参数
- `.github/vlaw-status.md` 记录最后保存的 checkpoint 路径

### 6.3 数据损坏

- HDF5 文件写入使用原子操作 (写入临时文件 → rename)
- 关键数据增量保存，不覆盖之前的数据

---

## 七、扩展指南

### 7.1 添加新 Agent

1. 在 `.github/agents/` 中创建 `{name}.agent.md`
2. 定义 YAML frontmatter (name, tools, model, handoffs)
3. 在 body 中编写详细指令
4. 如需被 Coordinator 调用，在 `vlaw-coordinator.agent.md` 的 `agents` 列表中添加
5. 更新 `.github/vlaw-status.md` 添加对应状态行

### 7.2 添加新 Prompt File

1. 在 `.github/prompts/` 中创建 `{name}.prompt.md`
2. 定义 YAML frontmatter (name, description, agent, tools)
3. 编写步骤化指令
4. 在 chat 中通过 `/{name}` 调用

### 7.3 添加新指令规则

1. 在 `.github/instructions/` 中创建 `{name}.instructions.md`
2. 使用 `applyTo` 指定适用的文件 glob 模式
3. 指令会自动注入匹配文件的 chat 上下文
