# project-operations

本技能整合了原先分散的 `check-status`、`update-memory` 与 `progress-agent` 工作流，统一负责项目状态检查、记忆更新与进度文件同步。

原则：
- 只做只读检查或状态文档更新
- 不执行训练、推理、数据处理等业务操作
- 任何更新都必须尽量基于仓库内实际文件、结果文件和系统快照

## 1. 使用场景

适用于以下请求：
- 快速汇总当前项目状态
- 更新项目记忆 / 状态文件
- 汇总 GPU、进程、checkpoint、数据目录状态
- 整理最近一次或最近几次实验执行结果

## 2. 只读状态检查模式

### Step 1：读取主状态文档

优先读取以下文件中实际存在的项：

```bash
ls .github/vlaw-status.md .github/VLAW_NEXT_STEPS.md docs/*progress*.md docs/*plan*.md 2>/dev/null
```

若存在 `.github/vlaw-status.md`，提取：
- 当前阶段
- 当前阻塞
- GPU 分配

若存在 `.github/VLAW_NEXT_STEPS.md`，提取：
- 进行中任务
- 最高优先级待办项

### Step 2：快速扫描资产状态

```bash
ls -lt checkpoints/vlaw/world_model/ 2>/dev/null | head -5
ls -lt checkpoints/vlaw/reward_model/ 2>/dev/null | head -3
ls -lt checkpoints/vlaw/policy/ 2>/dev/null | head -3
du -sh data/vlaw/*/ 2>/dev/null
ls -lt logs/vlaw/*-result*.md 2>/dev/null | head -5
```

### Step 3：GPU 与进程状态

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null

tmux list-sessions 2>/dev/null || true
ps aux | grep -E "train_|imagination|eval_|accelerate" | grep -v grep
```

### Step 4：输出格式

建议输出：
- 当前阶段
- 当前阻塞
- GPU 状态
- 最新 checkpoint
- 数据状态
- 最近结果文件摘要
- 下一步建议

## 3. 记忆更新模式

目标：
- 将本次会话关键进展同步到项目长期记忆文件

### Step 1：读取当前记忆与状态文件

优先查找并读取存在的文件，例如：

```bash
ls .github/vlaw-status.md .github/VLAW_NEXT_STEPS.md .github/knowledge/decisions.md .github/knowledge/bugs-and-fixes.md 2>/dev/null
```

若仓库中维护有单独记忆文件，也一并读取。

### Step 2：从当前上下文提取需要记录的信息

重点提取：
- 完成的实验或任务
- 新发现的 Bug / 根因 / 修复
- 路径或配置变更
- 新 blocker
- 新的架构决策
- GPU 资源分配变化

### Step 3：更新规则

- 优先更新已有 section，而不是重复追加
- 每条记录保持简洁，指向更详细的报告文件
- 不重复写入同一条结论
- 只改有变化的部分

### Step 4：状态文件更新点

如果以下文件存在，可同步更新：

1. `.github/vlaw-status.md`
   - 最后更新时间
   - 阶段状态
   - 阻塞描述
   - GPU 分配

2. `.github/VLAW_NEXT_STEPS.md`
   - 已完成任务标记为完成
   - 新增待办项
   - 调整优先级

3. `.github/knowledge/decisions.md`
   - 新 ADR

4. `.github/knowledge/bugs-and-fixes.md`
   - 新 Bug 记录

## 4. 统一进度同步模式

此模式兼容原 `progress-agent`。

### Step 1：收集系统状态

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

tmux list-sessions 2>/dev/null || echo "no tmux sessions"
ps aux | grep -E "train_|imagination|eval_|accelerate" | grep -v grep
ls -lt logs/vlaw/*-result*.md 2>/dev/null | head -10
```

### Step 2：读取最近结果文件

```bash
for f in $(ls -t logs/vlaw/*-result*.md 2>/dev/null | head -3); do
  tail -20 "$f"
done
```

### Step 3：收集 checkpoint / 数据目录

```bash
ls -lt checkpoints/vlaw/world_model/ 2>/dev/null
ls -lt checkpoints/vlaw/reward_model/ 2>/dev/null
ls -lt checkpoints/vlaw/policy/ 2>/dev/null
du -sh data/vlaw/demos/ data/vlaw/rollouts/ data/vlaw/encoded/ data/vlaw/synthetic/ 2>/dev/null
```

### Step 4：更新输出要求

最终汇总至少包含：

#### 变更摘要
- 哪些状态文件更新了
- 哪些任务从进行中变为完成
- 是否新增 ADR 或 Bug

#### 当前快照
- GPU 占用
- 活跃进程
- 最近结果文件摘要
- 可用资源

#### 下一步建议
- 当前最优先动作
- 可以并行做的动作
- 哪些任务仍被依赖阻塞

## 5. 注意事项

- 不在这个技能里执行训练、推理或数据处理命令
- 若仓库里不存在 `.github/` 状态文件，优先退化为读取 `docs/` 下最新计划和进度文档
- 若路径发生迁移，应先统一状态源，再继续自动更新逻辑
