---
name: vlaw-iteration
description: "执行一轮完整的 VLAW 迭代 (Step 1-8)"
agent: VLAW-Coordinator
tools: ['agent', 'runCommands', 'read', 'edit', 'search']
---

# VLAW 完整迭代

执行一轮完整的 VLAW Algorithm 1 迭代循环。

> 当前阶段策略：默认仅执行 LiftPegUpright-v1；PickCube-v1 与 StackCube-v1 在 Lift-only 验证通过后再执行。

## 迭代步骤
按照 [VLAW_REPRODUCTION_PLAN.md](../VLAW_REPRODUCTION_PLAN.md) 第 3.6 节:

1. **真实环境 Rollout** → 使用 Data-Agent 收集 ${input:num_rollouts:50} 条/任务
2. **VAE 离线编码** → 使用 Data-Agent 编码新轨迹
3. **VLM 标注 (真实)** → 使用 Reward-Agent 标注真实轨迹
4. **微调世界模型** → 使用 WM-Agent 在 D_real + λ·D_demo 上训练 50K steps
5. **Imagination** → 使用 Imagination-Agent 生成 ${input:num_synthetic:500} 条合成轨迹
6. **VLM 标注 (合成)** → 使用 Reward-Agent 标注合成轨迹
7. **策略更新** → 使用 Policy-Agent 在 D_real+ ∪ D_syn+ 上训练 2000 steps
8. **评估** → 使用 Eval-Agent 评估 (50 episodes/task)

## 参数
- 迭代轮次: ${input:iteration:1}
- 任务: ${input:task:LiftPegUpright-v1}

## 注意事项
- 确保前置依赖完成后再启动下游任务
- 每步完成后更新 `.github/vlaw-status.md`
- 如遇到 OOM，参考复现计划中的降级方案
