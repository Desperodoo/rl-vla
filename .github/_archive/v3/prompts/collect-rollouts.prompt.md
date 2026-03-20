---
name: collect-rollouts
description: "在 ManiSkill 中收集 rollout 数据"
agent: Data-Agent
tools: ['runCommands', 'read', 'edit']
---

# 收集 Rollout 数据

在 ManiSkill 环境中使用当前策略收集 rollout 数据。

## 步骤
1. 读取 `.github/vlaw-status.md` 确认当前策略 checkpoint 路径
2. 加载 ShortCut Flow 策略
3. 在 GPU 4-5 上运行 ManiSkill rollout:
   - `CUDA_VISIBLE_DEVICES=4,5`
   - `num_envs=64` per GPU
   - 采集 RGB(2cam) + state + actions
4. 保存为 HDF5 到 `data/vlaw/rollouts/`
5. 更新 `.github/vlaw-status.md`

## 参数
- 任务: ${input:task:LiftPegUpright-v1}
- 轨迹数: ${input:num_episodes:50}
- 策略 checkpoint: ${input:checkpoint:checkpoints/il/best_eval_success_once.pt}
