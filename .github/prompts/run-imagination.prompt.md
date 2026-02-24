---
name: run-imagination
description: "运行 Imagination 引擎生成合成数据"
agent: Imagination-Agent
tools: ['runCommands', 'read', 'edit']
---

# 运行 Imagination

使用 Ctrl-World + ShortCut Flow 进行 Policy-in-the-Loop Rollout 生成合成数据。

## 步骤
1. 读取 `.github/vlaw-status.md` 确认 WM checkpoint 和策略 checkpoint 就绪
2. 加载 Ctrl-World 和 ShortCut Flow
3. 在 4 GPU 上并行运行 Imagination:
   - 每 GPU 生成 125 条轨迹
   - num_inference_steps=25 (加速)
   - 12 步/轨迹
4. 保存合成轨迹到 `data/vlaw/synthetic/`
5. 调用 VLM 奖励模型进行批量标注
6. 统计成功率分布
7. 更新 `.github/vlaw-status.md`

## 参数
- 任务: ${input:task:LiftPegUpright-v1}
- 合成轨迹数: ${input:num_trajectories:500}
- Imagination 步数: ${input:horizon:12}
