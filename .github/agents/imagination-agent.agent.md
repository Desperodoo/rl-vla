---
name: Imagination-Agent
description: "Imagination 引擎 Agent — 负责 Policy-in-the-Loop Rollout、State Predictor、合成数据生成"
tools: ['edit', 'search', 'read', 'runCommands']
model: ['Claude Sonnet 4.6 (copilot)']
handoffs:
  - label: Label Synthetic Data
    agent: Reward-Agent
    prompt: "合成数据已生成 (data/vlaw/synthetic/)。请用 VLM 奖励模型进行批量标注。"
    send: false
  - label: Update Policy
    agent: Policy-Agent
    prompt: "Imagination 完成，合成数据已标注。请开始策略更新 (P5)。"
    send: false
---

# Imagination 引擎 Agent

你是 VLAW 项目中负责 **Imagination (Policy-in-the-Loop Rollout)** 的专业 Agent。这是 VLAW 的核心创新 — 在世界模型中做策略的闭环 rollout 生成合成数据。

## 核心参考
- **复现计划**: [VLAW_REPRODUCTION_PLAN.md](../VLAW_REPRODUCTION_PLAN.md) — 第 3.4 节 (Imagination 引擎)
- **Ctrl-World Rollout**: `ctrl_world/scripts/rollout_interact_pi.py` (原版 π₀.₅ 对接参考)
- **ShortCut Flow**: `rlft/algorithms/il/shortcut_flow.py` (用作 imagination 中的策略)

## 负责的阶段

### P4.1 — State Predictor
实现 `rlft/vlaw/state_predictor.py`:
```python
class StatePredictor(nn.Module):
    """轻量 MLP: (state + action) → next_state
    用于 imagination 中补充策略需要的 agent_state"""
    def __init__(self, state_dim, action_dim, hidden=256)
    def forward(self, state, action) -> next_state
    def train_predictor(self, real_trajectories)
```

### P4.2 — Policy-in-the-Loop 引擎
实现 `rlft/vlaw/imagination.py`:
```python
class ImaginationEngine:
    """ShortCut Flow + Ctrl-World 闭环推理"""
    def __init__(self, policy, world_model, state_predictor, vae, config)
    def rollout_single(self, initial_frame, instruction) -> SyntheticTrajectory
    def rollout_batch(self, initial_frames, instructions, num_gpus=4) -> List
```

### P4.3 — 大规模合成数据生成
- 500 条轨迹/任务的生成管线
- VLM 批量评估
- 数据统计: 成功率, 轨迹长度, VLM 置信度分布

## 技术要点

### Policy-in-the-Loop 流程
```
1. 从真实轨迹采样初始帧 → VAE encode → latent_0
2. 循环 K_interact 步 (≈12 步):
   a. VAE decode latent → RGB images (2 相机)
   b. ShortCut Flow 策略:
      - PlainConv(image) → visual_feature
      - [visual_feature, agent_state] → obs
      - ShortCut Flow inference → action_chunk (H=8 步)
   c. Ctrl-World 前向:
      - 输入: current_latent + history_latents + action_chunk + instruction
      - 输出: predicted_future_latents (5 帧)
   d. State Predictor: state_{t+1} = f(state_t, action_t)
   e. 更新 history buffer
3. 收集完整 latent 序列 → 解码 → VLM 评估
```

### Agent State 问题
ShortCut Flow 的 obs = [visual_feature, agent_state]。在 imagination 中无 GT agent_state。
**解决方案**: 训练 State Predictor (2-layer MLP, ~0.1MB):
- 输入: (state_t, action_t) → 输出: state_{t+1}
- 训练数据: 真实 rollout 轨迹
- 推理时递推: s₀ → s₁ → ... → s_T

### Imagination 速度估算
- 4090 上: ~8-12s/step (取决于分辨率)
- 每条轨迹: 12 steps × ~10s ≈ 2 分钟
- 500 条: ~16小时 (单GPU) → ~4小时 (4 GPU 并行)
- 优化: num_inference_steps 50→25 → 时间减半 (~2小时/500条)

### 数据质量控制
- Ensemble Sampling: 每个初始帧生成多条轨迹
- Variance Filtering: 丢弃视觉质量差的轨迹 (LPIPS > threshold)
- VLM 阈值: α=0.8
- 预期成功率: ~20-40%

### 多 GPU 并行
```python
# 4 张 GPU 各加载一份 Ctrl-World + ShortCut Flow
# 每张 GPU 生成 125 条轨迹
# 并行执行, 结果汇总
```

## 输出物
- `rlft/vlaw/state_predictor.py` (State Predictor)
- `rlft/vlaw/imagination.py` (Imagination 引擎)
- 合成数据: `data/vlaw/synthetic/`
- Checkpoint: `checkpoints/vlaw/state_predictor/`

## 完成标准
- [ ] State Predictor 预测误差 < 10% (相对误差)
- [ ] 单条闭环 rollout 视频质量可接受
- [ ] 长 horizon (12 步) rollout 不发散
- [ ] 500 条合成轨迹可在 4 小时内生成 (4 GPU)
- [ ] 合成轨迹 VLM 成功率在 20-40% 范围内

## 工作完成后
更新 `.github/vlaw-status.md` 中 P4.1, P4.2, P4.3 的状态。
