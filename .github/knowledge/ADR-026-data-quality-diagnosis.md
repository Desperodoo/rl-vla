# ADR-026: AWSC 数据质量诊断与 frame_skip 修正

> **日期**: 2026-03-04 | **状态**: 已诊断，待重新采集
> **关联**: ADR-023 (帧率分析), BUG-022 (ghost episodes)

---

## 一、问题描述

使用 AWSC 微调 checkpoint (`fair_comparison/awsc/best_s42`) 收集的 1200 条轨迹存在严重质量问题：
- T_max=21（预期 ≥35），成功轨迹仅 92 条 (7.7%)
- 160 条轨迹因 T<10 被丢弃

## 二、根因分析

### 根因 1（已确认）: frame_skip=5 而非脚本中的 3

**证据**: 收集日志 `logs/vlaw/collect_awsc_mixed.log` 第 5 行：
```
开始收集: 目标 1200 条轨迹, min_traj_length=10, frame_skip=5
```

脚本 `scripts/vlaw/collect_awsc_data.py` 中写的是 `frame_skip=3`，但 Data-Agent 在运行时修改了参数为 5。

**数学验证**:
- frame_skip=5, max_episode_steps=100
- 帧 = 1(first, step 1) + 20(regular, steps 5,10,...,100) = **21** ✅ 完美匹配 T_max=21

### 根因 2（可能）: EMA 权重未正确加载

**证据**: 日志显示 `"加载 ShortCut Flow"` 而非 `"检测到 AWSC checkpoint (EMA+config)"`。
这意味着 `_has_ema=False`，即 EMA 检测代码可能未被触发。AWSC checkpoint 中 velocity_net_ema 有 154 个key，应该被检测到。

**影响**: 若未使用 EMA 权重而使用了 online velocity_net 权重，策略性能会显著降低。

### 根因 3: max_episode_steps 不一致

| 数据集 | 实际 max_episode_steps | 说明 |
|--------|----------------------|------|
| IL 基线 (rollouts/) | 200 (ManiSkill 默认) | T_max=68, frame_skip=3 → 68 = 1+66+1 ✓ |
| AWSC (rollouts_awsc/) | 100 | AWSC 训练时 100，但 ManiSkill 默认 200 |
| 推荐 | 200 | 统一使用 ManiSkill 默认值 |

## 三、frame_skip 精确分析

| frame_skip | 采样帧率 | vs WM 5Hz | 100 步 T | 200 步 T | 评价 |
|-----------|---------|-----------|---------|---------|------|
| 3 | 6.67 Hz | +33% ❌ | 35 | 68 | WM 预训练先验偏差大 |
| **4** | **5.0 Hz** | **精确匹配 ✅** | **26** | **51** | **推荐** |
| 5 | 4.0 Hz | -20% ❌ | 21 | 41 | 帧间运动量过大 |

**结论**: frame_skip=4 是唯一正确选择（20Hz / 4 = 5Hz = DROID 降采样后 WM 预训练频率）。

## 四、IL 基线数据 (rollouts/) 同样需要重采集

IL 基线数据使用 frame_skip=3 (6.67Hz)，与 WM 预训练的 5Hz 存在 33% 偏差。
如果要正确对齐 WM，所有数据都需要以 frame_skip=4 重新采集。

## 五、受影响资产

| 路径 | 大小 | 状态 | 必须操作 |
|------|------|------|---------|
| `data/vlaw/rollouts_awsc/` | 615M | ❌ frame_skip=5 错误 | **删除** |
| `data/vlaw/encoded_awsc/` | 823M | ❌ 编码自错误数据 | **删除** |
| `data/vlaw/meta_info_awsc/` | 12K | ❌ 统计自错误数据 | **删除** |
| `data/vlaw/rollouts/` | 1.1G | ⚠️ frame_skip=3 (6.67Hz) | 归档或删除 |
| `data/vlaw/encoded/` | 1.7G | ⚠️ 编码自 6.67Hz 数据 | 归档或删除 |
| `data/vlaw/meta_info/` | 16K | ⚠️ 统计自 6.67Hz 数据 | 重新生成 |

## 六、AWSC Checkpoint 验证

```
路径: runs/fair_comparison/fair_comparison/awsc/best_s42__1772570560/checkpoints/final.pt
Agent 前缀: ['critic', 'critic_target', 'velocity_net', 'velocity_net_ema']
velocity_net: 154 keys, velocity_net_ema: 154 keys, critic: 280 keys
Config: pred_horizon=8, obs_horizon=2, max_episode_steps=100, normalize_actions=False
pretrain_path: runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt
visual_encoder: 存在，从 checkpoint 顶层 "visual_encoder" key 加载
EMA: 存在，`load_shortcut_flow_policy` 中有 EMA→velocity_net 重映射逻辑
```

**经验证的加载流程** (`rlft/utils/flow_wrapper.py:load_shortcut_flow_policy`):
1. 检测到 `velocity_net_ema.*` keys → 重映射为 `velocity_net.*`（替换 online weights）
2. 从 `velocity_net.cond_encoder.1.weight` 推断 state_dim
3. 创建 `ShortCutVelocityUNet1D` 并加载 EMA weights
4. 从 checkpoint["visual_encoder"] 加载 PlainConv
5. 返回 `(flow_wrapper, visual_encoder, state_dim)`

## 七、ManiSkill 环境行为确认

- **LiftPegUpright-v1 不存在 early termination**：所有 episode 恒定运行到 max_episode_steps 后 truncated
- **实验验证**: step 100 时 terminated=False, truncated=True
- **partial reset** (`env.reset(options={"env_idx"})`) 正常工作
