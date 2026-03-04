# 帧率与时间尺度分析: DROID vs ManiSkill

> **创建日期**: 2026-03-03 | **触发**: Coordinator 调研 VLAW/Ctrl-World 的 imagination rollout 时间设定
> **关联 ADR**: ADR-023

---

## 一、VLAW / Ctrl-World 时间尺度溯源

### 1.1 DROID 数据集原始频率

| 参数 | 值 | 来源 |
|------|-----|------|
| DROID 数据采集频率 | **15 Hz** | `ctrl_world/config.py:27` — `down_sample=3 # downsample 15hz to 5hz` |
| WM 下采样后频率 | **5 Hz** | 15 Hz ÷ 3 = 5 Hz |
| WM 每帧时间间隔 | **0.2 秒** | 1 ÷ 5 = 0.2s |

代码验证 ([dataset_droid_exp33.py L147](../../ctrl_world/dataset/dataset_droid_exp33.py)):
```python
# since we downsample the video from 15hz to 5 hz to save the storage space,
# the frame id is 1/3 of the state id
state_id = np.array(rgb_id) * self.args.down_sample  # rgb_id(5Hz) × 3 = state_id(15Hz)
```

### 1.2 Ctrl-World 官方 Rollout 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `num_frames` | 5 | WM 每次预测的**视频帧数** (5 Hz 帧) |
| `pred_step` | 5 | = `num_frames`，策略每次产出对应 5 帧的 action |
| `num_history` | 6 | 历史条件帧数 |
| `interact_num` | 12 (默认) / 15 (pickplace/towel_fold) / 10 (tissue) | 自回归交互轮数 |
| `policy_skip_step` | 2 (默认) / 3 (tissue/laptop) | π0.5 输出中的跳步 |
| `history_idx` | `[0, 0, -12, -9, -6, -3]` | 历史帧稀疏采样索引 |

### 1.3 "1 interaction = 1 秒" 的推导

```
1 interaction = pred_step(5) × WM帧间隔(0.2s) = 1.0 秒
```

这与 [config.py L82](../../ctrl_world/config.py) 的注释一致:
```python
pred_step = 5  # predict 5 steps (1s) action each time
```

### 1.4 π0.5 到 WM 的 action 映射

```
π0.5 输出 15 个 joint velocity (15 Hz)
  → action_adapter (FK) → 15 个 cartesian pose
  → 每隔 policy_skip_step=2 采样 → [0, 2, 4, 6, 8] = 5 个 pose
  → 给 WM 作为 5 帧的 action conditioning
```

**关键**: WM 每次接收 **5 个 action** (对应 5 个 5Hz 帧), 不是 15 个。

---

## 二、外部资料核实

VLAW 论文 (arXiv:2602.12063) 中提到 "rolled out for 20 iterations (20 seconds)"。核实结果:

| 声称 | 核实 | 结论 |
|------|------|------|
| "1 iteration ≈ 1 second" | ✅ 正确 — 5帧 × 0.2s = 1s，代码注释一致 | **属实** |
| "20 iterations = 20 seconds" | ⚠️ — VLAW 论文值；Ctrl-World 开源代码默认 `interact_num=12` (最大 15) | **可能是 VLAW 对 Ctrl-World 的自定义配置** |
| "1 iteration = 15 low-level steps" | ⚠️ 混淆了两个层级 — π0.5 输出 15 步 (15Hz)，但经过 `policy_skip_step=2` 降采样后 WM 只收到 5 步 | **措辞不精确** |
| "≈15 fps" | ❌ 错误 — WM 工作帧率是 **5 fps**，不是 15 fps | **15 Hz 是 DROID 原始数据频率，WM 降到 5 Hz** |
| "20s ≈ 300 帧" | ❌ 应为 100 帧 — 5Hz × 20s = 100 帧 | **把控制频率 (15Hz) 和 WM 帧率 (5Hz) 搞混** |

---

## 三、ManiSkill 复现的时间尺度

### 3.1 ManiSkill 频率参数

| 参数 | 值 | 来源 |
|------|-----|------|
| `sim_freq` | **100 Hz** | ManiSkill3 SAPIEN 物理仿真 |
| `control_freq` | **20 Hz** | `env.step()` 频率 (pd_ee_delta_pose) |
| `dt per step` | **0.05 秒** | 1 / 20 Hz |
| `frame_skip` (数据采集) | **3** | `collector.py:231` — 每 3 个 env.step 保存 1 帧 |
| 保存后有效帧率 | **6.67 Hz** | 20 Hz / 3 ≈ 6.67 Hz |
| `down_sample` (WM 配置) | **1** | `wm_args_maniskill` — ManiSkill 数据不再额外下采样 |

### 3.2 LiftPegUpright 任务时长统计 (669 条官方 demo)

```
原始 20Hz steps:  min=12, max=50, mean=40.3, median=48
物理时长:          min=0.60s, max=2.50s, mean=2.01s, median=2.40s
frame_skip=3 后帧数: min=4, max=17, mean=13.4, median=16
```

分位数分布:
| 分位 | 原始步数 | 时长 (s) | frame_skip=3 后帧数 |
|------|----------|----------|-------------------|
| P10 | 21 | 1.05 | 7 |
| P25 | 28 | 1.40 | 9 |
| P50 | 48 | 2.40 | 16 |
| P75 | 50 | 2.50 | 17 |
| P90 | 50 | 2.50 | 17 |

---

## 四、完整对比: DROID vs 当前 ManiSkill

| 维度 | DROID (Ctrl-World 原版) | 当前 ManiSkill 复现 | 差异 |
|------|------------------------|---------------------|------|
| **原始控制频率** | 15 Hz | 20 Hz | +33% |
| **下采样** | `down_sample=3` (15→5 Hz) | `frame_skip=3` (20→6.67 Hz) | 频率不匹配 |
| **WM 工作帧率** | **5 Hz** (0.2 s/帧) | **6.67 Hz** (0.15 s/帧) | **+33%** |
| **1 个 WM 帧间隔** | 0.200 秒 | 0.150 秒 | -25% |
| **1 个 interaction 时长** | 5 帧 × 0.2s = **1.0 秒** | 5 帧 × 0.15s = **0.75 秒** | -25% |
| **总 imagination 时长** | 12 × 1.0s = **12 秒** | 12 × 0.75s = **9 秒** | -25% |
| **典型任务完成时长** | **12-20 秒** (DROID 桌面任务) | **~2.4 秒** (LiftPeg 中位数) | **差 5-8 倍** |
| **imagination / 任务时长 比** | ≈ **1:1** | ≈ **3.75:1** | 严重过长 |

---

## 五、核心问题分析

### 问题 1: WM 帧率不匹配 (6.67 Hz vs 5 Hz)

Ctrl-World 预训练在 **5 Hz** DROID 视频上。WM 学到的 "两帧之间发生多大变化" 对应 **0.2 秒** 的时间间隔。

我们的 `frame_skip=3` 使数据为 **6.67 Hz** (帧间隔 0.15s)。WM 预期每帧间有 0.2s 的运动量，但实际只有 0.15s → **动作幅度被系统性低估 25%**。

虽然 WM 微调可能部分适应，但与 pretrained 权重的先验存在系统偏差。

### 问题 2 (最严重): Imagination 时长远超任务时长

```
LiftPegUpright 中位数完成时间: 2.4 秒 ≈ 16 帧 (6.67Hz)  ≈ 3.2 个 interaction
当前 Imagination 配置:         9.0 秒 = 60 帧 (6.67Hz)  = 12 个 interaction

→ 约 73% 的帧 (后 44 帧) 在描述任务已完成/偏离的无效状态
```

这些无效帧的影响:
1. **WM 在无意义地外推** → 图像持续模糊退化
2. **VLM 看到大量退化帧** → 整条轨迹的 p_yes 被拉低
3. **浪费计算资源** → 每条轨迹生成时间本可缩短 ~67%

对比 DROID: imagination 12-20s 覆盖 12-20s 任务 → **1:1 匹配**。

### 问题 3: D_syn+ 低通过率的重要解释

当前 D_syn+ 通过率: **7/200 = 3.5%** (α=0.4)

如果 VLM 评估的 16 帧中大部分是任务完成后 WM 继续外推的模糊帧，VLM 自然倾向判定失败。**这不是 WM 本身质量不够，而是我们让 WM 预测了远超必要的帧数**。

---

## 六、推荐修正方案

### 方案 A: 最小改动 — 缩短 `num_interact`

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `num_interact` | 12 | **4** (或 3-5) | 任务最长 2.5s ≈ 17 帧 ≈ 3.4 个 interaction; 加冗余 → 4 |

效果:
- 总 imagination: 4 × 5 × 0.15s = **3.0 秒, 20 帧** (覆盖最长 demo + 0.5s 冗余)
- 生成速度提升 **~3×** (WM forward 次数从 12 减至 4)
- VLM 看到的帧中大部分为任务关键阶段，模糊帧比例大幅下降

### 方案 B: 精确对齐帧率 — 改 `frame_skip=4`

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `frame_skip` | 3 | **4** | 20 Hz / 4 = **5 Hz**，精确匹配 DROID WM 预训练帧率 |
| `num_interact` | 12 | **4** (或 3-5) | 同方案 A |

效果:
- WM 帧率精确 5 Hz → 消除帧间隔偏差
- 总 imagination: 4 × 5 × 0.2s = **4.0 秒, 20 帧**
- **代价**: 需要重新采集/重新编码所有训练数据

### 方案 C: 同时对齐帧率 + 适配 num_frames

更激进的适配——根据 ManiSkill 任务特点重新设计 imagination 参数:

| 参数 | 值 | 理由 |
|------|-----|------|
| `frame_skip` | 4 | 对齐 5 Hz |
| `num_frames` | 3 | 任务更短, 减少每次 WM 预测的帧数 |
| `num_interact` | 5 | 5 × 3 × 0.2s = 3.0s, 覆盖任务 |
| `num_history` | 4 | 减少历史帧以匹配更短的轨迹 |

### ⚡ 推荐: 先执行方案 A (num_interact 缩短)

方案 A **无需重新采集数据, 只改 1 个参数**, 即可验证"缩短 imagination 是否提升 D_syn+ 产出率"。

如果 D_syn+ 显著提升 (如 3.5% → 15%+), 再考虑方案 B/C 做精确帧率对齐。

---

## 七、预期影响

| 指标 | 当前 (num_interact=12) | 预期 (num_interact=4) |
|------|----------------------|----------------------|
| Imagination 帧数/条 | 60 | 20 |
| 无效帧比例 | ~73% | ~15% |
| 生成时间/条 | ~3 min | ~1 min |
| VLM p_yes (预期) | mean=0.15-0.18 | **显著提升** (减少退化帧) |
| D_syn+ 通过率 (α=0.4) | 3.5% (7/200) | **预期 > 10%** |
| 200 条生成总时间 | ~10 h | ~3.3 h |

> **注**: 以上预期为定性估计, 需实验验证。推荐先用 `num_interact=4` 生成 50 条做快速验证。
