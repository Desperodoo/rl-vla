# VLAW 数据重采集计划 (v3 — frame_skip=4)

> **创建日期**: 2026-03-04 | **状态**: 待用户确认后执行
> **前置文档**: [`knowledge/ADR-026-data-quality-diagnosis.md`](knowledge/ADR-026-data-quality-diagnosis.md)
> **帧率分析**: [`docs/vlaw/frame_rate_timing_analysis.md`](../docs/vlaw/frame_rate_timing_analysis.md)

---

## 一、背景

前两轮数据采集均存在帧率问题：
- v2 IL 基线: frame_skip=3 → 6.67Hz (与 WM 5Hz 不匹配)
- v2 AWSC: frame_skip=5 → 4Hz (Data-Agent 擅自修改参数)

本计划以 **frame_skip=4 (5Hz)** 重新采集所有数据，精确对齐 Ctrl-World WM 预训练频率。

---

## 二、核心参数

| 参数 | 值 | 依据 |
|------|-----|------|
| **frame_skip** | **4** | 20Hz / 4 = 5Hz = DROID 15Hz/3 (WM 预训练帧率) |
| max_episode_steps | 200 | ManiSkill 默认值 (不人为截断) |
| num_envs | 64 | GPU 向量化并行 |
| camera_width/height | 128 | 与训练一致 |
| control_mode | pd_ee_delta_pose | 与 AWSC 训练一致 |
| obs_horizon | 2 | 与 AWSC 训练一致 |
| min_traj_length | 5 | 放宽 (frame_skip=4, 200步 → T=51, 远超阈值) |

### 预期轨迹长度

| max_episode_steps | frame_skip=4 下 T | 计算方式 |
|-------------------|-------------------|---------|
| 200 (默认) | **51** | 1(first) + 50(regular: 4,8,...,200) = 51 |
| 100 (AWSC 训练) | **26** | 1(first) + 25(regular: 4,8,...,100) = 26 |

---

## 三、策略选择

### 方案 A: AWSC 微调策略 (推荐)

```
checkpoint: runs/fair_comparison/fair_comparison/awsc/best_s42__1772570560/checkpoints/final.pt
预期 success_at_end: ~50% (fair_comparison 报告)
pred_horizon: 8, obs_horizon: 2, EMA: 有
max_episode_steps: 训练时用 100, 采集时用 200 (让策略跑满)
```

**关键加载要求**:
1. **必须使用 EMA 权重**: `velocity_net_ema.*` → 重映射为 `velocity_net.*`
2. **必须加载 visual_encoder**: 从 checkpoint 顶层 `"visual_encoder"` key
3. **pred_horizon=8**: 从 checkpoint config 自动读取
4. **act_steps = min(cfg.act_steps, pred_horizon) = min(8, 8) = 8**

### 方案 B: IL 基线策略 (对照)

```
checkpoint: checkpoints/il/best_eval_success_once.pt
预期 success_at_end: ~8-10% (基线水平)
pred_horizon: 16, obs_horizon: 2, EMA: 有
```

**决策**: 先用方案 A 采集。如果 success_at_end < 30%（说明 AWSC 加载有问题），回退到方案 B。

---

## 四、采集任务清单

### Step 1: 策略加载验证 (dry_run, 3 条轨迹)

**目的**: 确认 AWSC checkpoint 加载正确、EMA 权重生效、success_rate 合理。

```
配置:
  checkpoint_path: runs/fair_comparison/.../final.pt
  num_envs: 64
  num_episodes: 3 (dry_run=True)
  max_episode_steps: 200
  frame_skip: 4
  GPU: 4
```

**通过条件**:
- [ ] 日志显示 `"检测到 AWSC checkpoint (EMA+config)"` 或 `"Using velocity_net_ema weights"`
- [ ] 日志显示 `frame_skip=4`（不是 3 或 5）
- [ ] T ≈ 51 (失败轨迹) 或 T < 51 (成功轨迹)
- [ ] 至少 1/3 条 success_at_end（50% 基线策略在 3 条中应有 1-2 条成功）

### Step 2: 小批量采集 (50 条) + 质量快检

**目的**: 验证批量收集的数据质量。

```
配置:
  num_episodes: 50
  max_episode_steps: 200
  frame_skip: 4
  min_traj_length: 5
  output_dir: data/vlaw/rollouts/pilot/LiftPegUpright-v1
  GPU: 4
```

**通过条件** (自动化检查):
- [ ] T_max == 51 (失败轨迹恰好 51 帧)
- [ ] T_min ≥ 5 (无被过滤的短轨迹或极少 <3 条)
- [ ] success_at_end ≥ 30% (AWSC 策略应 ~50%)
- [ ] rgb_base vs rgb_render diff > 30 (双相机不重复)
- [ ] 0 条幽灵轨迹
- [ ] 0 条空轨迹
- [ ] action 范围在 [-1, 1]

### Step 3: 正式批量采集

通过 Step 2 后，三种数据一次性采集：

| 数据集 | num_episodes | output_dir | 说明 |
|--------|-------------|------------|------|
| mixed | 1200 | `data/vlaw/rollouts/mixed/LiftPegUpright-v1/` | 训练数据 (成功+失败混合) |
| eval | 20 | `data/vlaw/rollouts/eval/LiftPegUpright-v1/` | 评估数据 |
| high_suc | — | `data/vlaw/rollouts/high_suc/LiftPegUpright-v1/` | 从 mixed 中筛选 success_at_end=True |

### Step 4: 数据质量报告

生成完整质量报告，供用户审核确认：

```
报告内容:
  1. 采集参数回顾 (frame_skip, max_episode_steps, checkpoint, EMA, 实际log截取)
  2. 基础统计
     - mixed: N条, T分布 (min/max/mean/median/histogram), success_at_end 比率
     - eval: N条, T分布, success_at_end 比率
     - high_suc: N条, T分布
  3. 帧率验证
     - 确认 T_max == 51 (200步 / frame_skip=4)
     - 确认无异常 T 值 (所有失败轨迹 T==51, 成功轨迹 T < 51)
  4. 图像质量
     - rgb_base vs rgb_render mean diff (应 > 30)
     - 随机抽 5 条轨迹的首帧/末帧截图 (base64 或保存为 png)
  5. 动作分布
     - action 范围 (应在 [-1, 1])
     - action mean/std per dimension
  6. 与旧数据对比
     - IL 旧数据 (frame_skip=3): T_max=68, success_at_end=8.8%
     - AWSC 旧数据 (frame_skip=5): T_max=21, success_at_end=7.7%
     - 新数据预期: T_max=51, success_at_end≈50%
  7. Go/No-Go 建议

报告路径: results/vlaw/data_quality_report_v3.md
```

### Step 5: 用户确认

**用户确认后才能进入下一步。** 不可跳过此步骤。

### Step 6: VAE 编码

确认无误后：
1. 编码 mixed + high_suc → `data/vlaw/encoded/train/`
2. 编码 eval → `data/vlaw/encoded/eval/`
3. 重新生成 stat.json → `data/vlaw/meta_info/maniskill/stat.json`
4. 编码质量验证: latent shape (T,4,48,24), dtype fp16, top-bot diff > 0.5

---

## 五、数据质量自动化测试清单

> 以下检查项在 Step 2 和 Step 4 中必须全部通过。任一项失败则停止并排查。

### A. 轨迹长度检查

| 检查 | 预期 | 失败行动 |
|------|------|---------|
| T_max (failed traj) == 51 | frame_skip=4, max_episode_steps=200 | 检查实际 frame_skip 和 max_episode_steps |
| T_min ≥ 5 | min_traj_length=5 | 检查是否有 env 异常 |
| T_median (all) ≈ 30-45 | 成功和失败混合 | — |
| 丢弃率 < 5% | 少数极短轨迹 | 如 >5% 则检查策略行为 |

### B. 成功率检查

| 检查 | 预期 | 失败行动 |
|------|------|---------|
| AWSC: success_at_end ≥ 30% | fair_comparison ~50% | 检查 EMA 加载、visual encoder、obs 格式 |
| IL 基线: success_at_end ≥ 5% | 基线 ~8% | — |

### C. 加载验证检查

| 检查 | 预期 | 失败行动 |
|------|------|---------|
| 日志含 "velocity_net_ema" 或 "EMA" | EMA 权重被使用 | 检查 collector.py 检测逻辑 |
| 日志显示 `frame_skip=4` | 参数未被篡改 | 检查 CollectorConfig 实际值 |
| 日志显示 `pred_horizon=8` | AWSC config 读取正确 | 检查 checkpoint config |

### D. 图像质量检查

| 检查 | 预期 | 失败行动 |
|------|------|---------|
| rgb_base shape == (T, 128, 128, 3) | 正确分辨率 | — |
| rgb_base != rgb_render | diff > 30 | 检查是否重复 (BUG-020) |
| rgb_base 非全黑/全白 | pixel mean ∈ [20, 235] | 检查相机配置 |

### E. 动作检查

| 检查 | 预期 | 失败行动 |
|------|------|---------|
| action ∈ [-1, 1] | pd_ee_delta_pose 范围 | — |
| action std > 0.01 per dim | 非常量动作 | 检查策略是否退化 |

### F. HDF5 格式检查

| 检查 | 预期 | 失败行动 |
|------|------|---------|
| 每条轨迹含 keys: rgb_base, rgb_render, state, obs_agent, actions, env_success | 完整字段 | — |
| actions.shape == (T, 7) | 7-DoF 动作 | — |
| env_success.shape == (T,) | 逐步记录 | — |
| env_success dtype == bool | 布尔类型 | — |

---

## 六、风险与回退方案

| 风险 | 影响 | 应对 |
|------|------|------|
| AWSC success_at_end < 30% | 数据质量不足 | 回退到 IL 基线策略 (8% success)，或诊断 AWSC 加载问题 |
| max_episode_steps=200 导致 T=51 太长 | WM 训练效率低 | 考虑改用 max_episode_steps=100 (T=26) |
| frame_skip=4 数据与旧 VLM LoRA 不兼容 | VLM 需重训 | 预期行为，VLM LoRA 需要重新训练 |
| collector.py 又被 agent 修改参数 | 数据再次错误 | **Step 1 dry_run 必须验证日志中的实际参数** |

---

## 七、执行时间估算

| 步骤 | 预估时间 | GPU |
|------|---------|-----|
| Step 1: dry_run | < 1 分钟 | GPU 4 |
| Step 2: pilot 50 条 | ~2 分钟 | GPU 4 |
| Step 3: 正式采集 1200+20 条 | ~8 分钟 | GPU 4 |
| Step 4: 质量报告 | ~2 分钟 | CPU |
| Step 5: 用户确认 | 人工 | — |
| Step 6: VAE 编码 | ~15 分钟 | GPU 5 |
| **总计** | **~30 分钟 + 人工确认** | |

---

## 八、后续 Pipeline

数据确认无误后的完整 Pipeline:

```
Phase 0: 数据采集 (本计划) ← 当前
    ↓
Phase 1: WM 微调 (GPU 0-3, 2000步)  ─┐
Phase 2: VLM 微调 (GPU 6, 200步)     ─┤ 并行
    ↓                                 ─┘
Phase 3: Imagination + VLM 标注
    ↓
Phase 4: 策略更新 (Weighted FM)
    ↓
Phase 5: 评估
```

> **重要**: stat.json 必须在 Phase 0 数据确认后重新生成，因为 WM 训练依赖它做 action normalization。
