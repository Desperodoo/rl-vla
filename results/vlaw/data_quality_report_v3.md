# VLAW v3 数据质量报告

> **生成时间**: 2026-03-05 01:03:55
> **计划文档**: `.github/VLAW_DATA_COLLECTION_PLAN_V3.md`

## 1. 采集参数回顾

| 参数 | 值 |
|------|-----|
| checkpoint | `runs/fair_comparison/fair_comparison/awsc/best_s42__1772570560/checkpoints/final.pt` |
| frame_skip | **4** (BUG-023: 严禁修改) |
| max_episode_steps | 200 |
| num_envs | 64 (mixed) / 20 (eval) |
| camera | 128×128 |
| control_mode | pd_ee_delta_pose |
| obs_horizon | 2 |
| pred_horizon | 8 (AWSC config) |
| EMA权重 | ✅ 已加载 (velocity_net_ema) |
| min_traj_length | 5 |

**日志确认**:
```
[VLAW-P1.1] 检测到 AWSC checkpoint (EMA+config), pred_horizon=8
Using velocity_net_ema weights (154 tensors)
frame_skip=4
```

## 2. 基础统计

### Mixed

- **轨迹数**: 1200
- **success_at_end**: 552/1200 = **46.0%**
- **文件大小**: 702.1 MB
- **HDF5**: `/home/wjz/rl-vla/data/vlaw/rollouts/mixed/LiftPegUpright-v1/LiftPegUpright-v1_real_1772643507.h5`

| 统计量 | 全部 | 成功轨迹 | 失败轨迹 |
|--------|------|---------|---------|
| T_min | 5 | 5 | 50 |
| T_max | 51 | 49 | 51 |
| T_mean | 34.3 | 15.3 | 50.6 |
| T_median | 50.0 | 11.0 | 51.0 |

**T 分布直方图**:

| T 范围 | 计数 |
|--------|------|
| 0-5 | 0  |
| 5-10 | 240 ████████████████████████████████████████ |
| 10-15 | 96 ███████████████████ |
| 15-20 | 59 ███████████ |
| 20-25 | 49 █████████ |
| 25-30 | 42 ████████ |
| 30-35 | 24 ████ |
| 35-40 | 15 ███ |
| 40-45 | 11 ██ |
| 45-50 | 16 ███ |
| 50-55 | 648 ████████████████████████████████████████ |

### Eval

- **轨迹数**: 20
- **success_at_end**: 17/20 = **85.0%**
- **文件大小**: 9.0 MB
- **HDF5**: `/home/wjz/rl-vla/data/vlaw/rollouts/eval/LiftPegUpright-v1/LiftPegUpright-v1_real_1772643569.h5`

| 统计量 | 全部 | 成功轨迹 | 失败轨迹 |
|--------|------|---------|---------|
| T_min | 5 | 5 | 51 |
| T_max | 51 | 42 | 51 |
| T_mean | 21.8 | 16.6 | 51.0 |
| T_median | 15.5 | 13.0 | 51.0 |

**T 分布直方图**:

| T 范围 | 计数 |
|--------|------|
| 0-5 | 0  |
| 5-10 | 4  |
| 10-15 | 6 █ |
| 15-20 | 3  |
| 20-25 | 1  |
| 25-30 | 0  |
| 30-35 | 0  |
| 35-40 | 2  |
| 40-45 | 1  |
| 45-50 | 0  |
| 50-55 | 3  |

### High_suc

- **轨迹数**: 552
- **success_at_end**: 552/552 = **100.0%**
- **文件大小**: 203.7 MB
- **HDF5**: `/home/wjz/rl-vla/data/vlaw/rollouts/high_suc/LiftPegUpright-v1/LiftPegUpright-v1_high_suc_real_1772643507.h5`

| 统计量 | 全部 | 成功轨迹 | 失败轨迹 |
|--------|------|---------|---------|
| T_min | 5 | 5 | N/A |
| T_max | 49 | 49 | N/A |
| T_mean | 15.3 | 15.3 | N/A |
| T_median | 11.0 | 11.0 | N/A |

**T 分布直方图**:

| T 范围 | 计数 |
|--------|------|
| 0-5 | 0  |
| 5-10 | 240 ████████████████████████████████████████ |
| 10-15 | 96 ███████████████████ |
| 15-20 | 59 ███████████ |
| 20-25 | 49 █████████ |
| 25-30 | 42 ████████ |
| 30-35 | 24 ████ |
| 35-40 | 15 ███ |
| 40-45 | 11 ██ |
| 45-50 | 16 ███ |
| 50-55 | 0  |

## 3. 帧率验证

- **frame_skip=4**: 20Hz / 4 = **5Hz** — 精确匹配 Ctrl-World WM 预训练频率 ✅
- **Mixed T_max = 51**: 预期 51 (200/4+1), ✅ 正确
- **失败轨迹 T 范围**: 50-51
- **成功轨迹 T 范围**: 5-49

## 4. 图像质量

- **rgb_base vs rgb_render mean diff**: 56.5 ✅ > 30
- **diff range**: [55.5, 57.8]
- **RGB shape**: ['(128, 128, 3)']

## 5. 动作分布

- **action_dim**: 7
- **范围**: [-1.0000, 1.0000] ✅ 在 [-1, 1]

**Per-dimension 统计** (mixed):

| Dim | Mean | Std |
|-----|------|-----|
| 0 | -0.0258 | 0.2253 ✅ |
| 1 | 0.0125 | 0.2477 ✅ |
| 2 | -0.0504 | 0.3878 ✅ |
| 3 | 0.0350 | 0.3356 ✅ |
| 4 | -0.2352 | 0.4645 ✅ |
| 5 | -0.0628 | 0.3360 ✅ |
| 6 | -0.3438 | 0.4988 ✅ |

## 6. 与旧数据对比

| 指标 | IL 旧 (frame_skip=3) | AWSC 旧 (frame_skip=5) | **v3 新** (frame_skip=4) |
|------|---------------------|----------------------|----------------------|
| 帧率 | 6.67Hz | 4Hz | **5Hz** ✅ |
| T_max | 68 | 21 | **51** |
| success_at_end | 8.8% (IL) | 7.7% (AWSC bug) | **46.0%** |
| 轨迹数 | 1200 | 1200 | **1200** |
| EMA 权重 | ❌ 未使用 | ⚠️ 不确定 | **✅ 已验证** |
| frame_skip 验证 | ❌ 未验证 | ❌ 被篡改 | **✅ 日志确认=4** |

## 7. BUG-024 Selection Bias 分析

**问题**: ManiSkill3 中成功 episode terminated=True (提前结束)，
小样本采集时成功 episode 更容易被收集，导致 selection bias。

- **Mixed (1200条) success_at_end**: 46.0%
- **Eval (20条) success_at_end**: 85.0%
- **fair_comparison eval 真实值**: ~46%

**结论**: Mixed 1200条的 46.0% ✅ 与真实值匹配 (36-60% 范围内), 大量采集成功消除 selection bias

**Eval 注意**: 20条样本量太小, success_at_end=85.0% (因 selection bias 偏高，符合 BUG-024 预期)

## 8. 质量检查清单

- [✅] frame_skip=4 (日志确认) — frame_skip=4
- [✅] EMA 权重使用 (日志确认) — velocity_net_ema weights
- [✅] T_max=51 (失败轨迹) — T_max=51
- [✅] T_min≥5 — T_min=5
- [✅] 0 幽灵轨迹 — 日志确认: discarded_ghost=0
- [✅] 0 空轨迹 — 日志确认: discarded_empty=0
- [✅] actions ∈ [-1, 1] — [-1.0000, 1.0000]
- [✅] action std > 0.01 (all dims) — per-dim check
- [✅] rgb_base ≠ rgb_render (diff > 30) — diff=56.5
- [✅] success_at_end 在 36-60% (无 selection bias) — 46.0%
- [✅] action_dim=7 — dim=7

## 9. Go/No-Go 建议

### ✅ **GO** — 数据质量通过所有检查，可进入下一步

建议下一步:
1. 用户确认本报告
2. 执行 VAE 编码 (mixed + high_suc → train, eval → eval)
3. 重新生成 stat.json (action normalization)
4. 开始 Phase 1 WM 微调

## 10. 数据资产汇总

| 数据集 | 轨迹数 | success_at_end | T 范围 | 文件大小 | 路径 |
|--------|--------|---------------|--------|---------|------|
| mixed | 1200 | 46.0% | 5-51 | 702.1 MB | `data/vlaw/rollouts/mixed/LiftPegUpright-v1/LiftPegUpright-v1_real_1772643507.h5` |
| eval | 20 | 85.0% | 5-51 | 9.0 MB | `data/vlaw/rollouts/eval/LiftPegUpright-v1/LiftPegUpright-v1_real_1772643569.h5` |
| high_suc | 552 | 100.0% | 5-49 | 203.7 MB | `data/vlaw/rollouts/high_suc/LiftPegUpright-v1/LiftPegUpright-v1_high_suc_real_1772643507.h5` |
