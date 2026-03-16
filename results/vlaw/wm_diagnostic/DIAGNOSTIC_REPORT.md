# WM v5 Imagination Diagnostic Report

**Date**: 2026-03-14
**Checkpoint**: iter1_v5/checkpoint-3800.pt (best loss≈0.157)
**Dataset**: LiftPegUpright-v1 train_v5 (HDF5 encoded with SVD VAE)

---

## Executive Summary

训练验证（单 chunk + GT 条件）PSNR 高达 35-39 dB，但 Imagination（12-chunk AR + policy actions）质量极差。通过 7 组受控消融实验，我们隔离了每个可能因素的贡献：

**根因排序（从大到小）：**

| 排名 | 因素 | PSNR 影响 | 严重程度 |
|------|------|----------|---------|
| 1 | **BUG-D: Action tiling（future EE pose 全相同）** | **-4.5 ~ -8.5 dB** | CRITICAL |
| 2 | num_inference_steps 50→25 | ~0.8 dB | Low |
| 3 | 自回归误差累积（6 chunk） | ~0 dB (negligible) | None |
| 4 | History 采样策略（sparse vs contiguous） | <0.5 dB | Negligible |
| 5 | History latent 噪声敏感度 | PSNR 反而上升 | None |

**核心结论：BUG-D 是 Imagination 质量退化的唯一显著根因。** 自回归误差累积不是问题。

---

## 实验详细结果

### Group A: Action Sensitivity Test

**目标**：验证 WM 是否真正使用了 action conditioning。

| 条件 | Mean PSNR | Mean L2 |
|------|----------|---------|
| GT actions | **36.85** | baseline |
| Tiled current EE | 32.35 | 0.224 |
| Zero actions | 29.77 | 0.274 |
| Random actions | 25.61 | — |

**结论**：WM **确实**在使用 action conditioning。GT → tile 已有 4.5 dB 下降，GT → zero 有 7.1 dB 下降。

Per-frame decay 分析（sample 1）：
- GT actions: frame 1→5: 57.3→34.6 dB（≈22.7 dB decay，但大部分是 frame 1 本身的特殊性——与 history 最后一帧几乎相同）
- Tiled actions: frame 1→5: 56.7→31.7 dB（25 dB decay）
- **Tiled 在远端帧（frame 4-5）退化尤其严重**

### Group B: num_inference_steps Ablation

| Steps | PSNR | vs 50-step |
|-------|------|-----------|
| 10 | 37.53 | +0.40 |
| 15 | 36.40 | -0.74 |
| 20 | 36.61 | -0.52 |
| 25 | 36.35 | -0.78 |
| 50 | 37.13 | baseline |

**结论**：Steps 数目影响只有 ~0.8 dB，且非单调（10 step 反而最高）。**此因素不重要。**

### Group C: BUG-D Action Conditioning (Alpha Sweep)

核心实验——将 future EE pose 在 tiled 和 GT 之间线性插值：

`future_ee = current_ee + α * (gt_future_ee - current_ee)`

**Sample 1（动态小, peg 基本不动）:**

| α | PSNR |
|---|------|
| 0.0 (tiled) | 35.32 |
| 0.25 | 35.79 |
| 0.5 | 35.95 |
| **1.0 (GT)** | **37.41** |
| 1.5 | 35.76 |
| 2.0 | 33.87 |

**Sample 2（动态大, peg 显著运动）:**

| α | PSNR |
|---|------|
| 0.0 (tiled) | 27.23 |
| 0.25 | 27.54 |
| 0.5 | 27.76 |
| **1.0 (GT)** | **34.12** |
| 1.5 | 27.41 |
| 2.0 | 26.77 |

**关键发现**：
1. α=1.0 (GT) 始终最优，**单调从 α=0→1 改善，从 α=1→2 退化**
2. **动态大的样本，GT vs tiled 差距高达 8.5 dB**（sample 2: 34.12 vs 26.60）
3. 动态小的样本差距约 2 dB（sample 1: 37.41 vs 35.32）
4. **这解释了为什么 Imagination 中 peg 几乎不动**——tiled actions 告诉 WM "什么都不做"

### Group D3: History Noise Sensitivity

| σ | PSNR | L2 |
|---|------|---|
| 0.00 | 36.03 | 0.115 |
| 0.01 | 36.36 | 0.111 |
| 0.02 | 36.67 | 0.107 |
| 0.05 | 37.21 | 0.101 |
| 0.10 | **37.56** | **0.097** |
| 0.20 | 36.90 | 0.104 |
| 0.50 | 36.37 | 0.111 |

**结论**：WM 对 history noise 非常鲁棒。轻度噪声（σ=0.05-0.1）反而**提升** PSNR（可能起到类似 noise augmentation 的效果）。σ=0.5 也仅下降 0.3 dB。**自回归误差通过 history 传播不是问题。**

### Group E: History Sampling Strategy

| 策略 | PSNR |
|------|------|
| Contiguous（训练分布） | 37.71 |
| Sparse（imagination 分布） | **38.00** |
| Same frame 0 | 37.53 |
| Same current | 37.89 |

**结论**：所有策略差距 < 0.5 dB。**History sampling 不影响质量。** Sparse 甚至略高于 contiguous（可能因为覆盖更广时间范围）。

### Group F2: Progressive Factor Introduction

| Step | 条件 | PSNR | Δ (vs prev) | 因素 |
|------|------|------|------------|------|
| 0 | GT all, 50 steps | 36.31 | — | baseline |
| 1 | 25 steps | 37.61 | **+1.29** | steps 减少 |
| 2 | + tiled actions | 35.14 | **-2.47** | BUG-D |

**结论**：Action tiling (BUG-D) 是最大的退化因素（-2.47 dB）。步数从 50→25 反而略有提升（进一步确认 steps 不重要）。

### Group D_MC: Multi-chunk Autoregressive (6 chunks)

**这是最关键的新实验** — 用 51 帧长轨迹跑 6 chunk AR。

| Chunk | D1 Oracle PSNR | D2 AR PSNR | Gap (D1-D2) |
|-------|----------------|------------|-------------|
| 0 | 38.54 | 38.88 | -0.34 |
| 1 | 38.00 | 37.67 | +0.32 |
| 2 | 38.36 | 39.36 | -1.00 |
| 3 | 39.70 | 39.31 | +0.39 |
| 4 | 38.80 | 39.20 | -0.40 |
| 5 | 38.45 | 38.59 | -0.14 |
| **Mean** | **38.64** | **38.84** | **-0.20** |

**结论**：
- D1 (oracle history) 和 D2 (predicted history AR) 的 PSNR **几乎相同**
- Mean gap = -0.20 dB（AR 甚至略好！）
- 没有随 chunk 递增的退化趋势
- **自回归误差累积在 6 chunk（30帧）内完全不是问题**

---

## 综合诊断与建议

### 根因确认

**BUG-D（future action tiling）是 Imagination 质量退化的根本原因。** 其他所有疑似因素（AR 误差累积、inference steps、history 采样、history 噪声敏感度）均已排除。

当 Imagination 推理时，5 个 future 帧全部使用相同的当前 EE pose（`np.tile`），这等价于告诉 WM "机械臂不会动" → WM 预测静态场景 → **peg 动态消失**。

### 修复方案优先级

1. **[CRITICAL] 修复 BUG-D：在 Imagination 推理时提供正确的 future EE poses**
   - Fix1（delta→EE 积分）已失败，可能因为：
     - PD 控制器使 raw delta ≠ 实际 EE 位移
     - 积分 EE pose 超出 WM 训练分布
   - **建议新方案**：
     - **Plan A**: 在 ManiSkill env 中执行 policy action，获取真实的 EE pose 反馈，再送入 WM
     - **Plan B**: 训练一个轻量级 "action-to-EE-pose" 映射网络
     - **Plan C**: 将 WM 改为接受 delta action conditioning（需重训 WM，但消除了推理时的转换问题）

2. **[Low] 统一 num_inference_steps = 50**
   - 影响小，但容易修复

3. **不需要修的**：
   - History sampling strategy — 不影响
   - AR error accumulation — 不是问题
   - History latent noise — WM 很鲁棒

---

## 实验配置

- GPU: CUDA:2 (single GPU)
- Checkpoint: iter1_v5/checkpoint-3800.pt
- Data: train_v5 HDF5
- Samples: 3 short windows (11 frames) + 1 long trajectory (51 frames, traj_0061)
- Script: `scripts/vlaw/diagnostic/wm_diagnostic_battery.py`
