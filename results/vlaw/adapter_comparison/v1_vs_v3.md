# Dynamics Adapter V1 vs V3 对比数据

> 从 MEMORY.md 归档。V3 已弃用（delta_target euler wrapping 致命 bug）。

## 1. 测试条件

| 属性 | V1 | V3 |
|------|----|----|
| Params | 584K | ~584K |
| 架构 | 3-layer MLP (60→512→512→512→50) | 同 + sincos input + delta_target |
| 输出 | absolute EE (via sin/cos → atan2) | delta from current_ee (7D) |
| Checkpoint | `checkpoints/vlaw/dynamics_adapter/best.pt` | `checkpoints/vlaw/dynamics_adapter_v3/best.pt` |

## 2. EE 预测精度对比

| Test Set | V1 pos_mae | V3 pos_mae | V1 euler_mae | V3 euler_mae |
|----------|------------|------------|--------------|--------------|
| Clean data | 55.5mm | **39.2mm** (+29% ✅) | 0.180 rad | 0.222 rad (-24% ❌) |
| Mixed data | **20.0mm** | 41.5mm (-108% ❌) | **0.103 rad** | 0.294 rad (-185% ❌) |

> **注意**：上表中的 "Clean" 和 "Mixed" 含义与 Phase 0.2 的数据源不同。此处 "Clean" 指 adapter_clean (fs=3)，"Mixed" 指 mixed (fs=4)。

## 3. V3 致命 Bug：delta_target euler wrapping

当 `current_ee euler = 3.11 rad (≈π)`，`gt_ee euler = -3.09 rad (≈-π)` 时：
- `delta = -3.09 - 3.11 = -6.2 rad`（应为 ~0.05 rad）
- V3 被训练预测这些巨大的 euler 跳变 → 推理时严重偏差

## 4. 结论

- **V1 仍是最佳选择**，V3 架构改进方向正确但 delta_target 实现有缺陷
- V1 在分布内数据（mixed, fs=4）表现远优于分布外（adapter_*, fs=3）
- 这与 Phase 0.2 的 frame_skip 影响分析一致

## 5. Phase 0.2 新数据补充（2026-03-17）

| Source | frame_skip | pos_mae (mm) | euler_mae (rad) |
|--------|-----------|-------------|----------------|
| mixed (fs=4) | 4 | 26.61 | 0.0980 |
| adapter_clean (fs=3) | 3 | 50.63 | 0.1420 |
| adapter_teleop (fs=3) | 3 | 74.95 | 0.2082 |
| adapter_gaussian (fs=3) | 3 | 75.42 | 0.2325 |
| adapter_random (fs=3) | 3 | 105.37 | 0.5014 |

**关键发现**：
- mixed (fs=4) 上 pos_mae=26.61mm — 与 MEMORY 记录的 14.5mm 差异可能来源于评估范围（MEMORY 的 14.5mm 是训练集上的 eval loss，此处是 full dataset 上的推理）
- fs=3 数据上 pos_mae 是 fs=4 的 1.9-4x，确认 frame_skip 不匹配是重大性能损失源
- Per-step 误差单调递增：step1=18.5mm → step5=33.6mm（1.8x），说明存在误差累积但非爆炸性
