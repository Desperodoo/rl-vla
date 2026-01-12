# RTC 测试结果总结

## 测试环境

- **日期**: 2026-01-08
- **机械臂**: ARX5 (X5 @ can0)
- **Demo 数据**: `~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5`
- **轨迹长度**: 461 帧，回放 300 帧
- **速度缩放**: 30%
- **模拟推理延迟**: 80 ± 20 ms

## 测试配置

| 参数 | 值 | 说明 |
|------|-----|------|
| H (prediction_horizon) | 16 | 策略预测的动作序列长度 |
| s (execute_horizon) | 8 | RTC 模式每 chunk 执行的动作数 |
| action_horizon | 8 | 传统模式每 chunk 执行的动作数 |
| action_dt | 33.3ms | 动作时间间隔 (30Hz / 0.3 速度) |
| soft_mask_schedule | exp | 指数衰减掩码 |

## 测试结果对比

### 核心指标

| 指标 | Traditional | RTC | 改进 | 改进率 |
|------|-------------|-----|------|--------|
| **关节误差均值 (rad)** | 0.0361 | 0.0185 | +0.0176 | **+48.8%** ✓ |
| **关节误差最大 (rad)** | 0.0913 | 0.0252 | +0.0661 | **+72.4%** ✓ |
| 关节误差标准差 | 0.0205 | 0.0044 | +0.0161 | **+78.5%** ✓ |
| 夹爪误差均值 | 0.0018 | 0.0014 | +0.0004 | +22.2% |

### 时序特性

| 指标 | Traditional | RTC | 说明 |
|------|-------------|-----|------|
| time_diff 范围 | -147ms ~ -211ms | +41ms ~ +46ms | 负值=重叠，正值=间隙 |
| time_diff 均值 | ~-180ms | ~+43ms | RTC 消除了重叠 |
| 总时长 | 38.19s | 38.78s | 基本一致 |

### 动作连续性

| 指标 | Traditional | RTC | 变化 |
|------|-------------|-----|------|
| 动作不连续均值 | 0.0111 | 0.0219 | -0.0108 ⚠ |
| 动作不连续最大 | 0.0408 | 0.0765 | -0.0357 ⚠ |

### RTC 特有指标

| 指标 | 值 |
|------|-----|
| d 值均值 | 2.00 |
| d 值分布 | 全部为 2 (非常稳定) |
| 拼接次数 | 37 |

## 关键发现

### ✅ RTC 的优势

1. **显著降低跟踪误差**: 
   - 关节误差均值降低 **48.8%** (0.0361 → 0.0185 rad)
   - 关节误差最大值降低 **72.4%** (0.0913 → 0.0252 rad)
   - 误差标准差降低 **78.5%** (更稳定的跟踪)

2. **消除 chunk 重叠**:
   - Traditional 模式: time_diff ≈ -180ms (严重重叠)
   - RTC 模式: time_diff ≈ +43ms (轻微间隙)
   - 这意味着 RTC 成功避免了新旧 chunk 的冲突覆盖

3. **稳定的 d 值自适应**:
   - 推理延迟稳定 (80±20ms)，d 值保持在 2
   - 说明自适应估计工作正常

### ⚠ 需要关注的问题

1. **动作不连续性增加**:
   - 从 0.0111 增加到 0.0219 (约 +97%)
   - 可能原因: soft masking 的混合边界导致轻微跳变
   - 建议: 调整 `soft_mask_decay_rate` 或尝试 `linear` 掩码

## 结果解读

### time_diff 的含义

```
time_diff = 新chunk调度开始时间 - 旧chunk调度结束时间

Traditional: time_diff = -180ms
  |------ old chunk (778ms) ------|
                    |------ new chunk (778ms) ------|
                    ^               ^
                 start_new      end_old
                 (重叠 180ms)

RTC: time_diff = +43ms
  |------ old chunk (778ms) ------|
                                      |------ new chunk (778ms) ------|
                                  ^   ^
                              end_old start_new
                              (间隙 43ms)
```

### 为什么 RTC 跟踪误差更小？

1. **避免动作冲突**: Traditional 模式中，新 chunk 会覆盖旧 chunk 尚未执行完的部分，导致机械臂"跳跃"
2. **平滑过渡**: RTC 的 soft masking 在重叠区域进行混合，而不是硬切换
3. **时序对齐**: d 步冻结确保已执行的动作不会被新 chunk 覆盖

## 测试文件

- **模拟测试**: `python -m consistency_policy.tests.test_rtc_controller`
- **真机测试**: `python -m consistency_policy.tests.test_rtc_replay`
- **结果数据**: `./rtc_comparison_20260108_004157.npz`

## 后续优化建议

1. **减小动作不连续性**:
   ```bash
   # 尝试线性掩码
   python -m consistency_policy.tests.test_rtc_replay \
       --demo /path/to/demo.h5 \
       --soft-mask-schedule linear
   
   # 或调整衰减速率 (更平滑的过渡)
   # 在代码中修改 rtc_soft_mask_decay_rate = 1.5
   ```

2. **测试不同 execute_horizon**:
   ```bash
   # 尝试更小的 s 值
   python -m consistency_policy.tests.test_rtc_replay \
       --demo /path/to/demo.h5 \
       --execute-horizon 6
   ```

3. **实际策略推理测试**:
   - 当前测试使用模拟推理延迟 (sleep)
   - 下一步应在 `eval_real_mp.py` 中启用 RTC 进行完整评估

## 结论

RTC 实现 **显著改善了轨迹跟踪精度**，关节误差降低约 **50%**。主要通过消除 chunk 重叠冲突实现。虽然动作不连续性有所增加，但整体跟踪性能的提升是明显的。

建议在实际策略推理场景中进一步验证 RTC 的效果。
