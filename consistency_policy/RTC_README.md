# Real-Time Chunking (RTC) 实现

基于 Physical Intelligence 的论文 [Real-Time Chunking for Policies](https://arxiv.org/abs/2506.07339) 实现。

## 概述

RTC 解决了异步 Action Chunking 的时序问题：当策略推理时间与动作执行时间不匹配时，会导致动作 chunk 之间的重叠或间隙。

### 核心概念

- **H (prediction_horizon)**: 策略每次预测的动作序列长度 (例如 16)
- **s (execute_horizon)**: 每个 chunk 实际执行的动作数 (例如 8)
- **d (inference_delay)**: 推理期间执行的动作数，$d = \lceil \text{inference\_time} / \Delta t \rceil$
- **Δt (action_dt)**: 动作时间间隔 (例如 33.3ms @ 30Hz)

### 算法核心

1. **异步推理**: 在执行当前 chunk 的第 d 个动作时开始下一次推理
2. **Soft Masking**: 使用软掩码权重混合旧/新 chunk 的重叠部分
3. **Chunk 拼接**: 实际执行 `old_chunk[:d] + blended_chunk[d:s]`

```
时间线:
          |<--- d --->|<---- s-d ---->|
          |冻结区域     |混合区域        |
          v           v               v
old_chunk: [0, 1, ..., d-1, d, d+1, ..., H-1]
                           ↓ (soft mask 混合)
new_chunk: [0, 1, ..., d-1, d, d+1, ..., H-1]
```

## 文件结构

```
consistency_policy/
├── rtc_manager.py      # RTC 核心实现
├── eval_real_mp.py     # 真机评估 (已集成 RTC)
├── test_rtc.py         # RTC 测试脚本
└── RTC_README.md       # 本文档
```

## 使用方法

### 1. 启用 RTC 模式 (默认)

```bash
python -m consistency_policy.eval_real_mp \
    --output ./eval_output \
    -v
```

### 2. 禁用 RTC (使用传统模式)

```bash
python -m consistency_policy.eval_real_mp \
    --output ./eval_output \
    --no-rtc \
    -v
```

### 3. 调整 RTC 参数

```bash
python -m consistency_policy.eval_real_mp \
    --output ./eval_output \
    --rtc-execute-horizon 6 \       # 减少每 chunk 执行的动作数
    --rtc-min-d 2 \                 # 最小推理延迟
    --rtc-max-d 4 \                 # 最大推理延迟
    --rtc-mask-schedule exp \       # 使用指数衰减掩码
    -v
```

## 配置参数

在 `eval_real_mp.py` 的 `DEFAULT_CONFIG` 中:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_rtc` | `True` | 是否启用 RTC |
| `rtc_execute_horizon` | `8` | 每个 chunk 执行的动作数 (s) |
| `rtc_inference_delay_percentile` | `95` | 推理延迟估计的百分位数 |
| `rtc_inference_delay_margin` | `0.01` | 推理延迟余量 (秒) |
| `rtc_min_inference_delay_steps` | `2` | 最小 d 值 |
| `rtc_max_inference_delay_steps` | `5` | 最大 d 值 |
| `rtc_soft_mask_schedule` | `"exp"` | 软掩码类型: `"exp"` 或 `"linear"` |
| `rtc_soft_mask_decay_rate` | `2.0` | 指数衰减速率 |
| `rtc_enable_soft_masking` | `True` | 是否启用软掩码 |

## Soft Masking 权重计算

对于 $i \in [0, H)$:

$$
W[i] = \begin{cases}
1 & \text{if } i < d \\
\exp(-\alpha \cdot \frac{i - d}{H - s - d}) & \text{if } d \leq i < H - s \\
0 & \text{if } i \geq H - s
\end{cases}
$$

其中 $\alpha$ 是衰减速率 (默认 2.0)。

混合公式:
$$
\text{blended}[i] = W[i] \cdot \text{old\_chunk}[i] + (1 - W[i]) \cdot \text{new\_chunk}[i]
$$

## 测试结果

使用默认参数进行模拟测试:

| 指标 | Traditional | RTC |
|------|-------------|-----|
| 平均重叠时间 | 164ms | 132ms |
| 有效执行率 | 38.4% | 50.5% |
| 重叠减少 | - | 32ms |

运行测试:
```bash
python -m consistency_policy.test_rtc
```

## 注意事项

### 1. 参数调优建议

- **如果推理时间稳定** (例如 80±10ms): 可以减小 `rtc_max_inference_delay_steps`
- **如果推理时间波动大**: 增大 `rtc_inference_delay_margin`
- **如果动作不连续**: 调整 `rtc_soft_mask_decay_rate` (更大=更激进过渡)

### 2. 与 Consistency Policy 的兼容性

Consistency Policy 使用单步或少量步骤生成动作，不像 Diffusion Policy 需要多步迭代。
RTC 原论文的 "guided inpainting" 需要迭代去噪过程，但我们这里使用的是：
- **Post-processing 方案**: 在策略输出后进行 soft masking 混合
- 这种方案不需要修改策略网络，但效果略逊于 training-time RTC

### 3. 频率匹配

为获得最佳效果，建议:
- `eval_frequency × (execute_horizon × action_dt) ≈ 1`
- 例如: 10Hz × (8 × 33.3ms) = 10 × 267ms ≈ 2.67 (有重叠)
- 可以考虑减小 `execute_horizon` 到 4: 10Hz × (4 × 33.3ms) ≈ 1.33

## 参考

- [Real-Time Chunking (arXiv:2506.07339)](https://arxiv.org/abs/2506.07339)
- [Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
- [Training-Time RTC (arXiv:2512.05964)](https://arxiv.org/abs/2512.05964)
