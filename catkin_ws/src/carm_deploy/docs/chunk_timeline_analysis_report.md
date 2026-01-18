# Action Chunk 时间线分析报告

**生成时间**: 2026-01-18  
**数据来源**: inference + record timeline logs  
**分析工具**: `analyze_timeline.py v2`

---

## 1. 系统配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `pred_horizon` | 16 | 模型预测的动作序列长度 |
| `act_horizon` | 16 | 实际执行的动作步数（当前与 pred_horizon 相同） |
| `obs_horizon` | 2 | 历史观测帧数 |
| `desire_inference_freq` | 30 Hz | 期望推理频率 |
| `temporal_factor_k` | 0.05 | 时间加权融合因子 |
| 动作执行频率 | 30 Hz | 固定 action_interval = 33.3ms |

---

## 2. ActionChunkManager 架构与原理

### 2.1 双线程架构

当前系统采用 **推理-控制分离** 的双线程架构：

```
┌─────────────────────────────────────────────────────────────────┐
│                        主进程                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────┐         ┌──────────────────┐            │
│   │   推理线程        │         │   控制线程        │            │
│   │  (~18 Hz 实际)    │         │  (~200 Hz)       │            │
│   ├──────────────────┤         ├──────────────────┤            │
│   │ 1. 获取观测       │         │ 1. 查询融合动作   │            │
│   │ 2. 模型推理       │  共享    │ 2. 插值到当前时刻 │            │
│   │ 3. 生成 chunk    │ ──────► │ 3. 发送控制命令   │            │
│   │ 4. 添加到 Manager │         │                  │            │
│   └──────────────────┘         └──────────────────┘            │
│              │                          │                       │
│              ▼                          ▼                       │
│   ┌──────────────────────────────────────────────────┐         │
│   │           ActionChunkManager (共享)               │         │
│   │  ┌─────────┬─────────┬─────────┬─────────┐      │         │
│   │  │ Chunk 1 │ Chunk 2 │ Chunk 3 │  ...    │      │         │
│   │  │ (TF)    │ (TF)    │ (TF)    │         │      │         │
│   │  └─────────┴─────────┴─────────┴─────────┘      │         │
│   │            受 lock_tfs 保护                       │         │
│   └──────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

**关键特点**：
- **推理线程**：以 ~18Hz 的频率生成新的 action chunk，每个 chunk 包含 16 步动作预测
- **控制线程**：以 ~200Hz 的高频查询 ActionChunkManager，获取当前时刻的融合动作
- **线程安全**：通过 `lock_tfs` 互斥锁保护共享的 ActionChunkManager

### 2.2 Chunk 创建与存储

每当推理线程完成一次模型推理，会创建一个新的 `VecTF`（轨迹片段）并添加到 ActionChunkManager：

```python
# inference_ros.py 中的 chunk 创建逻辑
all_actions = policy_output["actions"]  # shape: (pred_horizon, action_dim)

# 创建轨迹片段，包含所有 pred_horizon 步
vec_tf = VecTF(time.time())
for i, action in enumerate(all_actions):
    vec_tf.extend(action, self.action_interval)

# 添加到 Manager，返回唯一的 chunk_id
chunk_id = self.action_chunk_manager.add_trajectory(vec_tf)
```

**重要发现**：当前实现中 `act_horizon` 参数 **并未被用于截断**！所有 `pred_horizon=16` 步都被添加到每个 chunk 中。

### 2.3 时间加权融合机制

控制线程调用 `get_fused_action()` 时，ActionChunkManager 会：

1. **收集有效 chunk**：找出所有在当前时刻 `t` 有效的轨迹片段
2. **计算时间权重**：使用指数衰减公式计算每个 chunk 的权重
3. **加权融合**：按权重加权平均得到最终动作

**时间权重公式**：

$$
w_i = e^{-k \cdot \Delta t_i}
$$

其中：
- $\Delta t_i = t - t_{start,i}$ 是当前时刻距离 chunk $i$ 开始时刻的时间差
- $k = 0.05$ 是时间衰减因子（`temporal_factor_k`）
- 权重归一化：$\hat{w}_i = w_i / \sum_j w_j$

**融合示意**：

```
时间轴 ────────────────────────────────────────►
      
Chunk 1: ████████████████  (较旧，权重低)
Chunk 2:   ████████████████  
Chunk 3:     ████████████████
...
Chunk 9:               ████████████████  (较新，权重高)

当前时刻 t: ──────────────────┼

融合结果 = Σ(action_i × weight_i)
```

### 2.4 与标准 Action Chunking 的区别

| 特性 | 标准 Action Chunking | 当前实现 |
|------|---------------------|----------|
| **截断机制** | 执行 act_horizon 步后丢弃 | 无截断，所有 pred_horizon 步都保留 |
| **chunk 切换** | 旧 chunk 被新 chunk 替换 | 多 chunk 同时有效（~9个） |
| **动作选择** | 直接使用最新 chunk 的动作 | 时间加权融合多个 chunk |
| **平滑性** | 可能有跳变 | 天然平滑过渡 |
| **响应性** | 快（直接使用新预测） | 较慢（融合稀释新预测） |

**标准 Action Chunking 流程**：
```
pred_horizon = 16, act_horizon = 8

Chunk 1: [a1, a2, a3, a4, a5, a6, a7, a8 | a9...a16 被丢弃]
                                          ↓ 执行完 8 步后
Chunk 2:                                 [b1, b2, b3, b4, b5, b6, b7, b8 | ...]
```

**当前实现流程**：
```
pred_horizon = 16, act_horizon = 16 (未截断)

Chunk 1: ████████████████████████████████  (16步，一直有效直到超时)
Chunk 2:     ████████████████████████████████
Chunk 3:         ████████████████████████████████
...              (多 chunk 同时有效，融合输出)
```

### 2.5 lock_tfs 的作用

`lock_tfs` 是一个 `threading.Lock()`，用于保护 ActionChunkManager 的线程安全访问：

```python
# 推理线程添加 chunk
with self.lock_tfs:
    chunk_id = self.action_chunk_manager.add_trajectory(vec_tf)

# 控制线程查询融合动作
with self.lock_tfs:
    action, valid = self.action_chunk_manager.get_fused_action(t)
```

---

## 3. 基础延迟统计

### 3.1 观测延迟 (`delta_obs`)

从相机 ROS 时间戳到观测数据就绪的延迟：

| 统计量 | 值 |
|--------|-----|
| Mean | **44.4 ms** |
| Std | 8.7 ms |
| P50 | 44.3 ms |
| P95 | 59.3 ms |
| Max | 101.3 ms |

**分析**: 相机采集+传输延迟稳定在 ~45ms，P95 约 60ms，属于正常范围。

### 3.2 推理时间 (`inference_time`)

模型推理耗时：

| 统计量 | 值 |
|--------|-----|
| Mean | **34.7 ms** |
| Std | 33.8 ms |
| P50 | 32.7 ms |
| P95 | 41.3 ms |
| Max | **839.6 ms** |

**分析**: 
- 中位数 ~33ms，满足 30Hz 推理需求
- **存在长尾问题**：Max 达 839ms，可能是 GPU 调度或首次推理的 warm-up 造成
- P99 仅 45ms，说明长尾事件较少

### 3.3 端到端反应延迟 (`delta_chunk_obs`)

从观测时间戳到 chunk 生成的总延迟（= 观测延迟 + 推理时间）：

| 统计量 | 值 |
|--------|-----|
| Mean | **90.0 ms** |
| Std | 43.0 ms |
| P50 | 89.2 ms |
| P95 | 106.3 ms |
| Max | 955.1 ms |

**分析**: 系统端到端反应延迟约 **90ms（~3帧@30Hz）**，这意味着模型看到的观测比实际晚 3 帧。

### 3.4 控制下发延迟 (`control_lag`)

从查询融合动作到发送控制命令的延迟：

| 统计量 | 值 |
|--------|-----|
| Mean | **0.32 ms** |
| Max | 2.6 ms |

**分析**: 控制下发延迟可忽略不计。

---

## 4. Chunk 重叠分析

### 4.1 相邻 Chunk 间隔 (`inter_chunk_gap`)

| 统计量 | 值 |
|--------|-----|
| Mean | **54.2 ms** |
| Std | 20.8 ms |
| P50 | 51.8 ms |
| P95 | 67.2 ms |
| Max | 290.6 ms |

**分析**: 
- 实际推理频率 ≈ **18.5 Hz**（1000/54.2），低于期望的 30Hz
- 这是因为推理耗时 (~35ms) + 观测准备 (~45ms) 导致的

### 4.2 重叠率 (`overlap_ratio`)

相邻 chunk 的时间重叠比例：

| 统计量 | 值 |
|--------|-----|
| Mean | **89.2%** |
| Std | 4.2% |
| P50 | 89.6% |
| P95 | 91.4% |
| Min | 41.9% |

**分析**: 
- 相邻 chunk 平均有 **89% 的时间重叠**
- 这意味着每个新 chunk 仅带来约 11% 的新信息
- 高重叠率有助于平滑动作过渡，但也意味着较高的计算冗余

### 4.3 act_horizon 内的 Chunk 切换 (`act_horizon_switches`)

在一个 chunk 的 act_horizon 时间范围内（16 × 33.3ms = 533ms），产生的新 chunk 数量：

| 统计量 | 值 |
|--------|-----|
| Mean | **9.3** |
| Max | 11 |
| P50 | 10 |

**分析**: 
- 在 act_horizon 期间，平均有 **9-10 个新 chunk** 产生
- 这意味着 act_horizon 内的动作会被后续 chunk 的预测所"覆盖"
- **关键洞察**：当前 `act_horizon=16` 与 `pred_horizon=16` 相同，实际上只有前 ~2 步的动作会被独占执行，其余都会被新 chunk 融合覆盖

### 4.4 同时活跃的 Chunk 数量 (`active_chunk_count`)

| 统计量 | 值 |
|--------|-----|
| Mean | **9.1** |
| Max | 11 |
| P50 | 9 |

**分析**: 控制线程查询时，平均有 **9 个 chunk 同时有效**，参与时间加权融合。

---

## 5. 时间线可视化解读

### 5.1 Gantt 时间线图

![Timeline](timeline.png)

- **深色区域**: act_horizon 范围（当前与 pred_horizon 相同，均为 16 步 = 0.53s）
- **浅色区域**: pred_horizon 剩余部分（当前为 0）
- **Y 轴**: Chunk ID（递增）
- **X 轴**: 时间（秒）

**观察**:
1. 每个 chunk 覆盖约 **0.5 秒**的时间范围
2. 相邻 chunk 高度重叠，形成"瓦片式"覆盖
3. 稳态时同时有 9-10 个 chunk 活跃

### 5.2 活跃 Chunk 数量曲线

- 启动阶段：从 1 个逐渐增加到 9-10 个
- 稳态阶段：稳定在 9-10 个
- 结束阶段：逐渐减少

### 5.3 控制步使用的 Chunk 数量

- 启动阶段有少量点位较低（chunk 积累中）
- 稳态时稳定在 9-10 个

---

## 6. 关键指标汇总

| 指标 | 值 | 含义 |
|------|-----|------|
| 端到端延迟 | **90 ms** | 从观测到动作开始执行的延迟 |
| 实际推理频率 | **18.5 Hz** | 低于期望的 30Hz |
| Chunk 重叠率 | **89%** | 相邻 chunk 时间重叠比例 |
| 活跃 Chunk 数 | **9.1** | 参与融合的 chunk 数量 |
| act_horizon 切换数 | **9.3** | act_horizon 内产生的新 chunk 数 |

---

## 7. 对 Training-time RTC 的启示

### 7.1 训练数据采集时的时间对齐

采集端统计 (`delta_action_obs`)：
- Mean: **41.8 ms**
- P95: **46.5 ms**

这是手柄遥操作时，动作命令相对于观测的延迟。与推理端的 `delta_chunk_obs` (90ms) 相比：
- **推理延迟 > 采集延迟**：模型在推理时"看到"的观测比采集时更旧
- 这种 **train-infer gap** 可能影响 RTC 训练效果

### 7.2 act_horizon 选择建议

当前配置 `act_horizon = pred_horizon = 16`，意味着：
- 每个 chunk 的所有动作都可能被执行
- 但由于 ~10 个 chunk 同时活跃，后续步骤会被大量融合
- 实际"独占"执行的步数仅约 **pred_horizon / active_chunks ≈ 1.7 步**

**建议**：
- 如果希望减少融合、增加新动作的影响力，可降低 `act_horizon`
- 例如设置 `act_horizon = 8`，则 chunk 后半段的预测会被丢弃，减少融合冗余

### 7.3 temporal_factor_k 调优

当前 `k = 0.05`，权重衰减较慢，意味着旧 chunk 的贡献较大。

| k 值 | 效果 |
|------|------|
| 0.01 | 融合更平滑，旧 chunk 权重更高 |
| **0.05** | 当前配置 |
| 0.1+ | 新 chunk 主导，响应更快但可能抖动 |

---

## 8. 改进建议

1. **降低推理延迟**
   - 当前瓶颈是推理耗时 (~35ms) + 观测准备 (~45ms)
   - 考虑使用更快的视觉编码器或降低图像分辨率

2. **减少 act_horizon**
   - 设置 `act_horizon < pred_horizon`（如 8）
   - 减少冗余融合，提高新预测的响应性

3. **调整推理频率**
   - 实际频率 18.5Hz < 期望 30Hz
   - 可接受，因为 30Hz 动作执行频率由 `action_interval` 保证

4. **长尾推理优化**
   - 最大推理时间 839ms 异常
   - 检查是否有 GPU 内存抖动或首次推理 warm-up

---

## 附录：数据文件

- 统计 JSON: `summary.json`
- 时间线图: `timeline.png`
- 重叠分析图: `overlap.png`
- 原始日志: `inference_logs/timeline_*.jsonl`, `timeline_record/timeline_record_*.jsonl`
