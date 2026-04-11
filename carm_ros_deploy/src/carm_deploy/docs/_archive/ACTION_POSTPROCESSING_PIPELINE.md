# Action Post-Processing Pipeline 分析报告

> Archive note
>
> 本文档描述的是较早阶段的 action post-processing 设计，其中包含
> `KeyboardInterventionHandler` / `InterventionApplier` 这条人工键盘干预链路。
> 该链路已经不再属于当前现役 `inference_ros` pipeline。
>
> 当前现役推理链路以 `action_model -> action_executed` 为主语义，
> Human-in-the-loop 相关能力也不会直接复用这里的旧 intervention 设计。

本报告详细分析真机部署 (`inference_ros.py`) 中 action 从模型输出到机械臂执行的完整后处理流程。

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Inference Thread (30Hz)                           │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │ Policy  │───▶│ Safety Check │───▶│ Intervention│───▶│ ActionChunk     │ │
│  │ Output  │    │ & Clip       │    │ (Optional)  │    │ Manager         │ │
│  │ [16,15] │    │              │    │             │    │ (add_trajectory)│ │
│  └─────────┘    └──────────────┘    └─────────────┘    └────────┬────────┘ │
└──────────────────────────────────────────────────────────────────┼──────────┘
                                                                   │
                                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Control Thread (200Hz)                              │
│  ┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────────┐ │
│  │ ActionChunkManager  │───▶│ get_fused_action │───▶│ Robot Execution     │ │
│  │ (query by time)     │    │ (interpolate/    │    │ (joint/end-effector)│ │
│  │                     │    │  ensemble)       │    │                     │ │
│  └─────────────────────┘    └──────────────────┘    └─────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心组件

### 2.1 TrajectoryInterpolator (VecTF)

**功能**: 存储带时间戳的动作序列，支持按时间查询。

**位置**: `utils/trajectory_interpolator.py`

```python
class TrajectoryInterpolator:
    def append(self, timestamp, action)     # 添加 (时间戳, 动作)
    def get_once(self, query_time)          # 返回第一个 >= query_time 的动作
    def get_once_with_timestamp(self, query_time)  # 同上，带时间戳
    def get_interpolated(self, query_time)  # 线性插值（可选）
    def clear_before(self, timestamp)       # 清理过期数据
```

**工作原理**:
- 每个 `VecTF` 实例存储一个 action chunk 的所有时间步
- 动作按 `(timestamp, action)` 对存储，时间戳递增
- 查询时返回第一个时间戳 >= 查询时间的动作

### 2.2 ActionChunkManager

**功能**: 管理多个 action chunk，支持两种执行模式。

**位置**: `utils/trajectory_interpolator.py`

```python
class ActionChunkManager:
    MODE_TEMPORAL_ENSEMBLE = 'temporal_ensemble'  # 多chunk时间加权融合
    MODE_RECEDING_HORIZON = 'receding_horizon'    # 标准action chunking
    
    def add_trajectory(self, trajectory)          # 添加新chunk
    def get_fused_action(self, query_time)        # 获取执行动作
    def get_fused_action_with_meta(self, query_time)  # 获取动作+元信息
```

---

## 3. 执行模式详解

### 3.1 Temporal Ensemble 模式（默认）

**原理**: 多个活跃 chunk 的动作按时间权重融合。

```python
# 权重计算公式
exp_weights = exp(-temporal_factor_k * [n-1, n-2, ..., 1, 0])
exp_weights = exp_weights / exp_weights.sum()  # 归一化

# 融合
fused_action = sum(actions * exp_weights)
```

**特点**:
- ✅ 输出平滑，动作连续
- ❌ 引入延迟，响应较慢
- ❌ 与训练时语义不一致（train-infer mismatch）
- ❌ 多 chunk 叠加可能导致误差累积

**示例**: 假设有 3 个活跃 chunk，`temporal_factor_k=0.05`
```
weights = exp(-0.05 * [2, 1, 0]) = [0.905, 0.951, 1.000]
normalized = [0.317, 0.333, 0.350]  # 最新chunk权重略高
```

### 3.2 Receding Horizon 模式（推荐用于 RLFT）

**原理**: 只执行最新的有效 chunk，符合标准 Action Chunking 语义。

```python
# 只使用最新chunk
for chunk in reversed(trajectories):
    action = chunk.get_once(query_time)
    if action is not None:
        return action  # 返回最新有效动作

# 无有效chunk时 hold position
return self._last_action
```

**特点**:
- ✅ 响应快，无延迟
- ✅ 与训练时语义一致
- ✅ 适合 RLFT（强化学习微调）
- ❌ chunk 切换时可能有跳变

**可选平滑**: 启用 `crossfade_steps` 进行 chunk 切换平滑
```python
# crossfade 公式
alpha = (crossfade_progress + 1) / crossfade_steps
final_action = (1 - alpha) * old_action + alpha * new_action
```

---

## 4. 超参数详解

### 4.1 推理频率参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `desire_inference_freq` | 30 | 推理线程频率 (Hz)，每秒生成多少个 action chunk |
| `control_period` | 0.005 | 控制线程周期 (s)，即 200Hz 控制频率 |

**影响**:
- `desire_inference_freq` 越高，action chunk 更新越频繁，但计算开销更大
- 30Hz 推理 + 200Hz 控制是常用配置

### 4.2 Action Chunk 构建参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pred_horizon` | 16 | 模型预测的动作序列长度 |
| `act_horizon` | 10 | 实际执行的动作步数（可配置） |
| `action_interval` | 1/30 ≈ 0.033s | 相邻动作的时间间隔 |
| `truncate_at_act_horizon` | True | 是否截断到 act_horizon |

**影响**:
- `action_interval = 1/30s` 意味着一个 chunk 的时间跨度 = `act_horizon * 0.033s`
- 例: `act_horizon=10` → chunk 持续 0.33s
- 若 `truncate_at_act_horizon=False`，chunk 持续 `pred_horizon * 0.033s = 0.53s`

**chunk 时间戳计算**:
```python
for i in range(num_actions_to_add):
    target_time = chunk_base_time + i * action_interval
    tf.append(target_time, all_actions[i])
```

### 4.3 时间融合参数 (Temporal Ensemble 模式)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temporal_factor_k` | 0.05 | 时间衰减因子，越大衰减越快 |
| `max_active_chunks` | None | 最大活跃 chunk 数量 |

**temporal_factor_k 的影响**:

| k 值 | 效果 |
|------|------|
| 0.01 | 衰减慢，多 chunk 权重接近，平滑但延迟大 |
| 0.05 | 默认值，平衡平滑与响应 |
| 0.1 | 衰减快，最新 chunk 权重高，响应快但可能跳变 |
| 0.5+ | 几乎只用最新 chunk，接近 receding_horizon |

**权重示例**（3个chunk）:
```
k=0.01: weights = [0.327, 0.333, 0.340]  # 几乎均匀
k=0.05: weights = [0.317, 0.333, 0.350]  # 轻微偏向新chunk
k=0.1:  weights = [0.300, 0.333, 0.367]  # 明显偏向新chunk
k=0.5:  weights = [0.186, 0.307, 0.507]  # 强烈偏向新chunk
```

### 4.4 Receding Horizon 模式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_active_chunks` | 2 | 保留的 chunk 数量（当前+备用） |
| `crossfade_steps` | 0 | chunk 切换平滑步数，0=直接切换 |

**crossfade_steps 的影响**:

| 值 | 效果 |
|----|------|
| 0 | 无平滑，chunk 切换时可能跳变 |
| 3 | 轻微平滑，过渡约 0.015s (200Hz × 3) |
| 10 | 明显平滑，过渡约 0.05s |
| 20+ | 强平滑，但会引入延迟 |

### 4.5 chunk 时间基准参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_time_base` | 'sys_time' | chunk 时间戳基准 |

**选项**:
- `sys_time`: 使用系统时间（推荐），稳定可靠
- `obs_stamp`: 使用观测时间戳，可能受 ROS 消息延迟影响

### 4.6 位置前瞻参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pos_lookahead_step` | 1 | 前瞻步数 |
| `pos_lookahead_duration` | 0.015 | 前瞻持续时间 (s) |

**说明**: 这些参数用于调整动作执行的时间偏移，通常保持默认即可。

---

## 5. 完整 Action 后处理流程

### Step 1: 模型输出
```python
ret = self.policy({"qpos": qpos, "image": curr_image})
all_actions = ret["a_hat"].squeeze(0).cpu().numpy()  # [pred_horizon, 15]
# 格式: [joint(6), gripper(1), relative_pose(7), gripper(1)]
```

### Step 2: 安全检查与裁剪
```python
# 末端位姿模式
for i in range(len(all_actions)):
    relative_pose = all_actions[i, 7:14]
    
    # 2.1 检查位移是否过大
    trans_norm = np.linalg.norm(relative_pose[:3])
    if trans_norm > max_trans:  # max_trans = 0.1m
        scale = max_trans / trans_norm
        all_actions[i, 7:10] *= scale  # 缩放位移
    
    # 2.2 检查工作空间边界
    target_pose = apply_relative_transform(relative_pose, qpos_end[:7], grip)
    clipped_pose, warnings = safety_controller.check_workspace(target_pose)
    
    # 2.3 检查夹爪限位
    clipped_gripper, grip_warnings = safety_controller.check_joint_limits(gripper_action)
```

### Step 3: 人工干预（历史设计，现役链路已移除）
```python
if self.intervention_enabled:
    intervention = self.intervention_handler.get_intervention()
    if intervention is not None:
        all_actions, intervention_mask = InterventionApplier.apply_to_action_chunk(
            all_actions, intervention, action_format='ee_delta'
        )
```

### Step 4: 坐标变换
```python
# 相对位姿 → 绝对位姿
for i in range(all_actions.shape[0]):
    relative_pose = all_actions[i][7:14]
    grip = all_actions[i][14]
    target_pose = apply_relative_transform(relative_pose, qpos_end[:7], grip)
    # target_pose: [x, y, z, qx, qy, qz, qw, gripper]
```

### Step 5: 构建 Action Chunk
```python
tf = VecTF({})
action_interval = 1.0 / 30.0  # 约 33ms

# 截断到 act_horizon
num_actions = min(act_horizon, len(all_actions)) if truncate_at_act_horizon else len(all_actions)

for i in range(num_actions):
    target_time = chunk_base_time + i * action_interval
    tf.append(target_time, all_actions[i].tolist())

chunk_id = action_manager.add_trajectory(tf)
```

### Step 6: 控制线程查询执行
```python
# 200Hz 控制循环
while running:
    tm = time.time()
    action = action_manager.get_fused_action(tm)  # 根据模式获取动作
    
    if action is not None:
        env.end_control_nostep(action)  # 发送到机械臂
    
    time.sleep(0.005)  # 200Hz
```

---

## 6. 参数配置建议

### 6.1 标准部署（平滑优先）
```bash
--execution_mode temporal_ensemble \
--temporal_factor_k 0.05 \
--truncate_at_act_horizon \
--act_horizon 10
```

### 6.2 RLFT 微调（语义一致性优先）
```bash
--execution_mode receding_horizon \
--max_active_chunks 2 \
--crossfade_steps 5 \
--truncate_at_act_horizon \
--act_horizon 8
```

### 6.3 快速响应（低延迟优先）
```bash
--execution_mode receding_horizon \
--max_active_chunks 1 \
--crossfade_steps 0 \
--act_horizon 4
```

---

## 7. 时序分析

### 典型时序 (temporal_ensemble, 30Hz 推理, 200Hz 控制)

```
时间 (ms)     推理线程                    控制线程
─────────────────────────────────────────────────────────
   0         [推理开始]                   
  20         生成 chunk_0 (t=0~330ms)    查询 t=20ms, 用 chunk_0
  25                                     查询 t=25ms, 用 chunk_0
  30                                     查询 t=30ms, 用 chunk_0
  33         [推理开始]
  53         生成 chunk_1 (t=33~363ms)   查询 t=53ms, 融合 chunk_0+chunk_1
  ...
```

### chunk 时间覆盖

```
chunk_0: |████████████████████|  (t=0 ~ t=330ms, 10步×33ms)
chunk_1:      |████████████████████|  (t=33 ~ t=363ms)
chunk_2:           |████████████████████|  (t=66 ~ t=396ms)

控制查询:   ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  (每5ms一次)
```

---

## 8. 总结

| 方面 | Temporal Ensemble | Receding Horizon |
|------|-------------------|------------------|
| 平滑性 | ✅ 好 | ⚠️ 需要 crossfade |
| 响应速度 | ⚠️ 有延迟 | ✅ 快 |
| 语义一致性 | ❌ 与训练不一致 | ✅ 与训练一致 |
| 适用场景 | 常规部署 | RLFT、精细任务 |
| 关键参数 | `temporal_factor_k` | `crossfade_steps` |

**核心超参数速查**:
- `temporal_factor_k`: 控制时间融合衰减速度 (0.01~0.5)
- `action_interval`: 动作时间间隔 (固定 1/30s ≈ 33ms)
- `act_horizon`: 实际执行的动作步数 (推荐 8~10)
- `crossfade_steps`: chunk 切换平滑步数 (0~10)
