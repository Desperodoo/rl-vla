# RTC 官方仓库架构分析

本文档分析 [Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) 官方仓库的核心设计，为本项目的实现提供参考。

## 1. 核心概念

### 1.1 Action Chunking 问题

传统 action chunking 的问题：策略每次输出 $H$ 个动作 (action chunk)，在执行完这 $H$ 个动作期间不进行新的推理。这导致：

1. **延迟问题**：推理完成后才能开始执行新动作
2. **连续性问题**：chunk 边界处动作不连续

### 1.2 RTC 解决方案

RTC 的核心思想是：**在执行当前 chunk 的同时，并行进行下一次推理**。

关键参数：
- `action_chunk_size` ($H$): 策略输出的动作序列长度 (本项目=8)
- `inference_delay` ($d$): 推理期间需要执行的动作数
- `execute_horizon` ($e$): 每轮执行的总动作数

时间线示意：
```
轮次 n:
  t=0                t=d*dt           t=e*dt
   |-----推理中------|                  |
   |执行 chunk[n-1][d:e]|执行 chunk[n][d:e]|
   
轮次 n+1:
                     t=0              t=d*dt           t=e*dt
                      |-----推理中------|                  |
                      |执行 chunk[n][d:e]|执行 chunk[n+1][d:e]|
```

### 1.3 动作拼接策略

来自 `eval_flow.py` 的核心逻辑：

```python
# 执行 inference_delay 个来自上一轮的动作
# 然后执行 execute_horizon - inference_delay 个来自新推理的动作
action_chunk_to_execute = jnp.concatenate([
    action_chunk[:, :config.inference_delay],           # 上一轮 chunk 的前 d 个
    next_action_chunk[:, config.inference_delay:config.execute_horizon],  # 新 chunk 的 [d:e]
], axis=1)
```

## 2. 模型架构

### 2.1 ModelConfig

```python
@dataclasses.dataclass(frozen=True)
class ModelConfig:
    channel_dim: int = 256
    channel_hidden_dim: int = 512
    token_hidden_dim: int = 64
    num_layers: int = 4
    action_chunk_size: int = 8          # H = 8
    simulated_delay: int | None = None  # 训练时模拟推理延迟
```

### 2.2 FlowPolicy

基于 Flow Matching 的策略网络，核心方法：

#### `action()` - 标准采样
```python
def action(self, rng, obs, num_steps):
    """标准 flow matching 采样，不考虑历史"""
    dt = 1 / num_steps
    noise = jax.random.normal(rng, shape=(batch, H, action_dim))
    
    for _ in range(num_steps):
        v_t = self(obs, x_t, time)
        x_t = x_t + dt * v_t
        time = time + dt
    
    return x_t  # [batch, H, action_dim]
```

#### `realtime_action()` - RTC 实时采样
```python
def realtime_action(self, rng, obs, num_steps, prev_action_chunk,
                    inference_delay, prefix_attention_horizon,
                    prefix_attention_schedule, max_guidance_weight):
    """
    RTC 采样：条件化于前一个 chunk 的前缀
    
    关键点：
    1. 使用 prev_action_chunk[:, :inference_delay] 作为条件
    2. 通过 pinv_corrected_velocity 引导新 chunk 与前缀一致
    """
```

#### `bid_action()` - BID 采样 (Backwards-In-Distance)
```python
def bid_action(self, rng, obs, num_steps, prev_action_chunk,
               inference_delay, prefix_attention_horizon,
               n_samples, bid_weak_policy=None, bid_k=None):
    """
    BID 采样：拒绝采样方法
    
    1. 生成 n_samples 个候选 chunk
    2. 计算每个候选与 prev_action_chunk 前缀的 backward loss
    3. 选择 loss 最小的候选
    """
```

### 2.3 训练时延迟模拟

`simulated_delay` 参数在训练时模拟推理延迟：

```python
def loss(self, rng, obs, action):
    # 模拟推理延迟：前 delay 个 action 是"已知"的
    if self.simulated_delay is not None:
        delay = jax.random.choice(delay_rng, self.simulated_delay, p=w)
        mask = jnp.arange(action_chunk_size) < delay
        time = jnp.where(mask, 1.0, time)  # 前 delay 个设为 t=1 (已知)
```

## 3. 评估配置

### 3.1 EvalConfig

```python
@dataclasses.dataclass(frozen=True)
class EvalConfig:
    num_evals: int = 2048
    num_flow_steps: int = 5       # flow 采样步数
    
    inference_delay: int = 0      # 推理延迟 (动作数)
    execute_horizon: int = 1      # 每轮执行的动作数
    
    method: NaiveMethodConfig | RealtimeMethodConfig | BIDMethodConfig
```

### 3.2 方法配置

```python
# 朴素方法：不考虑前缀一致性
class NaiveMethodConfig:
    pass

# RTC 实时方法：使用梯度引导
class RealtimeMethodConfig:
    prefix_attention_schedule: str = "exp"  # "linear", "exp", "ones", "zeros"
    max_guidance_weight: float = 5.0

# BID 方法：拒绝采样
class BIDMethodConfig:
    n_samples: int = 16
    bid_k: int | None = None
```

### 3.3 评估 sweep

官方评估遍历所有参数组合：
```python
for inference_delay in [0, 1, 2, 3, 4]:
    for execute_horizon in range(max(1, inference_delay), 8 - inference_delay + 1):
        # 测试各种 (delay, horizon) 组合
```

## 4. 与本项目的对应关系

### 4.1 参数映射

| 官方参数 | 本项目参数 | 值 |
|---------|----------|-----|
| `action_chunk_size` | `act_horizon` | 8 |
| `num_flow_steps` | `num_flow_steps` | 5 |
| action dt | `dt_key` | 33.3ms (30Hz) |
| inference dt | `policy_dt` | 100ms (10Hz) |
| servo dt | - | 2ms (500Hz) |

### 4.2 推荐参数设置

基于延迟测试结果：

| 推理延迟 (P95) | `inference_delay` | `execute_horizon` |
|---------------|-------------------|-------------------|
| < 33ms | 1 | 3-7 |
| < 66ms | 2 | 3-6 |
| < 100ms | 3 | 4-5 |

### 4.3 实现差异

**官方实现**（仿真环境）：
- 同步推理和执行
- 推理完成后立即执行
- 无共享内存/进程间通信

**本项目实现**（真机环境）：
- 异步推理和伺服
- 共享内存通信
- 500Hz 三次样条插值
- 安全限制和平滑滤波

## 5. 关键实现细节

### 5.1 前缀权重 (Prefix Weights)

```python
def get_prefix_weights(start, end, total, schedule):
    """
    计算前缀注意力权重
    
    - start: inference_delay
    - end: prefix_attention_horizon  
    - total: action_chunk_size
    
    返回权重数组，用于加权 backward loss
    """
    if schedule == "linear":
        # 线性衰减
    elif schedule == "exp":
        # 指数衰减
    elif schedule == "ones":
        # 全 1
    elif schedule == "zeros":
        # 全 0 (hard masking)
```

### 5.2 梯度引导 (Gradient Guidance)

`realtime_action` 中的核心：

```python
# 1. 计算预测 x_1 和 Jacobian
x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)

# 2. 计算与 prev_action_chunk 的误差
error = (y - x_1) * weights  # 加权误差

# 3. 通过 VJP 计算校正项
pinv_correction = vjp_fun(error)[0]

# 4. 计算引导权重
guidance_weight = min(c * inv_r2, max_guidance_weight)

# 5. 校正速度
v_t_corrected = v_t + guidance_weight * pinv_correction
```

### 5.3 执行逻辑

```python
def execute_chunk(carry, _):
    rng, obs, env_state, action_chunk, n = carry
    
    # 推理新 chunk
    if method == "naive":
        next_action_chunk = policy.action(key, obs, num_flow_steps)
    elif method == "realtime":
        next_action_chunk = policy.realtime_action(...)
    elif method == "bid":
        next_action_chunk = policy.bid_action(...)
    
    # 拼接执行序列
    action_chunk_to_execute = concatenate([
        action_chunk[:, :inference_delay],           # 上一轮前缀
        next_action_chunk[:, inference_delay:execute_horizon],  # 新chunk
    ])
    
    # 执行动作
    for action in action_chunk_to_execute:
        obs, env_state, reward, done = env.step(action)
    
    # 更新 chunk 引用（滑动窗口）
    next_action_chunk_shifted = concatenate([
        next_action_chunk[:, execute_horizon:],      # 移除已执行部分
        zeros(execute_horizon),                      # 填充零
    ])
    
    return (rng, obs, env_state, next_action_chunk_shifted, n_updated)
```

## 6. 本项目的实现建议

### 6.1 Phase A（当前）- 简化实现

```python
# 不使用 realtime/BID 方法，简单的滑动窗口
def inference_loop():
    while running:
        # 1. 采集观测
        images, state = get_observation()
        
        # 2. 推理（标准 flow）
        keyframes = policy.action(obs, num_flow_steps)
        
        # 3. 写入共享内存
        shm_writer.write(keyframes, t_write)
        
        # 4. 等待下一周期
        sleep_until(next_policy_time)
```

伺服进程使用三次样条平滑处理 chunk 边界。

### 6.2 Phase B（未来）- 完整 RTC

1. **实现 realtime_action**：添加前缀条件化采样
2. **优化 chunk 拼接**：committed/editable 双缓冲
3. **边界混合**：C1 连续 stitching

### 6.3 Phase C（未来）- 性能优化

1. **FP16 推理**
2. **异步相机采集**
3. **CUDA 流优化**

## 7. 参考文献

1. [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
2. [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964)
3. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
