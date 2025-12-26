# Real Robot Policy Inference Documentation

本文档说明 `inference_real_robot.py` 推理脚本的使用方法、模型架构、数据流以及相关配置。

## 目录

1. [概述](#概述)
2. [模型架构](#模型架构)
3. [数据格式](#数据格式)
4. [数据流详解](#数据流详解)
5. [使用方法](#使用方法)
6. [配置参数](#配置参数)
7. [示例输出](#示例输出)
8. [FAQ](#faq)

---

## 概述

`inference_real_robot.py` 是用于加载训练好的 checkpoint 并在真实机器人数据上进行推理的脚本。支持多种生成式策略算法，包括：

- **Diffusion Policy**: 基于 DDPM 的扩散策略
- **Flow Matching**: 流匹配策略
- **Reflected Flow**: 反射流匹配
- **Consistency Flow**: 一致性流（推荐，支持单步推理）
- **ShortCut Flow**: 快捷流

脚本功能：
- 自动检测算法类型
- 加载 checkpoint（支持 EMA 权重）
- 处理原始观测数据
- 运行推理并输出动作序列
- 可视化预测 vs Ground Truth

---

## 模型架构

### 整体结构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Real Robot Policy                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌───────────────┐                        │
│   │  RGB Image   │    │  Robot State  │                        │
│   │ (128×128×3)  │    │  (13D vector) │                        │
│   └──────┬───────┘    └──────┬────────┘                        │
│          │                   │                                  │
│          ▼                   ▼                                  │
│   ┌──────────────┐    ┌───────────────┐                        │
│   │  PlainConv   │    │ StateEncoder  │                        │
│   │  (CNN)       │    │ (2-layer MLP) │                        │
│   │  → 256D      │    │  → 256D       │                        │
│   └──────┬───────┘    └──────┬────────┘                        │
│          │                   │                                  │
│          └───────┬───────────┘                                  │
│                  ▼                                              │
│          ┌───────────────┐                                      │
│          │ Concat × T_obs│ = 512D × 2 = 1024D                  │
│          │ (global_cond) │                                      │
│          └──────┬────────┘                                      │
│                 │                                               │
│                 ▼                                               │
│          ┌───────────────┐                                      │
│          │    Agent      │                                      │
│          │(ConsistencyFlow)│                                    │
│          └──────┬────────┘                                      │
│                 │                                               │
│                 ▼                                               │
│          ┌───────────────┐                                      │
│          │ Action Seq    │                                      │
│          │ (16 × 7)      │                                      │
│          └───────────────┘                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 组件详解

#### 1. PlainConv (Visual Encoder)

CNN 视觉编码器，将 RGB 图像编码为固定维度特征向量。

```python
PlainConv(
    in_channels=3,           # RGB 通道数（或 4 如果包含深度）
    out_dim=256,             # 输出特征维度
    pool_feature_map=True,   # 使用全局平均池化
)
```

结构：
- Conv2d(3, 32, kernel=3, stride=2)
- Conv2d(32, 64, kernel=3, stride=2)  
- Conv2d(64, 128, kernel=3, stride=2)
- Conv2d(128, 256, kernel=3, stride=2)
- Conv2d(256, 256, kernel=3, stride=2)
- Global Average Pooling
- Linear(256, 256)

输入: `[B, 3, 128, 128]` → 输出: `[B, 256]`

#### 2. StateEncoder (State MLP)

2 层 MLP，将机器人状态编码为与视觉特征对齐的表示。

```python
StateEncoder(
    state_dim=13,      # 输入状态维度
    hidden_dim=128,    # 隐藏层维度
    out_dim=256,       # 输出维度（与视觉特征对齐）
)
```

结构：
- Linear(13, 128) → ReLU
- Linear(128, 256) → ReLU

输入: `[B, 13]` → 输出: `[B, 256]`

#### 3. Agent (Policy Network)

根据算法类型不同，使用不同的策略网络。以 ConsistencyFlowAgent 为例：

```python
ConsistencyFlowAgent(
    velocity_net=VelocityUNet1D(...),
    action_dim=7,
    obs_horizon=2,
    pred_horizon=16,
    num_flow_steps=10,
    ema_decay=0.999,
)
```

内部使用 `VelocityUNet1D`：
- 时间步嵌入维度: 64
- UNet 通道数: (64, 128, 256)
- 组归一化组数: 8

---

## 数据格式

### 输入数据格式（HDF5 存储）

```
trajectory.h5
├── obs/
│   ├── joint_pos      # [T, 6] float32 - 关节位置（弧度）
│   ├── joint_vel      # [T, 6] float32 - 关节速度
│   ├── gripper_pos    # [T, 1] float32 - 夹爪位置 [0, 1]
│   └── images/
│       └── wrist/
│           ├── rgb    # [T, 256, 256, 3] uint8 - RGB 图像
│           └── depth  # [T, 256, 256] float32 - 深度图（可选）
└── actions            # [T, 7] float32 - 动作序列
```

### 状态向量构成（13D）

| 索引 | 内容 | 维度 | 范围 |
|------|------|------|------|
| 0-5 | joint_pos | 6 | [-π, π] |
| 6-11 | joint_vel | 6 | ~ [-1, 1] |
| 12 | gripper_pos | 1 | [0, 1] |

### 动作向量构成（7D）

| 索引 | 内容 | 说明 |
|------|------|------|
| 0-5 | joint_action | 关节位置目标 |
| 6 | gripper_action | 夹爪位置目标 |

---

## 数据流详解

### 1. 观测预处理

原始数据 → 处理后数据

```python
obs_process_fn = create_real_robot_obs_process_fn(
    output_format="NCHW",          # 输出格式
    camera_name="wrist",           # 使用腕部相机
    include_depth=False,           # 不包含深度
    target_size=(128, 128),        # 目标尺寸
)

processed = obs_process_fn(raw_obs)
# processed["rgb"]   : [T, 3, 128, 128] uint8
# processed["state"] : [T, 13] float32
```

### 2. 观测窗口采样

从完整轨迹中采样 `obs_horizon` 帧：

```
轨迹: [0, 1, 2, ..., 99, 100, 101, ...]
                          ↑    ↑
                      obs_indices = [100, 101]
                      
采样后:
  rgb:   [B, 2, 3, 128, 128]
  state: [B, 2, 13]
```

### 3. 特征编码

```python
# 视觉编码
rgb_flat = rgb.view(B * T, 3, 128, 128).float() / 255.0
visual_feat = visual_encoder(rgb_flat)  # [B*T, 256]
visual_feat = visual_feat.view(B, T, 256)  # [B, 2, 256]

# 状态编码
state_flat = state.view(B * T, 13)
state_feat = state_encoder(state_flat)  # [B*T, 256]
state_feat = state_feat.view(B, T, 256)  # [B, 2, 256]

# 拼接
obs_features = cat([visual_feat, state_feat], dim=-1)  # [B, 2, 512]
global_cond = obs_features.view(B, -1)  # [B, 1024]
```

### 4. 动作生成

```python
# 生成完整预测序列
action_seq = agent.get_action(global_cond)  # [B, 16, 7]

# 提取可执行动作
start = obs_horizon - 1  # = 1
end = start + act_horizon  # = 1 + 8 = 9
executable_actions = action_seq[:, start:end]  # [B, 8, 7]
```

动作序列结构：

```
时间轴:   t-1    t    t+1   t+2   ...  t+14  t+15
         obs   obs   
               pred  pred  pred  ...  pred  pred
               [0]   [1]   [2]   ...  [14]  [15]
                     ↑                      ↑
                   start                  end
                   
可执行: actions[1:9] → 8 步动作
```

---

## 使用方法

### 基本用法

```bash
# 使用默认参数
python inference_real_robot.py \
    --checkpoint_path runs/consistency_flow-pick_cube-real__1__1766127719/checkpoints/iter_10000.pt

# 指定数据路径
python inference_real_robot.py \
    --checkpoint_path <checkpoint.pt> \
    --demo_path ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5

# 指定轨迹和起始步
python inference_real_robot.py \
    --checkpoint_path <checkpoint.pt> \
    --traj_idx 0 \
    --start_idx 100

# 不使用 EMA 权重
python inference_real_robot.py \
    --checkpoint_path <checkpoint.pt> \
    --no_ema
```

### 编程接口

```python
from inference_real_robot import RealRobotPolicyInference

# 初始化
inference = RealRobotPolicyInference(
    checkpoint_path="path/to/checkpoint.pt",
    demo_path="path/to/data.h5",
    device="cuda",
    use_ema=True,
)

# 从原始观测推理
raw_obs = ...  # 从 HDF5 加载
actions = inference.predict_from_raw(raw_obs)  # [1, 8, 7]

# 从处理后的数据推理
rgb = torch.randn(1, 2, 3, 128, 128)  # [B, obs_horizon, C, H, W]
state = torch.randn(1, 2, 13)         # [B, obs_horizon, state_dim]
actions = inference.predict(rgb, state)  # [1, 8, 7]
```

---

## 配置参数

### InferenceConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `obs_horizon` | 2 | 观测历史长度 |
| `act_horizon` | 8 | 可执行动作长度 |
| `pred_horizon` | 16 | 预测序列长度 |
| `diffusion_step_embed_dim` | 64 | 时间步嵌入维度 |
| `unet_dims` | (64, 128, 256) | UNet 通道数 |
| `n_groups` | 8 | 组归一化组数 |
| `visual_feature_dim` | 256 | 视觉特征维度 |
| `state_encoder_hidden_dim` | 128 | 状态编码器隐藏层 |
| `state_encoder_out_dim` | 256 | 状态编码器输出维度 |
| `num_diffusion_iters` | 100 | 扩散迭代次数 |
| `num_flow_steps` | 10 | 流匹配步数 |
| `camera_name` | "wrist" | 使用的相机 |
| `include_depth` | False | 是否包含深度 |
| `target_image_size` | (128, 128) | 目标图像尺寸 |

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint_path` | str | 必需 | Checkpoint 文件路径 |
| `--demo_path` | str | 默认数据路径 | 演示数据 HDF5 路径 |
| `--algorithm` | str | 自动检测 | 算法类型 |
| `--traj_idx` | int | 0 | 测试轨迹索引 |
| `--start_idx` | int | 100 | 起始时间步 |
| `--use_ema` | flag | True | 使用 EMA 权重 |
| `--no_ema` | flag | False | 不使用 EMA |
| `--save_fig` | str | 自动生成 | 图像保存路径 |
| `--device` | str | "cuda" | 运行设备 |

---

## 示例输出

运行脚本后，你将看到类似以下输出：

```
Using algorithm: consistency_flow
Inference module initialized:
  - Algorithm: consistency_flow
  - Action dim: 7
  - State dim: 13 -> 256
  - Visual feature dim: 256
  - Global cond dim: 1024
  - Obs horizon: 2
  - Pred horizon: 16
  - Act horizon: 8

============================================================
INFERENCE TEST
============================================================
Trajectory: 0
Observation indices: [100, 101]
Prediction for timesteps: 100 to 115

--- Input Data ---
RGB shape: (467, 3, 128, 128) (T, C, H, W)
State shape: (467, 13) (T, state_dim)

--- First 3 timesteps comparison ---

Timestep 100:
  Dim        Prediction  GroundTruth        Error
  ------------------------------------------------
  Joint0         0.1234       0.1256       0.0022
  Joint1        -0.5678      -0.5612       0.0066
  ...

--- Overall Metrics ---
MSE: 0.002345
MAE: 0.034567

--- Per-Dimension MAE ---
  Joint0: 0.023456
  ...
```

同时会生成可视化图表，展示每个动作维度的预测值和真实值对比。

---

## FAQ

### Q: 如何切换不同的算法？

A: 算法通常从 checkpoint 路径自动检测。也可以手动指定：
```bash
python inference_real_robot.py --checkpoint_path <path> --algorithm flow_matching
```

### Q: EMA 权重是什么？

A: EMA (Exponential Moving Average) 是训练过程中模型参数的指数移动平均。通常用于推理时获得更稳定的结果。推荐使用 EMA 权重（默认行为）。

### Q: 为什么预测长度是 16 但只执行 8 步？

A: 这是 Action Chunking 策略的设计。预测较长序列提供更多上下文，但只执行部分动作，然后重新观测和预测，以获得更好的闭环控制效果。

### Q: 如何在真实机器人上部署？

A: 
1. 初始化 `RealRobotPolicyInference` 对象
2. 在控制循环中：
   - 获取当前观测（RGB + 状态）
   - 维护最近 `obs_horizon` 帧的历史
   - 调用 `inference.predict()` 获取动作
   - 执行第一个动作或前几个动作
   - 重复

### Q: 如何调试输出？

A: 检查以下常见问题：
1. 图像归一化：确保输入为 `[0, 255]` 整数或 `[0, 1]` 浮点
2. 状态维度：确保状态向量维度匹配（13D）
3. 观测顺序：时间顺序应为从旧到新

---

## 相关文件

- [train_real_robot.py](train_real_robot.py) - 训练脚本
- [diffusion_policy/utils.py](diffusion_policy/utils.py) - StateEncoder 和数据处理工具
- [DATASET_COMPARISON.md](DATASET_COMPARISON.md) - 数据集格式对比
