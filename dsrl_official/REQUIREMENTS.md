# DSRL Official Implementation - 固定需求

## 概述

本模块基于官方 [ajwagen/dsrl](https://github.com/ajwagen/dsrl) 实现，针对 ManiSkill3 环境和 AW-ShortCut Flow 策略进行最小化适配。

## ⚠️ 重要发现：Checkpoint 维度不一致问题

### 问题描述

预训练 checkpoint (`best_eval_success_once.pt`) 中的 `velocity_net` 和 `critic` 使用了**不同的 `global_cond_dim`**：

| 模块 | 输入维度 | state_dim | 来源 |
|------|---------|-----------|------|
| `velocity_net` | 626 | 57 | `cond_encoder.weight: [512, 626]` |
| `critic` | 562 (obs) + 56 (action) = 618 | 25 | `q1_net.weight: [512, 618]` |

### 维度计算
```
global_cond_dim = obs_horizon * (visual_dim + state_dim)
                = 2 * (256 + state_dim)

velocity_net: 626 = 2 * 313 => state_dim = 57
critic:       562 = 2 * 281 => state_dim = 25
```

### 当前数据集/环境的 state_dim
```
qpos(9) + qvel(9) + tcp_pose(7) = 25  ✓ 匹配 critic
```

### 影响
- 这是一个**已知的不一致问题**，可能来自训练代码的 bug
- 需要使用 `state_dim = 25` (与环境/数据集匹配) 重新训练 checkpoint
- 或者在加载时只加载 `critic` 和 `visual_encoder`，重新初始化 `velocity_net`

## 固定需求

### 1. 仿真流程

- **对齐文件**: `rlft/dsrl_offpolicy/train/train_stage2_online.py`
- **环境包装**: 使用 `FlattenRGBDObservationWrapper` (from `mani_skill.utils.wrappers.flatten`)
- **仿真后端**: `physx_cuda` (GPU 并行)
- **验证标准**: 预训练 checkpoint 应达到 ~80% `success_once`

### 2. 数据集

```yaml
env_id: "LiftPegUpright-v1"
demo_path: "~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"
control_mode: "pd_ee_delta_pose"
obs_mode: "rgb"
state_dim: 25  # qpos(9) + qvel(9) + tcp_pose(7)
action_dim: 7
```

### 3. 预训练 Checkpoint 结构 (AWShortCutFlowAgent)

```yaml
checkpoint_path: "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"

# Checkpoint 结构
top_level_keys:
  - agent: 364 keys
  - ema_agent: 364 keys
  - visual_encoder: 12 keys

agent_modules:
  - velocity_net: 154 keys (ShortCutVelocityUNet1D)
  - velocity_net_ema: 154 keys
  - critic: 28 keys (DoubleQNetwork)
  - critic_target: 28 keys

# 维度信息
dimensions:
  global_cond_dim_velocity: 626  # ⚠️ 不匹配
  global_cond_dim_critic: 562    # ✓ 匹配 state_dim=25
  action_dim: 7
  act_horizon: 8
  pred_horizon: 16
```

### 4. 虚拟环境

```bash
conda activate rlft_ms3  # 注意：不是 arx-py310
# Python 3.10
# PyTorch >= 2.0
# stable-baselines3 (需要安装官方 fork)
```

## 官方 DSRL 算法概述

### 两种变体

| 变体 | 描述 | 动作空间 | 特点 |
|------|------|----------|------|
| **DSRL-SAC** | 标准 SAC，在噪声空间操作 | 噪声空间 | 计算效率高 |
| **DSRL-NA** | 从 Q^A 蒸馏 Q^W | 原始 + 噪声空间 | 样本效率高，**推荐** |

### 核心超参数（官方推荐）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `action_magnitude` | 1.5 | 噪声空间范围 [-mag, +mag] |
| `utd` | 20 | 每步梯度更新次数 |
| `net_arch` | [2048, 2048, 2048] | Actor/Critic MLP |
| `activation_fn` | Tanh | 激活函数 |
| `tau` | 0.005 | Target 网络软更新 |
| `n_critics` | 2 | Critic 数量 |

### 训练流程（官方）

```
1. 加载预训练 diffusion/flow policy (base_policy)
2. [DSRL-SAC] 包装环境: DiffusionPolicyEnvWrapper
   [DSRL-NA] 直接使用原始环境
3. [可选] 离线数据预填充 replay buffer
4. [可选] 初始 rollout 收集
5. 在线 SAC 训练 (model.learn)
```

## 本模块适配

### 与官方差异

| 方面 | 官方 | 本模块 |
|------|------|--------|
| Base Policy | DPPO (Diffusion) | ShortCut Flow |
| 环境 | Robomimic/Gym | ManiSkill3 |
| 配置系统 | Hydra | tyro |
| VecEnv | SubprocVecEnv | GPU 并行 (physx_cuda) |

### 需要适配的组件

1. **ShortCutFlowWrapper**: 适配 `DPPOBasePolicyWrapper` 接口
2. **ManiSkillEnvWrapper**: 适配 `DiffusionPolicyEnvWrapper`
3. **数据加载**: 从 HDF5 加载 ManiSkill demo
4. **评估逻辑**: 适配 ManiSkill 的 `success_once`/`success_at_end`

## 验证检查清单

- [ ] 预训练 checkpoint 在 `train_stage2_online.py` 环境中达到 ~80% success_once
- [ ] `ShortCutFlowWrapper` 与 `DPPOBasePolicyWrapper` 接口一致
- [ ] 环境包装器模式 vs 策略内部采样模式输出一致
- [ ] DSRL-NA Q^W 蒸馏损失与官方一致

---

## 调试计划 (Debug Plan)

### 阶段 0: 修复已知 Bug

#### 0.1 ShortCutFlowWrapper Flow 积分逻辑 (`utils.py`)

**问题**:
1. Flow 积分方向错误：应从 `t=0→1`，当前是 `t=1→0`
2. 缺少 `step_size` 参数：`velocity_net.forward()` 需要 4 个参数
3. 错误的 permute：UNet 期望 `(B, T, C)` 格式，不需要转置

#### 0.2 导入路径修复 (`train_dsrl.py`, `validate_baseline.py`)

```python
# 错误
from diffusion_policy.wrappers import FlattenRGBDObservationWrapper

# 正确
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
```

---

### 阶段 1: 导入测试

```bash
conda activate rlft_ms3

# 测试 1: SB3 标准版
python -c "from stable_baselines3 import SAC; print('SB3 OK')"

# 测试 2: ManiSkill3
python -c "import mani_skill.envs; print('ManiSkill3 OK')"

# 测试 3: diffusion_policy
python -c "from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D; print('ShortCut Flow OK')"

# 测试 4: dsrl_official 模块
python -c "from dsrl_official import utils, env_utils, callbacks; print('dsrl_official OK')"
```

**创建脚本**: `tests/test_imports.py`

---

### 阶段 2: 模型加载测试

验证 checkpoint 中的模块能正确加载：

| 模块 | Key 前缀 | 目标类 | 备注 |
|------|---------|--------|------|
| `velocity_net` | `velocity_net.*` | `ShortCutVelocityUNet1D` | ⚠️ `global_cond_dim=626` 不匹配 |
| `velocity_net_ema` | `velocity_net_ema.*` | `ShortCutVelocityUNet1D` | 同上 |
| `critic` | `critic.*` | `DoubleQNetwork` | ✓ `obs_dim=562` 匹配 |
| `critic_target` | `critic_target.*` | `DoubleQNetwork` | 同上 |
| `visual_encoder` | 顶层 `visual_encoder.*` | `PlainConv` | ✓ 12 keys |

**创建脚本**: `tests/test_model_loading.py`

---

### 阶段 3: 输入输出维度测试

固定配置:
```yaml
env_id: LiftPegUpright-v1
state_dim: 25  # qpos(9) + qvel(9) + tcp_pose(7)
visual_dim: 256  # PlainConv output
obs_horizon: 2
global_cond_dim: 562  # 2 * (256 + 25)
action_dim: 7
act_horizon: 8
pred_horizon: 16
```

**验证点**:
1. 环境观察空间维度
2. 数据集 state 维度
3. velocity_net 输入输出形状
4. critic 输入输出形状

**创建脚本**: `tests/test_dimensions.py`

---

### 阶段 4: 基准验证 + 双模式测试

```bash
# 运行双模式一致性测试
python -m pytest tests/test_dual_mode_consistency.py -v

# 运行基准验证 (需先修复维度问题)
python scripts/validate_baseline.py --num_episodes 50
```

**目标**: 预训练模型达到 ~80% `success_once`

---

### 阶段 5: SB3 Fork 安装

```bash
# 官方 DSRL 需要的 SB3 fork (包含 DSRL 算法类)
pip install git+https://github.com/ajwagen/stable-baselines3.git

# 验证
python -c "from stable_baselines3 import DSRL; print('SB3 DSRL Fork OK')"
```

---

### 阶段 6: 最小规模实验

#### 6.1 DSRL-SAC 模式
```bash
python train_dsrl.py \
    --env_id LiftPegUpright-v1 \
    --pretrained_checkpoint /path/to/checkpoint.pt \
    --total_timesteps 5000 \
    --use_env_wrapper \
    --num_envs 4 \
    --wandb_mode disabled
```

#### 6.2 DSRL-NA 模式
```bash
python train_dsrl.py \
    --env_id LiftPegUpright-v1 \
    --pretrained_checkpoint /path/to/checkpoint.pt \
    --total_timesteps 5000 \
    --no_use_env_wrapper \
    --num_envs 4 \
    --wandb_mode disabled
```

---

## 解决 Checkpoint 维度不一致的方案

### 方案 A: 重新训练 (推荐)

使用修正后的 `train_offline_rl.py` 重新训练：
```bash
CUDA_VISIBLE_DEVICES=0 python train_offline_rl.py \
    --algorithm aw_shortcut_flow \
    --env_id LiftPegUpright-v1 \
    --demo_path ~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --total_iters 30000
```

确保 `velocity_net` 和 `critic` 使用相同的 `global_cond_dim`。

### 方案 B: 部分加载 Checkpoint

只加载 `critic` 和 `visual_encoder`，重新初始化 `velocity_net`：
```python
# 伪代码
checkpoint = torch.load(path)
agent = AWShortCutFlowAgent(global_cond_dim=562, ...)
agent.critic.load_state_dict(extract_critic(checkpoint))
agent.visual_encoder.load_state_dict(checkpoint['visual_encoder'])
# velocity_net 保持随机初始化
```

### 方案 C: 适配代码到 626 维度

修改状态提取逻辑，添加额外的 32 维状态信息以匹配 `global_cond_dim=626`。
**不推荐**：需要修改环境和数据加载逻辑，容易引入更多问题。
