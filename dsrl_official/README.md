# DSRL Official - ManiSkill3 实现

基于官方 [ajwagen/dsrl](https://github.com/ajwagen/dsrl) 的实现，适配 ManiSkill3 和 ShortCut Flow。

## 概述

DSRL (Diffusion Steering with Reinforcement Learning) 是一种在预训练 Diffusion/Flow Policy 的噪声空间中进行强化学习的方法。

本模块实现了两种算法：
- **DSRL-SAC**: 使用环境包装器，SAC 在噪声空间操作
- **DSRL-NA**: 使用策略内部采样，蒸馏 Q^W（需要官方 SB3 fork）

## 快速开始

### 1. 验证组件

```bash
cd /home/amax/rl-vla/rlft

# 验证所有组件
python dsrl_official/tests/test_minimal_training.py

# 验证 DSRL-SAC 训练流程
python dsrl_official/tests/test_dsrl_sac_simple.py
```

### 2. 运行训练

```bash
# DSRL-SAC 模式（推荐先验证）
python dsrl_official/train_dsrl.py \
    --algorithm dsrl_sac \
    --env_id LiftPegUpright-v1 \
    --total_timesteps 100000 \
    --no-track

# 使用 wandb 跟踪
python dsrl_official/train_dsrl.py \
    --algorithm dsrl_sac \
    --env_id LiftPegUpright-v1 \
    --track
```

## 核心概念

### 噪声空间作为动作空间

DSRL-SAC 的核心思想：
1. SAC 输出"动作" = 噪声 $w \in [-\text{mag}, +\text{mag}]^{T \times D}$
2. 环境包装器将噪声通过 Flow Policy 解码为真实动作
3. SAC 学习在噪声空间中的最优策略

```
SAC Policy → Noise w → ShortCut Flow → Real Actions → Environment
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `action_magnitude` | 1.5 | 噪声范围 $[-\text{mag}, +\text{mag}]$ |
| `act_steps` | 8 | 动作执行步数 (action horizon) |
| `pred_horizon` | 16 | 预测动作序列长度 |
| `obs_horizon` | 2 | 观察历史长度 |
| `utd` | 20 | Update-To-Data ratio |

### 从 Checkpoint 推断 state_dim

本实现自动从预训练 checkpoint 推断 `state_dim`，避免手动配置：

```python
from dsrl_official.utils import load_shortcut_flow_policy

base_policy, visual_encoder = load_shortcut_flow_policy(
    checkpoint_path="path/to/checkpoint.pt",
    state_dim=None,  # 自动推断
    ...
)
```

## 模块结构

```
dsrl_official/
├── __init__.py           # 模块入口
├── train_dsrl.py         # 主训练脚本
├── utils.py              # 工具函数 (ShortCutFlowWrapper, load_shortcut_flow_policy)
├── env_utils.py          # 环境包装器 (ShortCutFlowEnvWrapper)
├── callbacks.py          # SB3 回调函数
├── REQUIREMENTS.md       # 详细技术文档
├── README.md             # 本文件
├── environment.yaml      # Conda 环境配置
├── scripts/              # 辅助脚本
│   ├── analyze_checkpoint_dims.py
│   └── ...
└── tests/                # 测试文件
    ├── test_minimal_training.py    # 组件验证
    ├── test_dsrl_sac_simple.py     # SAC 训练验证
    └── ...
```

## 依赖

### 必需
- Python 3.10+
- PyTorch 2.0+
- ManiSkill3
- stable-baselines3

### 可选（DSRL-NA 模式）
```bash
# 官方 DSRL fork of SB3
pip install git+https://github.com/ajwagen/stable-baselines3.git
```

## 已知问题

1. **ManiSkill3 GPU 环境兼容性**: ManiSkill3 的 GPU 并行环境与 SB3 VecEnv 接口不完全兼容，当前使用简化的适配方案

2. **视觉编码器集成**: 完整的 RGB 观察编码需要在环境包装器中集成 visual_encoder

## 参考

- [Official DSRL Paper](https://arxiv.org/abs/2406.01136)
- [Official DSRL Code](https://github.com/ajwagen/dsrl)
- [ManiSkill3 Documentation](https://maniskill.readthedocs.io/)
