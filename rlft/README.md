# RLFT: Reinforcement Learning and Flow-based Training

<p align="center">
  <b>A unified framework for robot learning with diffusion/flow policies</b>
</p>

RLFT 是一个统一的机器人学习框架，支持：
- **Imitation Learning (IL)**: Diffusion Policy, Flow Matching, ShortCut Flow, Consistency Flow
- **Offline Reinforcement Learning**: CPQL, AWCP, AW-ShortCut Flow
- **Online Reinforcement Learning**: SAC, RLPD, ReinFlow, AWSC

## 📁 项目结构

```
rlft/
├── algorithms/          # 策略学习算法
│   ├── il/              # 模仿学习算法
│   │   ├── diffusion_policy.py    # Diffusion Policy (DDPM)
│   │   ├── flow_matching.py       # Flow Matching (ODE-based)
│   │   ├── shortcut_flow.py       # ShortCut Flow (few-step sampling)
│   │   ├── consistency_flow.py    # Consistency Flow
│   │   └── reflected_flow.py      # Reflected Flow (bounded actions)
│   ├── offline_rl/      # 离线强化学习算法
│   │   ├── cpql.py                # CPQL (Conservative Policy Q-Learning)
│   │   ├── awcp.py                # AWCP (Advantage-Weighted Conservative Policy)
│   │   └── aw_shortcut_flow.py    # AW-ShortCut Flow
│   └── online_rl/       # 在线强化学习算法
│       ├── sac.py                 # SAC (Soft Actor-Critic)
│       ├── reinflow.py            # ReinFlow (PPO + Flow Matching)
│       └── awsc.py                # AWSC (Advantage-Weighted ShortCut Flow)
│
├── networks/            # 神经网络架构
│   ├── unet.py          # Conditional 1D U-Net
│   ├── velocity.py      # Velocity networks (VelocityUNet1D, ShortCutVelocityUNet1D)
│   ├── q_networks.py    # Q-networks (DoubleQ, EnsembleQ)
│   ├── actors.py        # Actor networks (Gaussian, Temperature)
│   └── encoders.py      # Visual/State encoders (PlainConv, ResNet)
│
├── buffers/             # 数据缓冲区
│   ├── replay_buffer.py    # Off-policy replay buffers
│   ├── success_buffer.py   # Success-filtered replay buffer
│   ├── rollout_buffer.py   # On-policy rollout buffer (PPO)
│   └── smdp.py             # SMDP cumulative reward computation
│
├── datasets/            # 数据集加载
│   ├── maniskill_dataset.py   # ManiSkill3 HDF5 demo loading
│   ├── carm_dataset.py        # CARM real robot demo loading
│   └── data_utils.py          # Data utilities
│
├── envs/                # 环境工具
│   ├── make_env.py      # Environment factory
│   └── evaluate.py      # Evaluation utilities
│
├── offline/             # 离线训练脚本
│   ├── train_carm.py       # CARM 真实机器人训练
│   └── train_maniskill.py  # ManiSkill 仿真训练
│
├── online/              # 在线训练脚本
│   ├── train_rlpd.py       # RLPD/AWSC 训练 (Off-policy)
│   └── train_reinflow.py   # ReinFlow 训练 (On-policy)
│
├── roboreward/          # RoboReward 标注工具
│
├── tests/               # 测试用例
│
└── utils/               # 通用工具
    ├── checkpoint.py    # 检查点保存/加载
    ├── ema.py           # EMA (Exponential Moving Average)
    └── schedulers.py    # 学习率调度器
```

## 🚀 快速开始

### 安装依赖

```bash
# 创建 conda 环境
conda create -n rlft python=3.10
conda activate rlft

# 安装 PyTorch
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装 ManiSkill3
pip install mani-skill

# 安装其他依赖
pip install tyro diffusers wandb tensorboard h5py
```

### 环境变量

```bash
# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/rl-vla
```

## 📖 使用指南

### 1. 离线模仿学习 (Imitation Learning)

#### ManiSkill 仿真环境

```bash
# Flow Matching
python -m rlft.offline.train_maniskill \
    --env_id LiftPegUpright-v1 \
    --demo_path ~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.state.pd_ee_delta_pose.physx_cuda.h5 \
    --algorithm flow_matching \
    --obs_mode state \
    --total_iters 100000

# ShortCut Flow (few-step sampling)
python -m rlft.offline.train_maniskill \
    --env_id LiftPegUpright-v1 \
    --demo_path ~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --algorithm shortcut_flow \
    --obs_mode rgb \
    --total_iters 100000

# Diffusion Policy
python -m rlft.offline.train_maniskill \
    --env_id PickCube-v1 \
    --algorithm diffusion_policy \
    --num_diffusion_iters 100
```

#### CARM 真实机器人

```bash
# Flow Matching
python -m rlft.offline.train_carm \
    --demo_path ~/recorded_data/pick_place \
    --algorithm flow_matching \
    --total_iters 100000

# ShortCut Flow
python -m rlft.offline.train_carm \
    --demo_path ~/recorded_data/pick_place \
    --algorithm shortcut_flow \
    --max_denoising_steps 8
```

支持的 IL 算法：
| 算法 | `--algorithm` | 描述 |
|------|---------------|------|
| Diffusion Policy | `diffusion_policy` | DDPM-based, 需要多步去噪 |
| Flow Matching | `flow_matching` | ODE-based, 连续时间流 |
| ShortCut Flow | `shortcut_flow` | 快速采样 (1-8步) |
| Consistency Flow | `consistency_flow` | 一致性模型 |
| Reflected Flow | `reflected_flow` | 边界反射处理 |

---

### 2. 离线强化学习 (Offline RL)

```bash
# CPQL (Conservative Policy Q-Learning)
python -m rlft.offline.train_maniskill \
    --env_id LiftPegUpright-v1 \
    --algorithm cpql \
    --obs_mode state \
    --lr_critic 3e-4

# AWCP (Advantage-Weighted Conservative Policy)
python -m rlft.offline.train_maniskill \
    --env_id LiftPegUpright-v1 \
    --algorithm awcp \
    --awac_beta 10.0

# AW-ShortCut Flow
python -m rlft.offline.train_maniskill \
    --env_id LiftPegUpright-v1 \
    --algorithm aw_shortcut_flow \
    --awac_beta 10.0 \
    --shortcut_weight 0.3
```

支持的 Offline RL 算法：
| 算法 | `--algorithm` | 描述 |
|------|---------------|------|
| CPQL | `cpql` | Conservative Q-Learning with Flow Policy |
| AWCP | `awcp` | Advantage-Weighted with Conservative Policy |
| AW-ShortCut | `aw_shortcut_flow` | Q-weighted ShortCut Flow |

---

### 3. 在线强化学习 (Online RL)

#### RLPD (Off-policy, 混合在线/离线数据)

```bash
# SAC (默认)
python -m rlft.online.train_rlpd \
    --env_id PickCube-v1 \
    --demo_path ~/.maniskill/demos/PickCube-v1/trajectory.state.h5 \
    --algorithm sac \
    --obs_mode state \
    --total_timesteps 1000000 \
    --online_ratio 0.5

# AWSC (Advantage-Weighted ShortCut Flow)
python -m rlft.online.train_rlpd \
    --env_id PickCube-v1 \
    --algorithm awsc \
    --pretrain_path runs/shortcut_bc/best.pt \
    --awsc_beta 10.0 \
    --awsc_bc_weight 1.0 \
    --awsc_shortcut_weight 0.3 \
    --total_timesteps 1000000
```

#### ReinFlow (On-policy, PPO + Flow)

```bash
# 从预训练 Flow Matching 模型微调
python -m rlft.online.train_reinflow \
    --env_id PushCube-v1 \
    --pretrained_path runs/flow_matching/checkpoint.pt \
    --obs_mode state \
    --total_updates 10000 \
    --lr 1e-6
```

支持的 Online RL 算法：
| 算法 | 脚本 | 描述 |
|------|------|------|
| SAC | `train_rlpd.py --algorithm sac` | Soft Actor-Critic + Action Chunking |
| AWSC | `train_rlpd.py --algorithm awsc` | Q-weighted ShortCut Flow (RLPD style) |
| ReinFlow | `train_reinflow.py` | PPO + Flow Matching |

---

## 🔧 关键参数说明

### 通用参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--seed` | 1 | 随机种子 |
| `--cuda` | True | 是否使用 GPU |
| `--track` | False | 是否使用 WandB 记录 |
| `--capture_video` | True | 是否录制评估视频 |

### Action Chunking 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--obs_horizon` | 2 | 观测历史长度 |
| `--act_horizon` | 8 | 执行动作长度 |
| `--pred_horizon` | 16 | 预测动作长度 |

### Flow/ShortCut 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--max_denoising_steps` | 8 | ShortCut Flow 最大步数 |
| `--num_inference_steps` | 8 | 推理采样步数 |
| `--shortcut_weight` | 0.3 | ShortCut 一致性损失权重 |

### AWSC 特有参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--awsc_beta` | 10.0 | Advantage weighting 温度 |
| `--awsc_bc_weight` | 1.0 | Flow BC 损失权重 |
| `--awsc_filter_policy_data` | False | 是否过滤低 advantage 样本 |
| `--awsc_advantage_threshold` | 0.0 | Advantage 过滤阈值 |
| `--pretrain_path` | None | 预训练检查点路径 |

---

## 🏗️ 架构设计

### 算法继承关系

```
nn.Module
├── DiffusionPolicyAgent
├── FlowMatchingAgent
│   └── ShortCutFlowAgent
│       ├── ConsistencyFlowAgent
│       └── ReflectedFlowAgent
├── CPQLAgent
│   └── AWCPAgent
│       └── AWShortCutFlowAgent
├── SACAgent
├── ReinFlowAgent
└── AWSCAgent
```

### 网络架构

```
Visual Encoder (PlainConv/ResNet)
        │
        ▼
    obs_features (B, T * feature_dim)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
VelocityUNet1D / ShortCutVelocityUNet1D   Q-Networks (Double/Ensemble)
        │                                  │
        ▼                                  ▼
   actions (B, pred_horizon, act_dim)    Q-values (B, 1)
```

### SMDP (Semi-Markov Decision Process) 公式

对于 action chunk 长度 τ：
- **累积奖励**: $R_t^{(\tau)} = \sum_{i=0}^{\tau-1} \gamma^i r_{t+i}$
- **折扣因子**: $\gamma^\tau$
- **Bellman 方程**: $Q(s_t, a_{t:t+\tau}) = R_t^{(\tau)} + \gamma^\tau (1 - d) Q(s_{t+\tau}, a')$

---

## 🧪 测试

```bash
# 运行所有测试
pytest rlft/tests/ -v

# 运行特定测试
pytest rlft/tests/test_awsc.py -v
pytest rlft/tests/test_awsc_rlpd.py -v
```

---

## 📊 实验结果记录

训练日志保存在 `runs/` 目录：

```
runs/
├── {exp_name}__{timestamp}/
│   ├── config.json          # 训练配置
│   ├── events.out.tfevents  # TensorBoard 日志
│   ├── checkpoints/
│   │   ├── best.pt          # 最佳模型
│   │   ├── step_*.pt        # 定期保存的检查点
│   │   └── final.pt         # 最终模型
│   └── videos/              # 评估视频 (如果 capture_video=True)
```

查看 TensorBoard：
```bash
tensorboard --logdir runs/
```

---

## 📚 参考文献

- **Diffusion Policy**: [Chi et al., RSS 2023](https://diffusion-policy.cs.columbia.edu/)
- **Flow Matching**: [Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747)
- **ShortCut Flow**: [Frans et al., 2024](https://arxiv.org/abs/2410.12557)
- **RLPD**: [Ball et al., ICML 2023](https://arxiv.org/abs/2302.02948)
- **ReinFlow**: [Ding et al., 2024](https://arxiv.org/abs/2402.14262)
- **CPQL**: [Nakamoto et al., ICLR 2024](https://arxiv.org/abs/2310.07297)

---

## 📝 License

MIT License
