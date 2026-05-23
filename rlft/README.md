# RLFT: Reinforcement Learning and Flow-based Training

<p align="center">
  <b>A unified framework for robot learning with diffusion/flow policies</b>
</p>

RLFT 是一个统一的机器人学习框架，支持：
- **Imitation Learning (IL)**: Diffusion Policy, Flow Matching, ShortCut Flow, Consistency Flow
- **Offline Reinforcement Learning**: CPQL, AWCP, AW-ShortCut Flow
- **Online Reinforcement Learning**: SAC, RLPD, AWSC

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
│   ├── rollout_buffer.py   # SMDP chunk reward collector
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
├── offline/             # 离线训练与评估脚本
│   ├── train_carm.py       # CARM 真实机器人训练
│   ├── eval_carm.py        # CARM 离线评估 (模型性能验证)
│   └── train_maniskill.py  # ManiSkill 仿真训练
│
├── online/              # 在线训练脚本
│   └── train_rlpd.py       # RLPD/AWSC 训练 (Off-policy)
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

### 方式一：一键运行（推荐）

我们提供了完整的自动化脚本，一键完成环境配置、数据下载、数据预处理和训练：

```bash
# 快速验证 (5000 步，验证全流程)
bash scripts/run_full_pipeline.sh --quick

# 完整训练 (4万步)
bash scripts/run_full_pipeline.sh --full

# 指定任务
bash scripts/run_full_pipeline.sh --quick --task PickCube-v1

# 跳过已完成的步骤
bash scripts/run_full_pipeline.sh --full --skip-env --skip-download
```

### 方式二：分步运行

#### Step 1: 配置环境

```bash
# 创建名为 'maniskill' 的 conda 环境
bash scripts/setup_maniskill_env.sh

# 激活环境
conda activate maniskill
```

#### Step 2: 下载演示数据

```bash
# 下载 LiftPegUpright-v1 任务的演示
bash scripts/download_demos.sh LiftPegUpright-v1

# 下载多个任务
bash scripts/download_demos.sh LiftPegUpright-v1 PickCube-v1 PushCube-v1
```

#### Step 3: Replay 生成训练数据

```bash
# Replay 生成 RGB 和 State 两种观测模式的数据集
# 使用 physx_cuda 后端，保存 sparse 奖励
bash scripts/replay_demos.sh LiftPegUpright-v1

# 指定控制模式和并行环境数
bash scripts/replay_demos.sh LiftPegUpright-v1 pd_ee_delta_pose 64
```

#### Step 4: 批量训练

```bash
# 快速验证所有算法 (5000 步)
bash scripts/run_all_algorithms.sh --quick

# 完整训练所有算法 (100万步)
bash scripts/run_all_algorithms.sh --full

# 指定 GPU 和算法
bash scripts/run_all_algorithms.sh --quick --gpus 0,1,2,3 --algorithms flow_matching,cpql

# 只训练 RGB 观测
bash scripts/run_all_algorithms.sh --quick --obs-mode rgb

# 预览命令（不执行）
bash scripts/run_all_algorithms.sh --quick --dry-run
```

#### Step 5: 监控训练

```bash
# 启动监控界面
bash scripts/monitor_training.sh logs/training_<timestamp>

# 查看 GPU 使用
watch -n 1 nvidia-smi

# 查看单个任务日志
tail -f logs/training_<timestamp>/LiftPegUpright-v1_flow_matching_rgb.log

# 终止所有训练
pkill -f 'rlft.offline.train_maniskill'
```

### 手动安装依赖

```bash
# 创建 conda 环境
conda create -n maniskill python=3.10
conda activate maniskill

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装 ManiSkill3
pip install mani-skill

# 安装其他依赖
pip install tyro diffusers wandb tensorboard h5py einops scikit-learn opencv-python
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

# Consistency Policy
python -m rlft.offline.train_carm \
    --algorithm consistency_flow \
    --exp_name consistency_flow_resnet10_discrete_gripper_weight_0.02_state_joint_action_full_norm_v2 \ --precompute_actions \
    --visual_encoder_type resnet10 \
    --gripper_ce_weight 0.02 \
    --state_mode joint_only \
    --action_mode full \
    --normalize_actions \
    --total_iters 200000
```

#### CARM 离线评估

使用录制的数据集评估已训练模型的性能（不需要机械臂），对比预测动作与 Ground Truth：

```bash
# 标准评估
python -m rlft.offline.eval_carm \
    --model_path runs/consistency_flow/checkpoints/final.pt \
    --data_dir ~/rl-vla/recorded_data/mix \
    --output_dir offline_results

# EMA vs Non-EMA 模型对比
python -m rlft.offline.eval_carm \
    --model_path runs/consistency_flow/checkpoints/final.pt \
    --data_dir ~/rl-vla/recorded_data/mix \
    --compare_ema

# 指定推理步数和 episode 数
python -m rlft.offline.eval_carm \
    --model_path runs/consistency_flow/checkpoints/final.pt \
    --data_dir ~/rl-vla/recorded_data/mix \
    --num_inference_steps 10 \
    --num_episodes 5
```

评估参数：
| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--model_path` | (必须) | 模型 checkpoint 路径 |
| `--data_dir` | `~/rl-vla/recorded_data` | 数据集目录 |
| `--output_dir` | `offline_results` | 输出目录（图表 + 指标） |
| `--use_ema` | False | 使用 EMA 模型推理 |
| `--compare_ema` | False | EMA vs Non-EMA 对比测试 |
| `--num_episodes` | all | 评估的 episode 数量 |
| `--num_inference_steps` | None | 推理步数（默认使用模型配置） |

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

支持的 Online RL 算法：
| 算法 | 脚本 | 描述 |
|------|------|------|
| SAC | `train_rlpd.py --algorithm sac` | Soft Actor-Critic + Action Chunking |
| AWSC | `train_rlpd.py --algorithm awsc` | Q-weighted ShortCut Flow (RLPD style) |

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
- **CPQL**: [Nakamoto et al., ICLR 2024](https://arxiv.org/abs/2310.07297)

---

## 📝 License

MIT License
