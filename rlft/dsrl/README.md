# DSRL: Dual-Stage Reinforcement Learning with Latent Steering

DSRL 实现了一种两阶段强化学习方法，通过在 latent（噪声）空间进行 steering 来改进预训练的 ShortCut Flow 策略，同时保持 flow 模型完全冻结。

## 核心思想

1. **保持 Flow 策略冻结**：预训练的 ShortCut Flow 模型不参与任何梯度更新
2. **Latent Steering**：训练一个轻量级的 `π_w(w | obs)` 来 steer flow 采样的初始噪声
3. **两阶段训练**：
   - **阶段1（离线）**：Advantage-Weighted MLE，利用 Q 网络蒸馏"好噪声"的偏好
   - **阶段2（在线）**：Macro-step PPO，在真实环境中继续优化 latent policy

## 文件结构

```
rlft/dsrl/
├── __init__.py                 # 模块导出
├── README.md                   # 本文档
├── latent_policy.py            # LatentGaussianPolicy: latent 空间的高斯策略
├── dsrl_agent.py               # DSRLAgent: 主 agent 类，整合所有组件
├── value_network.py            # ValueNetwork: 在线 PPO 的价值网络
├── macro_rollout_buffer.py     # MacroRolloutBuffer: macro-step 的 rollout buffer
├── train_latent_offline.py     # 阶段1：离线 AW-MLE 训练脚本（tyro CLI）
└── train_latent_online.py      # 阶段2：在线 PPO 训练脚本（ManiSkill 集成）
```

## 依赖的 diffusion_policy 组件

本模块复用了 `rlft/diffusion_policy` 中的以下组件：

- `ShortCutVelocityUNet1D`: 带 step size conditioning 的速度网络
- `EnsembleQNetwork`: 集成 Q 网络，支持 `get_ucb_q(kappa)` 方法
- `soft_update`: 目标网络软更新

## 使用方法

### 1. 准备预训练模型

需要以下预训练模型：
- AW-ShortCut Flow checkpoint（包含 velocity network + Q network）

训练脚本支持统一的 checkpoint 格式，兼容 AWShortCutFlowAgent 保存的检查点。

### 2. 阶段1：离线 AW-MLE 训练

```bash
cd rlft/dsrl

# 实际训练命令示例（与 AWSC 训练参数对齐）
CUDA_VISIBLE_DEVICES=0 python train_latent_offline.py \
    --env_id LiftPegUpright-v1 \
    --awsc_checkpoint checkpoints/best_eval_success_once.pt \
    --demo_path ~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --obs_mode rgb \
    --control_mode pd_ee_delta_pose \
    --sim_backend physx_cuda \
    --num_demos 1000 \
    --seed 0 \
    --total_iters 30000 \
    --eval_freq 2000 \
    --log_freq 100 \
    --num_eval_episodes 100 \
    --num_eval_envs 25 \
    --exp_name dsrl-stage1-LiftPegUpright-v1-seed0 \
    --track \
    --wandb_project_name maniskill_dsrl
```

主要参数说明：
- `--env_id`: ManiSkill 环境 ID
- `--awsc_checkpoint`: AWShortCutFlow 预训练检查点路径
- `--demo_path`: 离线演示数据路径（HDF5 格式，.h5）
- `--num_candidates`: 每个 observation 采样的 latent 候选数
- `--ucb_kappa`: UCB 探索系数（μ - κσ）
- `--advantage_temperature`: softmax advantage 温度

训练过程：
1. 对每个 observation，采样 M 个 latent candidates 从 prior N(0, I)
2. 通过冻结的 flow 模型生成 actions
3. 用 ensemble Q 打分（UCB: μ - κσ）
4. 计算 softmax advantage weights
5. 用 weighted MLE 训练 latent policy

### 3. 阶段2：在线 PPO 训练

```bash
# 实际训练命令示例（与 RLPD 训练参数对齐）
CUDA_VISIBLE_DEVICES=0 python train_latent_online.py \
    --env_id LiftPegUpright-v1 \
    --awsc_checkpoint checkpoints/best_eval_success_once.pt \
    --stage1_checkpoint runs/dsrl-stage1-LiftPegUpright-v1-seed0/checkpoints/best_eval_success_once.pt \
    --obs_mode rgb \
    --control_mode pd_ee_delta_pose \
    --sim_backend physx_cuda \
    --num_envs 50 \
    --num_eval_envs 25 \
    --gamma 0.9 \
    --total_timesteps 500000 \
    --batch_size 256 \
    --num_eval_episodes 100 \
    --max_episode_steps 100 \
    --lr 3e-4 \
    --value_lr 1e-3 \
    --seed 0 \
    --track \
    --wandb_project_name maniskill_dsrl \
    --exp_name dsrl-stage2-LiftPegUpright-v1-seed0
```

主要参数说明：
- `--checkpoint`: Stage 1 训练完成的检查点路径
- `--num_envs`: 并行环境数量
- `--rollout_steps`: 每次迭代收集的步数
- `--ppo_epochs`: PPO 更新轮数
- `--prior_mix_ratio`: prior mixing 比例

训练过程：
1. 从 latent policy 采样 w（with prior mixing）
2. 生成 action chunk 并执行 act_horizon 步
3. 收集 macro-step 轨迹
4. 计算 GAE advantage（γ_macro = γ^act_horizon）
5. PPO 更新 latent policy + value network

## 关键超参数

### 阶段1

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_candidates` | 16 | M: 采样的 latent 候选数 |
| `kappa` | 1.0 | UCB 系数 (μ - κσ) |
| `tau` | 5.0 | Soft baseline 温度 |
| `beta_latent` | 1.0 | Advantage weighting 温度 |
| `kl_coef` | 1e-3 | KL-to-prior 正则系数 |

### 阶段2

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ppo_clip` | 0.2 | PPO clip ratio |
| `entropy_coef` | 1e-3 | 熵奖励系数 |
| `prior_mix_ratio` | 0.3 | 初始 prior 混合比例 |
| `prior_mix_decay` | 0.995 | 混合比例衰减率 |
| `gamma_macro` | γ^8 | Macro-step 折扣因子 |

## API 参考

### LatentGaussianPolicy

```python
from dsrl import LatentGaussianPolicy

policy = LatentGaussianPolicy(
    obs_dim=512,
    pred_horizon=16,
    action_dim=7,
    steer_mode="full",      # 或 "act_horizon"
    act_horizon=8,
)

# 采样
w, log_prob = policy.sample(obs_cond, deterministic=False)

# 计算 KL 散度
kl = policy.kl_to_prior(obs_cond)
```

### DSRLAgent

```python
from dsrl import DSRLAgent

agent = DSRLAgent(
    velocity_net=velocity_net,
    q_network=q_network,
    latent_policy=latent_policy,
    value_network=value_network,
    ...
)

# 阶段1：离线损失
loss_dict = agent.compute_offline_loss(obs_cond)

# 阶段2：PPO 损失
loss_dict = agent.compute_ppo_loss(obs_cond, latent_w, old_log_prob, advantage, returns)

# 推理
actions, w, log_prob = agent.get_action(obs_features, use_latent_policy=True)

# 从指定 latent 采样（核心接口）
actions = agent.sample_actions_from_latent(obs_cond, w, use_ema=True)
```

## 与 AW-ShortCut Flow 的关系

DSRL 是 AW-ShortCut Flow 的自然延续：

1. **AW-ShortCut Flow**（阶段0）：
   - 训练 flow policy + Q-network
   - Q 用于 weight BC samples

2. **DSRL Stage 1**：
   - 冻结 flow + Q
   - 训练 latent policy 来 steer 噪声

3. **DSRL Stage 2**：
   - 冻结 flow（Q 可选冻结）
   - 在线 PPO 优化 latent policy

## 扩展

### 只 steering act_horizon

设置 `--steer_mode act_horizon` 只学习前 act_horizon 步的噪声：

```bash
python train_latent_offline.py \
    --steer_mode act_horizon \
    --act_horizon 8 \
    ...
```

这样 latent policy 只输出 (act_horizon, action_dim) 的噪声，其余步骤使用 prior。

### ManiSkill 环境支持

`train_latent_online.py` 已完整集成 ManiSkill 环境：

```bash
# 支持的环境示例
python train_latent_online.py --env_id LiftPegUpright-v1 ...
python train_latent_online.py --env_id StackCube-v1 ...
python train_latent_online.py --env_id PegInsertionSide-v1 ...
```

使用 `FlattenRGBObservationWrapper` 处理 RGB 观测，`SMDPChunkCollector` 处理 action chunk 的 SMDP 奖励计算。
