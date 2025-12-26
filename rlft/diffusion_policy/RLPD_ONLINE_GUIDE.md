# RLPD Online Training - 超参数配置指南

基于官方 RLPD 论文 (Ball et al., ICML 2023) 和 ManiSkill RLPD 实现。

## 核心发现

RLPD 论文的关键发现：
1. **Ensemble Q + Subsample Min**: 使用 10 个 Q 网络，每次随机选择 2 个取 min，大幅提升样本效率
2. **Critic Layer Norm**: 在 Q 网络中使用 Layer Normalization 提升稳定性
3. **backup_entropy=False**: 对于稀疏奖励任务，不在 Q-target 中加入熵项
4. **50/50 数据混合**: 50% 在线数据 + 50% 离线数据的混合比例效果最好

## 推荐默认参数

### 1. 基础 SAC 参数 (对齐官方)

```python
# 学习率
lr_actor: float = 3e-4
lr_critic: float = 3e-4
lr_temp: float = 3e-4

# 折扣因子
gamma: float = 0.9  # ManiSkill 短 episode 推荐 0.9, 长 episode 用 0.99

# 软更新
tau: float = 0.005

# 熵相关
init_temperature: float = 1.0
target_entropy: -action_dim * action_horizon  # 自动计算
backup_entropy: bool = False  # 重要！稀疏奖励必须 False
```

### 2. Ensemble Q 参数

```python
# Sample Efficient 配置 (推荐)
num_qs: int = 10         # Q 网络数量
num_min_qs: int = 2      # 取 min 的网络数量

# Walltime Efficient 配置 (更快但样本效率稍低)
num_qs: int = 2
num_min_qs: int = 2
```

### 3. 更新频率 (UTD Ratio)

```python
# Sample Efficient: 高 UTD，少量环境
utd_ratio: int = 16-20   # 每步多次梯度更新
num_envs: int = 1-8

# Walltime Efficient: 低 UTD，大量环境并行
utd_ratio: int = 4       # 每步少量梯度更新
num_envs: int = 32-50
```

### 4. 网络架构

```python
# Q 网络隐藏层
q_hidden_dims: List[int] = [256, 256, 256]  # 3 层 MLP

# 使用 Layer Norm (提升稳定性)
critic_layer_norm: bool = True  # 在 Q 网络中启用
```

### 5. 离线数据混合

```python
online_ratio: float = 0.5  # 50% 在线 + 50% 离线
```

## 两种配置模式

### Walltime Efficient (快速训练)

适用于：有大量 GPU 并行能力，想快速看到结果

```bash
./sweep_rlpd_online_parallel.sh --mode walltime --gpus "0 1 2 3" --env LiftPegUpright-v1
```

参数设置：
- `num_envs=50`: 大量并行环境
- `num_qs=2`: 少量 Q 网络
- `utd_ratio=4`: 较低更新频率
- 预计样本量：200K-500K steps

### Sample Efficient (样本高效)

适用于：样本宝贵，愿意用更长时间换取更少样本

```bash
./sweep_rlpd_online_parallel.sh --mode sample --gpus "0 1" --env LiftPegUpright-v1
```

参数设置：
- `num_envs=8`: 少量并行环境
- `num_qs=10`: REDQ 风格的大 ensemble
- `utd_ratio=16-20`: 高更新频率
- 预计样本量：50K-100K steps

## 按任务类型调整

### 简单任务 (PickCube, LiftPegUpright)
```python
gamma = 0.9
total_timesteps = 200_000
max_episode_steps = 100
```

### 中等任务 (StackCube, PegInsertion)
```python
gamma = 0.95
total_timesteps = 500_000
max_episode_steps = 150
```

### 困难任务 (Complex assembly)
```python
gamma = 0.99
total_timesteps = 1_000_000
max_episode_steps = 200
```

## 使用示例

### 1. 快速测试 (walltime 模式)
```bash
# 使用 GPU 0,1 进行 walltime efficient sweep
./sweep_rlpd_online_parallel.sh \
    --mode walltime \
    --gpus "0 1" \
    --env LiftPegUpright-v1 \
    --total-timesteps 200000

# 监控进度
./monitor_rlpd_online.sh --mode walltime
```

### 2. 完整 sweep (sample 模式)
```bash
# 使用 GPU 0,1,2,3 进行 sample efficient sweep
./sweep_rlpd_online_parallel.sh \
    --mode sample \
    --gpus "0 1 2 3" \
    --env LiftPegUpright-v1 \
    --total-timesteps 500000 \
    --wandb-project my_rlpd_sweep

# 监控进度
./monitor_rlpd_online.sh --mode sample
```

### 3. AWSC 算法 (带预训练模型)
```bash
./sweep_rlpd_online_parallel.sh \
    --algorithm awsc \
    --pretrained-path runs/awsc-LiftPegUpright-v1-seed0/checkpoints/best_eval_success_once.pt \
    --mode walltime \
    --gpus "0 1"
```

### 4. 单次训练 (调试)
```bash
CUDA_VISIBLE_DEVICES=0 python train_rlpd_online.py \
    --algorithm sac \
    --env_id LiftPegUpright-v1 \
    --obs_mode rgb \
    --control_mode pd_ee_delta_pose \
    --sim_backend physx_cuda \
    --num_envs 50 \
    --utd_ratio 4 \
    --num_qs 10 \
    --num_min_qs 2 \
    --gamma 0.9 \
    --backup_entropy False \
    --total_timesteps 200000 \
    --seed 0
```

## 关键参数敏感度

| 参数 | 敏感度 | 建议范围 | 说明 |
|------|--------|---------|------|
| `gamma` | 高 | 0.9-0.99 | 短 episode 用 0.9，长用 0.99 |
| `backup_entropy` | 高 | False | 稀疏奖励必须 False |
| `num_qs` | 中 | 2-10 | 越多越样本高效，但训练更慢 |
| `utd_ratio` | 中 | 4-20 | 越高越样本高效，但训练更慢 |
| `num_envs` | 低 | 8-50 | 根据 GPU 内存调整 |
| `lr` | 低 | 3e-4 | 通常不需要调整 |
| `batch_size` | 低 | 256 | 通常不需要调整 |

## 参考资料

- [RLPD 论文](https://arxiv.org/abs/2302.02948): Efficient Online Reinforcement Learning with Offline Data
- [官方 RLPD 代码](https://github.com/ikostrikov/rlpd)
- [ManiSkill RLPD 实现](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/rlpd)
- [REDQ 论文](https://arxiv.org/abs/2101.05982): 解释 ensemble Q + subsample min
