"""
DSRL Official Training Script for ManiSkill3

从官方 ajwagen/dsrl/train_dsrl.py 移植，适配 ManiSkill3 和 ShortCut Flow。

支持两种算法:
- DSRL-SAC: 使用环境包装器，SAC 在噪声空间操作
- DSRL-NA: 使用策略内部采样，蒸馏 Q^W

Usage:
    # DSRL-SAC (推荐先验证此模式)
    python train_dsrl.py --algorithm dsrl_sac --env_id LiftPegUpright-v1
    
    # DSRL-NA (更高样本效率)
    python train_dsrl.py --algorithm dsrl_na --env_id LiftPegUpright-v1

Reference: https://github.com/ajwagen/dsrl/blob/main/train_dsrl.py
"""

import os
import sys
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro

warnings.filterwarnings("ignore")

# 添加路径
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))  # 添加 rlft 目录以支持 dsrl_official 导入
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl"))
sys.path.insert(0, str(_root / "dsrl_offpolicy"))

# 导入 ManiSkill3
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

# 导入本地模块
from dsrl_official.utils import (
    ShortCutFlowWrapper,
    load_shortcut_flow_policy,
    collect_rollouts,
    load_offline_data,
)
from dsrl_official.env_utils import (
    ShortCutFlowEnvWrapper,
    ManiSkillGPUFlowEnvWrapper,
    ActionChunkWrapper,
    make_dsrl_env,
)
from dsrl_official.callbacks import LoggingCallback, ManiSkillEvalCallback

# 导入 diffusion_policy 组件
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.evaluate import evaluate
from diffusion_policy.utils import AgentWrapper, encode_observations
from diffusion_policy.plain_conv import PlainConv

# 尝试导入 SB3
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. Only custom training loop available.")

# 尝试导入官方 DSRL (SB3 fork)
try:
    from stable_baselines3 import DSRL
    HAS_DSRL = True
except ImportError:
    HAS_DSRL = False
    print("Warning: Official DSRL algorithm not found. Use standard SAC for dsrl_sac mode.")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class Args:
    """DSRL 训练参数 - 对齐官方配置格式"""
    
    # ===== 实验设置 =====
    exp_name: Optional[str] = None
    """实验名称"""
    seed: int = 1
    """随机种子"""
    cuda: bool = True
    """使用 CUDA"""
    track: bool = True
    """使用 wandb 跟踪"""
    wandb_project: str = "maniskill_dsrl_official"
    """wandb 项目名"""
    wandb_entity: Optional[str] = None
    """wandb 实体"""
    wandb_group: str = "dsrl_official"
    """wandb 分组"""
    
    # ===== 算法设置 =====
    algorithm: str = "dsrl_sac"
    """算法: 'dsrl_sac' 或 'dsrl_na'"""
    
    # ===== 环境设置 =====
    env_id: str = "LiftPegUpright-v1"
    """环境 ID"""
    n_envs: int = 16
    """训练环境数量"""
    n_eval_envs: int = 50
    """评估环境数量"""
    max_episode_steps: int = 100
    """最大 episode 步数"""
    control_mode: str = "pd_ee_delta_pose"
    """控制模式"""
    obs_mode: str = "rgb"
    """观察模式: 'state' 或 'rgb'"""
    sim_backend: str = "physx_cuda"
    """仿真后端"""
    reward_mode: str = "dense"
    """奖励模式"""
    
    # ===== 预训练 Checkpoint =====
    awsc_checkpoint: str = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
    """预训练 AW-ShortCut Flow checkpoint"""
    use_ema: bool = True
    """使用 EMA 权重"""
    
    # ===== 训练超参数 (官方推荐) =====
    total_timesteps: int = 20_000_000
    """总训练步数"""
    actor_lr: float = 3e-4
    """Actor 学习率"""
    critic_lr: float = 3e-4
    """Critic 学习率"""
    batch_size: int = 256
    """批大小"""
    buffer_size: int = 10_000_000
    """Replay buffer 大小"""
    gamma: float = 0.99
    """折扣因子"""
    tau: float = 0.005
    """Target 网络软更新率"""
    utd: int = 20
    """Update-To-Data ratio (每步梯度更新次数)"""
    train_freq: int = 1
    """训练频率"""
    ent_coef: float = -1
    """熵系数 (-1 表示自动调节)"""
    target_ent: float = -1
    """目标熵 (-1 表示自动)"""
    
    # ===== 网络架构 (官方推荐) =====
    num_layers: int = 3
    """网络层数"""
    layer_size: int = 2048
    """每层宽度"""
    n_critics: int = 2
    """Critic 数量"""
    use_layer_norm: bool = True
    """使用 LayerNorm"""
    
    # ===== Action/Observation 设置 =====
    obs_horizon: int = 2
    """观察历史长度"""
    pred_horizon: int = 16
    """预测动作序列长度"""
    act_steps: int = 8
    """动作执行步数 (action horizon)"""
    action_dim: int = 7
    """动作维度"""
    action_magnitude: float = 1.5
    """噪声范围 [-mag, +mag]"""
    
    # ===== 视觉编码器 =====
    visual_feature_dim: int = 256
    """视觉特征维度"""
    diffusion_step_embed_dim: int = 64
    """Diffusion 步嵌入维度"""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """UNet 通道维度"""
    n_groups: int = 8
    """GroupNorm 组数"""
    
    # ===== DSRL-NA 特有参数 =====
    noise_critic_grad_steps: int = 1
    """Q^W 梯度步数"""
    critic_backup_combine_type: str = "min"
    """Critic backup 组合类型"""
    
    # ===== 数据加载 =====
    load_offline_data: bool = False
    """是否加载离线数据"""
    offline_data_path: str = ""
    """离线数据路径"""
    init_rollout_steps: int = 0
    """初始 rollout 收集步数"""
    
    # ===== 日志和评估 =====
    log_interval: int = 1000
    """日志间隔"""
    eval_interval: int = 10000
    """评估间隔"""
    num_evals: int = 50
    """每次评估 episode 数"""
    save_model_interval: int = 100000
    """模型保存间隔"""
    save_replay_buffer: bool = False
    """是否保存 replay buffer"""
    deterministic_eval: bool = True
    """使用确定性评估"""
    capture_video: bool = True
    """录制视频"""


def make_maniskill_env(args: Args, is_eval: bool = False):
    """创建 ManiSkill3 环境。
    
    Args:
        args: 训练参数
        is_eval: 是否为评估环境
        
    Returns:
        env: 环境实例
    """
    n_envs = args.n_eval_envs if is_eval else args.n_envs
    
    env_kwargs = dict(
        obs_mode="rgbd" if "rgb" in args.obs_mode else "state",
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        num_envs=n_envs,
        reward_mode=args.reward_mode,
    )
    
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    
    env = gym.make(args.env_id, **env_kwargs)
    
    if "rgb" in args.obs_mode:
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    
    return env


def main():
    args = tyro.cli(Args)
    
    # 生成实验名称
    if args.exp_name is None:
        args.exp_name = f"dsrl_official-{args.algorithm}-{args.env_id}-seed{args.seed}"
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # 创建日志目录
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 初始化 wandb
    if args.track and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group,
            config=vars(args),
            save_code=True,
        )
    
    # ===== 计算维度 =====
    include_rgb = "rgb" in args.obs_mode
    
    # state_dim 将从 checkpoint 推断，避免创建临时环境导致 GPU PhysX 冲突
    # load_shortcut_flow_policy 会自动从 checkpoint 推断 state_dim
    
    # ===== 加载 Base Policy =====
    print("Loading ShortCut Flow base policy...")
    base_policy, visual_encoder = load_shortcut_flow_policy(
        checkpoint_path=args.awsc_checkpoint,
        visual_encoder_class=PlainConv if include_rgb else None,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_dim=args.action_dim,
        visual_feature_dim=args.visual_feature_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        unet_dims=tuple(args.unet_dims),
        n_groups=args.n_groups,
        state_dim=None,  # 从 checkpoint 自动推断
        include_rgb=include_rgb,
        use_ema=args.use_ema,
        device=device,
    )
    print("Base policy loaded successfully!")
    
    # 从 checkpoint 推断 state_dim
    # load_shortcut_flow_policy 会返回推断的 state_dim
    # 这里需要重新加载以获取 state_dim，或者在上面的调用中返回
    # 暂时使用固定值（从之前分析确认的值）
    state_dim = 25  # 从 checkpoint 推断得到 (cond_encoder 维度分析)
    visual_dim = args.visual_feature_dim if include_rgb else 0
    obs_dim = args.obs_horizon * (visual_dim + state_dim)
    single_obs_dim = visual_dim + state_dim  # 单帧观察维度
    
    MAX_STEPS = int(args.max_episode_steps / args.act_steps)
    
    print(f"\n{'='*50}")
    print(f"DSRL Official Training")
    print(f"{'='*50}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Environment: {args.env_id}")
    print(f"Obs dim: {obs_dim}")
    print(f"Action dim: {args.action_dim}")
    print(f"Action magnitude: {args.action_magnitude}")
    print(f"UTD ratio: {args.utd}")
    print(f"{'='*50}\n")
    
    # ===== 创建环境 =====
    print("Creating training environments...")
    train_env = make_maniskill_env(args, is_eval=False)
    
    print("Creating evaluation environments...")
    eval_env = make_maniskill_env(args, is_eval=True)
    
    # ===== 根据算法配置训练 =====
    if args.algorithm == "dsrl_sac":
        if not HAS_SB3:
            raise RuntimeError("DSRL-SAC requires stable-baselines3. Please install it.")
        
        print("\n=== DSRL-SAC Mode ===")
        print("Wrapping environment with ShortCutFlowEnvWrapper...")
        
        # 包装环境 (噪声空间作为动作空间)
        # 使用工厂函数 make_dsrl_env，它会自动处理 visual_encoder 和观察历史
        train_env = make_dsrl_env(
            env=train_env,
            base_policy=base_policy,
            visual_encoder=visual_encoder,
            action_magnitude=args.action_magnitude,
            act_steps=args.act_steps,
            action_dim=args.action_dim,
            state_dim=state_dim,
            visual_feature_dim=args.visual_feature_dim,
            obs_horizon=args.obs_horizon,
            include_rgb=include_rgb,
            use_gpu_env=False,  # 使用 VecEnv wrapper
            device=device,
        )
        
        eval_env = make_dsrl_env(
            env=eval_env,
            base_policy=base_policy,
            visual_encoder=visual_encoder,
            action_magnitude=args.action_magnitude,
            act_steps=args.act_steps,
            action_dim=args.action_dim,
            state_dim=state_dim,
            visual_feature_dim=args.visual_feature_dim,
            obs_horizon=args.obs_horizon,
            include_rgb=include_rgb,
            use_gpu_env=False,
            device=device,
        )
        
        # 配置网络架构
        net_arch = [args.layer_size] * args.num_layers
        
        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, qf=net_arch),
            activation_fn=torch.nn.Tanh,  # 官方推荐
            n_critics=args.n_critics,
        )
        # 注意: post_linear_modules 是官方 DSRL fork 特有的，标准 SB3 不支持
        # 如果需要 LayerNorm，需要安装官方 fork
        
        # 创建 SAC 模型
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=args.actor_lr,
            buffer_size=args.buffer_size,
            learning_starts=1,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.utd,
            action_noise=None,
            optimize_memory_usage=False,
            ent_coef="auto" if args.ent_coef == -1 else args.ent_coef,
            target_update_interval=1,
            target_entropy="auto" if args.target_ent == -1 else args.target_ent,
            tensorboard_log=log_dir,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=device,
        )
        
    elif args.algorithm == "dsrl_na":
        if not HAS_DSRL:
            raise RuntimeError(
                "DSRL-NA requires the official DSRL fork of stable-baselines3. "
                "Please install from: https://github.com/ajwagen/stable-baselines3"
            )
        
        print("\n=== DSRL-NA Mode ===")
        print("Using policy-internal sampling with Q^W distillation...")
        
        # 配置网络架构
        net_arch = [args.layer_size] * args.num_layers
        post_linear_modules = [torch.nn.LayerNorm] if args.use_layer_norm else None
        
        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, qf=net_arch),
            activation_fn=torch.nn.Tanh,
            log_std_init=0.0,
            n_critics=args.n_critics,
        )
        if post_linear_modules:
            policy_kwargs['post_linear_modules'] = post_linear_modules
        
        # 创建 DSRL 模型
        model = DSRL(
            "MlpPolicy",
            train_env,
            learning_rate=args.actor_lr,
            buffer_size=args.buffer_size,
            learning_starts=1,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.utd,
            action_noise=None,
            optimize_memory_usage=False,
            ent_coef="auto" if args.ent_coef == -1 else args.ent_coef,
            target_update_interval=1,
            target_entropy="auto" if args.target_ent == -1 else args.target_ent,
            tensorboard_log=log_dir,
            verbose=1,
            policy_kwargs=policy_kwargs,
            diffusion_policy=base_policy,
            diffusion_act_dim=(args.act_steps, args.action_dim),
            noise_critic_grad_steps=args.noise_critic_grad_steps,
            critic_backup_combine_type=args.critic_backup_combine_type,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # ===== 设置回调 =====
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_model_interval,
        save_path=f"{log_dir}/checkpoints/",
        name_prefix="dsrl_policy",
        save_replay_buffer=args.save_replay_buffer,
        save_vecnormalize=True,
    )
    
    logging_callback = LoggingCallback(
        action_chunk=args.act_steps,
        eval_episodes=int(args.num_evals / args.n_eval_envs),
        log_freq=MAX_STEPS,
        use_wandb=args.track,
        eval_env=eval_env,
        eval_freq=args.eval_interval,
        num_train_env=args.n_envs,
        num_eval_env=args.n_eval_envs,
        rew_offset=0,  # ManiSkill 不需要 reward offset
        algorithm=args.algorithm,
        max_steps=MAX_STEPS,
        deterministic_eval=args.deterministic_eval,
    )
    
    # ===== 初始评估 =====
    print("\nRunning initial evaluation...")
    logging_callback.evaluate(model, deterministic=False)
    if args.deterministic_eval:
        logging_callback.evaluate(model, deterministic=True)
    logging_callback.log_count += 1
    
    # ===== 可选: 加载离线数据 =====
    if args.load_offline_data and args.offline_data_path:
        print(f"\nLoading offline data from {args.offline_data_path}...")
        load_offline_data(model, args.offline_data_path, args.n_envs)
    
    # ===== 可选: 初始 rollout 收集 =====
    if args.init_rollout_steps > 0:
        print(f"\nCollecting initial rollouts ({args.init_rollout_steps} steps)...")
        collect_rollouts(
            model, train_env, args.init_rollout_steps, base_policy,
            algorithm=args.algorithm,
            action_magnitude=args.action_magnitude,
            act_steps=args.act_steps,
            action_dim=args.action_dim,
            n_envs=args.n_envs,
            device=device,
        )
        logging_callback.set_timesteps(args.init_rollout_steps * args.n_envs)
    
    # ===== 训练 =====
    print(f"\n{'='*50}")
    print("Starting training...")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"{'='*50}\n")
    
    callbacks = [checkpoint_callback, logging_callback]
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )
    
    # ===== 保存最终模型 =====
    model.save(f"{log_dir}/checkpoints/final")
    print(f"\nTraining complete! Model saved to {log_dir}/checkpoints/final")
    
    # ===== 清理 =====
    train_env.close()
    eval_env.close()
    
    if args.track and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
