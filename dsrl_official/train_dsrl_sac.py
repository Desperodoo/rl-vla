"""
DSRL-SAC Training Script for ManiSkill3

使用环境包装器模式，SAC 在噪声空间操作。
动作空间为 [−action_magnitude, +action_magnitude]^{act_steps * action_dim}

参考: https://github.com/ajwagen/dsrl/blob/main/train_dsrl.py

Usage:
    python train_dsrl_sac.py --total_timesteps 1000000
    
    # 自定义参数
    python train_dsrl_sac.py --n_envs 100 --utd 40 --seed 42
    
    # 使用 wandb 跟踪
    python train_dsrl_sac.py --track --wandb_project my_project
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
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl_offpolicy"))
sys.path.insert(0, str(_root / "dsrl_official"))

import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.make_env import make_eval_envs
from dsrl_official.utils import ShortCutFlowWrapper
from dsrl_official.env_utils import ManiSkillGPUFlowEnvWrapper

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ========== Wandb Logging Callback ==========
class WandbCallback(BaseCallback):
    """
    自定义 Callback，将 SB3 训练指标记录到 Wandb
    
    记录的指标：
    - rollout: ep_rew_mean, ep_len_mean
    - train: actor_loss, critic_loss, ent_coef, ent_coef_loss, learning_rate, n_updates
    - time: fps, total_timesteps
    - buffer: replay_buffer_size, replay_buffer_pos
    """
    
    def __init__(self, verbose: int = 0, log_freq: int = 100):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._last_log_step = 0
        self._episode_rewards = []
        self._episode_lengths = []
        self._n_episodes = 0
        
    def _on_step(self) -> bool:
        # 收集 episode 统计
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])
                self._n_episodes += 1
        
        # 周期性记录
        if self.num_timesteps - self._last_log_step >= self.log_freq:
            self._log_metrics()
            self._last_log_step = self.num_timesteps
            
        return True
    
    def _on_rollout_end(self) -> None:
        """在每次 rollout 结束时记录额外信息"""
        pass
    
    def _log_metrics(self):
        """记录训练指标到 wandb"""
        if not HAS_WANDB or wandb.run is None:
            return
            
        metrics = {}
        
        # 1. 时间和步数信息
        metrics["time/total_timesteps"] = self.num_timesteps
        metrics["time/n_updates"] = getattr(self.model, "_n_updates", 0)
        if hasattr(self.model, "start_time"):
            fps = self.num_timesteps / max(1, time.time() - self.model.start_time)
            metrics["time/fps"] = fps
        
        # 2. Rollout 统计
        if self._episode_rewards:
            metrics["rollout/ep_rew_mean"] = np.mean(self._episode_rewards[-100:])
            metrics["rollout/ep_rew_std"] = np.std(self._episode_rewards[-100:])
            metrics["rollout/ep_len_mean"] = np.mean(self._episode_lengths[-100:])
            metrics["rollout/n_episodes"] = self._n_episodes
        
        # 3. 训练损失 (从 logger 获取)
        if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
            logger_dict = self.model.logger.name_to_value
            
            # Actor/Policy loss
            for key in ["train/actor_loss", "train/policy_loss"]:
                if key in logger_dict:
                    metrics["train/actor_loss"] = logger_dict[key]
                    break
            
            # Critic loss
            for key in ["train/critic_loss", "train/value_loss", "train/q_loss"]:
                if key in logger_dict:
                    metrics["train/critic_loss"] = logger_dict[key]
                    break
            
            # Entropy coefficient
            if "train/ent_coef" in logger_dict:
                metrics["train/ent_coef"] = logger_dict["train/ent_coef"]
            
            # Entropy coefficient loss
            if "train/ent_coef_loss" in logger_dict:
                metrics["train/ent_coef_loss"] = logger_dict["train/ent_coef_loss"]
            
            # Learning rate
            for key in ["train/learning_rate", "train/lr"]:
                if key in logger_dict:
                    metrics["train/learning_rate"] = logger_dict[key]
                    break
            
            # Q values
            for key in ["train/q1_mean", "train/q_mean"]:
                if key in logger_dict:
                    metrics["train/q_mean"] = logger_dict[key]
                    break
                    
        # 4. Replay buffer 状态
        if hasattr(self.model, "replay_buffer"):
            rb = self.model.replay_buffer
            metrics["buffer/size"] = rb.size()
            metrics["buffer/pos"] = rb.pos if hasattr(rb, "pos") else 0
            metrics["buffer/full"] = int(rb.full if hasattr(rb, "full") else 0)
        
        # 5. 探索统计
        if hasattr(self.model, "ent_coef"):
            try:
                ent_coef = self.model.ent_coef
                if callable(ent_coef):
                    metrics["explore/ent_coef_value"] = ent_coef().item()
                elif hasattr(ent_coef, "item"):
                    metrics["explore/ent_coef_value"] = ent_coef.item()
            except:
                pass
        
        wandb.log(metrics, step=self.num_timesteps)


class SuccessInfoCallback(BaseCallback):
    """
    记录 ManiSkill3 环境中的 success 指标
    
    从 info dict 中提取 success_once, success_at_end 等
    """
    
    def __init__(self, verbose: int = 0, log_freq: int = 1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._last_log_step = 0
        self._success_once = []
        self._success_at_end = []
        
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "success_once" in info:
                self._success_once.append(float(info["success_once"]))
            if "success_at_end" in info:
                self._success_at_end.append(float(info["success_at_end"]))
        
        if self.num_timesteps - self._last_log_step >= self.log_freq:
            self._log_success()
            self._last_log_step = self.num_timesteps
            
        return True
    
    def _log_success(self):
        if not HAS_WANDB or wandb.run is None:
            return
            
        metrics = {}
        
        if self._success_once:
            metrics["rollout/success_once"] = np.mean(self._success_once[-500:])
            self._success_once = self._success_once[-500:]  # 只保留最近500个
            
        if self._success_at_end:
            metrics["rollout/success_at_end"] = np.mean(self._success_at_end[-500:])
            self._success_at_end = self._success_at_end[-500:]
        
        if metrics:
            wandb.log(metrics, step=self.num_timesteps)


class WandbEvalCallback(EvalCallback):
    """
    扩展 EvalCallback，将评估结果同步记录到 Wandb
    
    记录的指标：
    - eval/mean_reward: 平均评估奖励
    - eval/std_reward: 奖励标准差
    - eval/mean_ep_length: 平均 episode 长度
    - eval/success_rate: 成功率 (如果环境支持)
    - eval/best_mean_reward: 历史最佳奖励
    """
    
    def __init__(self, *args, log_to_wandb: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_to_wandb = log_to_wandb
        self._best_mean_reward = -np.inf
        self._eval_count = 0
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # 检查是否刚完成评估 (通过检查 last_mean_reward 是否更新)
        if hasattr(self, 'last_mean_reward') and self.last_mean_reward is not None:
            self._log_eval_to_wandb()
        
        return result
    
    def _log_eval_to_wandb(self):
        """将评估结果记录到 wandb"""
        if not self.log_to_wandb or not HAS_WANDB or wandb.run is None:
            return
        
        metrics = {}
        
        # 基础评估指标
        if hasattr(self, 'last_mean_reward'):
            metrics["eval/mean_reward"] = self.last_mean_reward
            
            # 更新最佳奖励
            if self.last_mean_reward > self._best_mean_reward:
                self._best_mean_reward = self.last_mean_reward
            metrics["eval/best_mean_reward"] = self._best_mean_reward
        
        # 从评估日志文件读取更多信息
        if self.log_path is not None:
            eval_log_path = Path(self.log_path) / "evaluations.npz"
            if eval_log_path.exists():
                try:
                    data = np.load(str(eval_log_path))
                    if "results" in data and len(data["results"]) > 0:
                        latest_results = data["results"][-1]  # 最近一次评估的所有 episode 奖励
                        metrics["eval/std_reward"] = np.std(latest_results)
                        metrics["eval/min_reward"] = np.min(latest_results)
                        metrics["eval/max_reward"] = np.max(latest_results)
                    
                    if "ep_lengths" in data and len(data["ep_lengths"]) > 0:
                        latest_lengths = data["ep_lengths"][-1]
                        metrics["eval/mean_ep_length"] = np.mean(latest_lengths)
                        metrics["eval/std_ep_length"] = np.std(latest_lengths)
                    
                    # 检查是否有 success 信息
                    if "successes" in data and len(data["successes"]) > 0:
                        latest_successes = data["successes"][-1]
                        metrics["eval/success_rate"] = np.mean(latest_successes)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Could not read eval log: {e}")
        
        # 评估计数
        self._eval_count += 1
        metrics["eval/eval_count"] = self._eval_count
        
        # 是否保存了新的最佳模型
        if hasattr(self, 'best_mean_reward'):
            metrics["eval/is_best"] = int(self.last_mean_reward >= self.best_mean_reward - 1e-6)
        
        wandb.log(metrics, step=self.num_timesteps)



@dataclass
class Args:
    """DSRL-SAC 训练参数"""
    
    # ===== 实验设置 =====
    exp_name: Optional[str] = None
    """实验名称 (自动生成如果为空)"""
    seed: int = 1
    """随机种子"""
    cuda: bool = True
    """使用 CUDA"""
    track: bool = False
    """使用 wandb 跟踪"""
    wandb_project: str = "maniskill_dsrl_sac"
    """wandb 项目名"""
    wandb_entity: Optional[str] = None
    """wandb 实体"""
    wandb_group: str = "dsrl_sac"
    """wandb 分组"""
    
    # ===== 环境设置 =====
    env_id: str = "LiftPegUpright-v1"
    """ManiSkill3 环境 ID"""
    n_envs: int = 50
    """训练环境数量"""
    n_eval_envs: int = 50
    """评估环境数量"""
    max_episode_steps: int = 100
    """最大 episode 步数"""
    control_mode: str = "pd_ee_delta_pose"
    """控制模式"""
    sim_backend: str = "physx_cuda"
    """仿真后端 (physx_cuda 或 cpu)"""
    reward_mode: str = "dense"
    """奖励模式"""
    
    # ===== 预训练 Checkpoint =====
    checkpoint: str = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
    """预训练 ShortCut Flow checkpoint"""
    use_ema: bool = True
    """使用 EMA 权重"""
    
    # ===== 模型参数 =====
    obs_horizon: int = 2
    """观察历史长度"""
    pred_horizon: int = 16
    """预测动作序列长度"""
    act_steps: int = 8
    """动作执行步数 (action horizon)"""
    action_dim: int = 7
    """动作维度"""
    state_dim: int = 25
    """状态维度"""
    visual_feature_dim: int = 256
    """视觉特征维度"""
    action_magnitude: float = 1.5
    """噪声范围 [-mag, +mag]"""
    
    # ===== 训练超参数 =====
    total_timesteps: int = 1_000_000
    """总训练步数"""
    learning_rate: float = 3e-4
    """学习率"""
    buffer_size: int = 1_000_000
    """Replay buffer 大小"""
    batch_size: int = 256
    """批大小"""
    gamma: float = 0.99
    """折扣因子"""
    tau: float = 0.005
    """Target 网络软更新率"""
    utd: int = 20
    """Update-To-Data ratio (每步梯度更新次数)"""
    learning_starts: int = 1000
    """开始训练前收集的步数"""
    
    # ===== 网络架构 =====
    num_layers: int = 3
    """Actor/Critic 网络层数"""
    layer_size: int = 2048
    """每层宽度"""
    n_critics: int = 2
    """Critic 数量"""
    log_std_init: float = -3.0
    """Actor log_std 初始化 (关键！-3 对应初始 std≈0.05，保护预训练策略)"""
    target_entropy: float = 0.0
    """目标熵 (0.0 = 鼓励低噪声，保护预训练策略)"""
    init_rollout_steps: int = 5001
    """预热 buffer 的步数 (使用预训练策略零噪声收集)"""
    
    # ===== 日志和评估 =====
    log_interval: int = 1000
    """日志间隔"""
    eval_freq: int = 10000
    """评估间隔"""
    n_eval_episodes: int = 50
    """每次评估 episode 数"""
    save_freq: int = 50000
    """模型保存间隔"""
    save_replay_buffer: bool = False
    """是否保存 replay buffer"""


class SB3EnvAdapter(VecEnv):
    """适配 ManiSkillGPUFlowEnvWrapper 到 SB3 VecEnv 接口"""
    
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._obs = None
        
    def reset(self):
        obs, _ = self.env.reset()
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        self._obs = obs
        return obs
    
    def step_async(self, actions):
        self._actions = actions
    
    def step_wait(self):
        obs, reward, terminated, truncated, info = self.env.step(self._actions)
        
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.cpu().numpy()
        
        dones = terminated | truncated
        
        # 构建 infos 列表
        infos = []
        for i in range(self.num_envs):
            env_info = {}
            if dones[i]:
                env_info["terminal_observation"] = obs[i].copy()
                env_info["TimeLimit.truncated"] = truncated[i] and not terminated[i]
            infos.append(env_info)
        
        self._obs = obs
        return obs, reward, dones, infos
    
    def close(self):
        self.env.close()
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if indices is None:
            indices = range(self.num_envs)
        results = []
        for i in indices:
            method = getattr(self.env, method_name, None)
            if method:
                results.append(method(*method_args, **method_kwargs))
            else:
                results.append(None)
        return results
    
    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self.env, attr_name, None)] * len(indices)
    
    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)
    
    def seed(self, seed=None):
        return [seed] * self.num_envs


def collect_initial_rollouts(
    env: VecEnv,
    model,
    n_steps: int,
    device: str = "cuda",
):
    """使用预训练策略（零噪声）收集初始 rollout 填充 replay buffer
    
    对于 DSRL-SAC，环境已经包装了 diffusion policy，
    所以动作空间就是噪声空间。使用零噪声 = 使用预训练策略。
    
    Args:
        env: SB3 VecEnv (已包装 ManiSkillGPUFlowEnvWrapper)
        model: SAC 模型
        n_steps: 收集的总步数
        device: 设备
    """
    print(f"[Init Rollout] Collecting {n_steps} steps with pretrained policy (zero noise)")
    
    obs = env.reset()
    collected_steps = 0
    episode_rewards = []
    current_rewards = np.zeros(env.num_envs)
    
    while collected_steps < n_steps:
        # DSRL-SAC: 动作空间就是噪声空间，零噪声 = 预训练策略
        actions = np.zeros((env.num_envs,) + env.action_space.shape, dtype=np.float32)
        
        # 执行动作
        new_obs, rewards, dones, infos = env.step(actions)
        
        # 添加到 replay buffer
        model.replay_buffer.add(
            obs,
            new_obs,
            actions,
            rewards,
            dones,
            infos,
        )
        
        # 统计
        current_rewards += rewards
        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(current_rewards[i])
                current_rewards[i] = 0
        
        obs = new_obs
        collected_steps += env.num_envs
        
        if collected_steps % 500 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            print(f"  [Init Rollout] {collected_steps}/{n_steps} steps, "
                  f"episodes: {len(episode_rewards)}, avg_reward: {avg_reward:.2f}")
    
    print(f"[Init Rollout] ✅ Collected {collected_steps} steps, "
          f"{len(episode_rewards)} episodes, "
          f"avg_reward: {np.mean(episode_rewards):.2f}")
    
    return collected_steps


def load_base_policy(args: Args, device: str):
    """加载预训练的 ShortCut Flow policy 和 visual encoder"""
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    agent_state = checkpoint.get("ema_agent" if args.use_ema else "agent", checkpoint)
    
    global_cond_dim = args.obs_horizon * (args.visual_feature_dim + args.state_dim)
    
    # 加载 velocity_net
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=args.action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=64,
        down_dims=(64, 128, 256),
        n_groups=8,
    ).to(device)
    
    velocity_net_state = {
        k.replace("velocity_net.", ""): v 
        for k, v in agent_state.items() 
        if k.startswith("velocity_net.")
    }
    velocity_net.load_state_dict(velocity_net_state)
    velocity_net.eval()
    
    # 加载 visual_encoder
    visual_encoder = PlainConv(
        in_channels=3, 
        out_dim=args.visual_feature_dim, 
        pool_feature_map=True
    ).to(device)
    
    if "visual_encoder" in checkpoint:
        visual_encoder.load_state_dict(checkpoint["visual_encoder"])
    visual_encoder.eval()
    
    # 创建 ShortCutFlowWrapper
    base_policy = ShortCutFlowWrapper(
        velocity_net=velocity_net,
        visual_encoder=None,  # visual_encoder 单独处理
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_dim=args.action_dim,
        num_inference_steps=8,
        device=device,
    )
    
    return base_policy, visual_encoder


def create_env(args: Args, visual_encoder, base_policy, device: str, is_eval: bool = False):
    """创建 DSRL-SAC 环境"""
    
    n_envs = args.n_eval_envs if is_eval else args.n_envs
    
    env_kwargs = dict(
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        obs_mode="rgbd",
        render_mode="rgb_array",
        reward_mode=args.reward_mode,
    )
    
    base_envs = make_eval_envs(
        env_id=args.env_id,
        num_envs=n_envs,
        sim_backend=args.sim_backend,
        env_kwargs=env_kwargs,
        other_kwargs=dict(obs_horizon=args.obs_horizon),
        video_dir=None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    
    wrapped_env = ManiSkillGPUFlowEnvWrapper(
        env=base_envs,
        base_policy=base_policy,
        visual_encoder=visual_encoder,
        action_magnitude=args.action_magnitude,
        act_steps=args.act_steps,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        visual_feature_dim=args.visual_feature_dim,
        obs_horizon=args.obs_horizon,
        include_rgb=True,
        device=device,
    )
    
    return SB3EnvAdapter(wrapped_env)


def main():
    args = tyro.cli(Args)
    
    # 生成实验名称
    if args.exp_name is None:
        args.exp_name = f"dsrl_sac-{args.env_id}-{args.n_envs}envs-utd{args.utd}-seed{args.seed}"
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # 创建日志目录
    log_dir = Path(__file__).parent / "runs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("DSRL-SAC Training")
    print("=" * 70)
    print(f"Experiment: {run_name}")
    print(f"Device: {device}")
    print(f"Environment: {args.env_id}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"N envs: {args.n_envs}")
    print(f"UTD ratio: {args.utd}")
    print(f"Action magnitude: {args.action_magnitude}")
    print("=" * 70)
    
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
    
    # ===== 加载 Base Policy =====
    print("\n[1/4] Loading base policy...")
    base_policy, visual_encoder = load_base_policy(args, device)
    print("  ✅ Loaded ShortCutFlowWrapper and visual encoder")
    
    # ===== 创建环境 =====
    print("\n[2/4] Creating environments...")
    train_env = create_env(args, visual_encoder, base_policy, device, is_eval=False)
    eval_env = create_env(args, visual_encoder, base_policy, device, is_eval=True)
    print(f"  ✅ Train env: {args.n_envs} envs")
    print(f"  ✅ Eval env: {args.n_eval_envs} envs")
    print(f"  Action space: {train_env.action_space}")
    print(f"  Observation space: {train_env.observation_space}")
    
    # ===== 创建 SAC 模型 =====
    print("\n[3/4] Creating SAC model...")
    
    net_arch = [args.layer_size] * args.num_layers
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        n_critics=args.n_critics,
        log_std_init=args.log_std_init,  # 关键：初始化小噪声，保护预训练策略
    )
    
    # learning_starts=1 因为我们使用 init_rollout_steps 预热 buffer
    effective_learning_starts = 1 if args.init_rollout_steps > 0 else args.learning_starts
    
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=effective_learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=1,
        gradient_steps=args.utd,
        action_noise=None,
        optimize_memory_usage=False,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy=args.target_entropy,  # 使用配置的 target_entropy
        tensorboard_log=str(log_dir),
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=device,
    )
    print(f"  ✅ Created SAC model")
    print(f"    - log_std_init: {args.log_std_init} (initial std ≈ {np.exp(args.log_std_init):.4f})")
    print(f"    - target_entropy: {args.target_entropy}")
    print(f"    - learning_starts: {effective_learning_starts}")
    
    # ===== 初始 Rollout (使用预训练策略填充 buffer) =====
    if args.init_rollout_steps > 0:
        print(f"\n[3.5/4] Collecting initial rollouts with pretrained policy...")
        collect_initial_rollouts(
            env=train_env,
            model=model,
            n_steps=args.init_rollout_steps,
            device=device,
        )
        print(f"  ✅ Replay buffer warmed up with {model.replay_buffer.size()} samples")
    
    # ===== 设置回调 =====
    callbacks = []
    
    # 1. Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="dsrl_sac",
        save_replay_buffer=args.save_replay_buffer,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # 2. Wandb logging callback
    if args.track and HAS_WANDB:
        wandb_callback = WandbCallback(
            verbose=1,
            log_freq=args.log_interval,
        )
        callbacks.append(wandb_callback)
        
        success_callback = SuccessInfoCallback(
            verbose=0,
            log_freq=args.log_interval,
        )
        callbacks.append(success_callback)
    
    # 3. Evaluation callback (记录评估指标，同步到 wandb)
    eval_callback = WandbEvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(log_dir / "eval_logs"),
        eval_freq=args.eval_freq // args.n_envs,  # 按 n_envs 调整
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
        log_to_wandb=(args.track and HAS_WANDB),
    )
    callbacks.append(eval_callback)
    
    # ===== 训练 =====
    print("\n[4/4] Starting training...")
    print(f"Training for {args.total_timesteps:,} timesteps...")
    print(f"  Wandb tracking: {'✅ Enabled' if (args.track and HAS_WANDB) else '❌ Disabled'}")
    print(f"  Eval frequency: every {args.eval_freq:,} steps")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=args.log_interval // args.n_envs,
    )
    
    # ===== 保存最终模型 =====
    final_path = checkpoint_dir / "final"
    model.save(str(final_path))
    print(f"\n✅ Training complete! Model saved to {final_path}")
    
    # ===== 清理 =====
    train_env.close()
    eval_env.close()
    
    if args.track and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
