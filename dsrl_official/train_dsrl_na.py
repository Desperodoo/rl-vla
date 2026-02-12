"""
DSRL-NA Training Script for ManiSkill3

使用 SB3 Fork 的 DSRL 算法，策略内部采样噪声，蒸馏 Q^W。
DSRL-NA: Noise as Action in diffusion policy's noise space

需要安装 ajwagen 的 SB3 fork:
    pip install git+https://github.com/ajwagen/stable-baselines3.git

参考: https://github.com/ajwagen/dsrl/blob/main/train_dsrl.py

Usage:
    python train_dsrl_na.py --total_timesteps 1000000
    
    # 自定义参数
    python train_dsrl_na.py --n_envs 100 --utd 40 --seed 42
    
    # 使用 wandb 跟踪
    python train_dsrl_na.py --track --wandb_project my_project
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
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

# 尝试导入官方 DSRL (SB3 fork)
try:
    from stable_baselines3 import DSRL
    HAS_DSRL = True
except ImportError:
    HAS_DSRL = False
    print("ERROR: DSRL-NA requires the official DSRL fork of stable-baselines3.")
    print("Install: pip install git+https://github.com/ajwagen/stable-baselines3.git")
    sys.exit(1)

from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.make_env import make_eval_envs
from dsrl_official.utils import ShortCutFlowWrapper

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ========== Wandb Logging Callback ==========
class WandbCallback(BaseCallback):
    """
    自定义 Callback，将 DSRL-NA 训练指标记录到 Wandb
    
    记录的指标：
    - rollout: ep_rew_mean, ep_len_mean
    - train: actor_loss, critic_loss, ent_coef, ent_coef_loss, noise_critic_loss
    - time: fps, total_timesteps
    - buffer: replay_buffer_size
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
            
            # Noise critic loss (DSRL-NA specific)
            if "train/noise_critic_loss" in logger_dict:
                metrics["train/noise_critic_loss"] = logger_dict["train/noise_critic_loss"]
            
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
            self._success_once = self._success_once[-500:]
            
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
    """DSRL-NA 训练参数"""
    
    # ===== 实验设置 =====
    exp_name: Optional[str] = None
    """实验名称 (自动生成如果为空)"""
    seed: int = 1
    """随机种子"""
    cuda: bool = True
    """使用 CUDA"""
    track: bool = False
    """使用 wandb 跟踪"""
    wandb_project: str = "maniskill_dsrl_na"
    """wandb 项目名"""
    wandb_entity: Optional[str] = None
    """wandb 实体"""
    wandb_group: str = "dsrl_na"
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
    use_layer_norm: bool = True
    """使用 LayerNorm (官方推荐)"""
    log_std_init: float = -3.0
    """Actor log_std 初始化 (关键！-3 对应初始 std≈0.05，保护预训练策略)"""
    
    # ===== DSRL-NA 特有参数 =====
    noise_critic_grad_steps: int = 10
    """Q^W 梯度步数 (原版推荐 10)"""
    critic_backup_combine_type: str = "min"
    """Critic backup 组合类型"""
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


class DSRLDiffusionPolicyWrapper:
    """包装 ShortCutFlowWrapper，使其与 DSRL 算法兼容。
    
    DSRL 调用: diffusion_policy(obs, noise, return_numpy=False)
    不传 act_steps 参数，所以我们需要在这里自动传入。
    """
    def __init__(self, base_policy, act_steps):
        self.base_policy = base_policy
        self.act_steps = act_steps
    
    def __call__(self, obs, noise, return_numpy=False):
        return self.base_policy(obs, noise, return_numpy=return_numpy, act_steps=self.act_steps)


class DSRLNAEnvWrapper(gym.Env):
    """DSRL-NA 环境包装器
    
    DSRL-NA 需要的接口：
    - 观测空间：obs_cond (562维 = obs_horizon * (visual_dim + state_dim))
    - 动作空间：diffused_actions (56维 = act_steps * action_dim)
    
    DSRL 算法会：
    1. Actor 输出噪声
    2. 使用 diffusion_policy 将噪声解码为 diffused_actions
    3. 将 diffused_actions 传给环境执行
    """
    
    def __init__(
        self,
        env,
        visual_encoder,
        act_steps: int,
        action_dim: int,
        state_dim: int,
        visual_feature_dim: int,
        obs_horizon: int,
        device: str,
    ):
        super().__init__()
        self.env = env
        self.visual_encoder = visual_encoder
        self.device = device
        
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.visual_feature_dim = visual_feature_dim
        self.obs_horizon = obs_horizon
        
        self.num_envs = getattr(env.unwrapped, "num_envs", 1)
        
        # 观测空间：编码后的 obs_cond
        obs_dim = obs_horizon * (visual_feature_dim + state_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 动作空间：diffused_actions (act_steps * action_dim)
        # DSRL 会在内部处理噪声到动作的转换
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_steps * action_dim,), dtype=np.float32
        )
        
        self._obs_history = None
        self._cached_obs_cond = None
        
    def _encode_rgb(self, rgb):
        """编码 RGB 观测"""
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        
        B, T, H, W, C = rgb.shape
        rgb_flat = rgb.reshape(B * T, H, W, C)
        
        rgb_tensor = torch.from_numpy(rgb_flat).to(self.device).float()
        if rgb_tensor.max() > 1.0:
            rgb_tensor = rgb_tensor / 255.0
        rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            features = self.visual_encoder(rgb_tensor)
        
        return features.reshape(B, T, -1)
    
    def _build_obs_cond(self, obs):
        """构建 obs_cond"""
        rgb = obs["rgb"]
        state = obs["state"]
        
        visual_features = self._encode_rgb(
            rgb.cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
        )
        
        if isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            state = torch.from_numpy(state).to(self.device).float()
        
        # 拼接 visual features 和 state
        obs_cond = torch.cat([visual_features, state], dim=-1)
        obs_cond = obs_cond.reshape(obs_cond.shape[0], -1)  # (B, obs_horizon * (visual_dim + state_dim))
        
        return obs_cond
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_cond = self._build_obs_cond(obs)
        self._cached_obs_cond = obs_cond
        return obs_cond.cpu().numpy(), info
    
    def step(self, diffused_actions):
        """执行 diffused_actions
        
        Args:
            diffused_actions: (B, act_steps * action_dim)
        """
        if isinstance(diffused_actions, np.ndarray):
            diffused_actions = torch.from_numpy(diffused_actions).to(self.device).float()
        
        # 重塑为 (B, act_steps, action_dim)
        B = diffused_actions.shape[0]
        actions = diffused_actions.reshape(B, self.act_steps, self.action_dim)
        
        # 执行动作序列
        total_reward = torch.zeros(B, device=self.device)
        terminated = torch.zeros(B, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(B, dtype=torch.bool, device=self.device)
        
        for t in range(self.act_steps):
            action_t = actions[:, t, :]
            obs, reward, term, trunc, info = self.env.step(action_t)
            
            if isinstance(reward, np.ndarray):
                reward = torch.from_numpy(reward).to(self.device)
            if isinstance(term, np.ndarray):
                term = torch.from_numpy(term).to(self.device)
            if isinstance(trunc, np.ndarray):
                trunc = torch.from_numpy(trunc).to(self.device)
            
            total_reward += reward * (~terminated).float()
            terminated = terminated | term
            truncated = truncated | trunc
            
            if terminated.all() or truncated.all():
                break
        
        obs_cond = self._build_obs_cond(obs)
        self._cached_obs_cond = obs_cond
        
        return (
            obs_cond.cpu().numpy(),
            total_reward.cpu().numpy(),
            terminated.cpu().numpy(),
            truncated.cpu().numpy(),
            info,
        )
    
    def close(self):
        self.env.close()


def collect_initial_rollouts(
    env: VecEnv,
    model,
    diffusion_policy,
    n_steps: int,
    act_steps: int,
    action_dim: int,
    noise_std: float = 0.0,
    device: str = "cuda",
):
    """使用预训练策略（接近零噪声）收集初始 rollout 填充 replay buffer
    
    这是 DSRL 原版的关键技巧：在训练开始前用预训练策略的高质量数据填充 buffer，
    避免随机 actor 产生的低质量数据污染初始训练。
    
    Args:
        env: SB3 VecEnv
        model: DSRL 模型
        diffusion_policy: 预训练的 diffusion policy
        n_steps: 收集的总步数
        act_steps: 动作执行步数
        action_dim: 动作维度
        noise_std: 注入的噪声标准差 (0.0 = 纯预训练策略)
        device: 设备
    """
    print(f"[Init Rollout] Collecting {n_steps} steps with pretrained policy (noise_std={noise_std})")
    
    obs = env.reset()
    collected_steps = 0
    episode_rewards = []
    current_rewards = np.zeros(env.num_envs)
    
    while collected_steps < n_steps:
        # 使用预训练策略生成噪声 (接近零)
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).to(device).float()
            
            # 生成接近零的噪声
            noise_shape = (env.num_envs, act_steps, action_dim)
            if noise_std > 0:
                noise = torch.randn(noise_shape, device=device) * noise_std
            else:
                noise = torch.zeros(noise_shape, device=device)
            
            # 使用 diffusion policy 解码噪声为动作
            diffused_actions = diffusion_policy(obs_tensor, noise, return_numpy=False)
            if isinstance(diffused_actions, torch.Tensor):
                actions = diffused_actions.reshape(env.num_envs, -1).cpu().numpy()
            else:
                actions = diffused_actions.reshape(env.num_envs, -1)
        
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


class SB3EnvAdapter(VecEnv):
    """适配 DSRLNAEnvWrapper 到 SB3 VecEnv 接口"""
    
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
    _base_policy = ShortCutFlowWrapper(
        velocity_net=velocity_net,
        visual_encoder=None,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_dim=args.action_dim,
        num_inference_steps=8,
        device=device,
    )
    
    # 包装为 DSRL 兼容的接口
    base_policy = DSRLDiffusionPolicyWrapper(_base_policy, args.act_steps)
    
    return base_policy, visual_encoder


def create_env(args: Args, visual_encoder, device: str, is_eval: bool = False):
    """创建 DSRL-NA 环境"""
    
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
    
    wrapped_env = DSRLNAEnvWrapper(
        env=base_envs,
        visual_encoder=visual_encoder,
        act_steps=args.act_steps,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        visual_feature_dim=args.visual_feature_dim,
        obs_horizon=args.obs_horizon,
        device=device,
    )
    
    return SB3EnvAdapter(wrapped_env)


def main():
    args = tyro.cli(Args)
    
    # 生成实验名称
    if args.exp_name is None:
        args.exp_name = f"dsrl_na-{args.env_id}-{args.n_envs}envs-utd{args.utd}-seed{args.seed}"
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
    print("DSRL-NA Training")
    print("=" * 70)
    print(f"Experiment: {run_name}")
    print(f"Device: {device}")
    print(f"Environment: {args.env_id}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"N envs: {args.n_envs}")
    print(f"UTD ratio: {args.utd}")
    print(f"diffusion_act_dim: ({args.act_steps}, {args.action_dim})")
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
    print("  ✅ Loaded ShortCutFlowWrapper (diffusion_policy)")
    print("  ✅ Loaded visual encoder")
    
    # ===== 创建环境 =====
    print("\n[2/4] Creating environments...")
    train_env = create_env(args, visual_encoder, device, is_eval=False)
    eval_env = create_env(args, visual_encoder, device, is_eval=True)
    print(f"  ✅ Train env: {args.n_envs} envs")
    print(f"  ✅ Eval env: {args.n_eval_envs} envs")
    print(f"  Action space: {train_env.action_space}")
    print(f"  Observation space: {train_env.observation_space}")
    
    # ===== 创建 DSRL 模型 =====
    print("\n[3/4] Creating DSRL model...")
    
    net_arch = [args.layer_size] * args.num_layers
    post_linear_modules = [torch.nn.LayerNorm] if args.use_layer_norm else None
    
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        n_critics=args.n_critics,
        log_std_init=args.log_std_init,  # 关键：初始化小噪声，保护预训练策略
    )
    if post_linear_modules:
        policy_kwargs['post_linear_modules'] = post_linear_modules
    
    # learning_starts=1 因为我们使用 init_rollout_steps 预热 buffer
    effective_learning_starts = 1 if args.init_rollout_steps > 0 else args.learning_starts
    
    model = DSRL(
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
        diffusion_policy=base_policy,
        diffusion_act_dim=(args.act_steps, args.action_dim),
        noise_critic_grad_steps=args.noise_critic_grad_steps,
        critic_backup_combine_type=args.critic_backup_combine_type,
        device=device,
    )
    print(f"  ✅ Created DSRL model with diffusion_policy")
    print(f"    - log_std_init: {args.log_std_init} (initial std ≈ {np.exp(args.log_std_init):.4f})")
    print(f"    - noise_critic_grad_steps: {args.noise_critic_grad_steps}")
    print(f"    - target_entropy: {args.target_entropy}")
    print(f"    - learning_starts: {effective_learning_starts}")
    
    # ===== 初始 Rollout (使用预训练策略填充 buffer) =====
    if args.init_rollout_steps > 0:
        print(f"\n[3.5/4] Collecting initial rollouts with pretrained policy...")
        collect_initial_rollouts(
            env=train_env,
            model=model,
            diffusion_policy=base_policy,
            n_steps=args.init_rollout_steps,
            act_steps=args.act_steps,
            action_dim=args.action_dim,
            noise_std=0.0,  # 纯预训练策略，零噪声
            device=device,
        )
        print(f"  ✅ Replay buffer warmed up with {model.replay_buffer.size()} samples")
    
    # ===== 设置回调 =====
    callbacks = []
    
    # 1. Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="dsrl_na",
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
        eval_freq=args.eval_freq // args.n_envs,
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
