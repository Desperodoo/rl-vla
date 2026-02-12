"""
DSRL Official Callbacks - 从官方 ajwagen/dsrl 移植

包含:
- LoggingCallback: 训练日志和评估回调
- ManiSkillEvalCallback: ManiSkill3 专用评估回调

Reference: https://github.com/ajwagen/dsrl/blob/main/utils.py
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path

try:
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    BaseCallback = object

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class LoggingCallback(BaseCallback if HAS_SB3 else object):
    """训练日志回调 - 从官方 DSRL 移植。
    
    功能:
    - 记录训练指标到 wandb
    - 定期评估并记录成功率
    - 支持 DSRL-SAC 和 DSRL-NA 两种算法
    
    Args:
        action_chunk: 动作分块大小
        log_freq: 日志记录频率
        use_wandb: 是否使用 wandb
        eval_env: 评估环境
        eval_freq: 评估频率
        eval_episodes: 评估 episode 数
        rew_offset: 奖励偏移 (用于计算成功率)
        num_train_env: 训练环境数量
        num_eval_env: 评估环境数量
        algorithm: 算法类型 ("dsrl_sac" 或 "dsrl_na")
        max_steps: 每 episode 最大步数
        deterministic_eval: 是否使用确定性评估
    """
    
    def __init__(
        self,
        action_chunk: int = 8,
        log_freq: int = 1000,
        use_wandb: bool = True,
        eval_env=None,
        eval_freq: int = 70,
        eval_episodes: int = 10,
        verbose: int = 0,
        rew_offset: float = 0,
        num_train_env: int = 1,
        num_eval_env: int = 1,
        algorithm: str = "dsrl_sac",
        max_steps: int = -1,
        deterministic_eval: bool = False,
    ):
        if HAS_SB3:
            super().__init__(verbose)
        else:
            # Fallback: 设置必要属性
            self.verbose = verbose
            self.n_calls = 0
            self.model = None
            self.locals = {}
        
        self.action_chunk = action_chunk
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.use_wandb = use_wandb and HAS_WANDB
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.log_count = 0
        self.total_reward = 0
        self.rew_offset = rew_offset
        self.total_timesteps = 0
        self.num_train_env = num_train_env
        self.num_eval_env = num_eval_env
        self.episode_success = np.zeros(num_train_env)
        self.episode_completed = np.zeros(num_train_env)
        self.algorithm = algorithm
        self.max_steps = max_steps
        self.deterministic_eval = deterministic_eval
    
    def _on_step(self) -> bool:
        """每步调用。"""
        # 记录 episode 信息
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        
        # 累积奖励
        rew = self.locals['rewards']
        self.total_reward += np.mean(rew)
        
        # 更新成功率统计
        self.episode_success[rew > -self.rew_offset] = 1
        self.episode_completed[self.locals['dones']] = 1
        
        # 更新总步数
        self.total_timesteps += self.action_chunk * self.model.n_envs
        
        # 定期日志记录
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                if self.use_wandb:
                    self.log_count += 1
                    
                    log_dict = {
                        "train/ep_len_mean": np.mean(self.episode_lengths),
                        "train/ep_rew_mean": np.mean(self.episode_rewards),
                        "train/rew_mean": np.mean(self.total_reward),
                        "train/timesteps": self.total_timesteps,
                    }
                    
                    # 添加 SB3 记录的指标
                    if hasattr(self.locals['self'], 'logger'):
                        logger = self.locals['self'].logger
                        if hasattr(logger, 'name_to_value'):
                            for key in ['train/ent_coef', 'train/actor_loss', 
                                        'train/critic_loss', 'train/ent_coef_loss']:
                                if key in logger.name_to_value:
                                    log_dict[key] = logger.name_to_value[key]
                    
                    wandb.log(log_dict, step=self.log_count)
                    
                    # 成功率
                    if np.sum(self.episode_completed) > 0:
                        wandb.log({
                            "train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
                        }, step=self.log_count)
                    
                    # DSRL-NA 特有指标
                    if self.algorithm == 'dsrl_na':
                        if hasattr(self.locals['self'], 'logger'):
                            logger = self.locals['self'].logger
                            if hasattr(logger, 'name_to_value'):
                                if 'train/noise_critic_loss' in logger.name_to_value:
                                    wandb.log({
                                        "train/noise_critic_loss": logger.name_to_value['train/noise_critic_loss'],
                                    }, step=self.log_count)
                
                # 重置统计
                self.episode_rewards = []
                self.episode_lengths = []
                self.total_reward = 0
                self.episode_success = np.zeros(self.num_train_env)
                self.episode_completed = np.zeros(self.num_train_env)
        
        # 定期评估
        if self.n_calls % self.eval_freq == 0:
            self.evaluate(self.locals['self'], deterministic=False)
            if self.deterministic_eval:
                self.evaluate(self.locals['self'], deterministic=True)
        
        return True
    
    def evaluate(self, agent, deterministic: bool = False):
        """评估 agent。"""
        if self.eval_episodes <= 0 or self.eval_env is None:
            return
        
        env = self.eval_env
        
        with torch.no_grad():
            success_list = []
            rew_list = []
            rew_total = 0
            total_ep = 0
            rew_ep = np.zeros(self.num_eval_env)
            
            for i in range(self.eval_episodes):
                obs = env.reset()
                success_i = np.zeros(obs.shape[0])
                r = []
                
                for _ in range(self.max_steps):
                    # 根据算法选择预测方法
                    if self.algorithm == 'dsrl_sac':
                        action, _ = agent.predict(obs, deterministic=deterministic)
                    elif self.algorithm == 'dsrl_na':
                        action, _ = agent.predict_diffused(obs, deterministic=deterministic)
                    else:
                        action, _ = agent.predict(obs, deterministic=deterministic)
                    
                    next_obs, reward, done, info = env.step(action)
                    obs = next_obs
                    
                    rew_ep += reward
                    rew_total += sum(rew_ep[done])
                    rew_ep[done] = 0
                    total_ep += np.sum(done)
                    success_i[reward > -self.rew_offset] = 1
                    r.append(reward)
                
                success_list.append(success_i.mean())
                rew_list.append(np.mean(np.array(r)))
                print(f'eval episode {i} at timestep {self.total_timesteps}')
            
            success_rate = np.mean(success_list)
            avg_rew = rew_total / total_ep if total_ep > 0 else 0
            
            if self.use_wandb:
                name = 'eval'
                if deterministic:
                    wandb.log({
                        f"{name}/success_rate_deterministic": success_rate,
                        f"{name}/reward_deterministic": avg_rew,
                    }, step=self.log_count)
                else:
                    wandb.log({
                        f"{name}/success_rate": success_rate,
                        f"{name}/reward": avg_rew,
                        f"{name}/timesteps": self.total_timesteps,
                    }, step=self.log_count)
    
    def set_timesteps(self, timesteps: int):
        """设置总步数（用于离线数据预填充后）。"""
        self.total_timesteps = timesteps


class ManiSkillEvalCallback:
    """ManiSkill3 专用评估回调。
    
    不依赖 SB3，用于自定义训练循环。
    
    Args:
        eval_envs: ManiSkill3 评估环境
        agent_wrapper: Agent 包装器
        visual_encoder: 视觉编码器
        include_rgb: 是否包含 RGB
        obs_horizon: 观察历史长度
        act_horizon: 动作执行步数
        device: 设备
    """
    
    def __init__(
        self,
        eval_envs,
        agent_wrapper,
        visual_encoder=None,
        include_rgb: bool = True,
        obs_horizon: int = 2,
        act_horizon: int = 8,
        device: str = "cuda",
        use_wandb: bool = True,
    ):
        self.eval_envs = eval_envs
        self.agent_wrapper = agent_wrapper
        self.visual_encoder = visual_encoder
        self.include_rgb = include_rgb
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.device = device
        self.use_wandb = use_wandb and HAS_WANDB
    
    def evaluate(
        self,
        num_episodes: int = 50,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """运行评估。
        
        Args:
            num_episodes: 评估 episode 数
            deterministic: 是否使用确定性策略
            
        Returns:
            评估指标字典
        """
        # 使用 diffusion_policy 的 evaluate 函数
        _root = Path(__file__).parent.parent
        sys.path.insert(0, str(_root / "diffusion_policy"))
        from diffusion_policy.evaluate import evaluate
        
        eval_metrics = evaluate(
            num_episodes,
            self.agent_wrapper,
            self.eval_envs,
            self.device,
            sim_backend="physx_cuda",
        )
        
        # 计算平均值
        for k in eval_metrics.keys():
            eval_metrics[k] = np.mean(eval_metrics[k])
        
        # 记录到 wandb
        if self.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
        
        return eval_metrics
