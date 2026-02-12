#!/usr/bin/env python3
"""
DSRL-SAC 最小可运行验证

使用简化的 state-only 模式验证 DSRL-SAC 训练流程。
避免 ManiSkill3 GPU 环境与 SB3 VecEnv 的兼容性问题。

Usage:
    python test_dsrl_sac_simple.py
"""

import os
import sys
from pathlib import Path

# 添加路径
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "diffusion_policy"))

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

print("=" * 60)
print("DSRL-SAC 简化验证 (State-only)")
print("=" * 60)

# ===== 检查依赖 =====
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    print("✅ stable-baselines3 available")
except ImportError:
    print("❌ stable-baselines3 not installed")
    sys.exit(1)

# ===== 配置 =====
CHECKPOINT_PATH = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
DEVICE = "cuda"

# 模型参数
OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_STEPS = 8
ACTION_DIM = 7
VISUAL_FEATURE_DIM = 256
STATE_DIM = 25  # 从 checkpoint 推断
ACTION_MAGNITUDE = 1.5

# 计算维度
OBS_DIM = OBS_HORIZON * (VISUAL_FEATURE_DIM + STATE_DIM)  # 562

print(f"\nConfig:")
print(f"  obs_dim: {OBS_DIM}")
print(f"  action_dim: {ACTION_DIM}")
print(f"  act_steps: {ACT_STEPS}")
print(f"  action_magnitude: {ACTION_MAGNITUDE}")

# ===== 加载 Base Policy =====
print("\n[1/3] Loading base policy...")

from dsrl_official.utils import load_shortcut_flow_policy
from diffusion_policy.plain_conv import PlainConv

base_policy, _ = load_shortcut_flow_policy(
    checkpoint_path=CHECKPOINT_PATH,
    visual_encoder_class=PlainConv,
    obs_horizon=OBS_HORIZON,
    pred_horizon=PRED_HORIZON,
    action_dim=ACTION_DIM,
    visual_feature_dim=VISUAL_FEATURE_DIM,
    state_dim=None,  # 从 checkpoint 推断
    include_rgb=True,
    use_ema=True,
    device=DEVICE,
)
print("  ✅ Base policy loaded")


# ===== 创建简化的测试环境 =====
print("\n[2/3] Creating simplified test environment...")

class SimpleDSRLEnv(gym.Env):
    """简化的 DSRL 测试环境。
    
    模拟 DSRL-SAC 环境：
    - 动作空间：噪声空间 [-mag, +mag]^(act_steps * action_dim)
    - 观察空间：[-1, 1]^obs_dim
    """
    
    def __init__(self, base_policy, obs_dim, action_dim, act_steps, action_magnitude, device):
        super().__init__()
        self.base_policy = base_policy
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.act_steps = act_steps
        self.action_magnitude = action_magnitude
        self.device = device
        
        # 噪声空间作为动作空间
        self.action_space = spaces.Box(
            low=-action_magnitude,
            high=action_magnitude,
            shape=(act_steps * action_dim,),
            dtype=np.float32,
        )
        
        # 观察空间
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        self._obs = None
        self._step_count = 0
        self._max_steps = 12  # 100 / 8 ≈ 12
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._obs = np.random.randn(self.obs_dim).astype(np.float32) * 0.1
        self._step_count = 0
        return self._obs, {}
    
    def step(self, action):
        """执行动作（噪声）-> 通过 base_policy 解码 -> 返回结果。"""
        self._step_count += 1
        
        # 将噪声转换为动作
        noise = torch.tensor(action, device=self.device, dtype=torch.float32).unsqueeze(0)
        noise = noise.view(1, self.act_steps, self.action_dim)
        obs_t = torch.tensor(self._obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            real_actions = self.base_policy(obs_t, noise)  # 返回 numpy array
        
        # 简单的奖励函数
        reward = -np.mean(np.abs(real_actions[0, 0]))  # 鼓励小动作
        
        # 更新观察
        self._obs = np.random.randn(self.obs_dim).astype(np.float32) * 0.1
        
        terminated = False
        truncated = self._step_count >= self._max_steps
        
        return self._obs, reward, terminated, truncated, {}


def make_env():
    return SimpleDSRLEnv(
        base_policy=base_policy,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        act_steps=ACT_STEPS,
        action_magnitude=ACTION_MAGNITUDE,
        device=DEVICE,
    )


env = DummyVecEnv([make_env])
print("  ✅ Test environment created")

# ===== 创建并训练 SAC =====
print("\n[3/3] Training SAC for 1000 steps...")

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=10000,
    learning_starts=100,
    batch_size=64,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    verbose=1,
    device=DEVICE,
)

model.learn(total_timesteps=1000)

print("\n" + "=" * 60)
print("✅ DSRL-SAC 简化验证成功!")
print("=" * 60)
print("\nBase policy 成功集成到 SAC 训练流程中。")
print("下一步: 实现完整的 ManiSkill3 环境包装器。")

env.close()
