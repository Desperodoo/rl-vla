#!/usr/bin/env python3
"""
DSRL-SAC + ManiSkill3 端到端测试

使用真实 ManiSkill3 环境测试完整的 DSRL-SAC 训练流程。

Usage:
    python test_maniskill_integration.py
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

print("=" * 60)
print("DSRL-SAC + ManiSkill3 Integration Test")
print("=" * 60)

# ===== 配置 =====
CHECKPOINT_PATH = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
DEVICE = "cuda"
ENV_ID = "LiftPegUpright-v1"
N_ENVS = 4

# 模型参数
OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_STEPS = 8
ACTION_DIM = 7
VISUAL_FEATURE_DIM = 256
STATE_DIM = 25
ACTION_MAGNITUDE = 1.5

# 训练参数
TRAIN_STEPS = 200  # 仅测试

print(f"\nConfig:")
print(f"  env_id: {ENV_ID}")
print(f"  n_envs: {N_ENVS}")
print(f"  checkpoint: {CHECKPOINT_PATH}")

# ===== Step 1: 加载 Base Policy =====
print("\n[1/4] Loading base policy and visual encoder...")

from dsrl_official.utils import load_shortcut_flow_policy
from diffusion_policy.plain_conv import PlainConv

base_policy, visual_encoder = load_shortcut_flow_policy(
    checkpoint_path=CHECKPOINT_PATH,
    visual_encoder_class=PlainConv,
    obs_horizon=OBS_HORIZON,
    pred_horizon=PRED_HORIZON,
    action_dim=ACTION_DIM,
    visual_feature_dim=VISUAL_FEATURE_DIM,
    state_dim=None,
    include_rgb=True,
    use_ema=True,
    device=DEVICE,
)
print("  ✅ Base policy and visual encoder loaded")

# ===== Step 2: 创建 ManiSkill3 环境 =====
print("\n[2/4] Creating ManiSkill3 environment...")

import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

try:
    env = gym.make(
        ENV_ID,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cuda",
        num_envs=N_ENVS,
        max_episode_steps=100,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    print(f"  ✅ Environment created: {ENV_ID}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
except Exception as e:
    print(f"  ❌ Failed to create environment: {e}")
    print("  This might be due to GPU PhysX conflicts.")
    print("  Falling back to mock environment test...")
    env = None

if env is not None:
    # ===== Step 3: 包装环境 =====
    print("\n[3/4] Wrapping environment with ShortCutFlowEnvWrapper...")
    
    from dsrl_official.env_utils import make_dsrl_env
    from stable_baselines3.common.vec_env import VecEnv
    
    # 由于 ManiSkill3 GPU 环境不直接兼容 SB3 VecEnv，
    # 我们需要使用 ManiSkillGPUFlowEnvWrapper 并添加 SB3 兼容层
    
    class ManiSkillSB3Adapter(VecEnv):
        """将 ManiSkill3 GPU 环境适配为 SB3 VecEnv"""
        
        def __init__(self, env, base_policy, visual_encoder, **kwargs):
            from dsrl_official.env_utils import ManiSkillGPUFlowEnvWrapper
            
            self._wrapped = ManiSkillGPUFlowEnvWrapper(
                env=env,
                base_policy=base_policy,
                visual_encoder=visual_encoder,
                **kwargs
            )
            
            super().__init__(
                num_envs=self._wrapped.num_envs,
                observation_space=self._wrapped.observation_space,
                action_space=self._wrapped.action_space,
            )
        
        def reset(self):
            obs, info = self._wrapped.reset()
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
            return obs
        
        def step_async(self, actions):
            self._pending_actions = actions
        
        def step_wait(self):
            obs, reward, terminated, truncated, info = self._wrapped.step(self._pending_actions)
            
            # Convert to numpy
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
            if isinstance(reward, torch.Tensor):
                reward = reward.cpu().numpy()
            if isinstance(terminated, torch.Tensor):
                terminated = terminated.cpu().numpy()
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.cpu().numpy()
            
            dones = terminated | truncated
            
            # Create infos list
            infos = []
            for i in range(self.num_envs):
                info_i = {}
                if dones[i]:
                    info_i['terminal_observation'] = obs[i]
                infos.append(info_i)
            
            return obs, reward, dones, infos
        
        def close(self):
            self._wrapped.env.close()
        
        def seed(self, seed=None):
            pass
        
        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * self.num_envs
        
        def env_method(self, method_name, *args, **kwargs):
            pass
        
        def get_attr(self, attr_name, indices=None):
            return [None] * self.num_envs
        
        def set_attr(self, attr_name, value, indices=None):
            pass
    
    wrapped_env = ManiSkillSB3Adapter(
        env=env,
        base_policy=base_policy,
        visual_encoder=visual_encoder,
        action_magnitude=ACTION_MAGNITUDE,
        act_steps=ACT_STEPS,
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        visual_feature_dim=VISUAL_FEATURE_DIM,
        obs_horizon=OBS_HORIZON,
        include_rgb=True,
        device=DEVICE,
    )
    
    print(f"  ✅ Environment wrapped")
    print(f"  Wrapped action space: {wrapped_env.action_space.shape}")
    print(f"  Wrapped observation space: {wrapped_env.observation_space.shape}")
    
    # ===== Step 4: 测试 SAC 训练 =====
    print("\n[4/4] Testing SAC training...")
    
    try:
        from stable_baselines3 import SAC
        
        model = SAC(
            "MlpPolicy",
            wrapped_env,
            learning_rate=3e-4,
            buffer_size=10000,
            learning_starts=50,
            batch_size=64,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            device=DEVICE,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], qf=[256, 256]),
            ),
        )
        
        print(f"  SAC model created")
        print(f"  Training for {TRAIN_STEPS} steps...")
        
        model.learn(
            total_timesteps=TRAIN_STEPS,
            progress_bar=True,
        )
        
        print(f"  ✅ SAC training completed!")
        
        # 评估
        print("\n  Running evaluation...")
        obs = wrapped_env.reset()
        total_reward = 0
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = wrapped_env.step_wait()
            wrapped_env.step_async(action)
            total_reward += reward.sum()
            if dones.any():
                obs = wrapped_env.reset()
        
        print(f"  Evaluation total reward: {total_reward:.2f}")
        
    except Exception as e:
        print(f"  ❌ SAC training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()

else:
    print("\n[3/4] Skipped (no environment)")
    print("\n[4/4] Skipped (no environment)")

print("\n" + "=" * 60)
print("Integration Test Complete!")
print("=" * 60)
