#!/usr/bin/env python3
"""
DSRL-SAC 小规模训练脚本

使用 EnvWrapper 模式训练 SAC，动作空间为噪声空间。

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_dsrl_sac_small.py
"""

import os
import sys
from pathlib import Path

# 添加路径
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl_offpolicy"))
sys.path.insert(0, str(_root / "dsrl_official"))

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
import time

print("=" * 70)
print("DSRL-SAC Small Scale Training")
print("=" * 70)

# ===== 配置 =====
CHECKPOINT_PATH = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
ENV_ID = "LiftPegUpright-v1"
DEVICE = "cuda"
SEED = 42

# 训练参数
TOTAL_TIMESTEPS = 100_000  # 减少到 20k 用于快速测试
N_ENVS = 50
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 50

# 模型参数
OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_HORIZON = 8
ACTION_DIM = 7
VISUAL_FEATURE_DIM = 256
STATE_DIM = 25
ACTION_MAGNITUDE = 1.5

# SAC 超参数
LEARNING_RATE = 3e-4
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.99
LEARNING_STARTS = 1000
TRAIN_FREQ = 1
GRADIENT_STEPS = 20  # UTD ratio

print(f"\nConfig:")
print(f"  total_timesteps: {TOTAL_TIMESTEPS}")
print(f"  n_envs: {N_ENVS}")
print(f"  learning_rate: {LEARNING_RATE}")
print(f"  gradient_steps (UTD): {GRADIENT_STEPS}")

# ===== Step 1: 加载 Checkpoint =====
print("\n[1/5] Loading checkpoint...")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
agent_state = checkpoint.get("ema_agent", checkpoint.get("agent", checkpoint))

# ===== Step 2: 加载模型组件 =====
print("\n[2/5] Loading model components...")

from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.plain_conv import PlainConv
from utils import ShortCutFlowWrapper

global_cond_dim = OBS_HORIZON * (VISUAL_FEATURE_DIM + STATE_DIM)

velocity_net = ShortCutVelocityUNet1D(
    input_dim=ACTION_DIM,
    global_cond_dim=global_cond_dim,
    diffusion_step_embed_dim=64,
    down_dims=(64, 128, 256),
    n_groups=8,
).to(DEVICE)

velocity_net_state = {k.replace("velocity_net.", ""): v for k, v in agent_state.items() if k.startswith("velocity_net.")}
velocity_net.load_state_dict(velocity_net_state)
velocity_net.eval()
print(f"  ✅ Loaded velocity_net")

visual_encoder = PlainConv(in_channels=3, out_dim=VISUAL_FEATURE_DIM, pool_feature_map=True).to(DEVICE)
if "visual_encoder" in checkpoint:
    visual_encoder.load_state_dict(checkpoint["visual_encoder"])
visual_encoder.eval()
print(f"  ✅ Loaded visual_encoder")

base_policy = ShortCutFlowWrapper(
    velocity_net=velocity_net,
    visual_encoder=None,
    obs_horizon=OBS_HORIZON,
    pred_horizon=PRED_HORIZON,
    action_dim=ACTION_DIM,
    num_inference_steps=8,
    device=DEVICE,
)
print(f"  ✅ Created ShortCutFlowWrapper")

# ===== Step 3: 创建训练环境 =====
print("\n[3/5] Creating training environments...")

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs
from env_utils import ManiSkillGPUFlowEnvWrapper

env_kwargs = dict(
    control_mode="pd_ee_delta_pose",
    max_episode_steps=100,
    obs_mode="rgbd",
    render_mode="rgb_array",
    reward_mode="dense",
)

base_envs = make_eval_envs(
    env_id=ENV_ID,
    num_envs=N_ENVS,
    sim_backend="physx_cuda",
    env_kwargs=env_kwargs,
    other_kwargs=dict(obs_horizon=OBS_HORIZON),
    video_dir=None,
    wrappers=[FlattenRGBDObservationWrapper],
)

train_env = ManiSkillGPUFlowEnvWrapper(
    env=base_envs,
    base_policy=base_policy,
    visual_encoder=visual_encoder,
    action_magnitude=ACTION_MAGNITUDE,
    act_steps=ACT_HORIZON,
    action_dim=ACTION_DIM,
    state_dim=STATE_DIM,
    visual_feature_dim=VISUAL_FEATURE_DIM,
    obs_horizon=OBS_HORIZON,
    include_rgb=True,
    device=DEVICE,
)
print(f"  ✅ Created training env with {N_ENVS} parallel envs")
print(f"  Action space: {train_env.action_space}")
print(f"  Observation space: {train_env.observation_space}")

# ===== Step 4: 创建 SAC 模型 =====
print("\n[4/5] Creating SAC model...")

from stable_baselines3 import SAC

# 为 SB3 创建一个适配的环境
from stable_baselines3.common.vec_env import VecEnv

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
            info_i = {}
            if dones[i]:
                # 处理 final_info
                if "final_info" in info:
                    final_info = info["final_info"]
                    if isinstance(final_info, dict) and "episode" in final_info:
                        ep_info = final_info["episode"]
                        info_i["episode"] = {
                            "r": ep_info.get("return", ep_info.get("reward", [0]))[i].item() if hasattr(ep_info.get("return", ep_info.get("reward", [0])), '__getitem__') else 0,
                            "l": ep_info.get("episode_len", [100])[i].item() if hasattr(ep_info.get("episode_len", [100]), '__getitem__') else 100,
                        }
            infos.append(info_i)
        
        self._obs = obs
        return obs, reward, dones, infos
    
    def close(self):
        self.env.close()
    
    def seed(self, seed=None):
        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
    
    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs
    
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name, None)] * self.num_envs
    
    def set_attr(self, attr_name, value, indices=None):
        pass

sb3_env = SB3EnvAdapter(train_env)

# 创建 SAC 模型
policy_kwargs = dict(
    net_arch=dict(pi=[512, 512, 512], qf=[512, 512, 512]),
    activation_fn=nn.Tanh,
)

model = SAC(
    "MlpPolicy",
    sb3_env,
    learning_rate=LEARNING_RATE,
    buffer_size=BUFFER_SIZE,
    learning_starts=LEARNING_STARTS,
    batch_size=BATCH_SIZE,
    tau=TAU,
    gamma=GAMMA,
    train_freq=TRAIN_FREQ,
    gradient_steps=GRADIENT_STEPS,
    verbose=1,
    policy_kwargs=policy_kwargs,
    seed=SEED,
    device=DEVICE,
)
print(f"  ✅ Created SAC model")

# ===== Step 5: 训练 =====
print("\n[5/5] Starting training...")

from stable_baselines3.common.callbacks import BaseCallback

class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_success_rate = 0.0
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            success_rate = self._evaluate()
            if self.verbose:
                print(f"\nStep {self.n_calls}: Success rate = {success_rate:.2%}")
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                print(f"  New best: {success_rate:.2%}")
        return True
    
    def _evaluate(self):
        obs = self.eval_env.reset()
        success_count = 0
        episode_count = 0
        
        while episode_count < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, dones, infos = self.eval_env.step_wait()
            self.eval_env.step_async(action)
            
            for i, done in enumerate(dones):
                if done:
                    episode_count += 1
                    if "episode" in infos[i]:
                        # 检查 success
                        pass
        
        return success_count / max(episode_count, 1)

# 简单训练
print(f"\nTraining for {TOTAL_TIMESTEPS} timesteps...")
start_time = time.time()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
    )
except KeyboardInterrupt:
    print("\nTraining interrupted by user")

elapsed = time.time() - start_time
print(f"\nTraining completed in {elapsed:.1f}s")
print(f"  Steps per second: {TOTAL_TIMESTEPS / elapsed:.1f}")

# 保存模型
save_path = Path(__file__).parent / "checkpoints" / "dsrl_sac_small"
save_path.mkdir(parents=True, exist_ok=True)
model.save(save_path / "final_model")
print(f"\n✅ Model saved to {save_path}")

# 最终评估
print("\n" + "=" * 70)
print("Final Evaluation")
print("=" * 70)

sb3_env.reset()
obs = sb3_env._obs
eval_metrics = defaultdict(list)
eps_count = 0

pbar = tqdm(total=N_EVAL_EPISODES, desc="Final Evaluation")
while eps_count < N_EVAL_EPISODES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = sb3_env.step_wait()
    sb3_env.step_async(action)
    
    for i, done in enumerate(dones):
        if done and eps_count < N_EVAL_EPISODES:
            eps_count += 1
            pbar.update(1)

pbar.close()

train_env.close()
print("\n✅ Training complete!")
