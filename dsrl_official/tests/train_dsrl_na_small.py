#!/usr/bin/env python3
"""
DSRL-NA 小规模训练脚本

使用 SB3 Fork 的 DSRL 算法，在原始动作空间中训练，使用噪声作为潜变量。
DSRL-NA: Noise as Action in diffusion policy's noise space

Usage:
    CUDA_VISIBLE_DEVICES=1 python train_dsrl_na_small.py
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
import gymnasium as gym
from gymnasium import spaces

print("=" * 70)
print("DSRL-NA Small Scale Training")
print("=" * 70)

# ===== 配置 =====
CHECKPOINT_PATH = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
ENV_ID = "LiftPegUpright-v1"
DEVICE = "cuda"
SEED = 42

# 训练参数
TOTAL_TIMESTEPS = 100_000  # Reduced for quick test
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

# DSRL 超参数
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

# ShortCutFlowWrapper 作为 diffusion_policy
_base_policy = ShortCutFlowWrapper(
    velocity_net=velocity_net,
    visual_encoder=None,  # 我们单独处理视觉编码
    obs_horizon=OBS_HORIZON,
    pred_horizon=PRED_HORIZON,
    action_dim=ACTION_DIM,
    num_inference_steps=8,
    device=DEVICE,
)

# 创建一个包装类，自动传 act_steps 参数给 DSRL 使用
class DSRLDiffusionPolicyWrapper:
    """包装 ShortCutFlowWrapper，使其与 DSRL 算法兼容。
    
    DSRL 调用: diffusion_policy(obs, noise, return_numpy=False)
    不传 act_steps 参数，所以我们需要在这里自动传入。
    """
    def __init__(self, base_policy, act_steps):
        self.base_policy = base_policy
        self.act_steps = act_steps
    
    def __call__(self, obs, noise, return_numpy=False):
        # 调用 base_policy 并传入 act_steps
        return self.base_policy(obs, noise, return_numpy=return_numpy, act_steps=self.act_steps)

base_policy = DSRLDiffusionPolicyWrapper(_base_policy, ACT_HORIZON)
print(f"  ✅ Created ShortCutFlowWrapper (base_policy) with act_steps={ACT_HORIZON}")

# ===== Step 3: 创建 DSRL-NA 环境 =====
print("\n[3/5] Creating DSRL-NA environments...")

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs

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

# DSRL-NA 环境：动作空间是 diffused_actions（56维=8*7）
class DSRLNAEnvWrapper(gym.Env):
    """
    DSRL-NA 环境包装器
    
    DSRL-NA 需要的接口：
    - 观测空间：obs_cond (562维)
    - 动作空间：diffused_actions (56维 = act_chunk * action_dim)
    
    DSRL 算法会：
    1. Actor 输出噪声 (56维，但通过 policy.scale_action 缩放)
    2. 使用 diffusion_policy 将噪声解码为 diffused_actions
    3. 将 diffused_actions 传给环境执行
    """
    
    def __init__(
        self,
        env,
        visual_encoder,
        act_steps=ACT_HORIZON,
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        visual_feature_dim=VISUAL_FEATURE_DIM,
        obs_horizon=OBS_HORIZON,
        device=DEVICE,
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
        
        # 动作空间：diffused_actions (act_chunk * action_dim)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_steps * action_dim,), dtype=np.float32
        )
        
        # 历史缓冲
        self._obs_history = None
        self._cached_obs_cond = None
        
    def _encode_rgb(self, rgb):
        """编码 RGB 观测
        
        Args:
            rgb: numpy array (B, T, H, W, C) 或 torch tensor
        """
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        
        B, T, H, W, C = rgb.shape
        rgb_flat = rgb.reshape(B * T, H, W, C)
        
        # (B*T, H, W, C) -> (B*T, C, H, W)
        rgb_tensor = torch.from_numpy(rgb_flat).to(self.device).float()
        if rgb_tensor.max() > 1.0:
            rgb_tensor = rgb_tensor / 255.0
        rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            features = self.visual_encoder(rgb_tensor)  # (B*T, visual_dim)
        
        features = features.reshape(B, T, -1)
        
        return features
    
    def _build_obs_cond(self, obs):
        """构建 obs_cond
        
        展平后的观测结构：
        - state: (B, obs_horizon, 25)
        - rgb: (B, obs_horizon, 128, 128, 3)
        """
        rgb = obs["rgb"]  # (B, T, H, W, C)
        state = obs["state"]  # (B, T, state_dim)
        
        # 编码 RGB
        visual_features = self._encode_rgb(rgb.cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb)
        # visual_features: (B, T, visual_dim)
        if visual_features.ndim == 2:
            B = rgb.shape[0]
            T = rgb.shape[1]
            visual_features = visual_features.reshape(B, T, -1)
        
        # 处理状态
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device).float()
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device).float()
        
        # 确保状态维度正确
        if state.shape[-1] < self.state_dim:
            padding = torch.zeros(
                state.shape[0], state.shape[1], self.state_dim - state.shape[-1],
                device=self.device
            )
            state = torch.cat([state, padding], dim=-1)
        else:
            state = state[:, :, :self.state_dim]
        
        # 合并特征
        obs_feature = torch.cat([visual_features, state], dim=-1)  # (B, T, visual_dim + state_dim)
        
        # 展平为 obs_cond
        obs_cond = obs_feature.reshape(obs_feature.shape[0], -1)  # (B, T * (visual_dim + state_dim))
        
        return obs_cond
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_cond = self._build_obs_cond(obs)
        self._cached_obs_cond = obs_cond
        return obs_cond.cpu().numpy(), info
    
    def step(self, diffused_actions):
        """执行 diffused_actions（已解码的动作序列）
        
        Args:
            diffused_actions: (B, act_steps * action_dim) 已解码的动作序列
        """
        B = self.num_envs
        
        # 重塑为 (B, act_steps, action_dim)
        if isinstance(diffused_actions, np.ndarray):
            diffused_actions = torch.from_numpy(diffused_actions).to(self.device).float()
        
        # 确保形状正确
        if diffused_actions.shape[0] != B:
            # 可能是 (B * act_steps * action_dim,) 的扁平数组
            diffused_actions = diffused_actions.reshape(B, -1)
        
        actions = diffused_actions.reshape(B, self.act_steps, self.action_dim)
        
        # 执行 act_steps 步
        total_reward = torch.zeros(B, device=self.device)
        terminated = torch.zeros(B, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(B, dtype=torch.bool, device=self.device)
        info = {}
        
        for t in range(self.act_steps):
            action_t = actions[:, t, :].cpu().numpy()
            obs, reward, term, trunc, step_info = self.env.step(action_t)
            
            if isinstance(reward, torch.Tensor):
                reward = reward.to(self.device)
            else:
                reward = torch.tensor(reward, device=self.device)
            
            total_reward += reward
            
            if isinstance(term, torch.Tensor):
                terminated = terminated | term.to(self.device)
            else:
                terminated = terminated | torch.tensor(term, device=self.device, dtype=torch.bool)
                
            if isinstance(trunc, torch.Tensor):
                truncated = truncated | trunc.to(self.device)
            else:
                truncated = truncated | torch.tensor(trunc, device=self.device, dtype=torch.bool)
            
            info = step_info
        
        obs_cond = self._build_obs_cond(obs)
        self._cached_obs_cond = obs_cond
        
        return obs_cond.cpu().numpy(), total_reward.cpu().numpy(), terminated.cpu().numpy(), truncated.cpu().numpy(), info
    
    def get_obs_cond(self):
        """获取当前观测的 obs_cond（给 DSRL 算法使用）"""
        return self._cached_obs_cond
    
    def close(self):
        self.env.close()

dsrl_env = DSRLNAEnvWrapper(
    env=base_envs,
    visual_encoder=visual_encoder,
    act_steps=ACT_HORIZON,
    action_dim=ACTION_DIM,
    state_dim=STATE_DIM,
    visual_feature_dim=VISUAL_FEATURE_DIM,
    obs_horizon=OBS_HORIZON,
    device=DEVICE,
)
print(f"  ✅ Created DSRL-NA env with {N_ENVS} parallel envs")
print(f"  Action space: {dsrl_env.action_space} (diffused_actions: {ACT_HORIZON}*{ACTION_DIM}={ACT_HORIZON*ACTION_DIM})")
print(f"  Observation space: {dsrl_env.observation_space}")

# ===== Step 4: 创建 DSRL 模型 =====
print("\n[4/5] Creating DSRL model...")

from stable_baselines3 import DSRL
from stable_baselines3.common.vec_env import VecEnv

class SB3DSRLEnvAdapter(VecEnv):
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
            info_i = {}
            if dones[i]:
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
    
    def get_obs_cond(self):
        """提供给 DSRL 算法获取当前 obs_cond"""
        return self.env.get_obs_cond()

sb3_env = SB3DSRLEnvAdapter(dsrl_env)

# 创建 DSRL 模型
policy_kwargs = dict(
    net_arch=dict(pi=[512, 512, 512], qf=[512, 512, 512]),
    activation_fn=nn.Tanh,
)

model = DSRL(
    "MlpPolicy",
    sb3_env,
    diffusion_policy=base_policy,
    diffusion_act_dim=(ACT_HORIZON, ACTION_DIM),
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
print(f"  ✅ Created DSRL model with diffusion_policy")
print(f"  diffusion_act_dim: ({ACT_HORIZON}, {ACTION_DIM})")

# ===== Step 5: 训练 =====
print("\n[5/5] Starting training...")

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
save_path = Path(__file__).parent / "checkpoints" / "dsrl_na_small"
save_path.mkdir(parents=True, exist_ok=True)
model.save(save_path / "final_model")
print(f"\n✅ Model saved to {save_path}")

# 最终评估 (使用 predict_diffused)
print("\n" + "=" * 70)
print("Final Evaluation (using predict_diffused)")
print("=" * 70)

sb3_env.reset()
obs = sb3_env._obs
eval_count = 0

pbar = tqdm(total=N_EVAL_EPISODES, desc="Final Evaluation")
while eval_count < N_EVAL_EPISODES:
    # 使用 predict_diffused 进行推理
    # DSRL.predict_diffused 会：
    # 1. 使用 policy 预测噪声
    # 2. 使用 diffusion_policy 将噪声解码为 diffused_actions
    action, _ = model.predict_diffused(obs, deterministic=True)
    
    obs, reward, dones, infos = sb3_env.step_wait()
    sb3_env.step_async(action)
    
    for i, done in enumerate(dones):
        if done and eval_count < N_EVAL_EPISODES:
            eval_count += 1
            pbar.update(1)

pbar.close()

dsrl_env.close()
print("\n✅ Training complete!")
