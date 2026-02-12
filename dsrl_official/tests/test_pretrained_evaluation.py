#!/usr/bin/env python3
"""
Pretrained Model Evaluation - 交叉验证 AgentWrapper 和 EnvWrapper 模式

测试场景：
1. AgentWrapper 模式：使用 velocity_net 从零噪声生成动作，测试 base policy 性能
2. EnvWrapper 模式：使用 ShortCutFlowEnvWrapper，验证 w=0 时的性能

目标：两种模式下 success_once 应该接近 80%

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_pretrained_evaluation.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# 添加路径
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl_offpolicy"))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

print("=" * 70)
print("Pretrained Model Evaluation - Cross Validation")
print("=" * 70)

# ===== 配置 =====
CHECKPOINT_PATH = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
ENV_ID = "LiftPegUpright-v1"
DEVICE = "cuda"
NUM_EVAL_EPISODES = 50
N_ENVS = 50  # 并行评估

# 模型参数
OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_HORIZON = 8
ACTION_DIM = 7
VISUAL_FEATURE_DIM = 256
STATE_DIM = 25

print(f"\nConfig:")
print(f"  checkpoint: {CHECKPOINT_PATH}")
print(f"  env_id: {ENV_ID}")
print(f"  num_eval_episodes: {NUM_EVAL_EPISODES}")
print(f"  n_envs: {N_ENVS}")

# ===== Step 1: 加载 Checkpoint =====
print("\n[1/4] Loading checkpoint...")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
print(f"  Checkpoint keys: {list(checkpoint.keys())}")

# 使用 ema_agent
if "ema_agent" in checkpoint:
    agent_state = checkpoint["ema_agent"]
    print("  Using ema_agent weights")
else:
    agent_state = checkpoint.get("agent", checkpoint)
    print("  Using agent weights")

# ===== Step 2: 加载模型组件 =====
print("\n[2/4] Loading model components...")

from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import AgentWrapper

# 创建 velocity_net
global_cond_dim = OBS_HORIZON * (VISUAL_FEATURE_DIM + STATE_DIM)

velocity_net = ShortCutVelocityUNet1D(
    input_dim=ACTION_DIM,
    global_cond_dim=global_cond_dim,
    diffusion_step_embed_dim=64,
    down_dims=(64, 128, 256),
    n_groups=8,
).to(DEVICE)

# 提取 velocity_net 权重
velocity_net_state = {}
for key, value in agent_state.items():
    if key.startswith("velocity_net."):
        velocity_net_state[key.replace("velocity_net.", "")] = value

velocity_net.load_state_dict(velocity_net_state)
velocity_net.eval()
print(f"  ✅ Loaded velocity_net ({len(velocity_net_state)} keys)")

# 创建 visual_encoder
visual_encoder = PlainConv(
    in_channels=3,
    out_dim=VISUAL_FEATURE_DIM,
    pool_feature_map=True,
).to(DEVICE)

if "visual_encoder" in checkpoint:
    visual_encoder.load_state_dict(checkpoint["visual_encoder"])
    print("  ✅ Loaded visual_encoder")
else:
    print("  ⚠️ No visual_encoder in checkpoint, using random init")

visual_encoder.eval()

# ===== Step 3: 创建评估环境 =====
print("\n[3/4] Creating evaluation environments...")

import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils import common
from diffusion_policy.make_env import make_eval_envs

eval_env_kwargs = dict(
    control_mode="pd_ee_delta_pose",
    max_episode_steps=100,
    obs_mode="rgbd",
    render_mode="rgb_array",
    reward_mode="dense",
)
eval_other_kwargs = dict(obs_horizon=OBS_HORIZON)

eval_envs = make_eval_envs(
    env_id=ENV_ID,
    num_envs=N_ENVS,
    sim_backend="physx_cuda",
    env_kwargs=eval_env_kwargs,
    other_kwargs=eval_other_kwargs,
    video_dir=None,
    wrappers=[FlattenRGBDObservationWrapper],
)

print(f"  ✅ Created {N_ENVS} parallel environments")


# ===== 创建 AgentWrapper 用于评估 =====
class PretrainedAgentWrapper(nn.Module):
    """用于评估 pretrained velocity_net (零噪声输入)"""
    
    def __init__(self, velocity_net, visual_encoder, obs_horizon, pred_horizon, act_horizon, action_dim, device):
        super().__init__()
        self.velocity_net = velocity_net
        self.visual_encoder = visual_encoder
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.action_dim = action_dim
        self.device = device
        
        # 推理步数
        self.num_inference_steps = 8
    
    def encode_obs(self, obs):
        """编码观察"""
        features_list = []
        
        # Visual encoding
        if self.visual_encoder is not None and "rgb" in obs:
            rgb = obs["rgb"]
            if isinstance(rgb, np.ndarray):
                rgb = torch.from_numpy(rgb).to(self.device)
            
            B = rgb.shape[0]
            T = self.obs_horizon
            
            # (B, T, H, W, C) -> (B*T, C, H, W)
            if rgb.dim() == 5:
                rgb = rgb.reshape(B * T, *rgb.shape[2:])
            if rgb.shape[-1] in [3, 4]:  # NHWC -> NCHW
                rgb = rgb.permute(0, 3, 1, 2)
            
            rgb = rgb.float()
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            
            with torch.no_grad():
                visual_feat = self.visual_encoder(rgb)  # (B*T, visual_dim)
            visual_feat = visual_feat.reshape(B, T, -1)
            features_list.append(visual_feat)
        
        # State encoding
        state = obs.get("state", obs.get("agent", None))
        if state is not None:
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).to(self.device)
            state = state.float()
            if state.dim() == 2:
                state = state.unsqueeze(1).expand(-1, self.obs_horizon, -1)
            features_list.append(state)
        
        # Concatenate and flatten
        obs_features = torch.cat(features_list, dim=-1)  # (B, T, visual_dim + state_dim)
        obs_cond = obs_features.reshape(obs_features.shape[0], -1)  # (B, T * (v+s))
        
        return obs_cond
    
    def get_action(self, obs, **kwargs):
        """从零噪声生成动作"""
        with torch.no_grad():
            # 编码观察
            obs_cond = self.encode_obs(obs)
            B = obs_cond.shape[0]
            
            # 初始化零噪声
            x = torch.zeros(B, self.pred_horizon, self.action_dim, device=self.device)
            
            # Flow 积分 (从 t=0 到 t=1)
            dt = 1.0 / self.num_inference_steps
            step_size = torch.full((B,), dt, device=self.device)
            
            for i in range(self.num_inference_steps):
                t = torch.full((B,), i * dt, device=self.device)
                v = self.velocity_net(x, t, step_size, obs_cond)
                x = x + v * dt
            
            # Clamp and return
            actions = torch.clamp(x, -1.0, 1.0)
            
            # 返回 act_horizon 步
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            return actions[:, start:end]


# ===== Step 4: 评估 =====
print("\n[4/4] Running evaluation...")

# 创建 agent wrapper
agent_wrapper = PretrainedAgentWrapper(
    velocity_net=velocity_net,
    visual_encoder=visual_encoder,
    obs_horizon=OBS_HORIZON,
    pred_horizon=PRED_HORIZON,
    act_horizon=ACT_HORIZON,
    action_dim=ACTION_DIM,
    device=DEVICE,
).to(DEVICE)
agent_wrapper.eval()

# 评估循环
eval_metrics = defaultdict(list)
obs, info = eval_envs.reset()
eps_count = 0

pbar = tqdm(total=NUM_EVAL_EPISODES, desc="AgentWrapper Evaluation")
while eps_count < NUM_EVAL_EPISODES:
    obs = common.to_tensor(obs, DEVICE)
    action_seq = agent_wrapper.get_action(obs)
    
    for i in range(action_seq.shape[1]):
        obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
        if truncated.any():
            break
    
    if truncated.any():
        if isinstance(info["final_info"], dict):
            for k, v in info["final_info"]["episode"].items():
                eval_metrics[k].append(v.float().cpu().numpy())
        else:
            for final_info in info["final_info"]:
                for k, v in final_info["episode"].items():
                    eval_metrics[k].append(v)
        eps_count += eval_envs.num_envs
        pbar.update(eval_envs.num_envs)

pbar.close()

# 打印结果
print("\n" + "=" * 70)
print("AgentWrapper Evaluation Results (Zero Noise)")
print("=" * 70)
for k in eval_metrics.keys():
    values = np.concatenate(eval_metrics[k]) if isinstance(eval_metrics[k][0], np.ndarray) else eval_metrics[k]
    mean_val = np.mean(values)
    print(f"  {k}: {mean_val:.4f}")

success_once = np.mean(eval_metrics.get("success_once", [0]))
success_at_end = np.mean(eval_metrics.get("success_at_end", [0]))
print(f"\n  >>> success_once: {success_once:.2%}")
print(f"  >>> success_at_end: {success_at_end:.2%}")

if success_once >= 0.75:
    print("\n  ✅ SUCCESS: Pretrained model achieves >= 75% success_once!")
else:
    print(f"\n  ⚠️ WARNING: Pretrained model achieves only {success_once:.2%} success_once")

eval_envs.close()

print("\n" + "=" * 70)
print("Evaluation Complete!")
print("=" * 70)
