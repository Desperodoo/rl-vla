#!/usr/bin/env python3
"""
EnvWrapper 评估脚本 - 验证 w=0 时 pretrained model 性能

测试场景:
    EnvWrapper 模式下，SAC 的动作输出是噪声 w，环境内部通过 ShortCut Flow 解码为真实动作。
    当 w=0 时，应该得到与 AgentWrapper 模式（零噪声输入）相同的结果。

目标: w=0 时 success_once 约 80%

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_envwrapper_evaluation.py
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
sys.path.insert(0, str(_root / "dsrl_official"))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

print("=" * 70)
print("EnvWrapper Evaluation - Zero Noise (w=0)")
print("=" * 70)

# ===== 配置 =====
CHECKPOINT_PATH = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
ENV_ID = "LiftPegUpright-v1"
DEVICE = "cuda"
NUM_EVAL_EPISODES = 50
N_ENVS = 50

# 模型参数
OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_HORIZON = 8
ACTION_DIM = 7
VISUAL_FEATURE_DIM = 256
STATE_DIM = 25
ACTION_MAGNITUDE = 1.5

print(f"\nConfig:")
print(f"  checkpoint: {CHECKPOINT_PATH}")
print(f"  env_id: {ENV_ID}")
print(f"  num_eval_episodes: {NUM_EVAL_EPISODES}")
print(f"  n_envs: {N_ENVS}")
print(f"  action_magnitude: {ACTION_MAGNITUDE}")

# ===== Step 1: 加载 Checkpoint =====
print("\n[1/5] Loading checkpoint...")

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
print("\n[2/5] Loading model components...")

from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.plain_conv import PlainConv

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

# ===== Step 3: 创建 ShortCutFlowWrapper =====
print("\n[3/5] Creating ShortCutFlowWrapper...")

from utils import ShortCutFlowWrapper

base_policy = ShortCutFlowWrapper(
    velocity_net=velocity_net,
    visual_encoder=None,  # visual_encoder 在 env_wrapper 中处理
    obs_horizon=OBS_HORIZON,
    pred_horizon=PRED_HORIZON,
    action_dim=ACTION_DIM,
    num_inference_steps=8,
    device=DEVICE,
)
print("  ✅ ShortCutFlowWrapper created")

# ===== Step 4: 创建包装环境 =====
print("\n[4/5] Creating wrapped environments...")

import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs
from env_utils import ManiSkillGPUFlowEnvWrapper

# 创建基础环境
eval_env_kwargs = dict(
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
    env_kwargs=eval_env_kwargs,
    other_kwargs=dict(obs_horizon=OBS_HORIZON),
    video_dir=None,
    wrappers=[FlattenRGBDObservationWrapper],
)

# 包装为 ManiSkillGPUFlowEnvWrapper
# 注意：这个版本的 EnvWrapper 会将观察编码为特征向量
wrapped_env = ManiSkillGPUFlowEnvWrapper(
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
print(f"  ✅ ManiSkillGPUFlowEnvWrapper created")
print(f"  Action space: {wrapped_env.action_space}")
print(f"  Observation space: {wrapped_env.observation_space}")

# ===== Step 5: 评估 =====
print("\n[5/5] Running evaluation with w=0...")

eval_metrics = defaultdict(list)
eps_count = 0

# Reset 环境
obs, info = wrapped_env.reset()

# 零噪声动作
zero_noise = np.zeros((N_ENVS, ACT_HORIZON * ACTION_DIM), dtype=np.float32)

pbar = tqdm(total=NUM_EVAL_EPISODES, desc="EnvWrapper Evaluation (w=0)")
while eps_count < NUM_EVAL_EPISODES:
    # 执行零噪声动作
    obs, rew, terminated, truncated, info = wrapped_env.step(zero_noise)
    
    # 检查 episode 结束 - 处理 tensor 和 ndarray 类型
    if isinstance(terminated, torch.Tensor):
        done = (terminated | truncated)
        done_any = done.any().item()
    elif isinstance(terminated, np.ndarray):
        done = np.logical_or(terminated, truncated)
        done_any = done.any()
    else:
        done = terminated or truncated
        done_any = done
    
    # 处理完成的 episode
    if done_any:
        # ManiSkill3 返回的 info 格式
        if "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, dict) and "episode" in final_info:
                for k, v in final_info["episode"].items():
                    if hasattr(v, "cpu"):
                        v = v.float().cpu().numpy()
                    eval_metrics[k].append(v)
            elif isinstance(final_info, list):
                for fi in final_info:
                    if fi is not None and "episode" in fi:
                        for k, v in fi["episode"].items():
                            eval_metrics[k].append(v)
        
        # 更新计数
        if isinstance(done, (torch.Tensor, np.ndarray)):
            num_done = done.sum().item() if isinstance(done, torch.Tensor) else int(done.sum())
        else:
            num_done = 1
        eps_count += num_done
        pbar.update(num_done)

pbar.close()

# 打印结果
print("\n" + "=" * 70)
print("EnvWrapper Evaluation Results (w=0)")
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
    print("\n  ✅ SUCCESS: EnvWrapper (w=0) achieves >= 75% success_once!")
else:
    print(f"\n  ⚠️ WARNING: EnvWrapper (w=0) achieves only {success_once:.2%} success_once")

wrapped_env.close()

print("\n" + "=" * 70)
print("Evaluation Complete!")
print("=" * 70)
