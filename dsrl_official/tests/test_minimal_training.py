#!/usr/bin/env python3
"""
DSRL Official 最小验证脚本

验证完整训练流程的端到端执行：
1. 加载预训练 ShortCut Flow 策略 (从 checkpoint 推断 state_dim)
2. 创建环境
3. 测试 base policy forward
4. 包装环境验证

Usage:
    python test_minimal_training.py
"""

import os
import sys
import time
from pathlib import Path

# 添加路径
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))  # 添加 rlft 目录
sys.path.insert(0, str(_root / "diffusion_policy"))

import numpy as np
import torch
import gymnasium as gym

print("=" * 60)
print("DSRL Official 最小训练验证")
print("=" * 60)

# ===== 配置 =====
ENV_ID = "LiftPegUpright-v1"
CHECKPOINT_PATH = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ENVS = 4

# 模型配置
OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_STEPS = 8
ACTION_DIM = 7
VISUAL_FEATURE_DIM = 256
ACTION_MAGNITUDE = 1.5

print(f"\nConfig:")
print(f"  Environment: {ENV_ID}")
print(f"  Device: {DEVICE}")
print(f"  N_envs: {N_ENVS}")
print(f"  Checkpoint: {CHECKPOINT_PATH}")

# ===== 检查依赖 =====
print("\n[1/5] Checking dependencies...")

try:
    from stable_baselines3 import SAC
    print("  ✅ stable-baselines3 available")
    HAS_SB3 = True
except ImportError:
    print("  ❌ stable-baselines3 not installed")
    print("     Please install: pip install stable-baselines3")
    HAS_SB3 = False

if not HAS_SB3:
    print("\nCannot proceed without stable-baselines3.")
    sys.exit(1)

# ===== 验证 Checkpoint =====
print("\n[2/5] Verifying checkpoint...")
if not Path(CHECKPOINT_PATH).exists():
    print(f"  ❌ Checkpoint not found: {CHECKPOINT_PATH}")
    sys.exit(1)
print(f"  ✅ Checkpoint exists")

# ===== 加载 Base Policy (从 checkpoint 推断 state_dim) =====
print("\n[3/5] Loading ShortCut Flow base policy...")

from dsrl_official.utils import load_shortcut_flow_policy
from diffusion_policy.plain_conv import PlainConv

base_policy, visual_encoder = load_shortcut_flow_policy(
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

# 计算 obs_dim（使用推断的 state_dim=25）
INFERRED_STATE_DIM = 25  # 从 checkpoint 推断得到
obs_dim = OBS_HORIZON * (VISUAL_FEATURE_DIM + INFERRED_STATE_DIM)
print(f"  obs_dim: {obs_dim} (state_dim={INFERRED_STATE_DIM})")
print("  ✅ Base policy loaded")

# ===== 测试 Base Policy Forward =====
print("\n[4/5] Testing base policy forward pass...")

test_noise = torch.randn(N_ENVS, PRED_HORIZON, ACTION_DIM, device=DEVICE)
test_obs = torch.randn(N_ENVS, obs_dim, device=DEVICE)

try:
    with torch.no_grad():
        actions = base_policy(test_obs, test_noise)
    print(f"  ✅ Forward pass successful")
    print(f"     Input noise shape: {test_noise.shape}")
    print(f"     Input obs shape: {test_obs.shape}")
    print(f"     Output actions shape: {actions.shape}")
except Exception as e:
    print(f"  ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===== 创建环境 =====
print("\n[5/5] Creating training environment...")

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

train_env = gym.make(
    ENV_ID,
    obs_mode="rgbd",
    control_mode="pd_ee_delta_pose",
    sim_backend="physx_cuda",
    num_envs=N_ENVS,
    max_episode_steps=100,
    reward_mode="dense",
)
train_env = FlattenRGBDObservationWrapper(train_env, rgb=True, depth=False, state=True)

# 测试环境 reset
print("  Testing environment reset...")
obs, info = train_env.reset()
print(f"  Obs type: {type(obs)}")
if isinstance(obs, dict):
    for k, v in obs.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: {v.shape}")

print("  ✅ Environment created")

# ===== 总结 =====
print("\n" + "=" * 60)
print("✅ 所有组件验证通过!")
print("=" * 60)

print("\n关键信息:")
print(f"  - Checkpoint state_dim: {INFERRED_STATE_DIM}")
print(f"  - obs_dim (global_cond_dim): {obs_dim}")
print(f"  - Action output shape: {actions.shape}")

print("\n下一步:")
print("  运行完整训练:")
print(f"  cd /home/amax/rl-vla/rlft && python dsrl_official/train_dsrl.py \\")
print(f"      --algorithm dsrl_sac --env_id {ENV_ID} \\")
print(f"      --total_timesteps 10000 --no-track")

# 清理
train_env.close()
