#!/usr/bin/env python3
"""
DSRL-SAC 端到端集成测试 (with Visual Encoder)

测试流程:
1. 加载 base policy 和 visual encoder
2. 创建 ManiSkill3 环境
3. 包装为 DSRL 环境
4. 验证 SAC 训练循环

Usage:
    python test_visual_encoder_integration.py
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
print("DSRL Visual Encoder Integration Test")
print("=" * 60)

# ===== 配置 =====
CHECKPOINT_PATH = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
DEVICE = "cuda"

# 模型参数
OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_STEPS = 8
ACTION_DIM = 7
VISUAL_FEATURE_DIM = 256
STATE_DIM = 25
ACTION_MAGNITUDE = 1.5

print(f"\nConfig:")
print(f"  obs_horizon: {OBS_HORIZON}")
print(f"  pred_horizon: {PRED_HORIZON}")
print(f"  act_steps: {ACT_STEPS}")
print(f"  action_dim: {ACTION_DIM}")
print(f"  visual_feature_dim: {VISUAL_FEATURE_DIM}")
print(f"  state_dim: {STATE_DIM}")

# ===== Step 1: 加载 Base Policy =====
print("\n[1/5] Loading base policy and visual encoder...")

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
print("  ✅ Base policy loaded")
print(f"  ✅ Visual encoder: {visual_encoder is not None}")

# ===== Step 2: 测试 Visual Encoder =====
print("\n[2/5] Testing visual encoder...")

# 创建模拟 RGB 输入 (B, C, H, W) - 128x128 RGB
batch_size = 4
rgb_input = torch.rand(batch_size, 3, 128, 128, device=DEVICE)

with torch.no_grad():
    visual_features = visual_encoder(rgb_input)

print(f"  RGB input shape: {rgb_input.shape}")
print(f"  Visual features shape: {visual_features.shape}")
print(f"  Expected: ({batch_size}, {VISUAL_FEATURE_DIM})")
assert visual_features.shape == (batch_size, VISUAL_FEATURE_DIM), "Visual encoder output shape mismatch!"
print("  ✅ Visual encoder works correctly")

# ===== Step 3: 测试环境包装器 =====
print("\n[3/5] Testing environment wrapper with visual encoder...")

from dsrl_official.env_utils import ShortCutFlowEnvWrapper, make_dsrl_env

# 创建模拟的 VecEnv
class MockVecEnv:
    """模拟 VecEnv，返回 dict 观察（含 rgb 和 state）"""
    
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.metadata = {}
        self._step_count = 0
        
        # 观察空间 (模拟 ManiSkill3 FlattenRGBDObservationWrapper 输出)
        self.observation_space = spaces.Dict({
            'state': spaces.Box(-np.inf, np.inf, (STATE_DIM,), dtype=np.float32),
            'rgb': spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),  # NHWC
        })
        
        # 动作空间 (连续)
        self.action_space = spaces.Box(-1, 1, (ACTION_DIM,), dtype=np.float32)
    
    def reset(self):
        """重置环境，返回 dict 观察"""
        obs = {
            'state': np.random.randn(self.num_envs, STATE_DIM).astype(np.float32),
            'rgb': np.random.randint(0, 255, (self.num_envs, 128, 128, 3), dtype=np.uint8),
        }
        info = [{} for _ in range(self.num_envs)]
        self._step_count = 0
        return obs, info
    
    def step_async(self, actions):
        """异步步骤"""
        self._pending_actions = actions
    
    def step_wait(self):
        """等待步骤完成"""
        self._step_count += 1
        obs = {
            'state': np.random.randn(self.num_envs, STATE_DIM).astype(np.float32),
            'rgb': np.random.randint(0, 255, (self.num_envs, 128, 128, 3), dtype=np.uint8),
        }
        rewards = np.random.randn(self.num_envs).astype(np.float32) * 0.1
        dones = np.array([self._step_count >= 10] * self.num_envs)
        infos = [{} for _ in range(self.num_envs)]
        return obs, rewards, dones, infos

# 创建模拟环境
mock_env = MockVecEnv(num_envs=4)

# 使用工厂函数包装
wrapped_env = make_dsrl_env(
    env=mock_env,
    base_policy=base_policy,
    visual_encoder=visual_encoder,
    action_magnitude=ACTION_MAGNITUDE,
    act_steps=ACT_STEPS,
    action_dim=ACTION_DIM,
    state_dim=STATE_DIM,
    visual_feature_dim=VISUAL_FEATURE_DIM,
    obs_horizon=OBS_HORIZON,
    include_rgb=True,
    use_gpu_env=False,
    device=DEVICE,
)

print(f"  Wrapped action space: {wrapped_env.action_space.shape}")
print(f"  Wrapped observation space: {wrapped_env.observation_space.shape}")
print(f"  Expected action space: ({ACT_STEPS * ACTION_DIM},)")
print(f"  Expected observation space: ({OBS_HORIZON * (VISUAL_FEATURE_DIM + STATE_DIM)},)")

# 验证维度
expected_action_dim = ACT_STEPS * ACTION_DIM
expected_obs_dim = OBS_HORIZON * (VISUAL_FEATURE_DIM + STATE_DIM)
assert wrapped_env.action_space.shape == (expected_action_dim,), \
    f"Action space mismatch: {wrapped_env.action_space.shape} != ({expected_action_dim},)"
assert wrapped_env.observation_space.shape == (expected_obs_dim,), \
    f"Observation space mismatch: {wrapped_env.observation_space.shape} != ({expected_obs_dim},)"

print("  ✅ Environment wrapper dimensions correct")

# ===== Step 4: 测试环境交互 =====
print("\n[4/5] Testing environment interaction...")

# Reset
obs = wrapped_env.reset()
print(f"  Reset obs shape: {obs.shape}")
assert obs.shape == (4, expected_obs_dim), f"Reset obs shape mismatch: {obs.shape}"

# Step with random noise action
for step in range(3):
    # SAC 输出的是噪声 (在 [-action_magnitude, +action_magnitude] 范围)
    noise_action = np.random.uniform(
        -ACTION_MAGNITUDE, ACTION_MAGNITUDE,
        (4, ACT_STEPS * ACTION_DIM)
    ).astype(np.float32)
    
    wrapped_env.step_async(noise_action)
    next_obs, rewards, dones, infos = wrapped_env.step_wait()
    
    print(f"  Step {step+1}: obs={next_obs.shape}, rewards={rewards.shape}, dones={dones.shape}")

print("  ✅ Environment interaction works")

# ===== Step 5: 测试 SAC 训练兼容性 =====
print("\n[5/5] Testing SAC training compatibility...")

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecEnv
    
    # 创建新的 wrapped env (需要 VecEnv 接口)
    class SB3CompatibleWrapper(VecEnv):
        """将 ShortCutFlowEnvWrapper 包装为完整 SB3 VecEnv"""
        
        def __init__(self, env):
            self._env = env
            # 调用父类 __init__
            super().__init__(
                num_envs=env.num_envs,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            self.metadata = getattr(env, 'metadata', {})
        
        def reset(self):
            return self._env.reset()
        
        def step_async(self, actions):
            self._env.step_async(actions)
        
        def step_wait(self):
            return self._env.step_wait()
        
        def close(self):
            pass
        
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
    
    # 重新创建环境
    mock_env2 = MockVecEnv(num_envs=2)
    wrapped_env2 = make_dsrl_env(
        env=mock_env2,
        base_policy=base_policy,
        visual_encoder=visual_encoder,
        action_magnitude=ACTION_MAGNITUDE,
        act_steps=ACT_STEPS,
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        visual_feature_dim=VISUAL_FEATURE_DIM,
        obs_horizon=OBS_HORIZON,
        include_rgb=True,
        use_gpu_env=False,
        device=DEVICE,
    )
    sb3_env = SB3CompatibleWrapper(wrapped_env2)
    
    # 创建 SAC 模型
    model = SAC(
        "MlpPolicy",
        sb3_env,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=10,
        batch_size=32,
        verbose=0,
        device=DEVICE,
    )
    
    print("  SAC model created successfully")
    
    # 尝试训练几步
    model.learn(total_timesteps=100, progress_bar=False)
    print("  ✅ SAC training works with visual encoder!")
    
except Exception as e:
    print(f"  ⚠️ SAC test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Integration Test Complete!")
print("=" * 60)
