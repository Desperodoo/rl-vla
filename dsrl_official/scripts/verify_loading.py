"""验证 train_stage1_offline.py 是如何加载维度不匹配的 checkpoint 的"""

import os
import sys
import torch
from pathlib import Path

# Setup imports
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl"))
sys.path.insert(0, str(_root / "dsrl_offpolicy"))

from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.algorithms.networks import DoubleQNetwork

checkpoint_path = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"

print("=" * 60)
print("验证模型创建和 checkpoint 加载")
print("=" * 60)

checkpoint = torch.load(checkpoint_path, map_location="cpu")
agent_state = checkpoint.get("ema_agent", checkpoint.get("agent"))

# 提取 velocity_net 和 q_network weights
velocity_net_state = {}
q_network_state = {}

for key, value in agent_state.items():
    if key.startswith("velocity_net."):
        velocity_net_state[key.replace("velocity_net.", "")] = value
    elif key.startswith("q_network.") or key.startswith("critic."):
        new_key = key.replace("q_network.", "").replace("critic.", "")
        q_network_state[new_key] = value

print(f"\n1. Checkpoint 中的权重信息:")
print(f"   velocity_net keys: {len(velocity_net_state)}")
print(f"   q_network keys: {len(q_network_state)}")

# 检查 checkpoint 中的关键维度参数
# 需要找到正确的 key 来获取 global_cond_dim
print(f"\n2. Checkpoint 中的部分 key 示例:")
vel_keys = list(velocity_net_state.keys())[:5]
print(f"   velocity_net keys (前5个): {vel_keys}")
q_keys = list(q_network_state.keys())[:5]
print(f"   q_network keys (前5个): {q_keys}")

# 寻找包含维度信息的 key
checkpoint_vel_global_cond_dim = None
checkpoint_q_input_dim = None
for key, value in velocity_net_state.items():
    if "encoder" in key and "0.weight" in key and len(value.shape) == 2:
        print(f"   velocity_net.{key}: {value.shape}")
        checkpoint_vel_global_cond_dim = value.shape[1]
        break

for key, value in q_network_state.items():
    if key == "nets.0.weight":
        print(f"   q_network.{key}: {value.shape}")
        checkpoint_q_input_dim = value.shape[1]
        break


# 模拟 train_stage1_offline.py 的行为
state_dim = 25  # 从数据集获取
visual_feature_dim = 256
obs_horizon = 2
action_dim = 7
act_horizon = 8

global_cond_dim = obs_horizon * (visual_feature_dim + state_dim)
print(f"\n3. train_stage1_offline.py 计算的维度:")
print(f"   state_dim = {state_dim}")
print(f"   visual_feature_dim = {visual_feature_dim}")
print(f"   global_cond_dim = {obs_horizon} * ({visual_feature_dim} + {state_dim}) = {global_cond_dim}")

# 创建新模型
print("\n4. 创建新模型 (global_cond_dim={})...".format(global_cond_dim))

velocity_net = ShortCutVelocityUNet1D(
    input_dim=action_dim,
    global_cond_dim=global_cond_dim,
    diffusion_step_embed_dim=64,
    down_dims=(64, 128, 256),
    n_groups=8,
)

q_network = DoubleQNetwork(
    action_dim=action_dim,
    obs_dim=global_cond_dim,
    action_horizon=act_horizon,
    hidden_dims=[512, 512, 512],
)

print(f"   新 velocity_net 参数量: {sum(p.numel() for p in velocity_net.parameters()) / 1e6:.2f}M")
print(f"   新 q_network.q1_net 参数量: {sum(p.numel() for p in q_network.q1_net.parameters()) / 1e6:.2f}M")
print(f"   新 q_network 参数量: {sum(p.numel() for p in q_network.parameters()) / 1e6:.2f}M")

# 尝试加载
print("\n5. 尝试加载 checkpoint 权重...")

try:
    velocity_net.load_state_dict(velocity_net_state)
    print("   ✅ velocity_net 加载成功")
except RuntimeError as e:
    print(f"   ❌ velocity_net 加载失败: {e}")
    
    # 使用 strict=False 尝试
    print("\n   尝试 strict=False...")
    result = velocity_net.load_state_dict(velocity_net_state, strict=False)
    print(f"   missing_keys: {result.missing_keys[:5]}..." if len(result.missing_keys) > 5 else f"   missing_keys: {result.missing_keys}")
    print(f"   unexpected_keys: {result.unexpected_keys[:5]}..." if len(result.unexpected_keys) > 5 else f"   unexpected_keys: {result.unexpected_keys}")

try:
    q_network.load_state_dict(q_network_state)
    print("   ✅ q_network 加载成功")
except RuntimeError as e:
    print(f"   ❌ q_network 加载失败: {e}")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
if checkpoint_vel_global_cond_dim:
    print(f"Checkpoint global_cond_dim: {checkpoint_vel_global_cond_dim} (velocity_net)")
print(f"Dataset-based global_cond_dim: {global_cond_dim}")
if checkpoint_vel_global_cond_dim:
    print(f"维度差异: {checkpoint_vel_global_cond_dim - global_cond_dim}")
print()
print("如果 velocity_net 加载失败但 q_network 成功:")
print("  --> 说明 checkpoint 中 velocity_net 和 q_network 是用不同维度训练的")
print("  --> 这可能是 checkpoint 混合或训练配置不一致导致的")
