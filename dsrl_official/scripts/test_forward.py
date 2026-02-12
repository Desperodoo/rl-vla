"""验证加载后的模型是否能正常 forward"""

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
print("验证加载后的模型是否能正常 forward")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(checkpoint_path, map_location=device)
agent_state = checkpoint.get("ema_agent", checkpoint.get("agent"))

# 提取权重
velocity_net_state = {}
q_network_state = {}

for key, value in agent_state.items():
    if key.startswith("velocity_net."):
        velocity_net_state[key.replace("velocity_net.", "")] = value
    elif key.startswith("q_network.") or key.startswith("critic."):
        new_key = key.replace("q_network.", "").replace("critic.", "")
        q_network_state[new_key] = value

# 测试参数
state_dim = 25
visual_feature_dim = 256
obs_horizon = 2
action_dim = 7
act_horizon = 8
pred_horizon = 16
batch_size = 2

global_cond_dim = obs_horizon * (visual_feature_dim + state_dim)
print(f"\n使用 global_cond_dim = {global_cond_dim} 创建模型...")

# 创建模型
velocity_net = ShortCutVelocityUNet1D(
    input_dim=action_dim,
    global_cond_dim=global_cond_dim,
    diffusion_step_embed_dim=64,
    down_dims=(64, 128, 256),
    n_groups=8,
).to(device)

q_network = DoubleQNetwork(
    action_dim=action_dim,
    obs_dim=global_cond_dim,
    action_horizon=act_horizon,
    hidden_dims=[512, 512, 512],
).to(device)

# 加载权重
print("\n加载权重...")
velocity_net.load_state_dict(velocity_net_state)
q_network.load_state_dict(q_network_state)
print("  ✅ 加载成功")

# 测试 forward
print("\n测试 forward...")

# 创建测试输入
obs_cond = torch.randn(batch_size, global_cond_dim, device=device)
action = torch.randn(batch_size, pred_horizon, action_dim, device=device)
timestep = torch.randint(0, 100, (batch_size,), device=device)
dt = torch.rand(batch_size, 1, device=device)

print(f"  obs_cond shape: {obs_cond.shape}")
print(f"  action shape: {action.shape}")
print(f"  timestep shape: {timestep.shape}")
print(f"  dt shape: {dt.shape}")

# 测试 velocity_net
try:
    with torch.no_grad():
        velocity = velocity_net(action, timestep, dt, global_cond=obs_cond)
    print(f"\n  ✅ velocity_net forward 成功!")
    print(f"     输出 shape: {velocity.shape}")
except Exception as e:
    print(f"\n  ❌ velocity_net forward 失败: {e}")

# 测试 q_network
try:
    with torch.no_grad():
        action_for_q = action[:, :act_horizon, :]  # 只用 act_horizon
        q_value = q_network(obs_cond, action_for_q)
    print(f"\n  ✅ q_network forward 成功!")
    print(f"     输出 shape: {q_value.shape}")
except Exception as e:
    print(f"\n  ❌ q_network forward 失败: {e}")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)

# 现在用 checkpoint 中的实际维度创建模型
checkpoint_global_cond_dim = 626  # 从之前分析得知
checkpoint_state_dim = (checkpoint_global_cond_dim // obs_horizon) - visual_feature_dim
print(f"\nCheckpoint 使用的维度:")
print(f"  global_cond_dim = {checkpoint_global_cond_dim}")
print(f"  推断 state_dim = {checkpoint_state_dim}")
print(f"  Dataset state_dim = {state_dim}")
print(f"  差异 = {checkpoint_state_dim - state_dim}")

print("\n如果 velocity_net forward 失败:")
print("  --> 需要用 global_cond_dim=626 创建 velocity_net")
print("  --> 或者需要调整 state 提取方式以包含额外 32 维")
