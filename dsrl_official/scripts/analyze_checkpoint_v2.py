"""深入分析 checkpoint 维度"""

import torch
from pathlib import Path

checkpoint_path = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"

print("=" * 60)
print("深入分析 Checkpoint 维度")
print("=" * 60)

checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("\n1. Checkpoint 顶层 keys:")
for key in checkpoint.keys():
    if isinstance(checkpoint[key], dict):
        print(f"   {key}: dict with {len(checkpoint[key])} keys")
    elif isinstance(checkpoint[key], torch.Tensor):
        print(f"   {key}: tensor {checkpoint[key].shape}")
    else:
        print(f"   {key}: {type(checkpoint[key])}")

# 分析 agent 和 ema_agent
for agent_key in ["agent", "ema_agent"]:
    if agent_key not in checkpoint:
        continue
    agent_state = checkpoint[agent_key]
    print(f"\n2. {agent_key} 中的子模块:")
    
    # 分类 keys
    modules = {}
    for key in agent_state.keys():
        module_name = key.split(".")[0]
        if module_name not in modules:
            modules[module_name] = []
        modules[module_name].append(key)
    
    for mod, keys in modules.items():
        print(f"   {mod}: {len(keys)} keys")
    
    # 查找能够确定 global_cond_dim 的 weight
    print(f"\n3. 从 {agent_key} 推断维度:")
    
    # 查找 velocity_net 中与 global_cond 相关的 layer
    for key, value in agent_state.items():
        if "velocity_net" in key and "global_cond" in key.lower():
            print(f"   {key}: {value.shape}")
        # 检查 encoder 的第一层 (包含 global_cond_dim)
        if "velocity_net.unet.encoder.0" in key and len(value.shape) == 2:
            print(f"   {key}: {value.shape}")
    
    # 查找 critic/q_network 输入维度
    for key, value in agent_state.items():
        if ("critic." in key or "q_network." in key) and ("0.weight" in key or "q1_net.0.weight" in key):
            if len(value.shape) == 2:
                print(f"   {key}: {value.shape}")
                input_dim = value.shape[1]
                # 假设 action_horizon=8, action_dim=7
                act_horizon = 8
                action_dim = 7
                obs_dim = input_dim - act_horizon * action_dim
                print(f"      --> input_dim={input_dim}, 推断 obs_dim={obs_dim}")
                break
    break

# 直接打印可能包含维度信息的 keys
print("\n4. 所有包含 weight 的 key (linear layers):")
agent_state = checkpoint.get("ema_agent", checkpoint.get("agent"))
count = 0
for key, value in agent_state.items():
    if ".weight" in key and len(value.shape) == 2:
        count += 1
        if count <= 20:  # 只打印前20个
            print(f"   {key}: {value.shape}")
        elif count == 21:
            print("   ... (more keys)")
