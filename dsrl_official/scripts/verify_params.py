"""深入验证 - 检查模型参数是否真正匹配"""

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

checkpoint_path = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"

print("=" * 60)
print("详细检查模型参数维度")
print("=" * 60)

device = "cpu"
checkpoint = torch.load(checkpoint_path, map_location=device)
agent_state = checkpoint.get("ema_agent", checkpoint.get("agent"))

# 提取 velocity_net 权重
velocity_net_state = {}
for key, value in agent_state.items():
    if key.startswith("velocity_net."):
        velocity_net_state[key.replace("velocity_net.", "")] = value

# 创建两个模型
print("\n创建两个 velocity_net:")
print("  Model A: global_cond_dim = 562 (dataset)")
print("  Model B: global_cond_dim = 626 (checkpoint)")

model_a = ShortCutVelocityUNet1D(
    input_dim=7,
    global_cond_dim=562,
    diffusion_step_embed_dim=64,
    down_dims=(64, 128, 256),
    n_groups=8,
)

model_b = ShortCutVelocityUNet1D(
    input_dim=7,
    global_cond_dim=626,
    diffusion_step_embed_dim=64,
    down_dims=(64, 128, 256),
    n_groups=8,
)

# 比较关键层的维度
print("\n关键层维度对比:")
print("-" * 60)

# 找到与 global_cond_dim 相关的层
key_layers = [
    "unet.mid_modules.0.cond_encoder.1.weight",
    "unet.down_modules.0.0.cond_encoder.1.weight",
    "unet.up_modules.0.0.cond_encoder.1.weight",
]

for key in key_layers:
    # 获取模型参数
    parts = key.split(".")
    param_a = model_a
    param_b = model_b
    for p in parts:
        if p.isdigit():
            param_a = param_a[int(p)]
            param_b = param_b[int(p)]
        else:
            param_a = getattr(param_a, p)
            param_b = getattr(param_b, p)
    
    # 获取 checkpoint 参数
    ckpt_shape = velocity_net_state[key].shape if key in velocity_net_state else "NOT FOUND"
    
    print(f"  {key}:")
    print(f"    Model A (562): {param_a.shape}")
    print(f"    Model B (626): {param_b.shape}")
    print(f"    Checkpoint:    {ckpt_shape}")

# 尝试严格加载
print("\n" + "=" * 60)
print("尝试严格加载 (strict=True)")
print("=" * 60)

try:
    result = model_a.load_state_dict(velocity_net_state, strict=True)
    print("  Model A 加载成功!")
except RuntimeError as e:
    err_str = str(e)
    # 只打印关键错误信息
    if "size mismatch" in err_str:
        print(f"  Model A 加载失败: size mismatch detected")
        # 找出不匹配的 key
        lines = err_str.split("\n")
        for line in lines:
            if "size mismatch" in line.lower() or "cond_encoder" in line:
                print(f"    {line.strip()}")
    else:
        print(f"  Model A 加载失败: {err_str[:200]}...")

try:
    result = model_b.load_state_dict(velocity_net_state, strict=True)
    print("  Model B 加载成功!")
except RuntimeError as e:
    print(f"  Model B 加载失败: {str(e)[:200]}...")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print("Checkpoint velocity_net 需要 global_cond_dim = 626")
print("但 train_stage1_offline.py 使用 global_cond_dim = 562")
print()
print("如果加载成功，说明:")
print("  1. PyTorch load_state_dict 可能忽略了某些层")
print("  2. 或使用了 strict=False")
print()
print("如果 forward 成功，说明:")
print("  1. 模型结构正确（无 shape mismatch 在 forward 中）")
print("  2. 但 cond_encoder 权重可能不正确！")
