#!/usr/bin/env python3
"""
分析预训练 checkpoint 中各模块的维度

用于调试 velocity_net.global_cond_dim=626 vs critic.obs_dim=562 不一致问题
"""

import sys
from pathlib import Path
import torch

# 默认 checkpoint 路径
DEFAULT_CHECKPOINT = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"


def analyze_checkpoint(checkpoint_path: str = DEFAULT_CHECKPOINT):
    """分析 checkpoint 中各模块的维度"""
    print("=" * 70)
    print("Checkpoint 维度分析")
    print("=" * 70)
    print(f"Path: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint 不存在: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # 1. 顶层结构
    print("\n" + "-" * 70)
    print("1. 顶层结构")
    print("-" * 70)
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: {len(checkpoint[key])} keys")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # 2. 分析 agent 和 ema_agent 结构
    for agent_key in ["agent", "ema_agent"]:
        if agent_key not in checkpoint:
            continue
        
        print(f"\n" + "-" * 70)
        print(f"2. {agent_key} 模块前缀统计")
        print("-" * 70)
        
        agent_state = checkpoint[agent_key]
        prefixes = {}
        for key in agent_state.keys():
            prefix = key.split(".")[0]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        for prefix, count in sorted(prefixes.items()):
            print(f"  {prefix}: {count} keys")
        
        # 3. 提取关键维度信息
        print(f"\n" + "-" * 70)
        print(f"3. {agent_key} 关键维度")
        print("-" * 70)
        
        # 3.1 velocity_net 的 cond_encoder (global_cond_dim)
        print("\n  [velocity_net global_cond_dim]")
        for key, value in agent_state.items():
            if "velocity_net" in key and "cond_encoder" in key and "weight" in key:
                if len(value.shape) == 2 and value.shape[1] > 500:
                    print(f"    {key}: {tuple(value.shape)}")
                    print(f"      → global_cond_dim = {value.shape[1]}")
                    # 推断 state_dim
                    # global_cond_dim = obs_horizon * (visual_dim + state_dim)
                    # 假设 obs_horizon=2, visual_dim=256
                    inferred_state_dim = value.shape[1] // 2 - 256
                    print(f"      → 推断 state_dim = {inferred_state_dim} (假设 obs_horizon=2, visual_dim=256)")
                    break
        
        # 3.2 critic 的输入层 (obs_dim + action_input)
        print("\n  [critic obs_dim + action_input]")
        for key, value in agent_state.items():
            if ("critic" in key or "q_network" in key) and "q1_net" in key and ".0.weight" in key:
                print(f"    {key}: {tuple(value.shape)}")
                input_dim = value.shape[1]
                # action_input = action_dim * act_horizon = 7 * 8 = 56
                action_input = 56
                obs_dim = input_dim - action_input
                print(f"      → 总输入维度 = {input_dim}")
                print(f"      → obs_dim = {obs_dim} (假设 action_input={action_input})")
                inferred_state_dim = obs_dim // 2 - 256
                print(f"      → 推断 state_dim = {inferred_state_dim} (假设 obs_horizon=2, visual_dim=256)")
                break
        
        # 只分析一个 agent key
        break
    
    # 4. visual_encoder 分析
    if "visual_encoder" in checkpoint:
        print("\n" + "-" * 70)
        print("4. visual_encoder 维度")
        print("-" * 70)
        
        ve_state = checkpoint["visual_encoder"]
        for key, value in ve_state.items():
            if "weight" in key and len(value.shape) >= 2:
                print(f"  {key}: {tuple(value.shape)}")
        
        # 检查 fc 层输入维度
        for key, value in ve_state.items():
            if "fc" in key and "weight" in key and len(value.shape) == 2:
                print(f"\n  [FC 层输入维度]")
                print(f"    {key}: {tuple(value.shape)}")
                print(f"      → fc 输入维度 = {value.shape[1]}")
    
    # 5. 维度一致性总结
    print("\n" + "=" * 70)
    print("5. 维度一致性总结")
    print("=" * 70)
    
    # 提取具体数值
    velocity_cond_dim = None
    critic_obs_dim = None
    
    agent_state = checkpoint.get("agent", checkpoint.get("ema_agent", {}))
    for key, value in agent_state.items():
        if "velocity_net" in key and "cond_encoder" in key and "weight" in key:
            if len(value.shape) == 2 and value.shape[1] > 500:
                velocity_cond_dim = value.shape[1]
                break
    
    for key, value in agent_state.items():
        if ("critic" in key or "q_network" in key) and "q1_net" in key and ".0.weight" in key:
            critic_obs_dim = value.shape[1] - 56  # 减去 action_input
            break
    
    if velocity_cond_dim and critic_obs_dim:
        print(f"  velocity_net.global_cond_dim = {velocity_cond_dim}")
        print(f"  critic.obs_dim = {critic_obs_dim}")
        
        if velocity_cond_dim == critic_obs_dim:
            print(f"\n  ✅ 维度一致!")
        else:
            print(f"\n  ❌ 维度不一致!")
            print(f"     差异 = {velocity_cond_dim - critic_obs_dim}")
            
            # 推断各自的 state_dim
            vel_state_dim = velocity_cond_dim // 2 - 256
            cri_state_dim = critic_obs_dim // 2 - 256
            print(f"\n  推断 state_dim:")
            print(f"    velocity_net 使用的 state_dim = {vel_state_dim}")
            print(f"    critic 使用的 state_dim = {cri_state_dim}")
            print(f"    差异 = {vel_state_dim - cri_state_dim} 维")
    
    print("\n" + "=" * 70)


def compare_state_extractors():
    """比较不同 state 提取方式的维度"""
    print("\n" + "=" * 70)
    print("State 提取逻辑分析")
    print("=" * 70)
    
    # 添加路径
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "diffusion_policy"))
    
    try:
        from diffusion_policy.utils import build_state_obs_extractor
        
        env_id = "LiftPegUpright-v1"
        extractor = build_state_obs_extractor(env_id)
        print(f"\n环境: {env_id}")
        print(f"state_obs_extractor 函数: {extractor}")
        
        # 尝试创建环境获取实际 state 维度
        import gymnasium as gym
        import mani_skill.envs
        
        env = gym.make(
            env_id,
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            render_mode=None,
        )
        
        obs, _ = env.reset()
        
        print(f"\n原始 obs 结构:")
        def print_obs_structure(obs, prefix=""):
            if isinstance(obs, dict):
                for k, v in obs.items():
                    print_obs_structure(v, prefix + k + ".")
            else:
                print(f"  {prefix[:-1]}: shape={obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        print_obs_structure(obs)
        
        # 使用 extractor 提取 state
        if extractor:
            extracted = extractor(obs)
            if isinstance(extracted, list):
                total_dim = sum(e.shape[-1] if hasattr(e, 'shape') else len(e) for e in extracted)
                print(f"\n提取后的 state (列表):")
                for i, e in enumerate(extracted):
                    print(f"  [{i}]: shape={e.shape if hasattr(e, 'shape') else len(e)}")
                print(f"  总维度 = {total_dim}")
        
        env.close()
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--compare-extractors", action="store_true", help="比较 state 提取逻辑")
    args = parser.parse_args()
    
    analyze_checkpoint(args.checkpoint)
    
    if args.compare_extractors:
        compare_state_extractors()
