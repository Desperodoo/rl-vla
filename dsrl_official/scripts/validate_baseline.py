#!/usr/bin/env python3
"""
基准验证脚本: 验证预训练 checkpoint 在新环境设置下的性能

目标: 确认预训练模型能达到约 80% success_once

Usage:
    python scripts/validate_baseline.py
    python scripts/validate_baseline.py --num_episodes 100 --num_envs 16
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import time
from dataclasses import dataclass
from typing import Optional

# 添加路径
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl_official"))

try:
    import tyro
except ImportError:
    print("Please install tyro: pip install tyro")
    sys.exit(1)


@dataclass
class ValidateArgs:
    """基准验证参数。"""
    
    # Checkpoint
    checkpoint_path: str = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
    
    # Environment
    env_id: str = "LiftPegUpright-v1"
    num_envs: int = 16
    max_episode_steps: int = 100
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "physx_cuda"
    
    # Evaluation
    num_episodes: int = 50
    render: bool = False
    save_video: bool = False
    video_dir: str = "eval_videos"
    
    # Policy
    obs_horizon: int = 2
    pred_horizon: int = 16
    act_steps: int = 8
    action_dim: int = 7
    visual_feature_dim: int = 256
    state_dim: int = 25
    include_rgb: bool = True
    use_ema: bool = True
    num_inference_steps: int = 8
    
    # Device
    device: str = "cuda"
    seed: int = 42


def make_eval_env(args: ValidateArgs):
    """创建评估环境。"""
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    
    # 创建向量化环境
    env_kwargs = dict(
        control_mode=args.control_mode,
        obs_mode="rgbd" if args.include_rgb else "state_dict",
        render_mode="rgb_array" if args.render or args.save_video else None,
        sim_backend=args.sim_backend,
    )
    
    if args.sim_backend == "physx_cuda":
        # GPU 并行环境
        env = gym.make(
            args.env_id,
            num_envs=args.num_envs,
            max_episode_steps=args.max_episode_steps,
            **env_kwargs,
        )
        env = ManiSkillVectorEnv(env, ignore_terminations=False)
    else:
        # CPU 环境
        from mani_skill.vector import VecEnv
        env = VecEnv([
            lambda: gym.make(args.env_id, max_episode_steps=args.max_episode_steps, **env_kwargs)
            for _ in range(args.num_envs)
        ])
    
    # 添加观察处理 wrapper
    env = FlattenRGBDObservationWrapper(
        env,
        rgb=args.include_rgb,
        depth=False,
        state=True,
    )
    
    return env


def load_policy(args: ValidateArgs):
    """加载预训练策略。"""
    from dsrl_official.utils import load_shortcut_flow_policy
    from diffusion_policy.plain_conv import PlainConv
    
    wrapper, visual_encoder = load_shortcut_flow_policy(
        checkpoint_path=args.checkpoint_path,
        visual_encoder_class=PlainConv,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_dim=args.action_dim,
        visual_feature_dim=args.visual_feature_dim,
        state_dim=args.state_dim,
        include_rgb=args.include_rgb,
        num_inference_steps=args.num_inference_steps,
        use_ema=args.use_ema,
        device=args.device,
    )
    
    return wrapper, visual_encoder


def evaluate_policy(
    env,
    wrapper,
    visual_encoder,
    args: ValidateArgs,
):
    """评估策略性能。"""
    device = args.device
    
    all_episode_rewards = []
    all_episode_lengths = []
    all_success_once = []
    all_success_at_end = []
    
    episode_count = 0
    
    # Reset
    obs, info = env.reset()
    current_rewards = np.zeros(args.num_envs)
    current_lengths = np.zeros(args.num_envs, dtype=np.int32)
    current_success_once = np.zeros(args.num_envs, dtype=bool)
    
    # 观察缓存
    obs_buffer = []
    
    print(f"\nRunning evaluation for {args.num_episodes} episodes...")
    pbar_total = args.num_episodes
    completed = 0
    
    while episode_count < args.num_episodes:
        # 更新观察缓存
        obs_buffer.append(obs)
        if len(obs_buffer) > args.obs_horizon:
            obs_buffer = obs_buffer[-args.obs_horizon:]
        
        # 处理观察
        if len(obs_buffer) < args.obs_horizon:
            # 填充
            padded_obs = [obs_buffer[0]] * (args.obs_horizon - len(obs_buffer)) + obs_buffer
        else:
            padded_obs = obs_buffer
        
        # 编码观察
        with torch.no_grad():
            # Stack observations
            stacked_obs = np.stack(padded_obs, axis=1)  # (num_envs, obs_horizon, ...)
            
            if args.include_rgb:
                # 提取 RGB 和 state
                # 假设 obs 是 dict with 'rgb' and 'state' keys
                # 需要根据实际结构调整
                if isinstance(obs, dict):
                    rgb = obs["rgb"]  # (num_envs, H, W, C)
                    state = obs["state"]  # (num_envs, state_dim)
                    
                    # 编码 RGB
                    rgb_tensor = torch.from_numpy(rgb).to(device)
                    rgb_tensor = rgb_tensor.permute(0, 3, 1, 2).float() / 255.0
                    visual_features = visual_encoder(rgb_tensor)  # (num_envs, visual_dim)
                    
                    # 拼接特征
                    state_tensor = torch.from_numpy(state).to(device).float()
                    obs_features = torch.cat([visual_features, state_tensor], dim=-1)
                else:
                    # 扁平化观察
                    obs_tensor = torch.from_numpy(obs).to(device).float()
                    obs_features = obs_tensor
            else:
                obs_tensor = torch.from_numpy(obs).to(device).float()
                obs_features = obs_tensor
            
            # 生成噪声 (这里使用零噪声作为基准)
            noise = torch.zeros(
                args.num_envs, args.pred_horizon, args.action_dim,
                device=device
            )
            
            # 获取动作
            actions = wrapper(obs_features, noise, return_numpy=True)
        
        # 执行 act_steps 步动作
        for step in range(args.act_steps):
            action = actions[:, step, :]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            current_rewards += reward
            current_lengths += 1
            
            # 检查 success
            if "success" in info:
                current_success_once |= info["success"]
            
            # 处理完成的 episode
            done = terminated | truncated
            for i in range(args.num_envs):
                if done[i] and episode_count < args.num_episodes:
                    all_episode_rewards.append(current_rewards[i])
                    all_episode_lengths.append(current_lengths[i])
                    all_success_once.append(current_success_once[i])
                    all_success_at_end.append(info.get("success", [False])[i] if isinstance(info.get("success"), np.ndarray) else False)
                    
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    current_success_once[i] = False
                    
                    episode_count += 1
                    
                    if episode_count % 10 == 0:
                        print(f"  Completed {episode_count}/{args.num_episodes} episodes")
            
            # 更新观察缓存
            obs_buffer.append(obs)
            if len(obs_buffer) > args.obs_horizon:
                obs_buffer = obs_buffer[-args.obs_horizon:]
    
    # 计算统计
    results = {
        "num_episodes": episode_count,
        "mean_reward": np.mean(all_episode_rewards),
        "std_reward": np.std(all_episode_rewards),
        "mean_length": np.mean(all_episode_lengths),
        "success_once": np.mean(all_success_once),
        "success_at_end": np.mean(all_success_at_end),
    }
    
    return results


def main():
    """主函数。"""
    args = tyro.cli(ValidateArgs)
    
    print("="*70)
    print("DSRL Baseline Validation")
    print("="*70)
    
    print(f"\nCheckpoint: {args.checkpoint_path}")
    print(f"Environment: {args.env_id}")
    print(f"Num envs: {args.num_envs}")
    print(f"Num episodes: {args.num_episodes}")
    
    # 检查 checkpoint 存在
    if not os.path.exists(args.checkpoint_path):
        print(f"\nError: Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载策略
    print("\nLoading policy...")
    try:
        wrapper, visual_encoder = load_policy(args)
        print("Policy loaded successfully!")
    except Exception as e:
        print(f"Error loading policy: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 创建环境
    print("\nCreating environment...")
    try:
        env = make_eval_env(args)
        print(f"Environment created: {env}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 评估
    print("\nStarting evaluation...")
    start_time = time.time()
    
    try:
        results = evaluate_policy(env, wrapper, visual_encoder, args)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        sys.exit(1)
    
    elapsed = time.time() - start_time
    
    # 打印结果
    print("\n" + "="*70)
    print("Validation Results")
    print("="*70)
    print(f"Episodes evaluated: {results['num_episodes']}")
    print(f"Mean reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"Mean episode length: {results['mean_length']:.1f}")
    print(f"Success once rate: {results['success_once']*100:.2f}%")
    print(f"Success at end rate: {results['success_at_end']*100:.2f}%")
    print(f"\nEvaluation time: {elapsed:.1f}s")
    
    # 验证目标
    target_success = 0.80
    if results['success_once'] >= target_success:
        print(f"\n✓ Target success rate ({target_success*100:.0f}%) ACHIEVED!")
    else:
        print(f"\n✗ Target success rate ({target_success*100:.0f}%) NOT achieved.")
        print(f"  Current: {results['success_once']*100:.2f}%")
        print(f"  Gap: {(target_success - results['success_once'])*100:.2f}%")
    
    print("="*70)
    
    env.close()
    
    return results['success_once'] >= target_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
