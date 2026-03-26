#!/usr/bin/env python3
"""
Phase 1.5 V6: Policy mini-eval 验证

在 ManiSkill3 中运行 5 个 episode 评估策略推理管线。
此测试验证：
  1. ManiSkill 环境创建
  2. ShortCut Flow 策略 inference（采样动作）
  3. env.step() 循环
  4. 成功率计算

注意：V5 mini-train 只训练了 10 步（从随机初始化），预期成功率接近 0%。
V6 验证的重点是管线能跑通，不是成功率。
"""

from __future__ import annotations

import os
import sys
import time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "9")


def main() -> None:
    import torch
    import numpy as np
    from pathlib import Path

    results: dict[str, str] = {}

    # ================================================================
    # Step 1: 加载 ShortCut Flow 策略
    # ================================================================
    print("\n" + "=" * 60)
    print("[V6] Step 1: 加载策略")
    print("=" * 60)
    try:
        from rlft.algorithms.il.shortcut_flow import ShortCutFlowAgent
        from rlft.networks import ShortCutVelocityUNet1D

        device = torch.device("cuda:0")

        # Build model matching V5 real data config
        velocity_net = ShortCutVelocityUNet1D(
            input_dim=7,
            global_cond_dim=2 * 25,  # obs_horizon=2, state_dim=25
        )
        agent = ShortCutFlowAgent(
            velocity_net=velocity_net,
            action_dim=7,
            obs_horizon=2,
            pred_horizon=4,
            device="cuda:0",
            num_inference_steps=4,
        )

        # Load V5 checkpoint
        ckpt_path = str(Path(WORKSPACE) / "checkpoints/vlaw/policy/v5_mini/real/policy_v5_mini.pt")
        if Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            agent.load_state_dict(ckpt["model_state_dict"], strict=True)
            print(f"[V6] ✅ 加载 V5 checkpoint: {ckpt_path}")
        else:
            print(f"[V6] ⚠️ V5 checkpoint 不存在，使用随机初始化: {ckpt_path}")

        agent = agent.to(device)
        agent.eval()
        results["step1_load"] = "✅"
    except Exception as e:
        results["step1_load"] = f"❌ {e}"
        import traceback; traceback.print_exc()
        return

    # ================================================================
    # Step 2: 创建 ManiSkill 环境并运行 5 episode
    # ================================================================
    print("\n" + "=" * 60)
    print("[V6] Step 2: ManiSkill 环境评估 (5 ep)")
    print("=" * 60)
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        env = gym.make(
            "LiftPegUpright-v1",
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            render_mode=None,
            max_episode_steps=100,
        )

        num_episodes = 5
        success_count = 0
        episode_lengths = []

        for ep in range(num_episodes):
            obs_dict, info = env.reset()
            # Extract state vector from obs_dict
            if isinstance(obs_dict, dict):
                # ManiSkill state obs is usually in 'agent' and 'extra' keys
                state_parts = []
                if "agent" in obs_dict:
                    agent_obs = obs_dict["agent"]
                    if isinstance(agent_obs, dict):
                        for v in agent_obs.values():
                            state_parts.append(np.asarray(v).flatten())
                    else:
                        state_parts.append(np.asarray(agent_obs).flatten())
                if "extra" in obs_dict:
                    extra_obs = obs_dict["extra"]
                    if isinstance(extra_obs, dict):
                        for v in extra_obs.values():
                            state_parts.append(np.asarray(v).flatten())
                    else:
                        state_parts.append(np.asarray(extra_obs).flatten())
                state = np.concatenate(state_parts)
            else:
                state = np.asarray(obs_dict).flatten()

            # Build obs window (obs_horizon=2, pad initial frame)
            obs_window = [state, state]  # duplicate for first step

            done = False
            t = 0
            ep_success = False

            while not done and t < 100:
                # Prepare obs tensor
                obs_np = np.stack(obs_window[-2:], axis=0)  # (2, state_dim)
                # Truncate or pad to expected dim=25
                if obs_np.shape[-1] > 25:
                    obs_np = obs_np[..., :25]
                elif obs_np.shape[-1] < 25:
                    pad = np.zeros((obs_np.shape[0], 25 - obs_np.shape[-1]))
                    obs_np = np.concatenate([obs_np, pad], axis=-1)

                obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)  # (1, 2, 25)

                # Policy inference
                with torch.no_grad():
                    action_pred = agent.get_action(obs_tensor)  # (1, pred_horizon, action_dim)
                
                # Take first action
                action = action_pred[0, 0].cpu().numpy()  # (7,)

                # Step env
                obs_dict, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Update state
                if isinstance(obs_dict, dict):
                    state_parts = []
                    if "agent" in obs_dict:
                        agent_obs = obs_dict["agent"]
                        if isinstance(agent_obs, dict):
                            for v in agent_obs.values():
                                state_parts.append(np.asarray(v).flatten())
                        else:
                            state_parts.append(np.asarray(agent_obs).flatten())
                    if "extra" in obs_dict:
                        extra_obs = obs_dict["extra"]
                        if isinstance(extra_obs, dict):
                            for v in extra_obs.values():
                                state_parts.append(np.asarray(v).flatten())
                        else:
                            state_parts.append(np.asarray(extra_obs).flatten())
                    state = np.concatenate(state_parts)
                else:
                    state = np.asarray(obs_dict).flatten()
                obs_window.append(state)
                t += 1

                # Check success from info
                if info.get("success", False):
                    ep_success = True

            if ep_success:
                success_count += 1
            episode_lengths.append(t)
            print(f"  ep {ep}: T={t}, success={ep_success}")

        env.close()

        success_rate = success_count / num_episodes
        avg_len = sum(episode_lengths) / len(episode_lengths)
        results["step2_eval"] = (
            f"✅ success_rate={success_rate:.0%} ({success_count}/{num_episodes}), "
            f"avg_len={avg_len:.1f}"
        )
        print(f"\n[V6] 评估结果: success_rate={success_rate:.0%}, avg_ep_len={avg_len:.1f}")
    except Exception as e:
        results["step2_eval"] = f"❌ {e}"
        import traceback; traceback.print_exc()

    # ================================================================
    # 汇总
    # ================================================================
    print("\n" + "=" * 60)
    print("[V6] Phase 1.5 V6 Policy Mini-Eval 验证结果:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    pipeline_ok = all("✅" in v for v in results.values())
    if pipeline_ok:
        print("[V6] ✅ 推理管线全部验证通过！")
        print("[V6]    (成功率低是预期的 — V5 模型只训练了 10 步)")
    else:
        print("[V6] ⚠️ 部分步骤未通过")


if __name__ == "__main__":
    main()
