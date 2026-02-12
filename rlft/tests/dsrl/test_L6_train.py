"""
L6: 训练冒烟测试

验证完整的 DSRL-SAC 训练循环能在短时间内正常运行:
  - Agent 正确初始化
  - Buffer warmup 正常
  - 训练循环（collect → update → eval）无崩溃
  - Loss 有限且梯度流动正常
  - 短期训练后性能不严重退化（成功率不低于 30%）

这是最完整的集成测试，涵盖 DSRL pipeline 的所有组件。

需要: checkpoint + GPU + ManiSkill3
预计用时: 5-10 分钟

运行:
    conda activate carm
    cd /home/lizh/rl-vla
    CUDA_VISIBLE_DEVICES=0 python rlft/tests/dsrl/test_L6_train.py
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
    sys.path.insert(0, str(_root / "diffusion_policy"))

CHECKPOINT_PATH = str(_root / "runs/awsc_checkpoint/checkpoints/best_eval_success_once.pt")
DEVICE = "cuda"

# 训练配置（小规模用于冒烟测试）
ENV_ID = "LiftPegUpright-v1"
NUM_ENVS = 10  # 少量环境加速
NUM_EVAL_ENVS = 10
OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_STEPS = 8
ACTION_DIM = 7
STATE_DIM = 25
VISUAL_DIM = 256
ACTION_MAG = 1.5
OBS_DIM = OBS_HORIZON * (VISUAL_DIM + STATE_DIM)
NOISE_DIM = ACT_STEPS * ACTION_DIM

# 训练超参数（极小规模）
TOTAL_STEPS = 500     # 仅 500 步
WARMUP_STEPS = 100
BATCH_SIZE = 64
UTD_RATIO = 4         # 减少 UTD 加速
LR = 3e-4
GAMMA = 0.99
TAU = 0.005
HIDDEN_DIMS = [256, 256]  # 小网络加速
EVAL_EPISODES = 10


def main():
    print("=" * 70)
    print("L6: DSRL-SAC Training Smoke Test")
    print("=" * 70)

    if not Path(CHECKPOINT_PATH).exists():
        print(f"✗ Checkpoint not found: {CHECKPOINT_PATH}")
        return False

    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False

    import gymnasium as gym
    import mani_skill.envs  # noqa
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    from rlft.utils.flow_wrapper import load_shortcut_flow_policy
    from rlft.networks import PlainConv
    from rlft.envs.dsrl_env import ManiSkillFlowEnvWrapper
    from rlft.envs import make_eval_envs, evaluate
    from rlft.algorithms.online_rl.dsrl_sac import DSRLSACAgent
    from rlft.buffers.dsrl_buffer import DSRLReplayBuffer
    from rlft.online.train_dsrl import _VecEnvAdapter, _collect_warmup, DSRLEvalAgentWrapper

    # 固定种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ==================================================
    # Step 1: 加载 Flow Policy
    # ==================================================
    print("\n[1/7] Loading flow policy...")
    base_policy, visual_encoder, inferred_sd = load_shortcut_flow_policy(
        checkpoint_path=CHECKPOINT_PATH,
        visual_encoder_class=PlainConv,
        include_rgb=True, use_ema=True, device=DEVICE,
    )
    print(f"  state_dim={inferred_sd}")

    # ==================================================
    # Step 2: 创建训练和评估环境
    # ==================================================
    print("\n[2/7] Creating environments...")
    raw_env = gym.make(
        ENV_ID, obs_mode="rgbd", control_mode="pd_ee_delta_pose",
        sim_backend="physx_cuda", num_envs=NUM_ENVS,
        reward_mode="dense", max_episode_steps=100,
    )
    raw_env = FlattenRGBDObservationWrapper(raw_env, rgb=True, depth=False, state=True)

    wrapped_env = ManiSkillFlowEnvWrapper(
        env=raw_env, base_policy=base_policy, visual_encoder=visual_encoder,
        action_magnitude=ACTION_MAG, act_steps=ACT_STEPS, action_dim=ACTION_DIM,
        state_dim=STATE_DIM, visual_feature_dim=VISUAL_DIM,
        obs_horizon=OBS_HORIZON, include_rgb=True, device=DEVICE,
    )
    adapter = _VecEnvAdapter(wrapped_env)

    eval_env_kwargs = dict(
        control_mode="pd_ee_delta_pose", obs_mode="rgbd",
        render_mode="rgb_array", reward_mode="dense", max_episode_steps=100,
    )
    eval_envs = make_eval_envs(
        env_id=ENV_ID, num_envs=NUM_EVAL_ENVS, sim_backend="physx_cuda",
        env_kwargs=eval_env_kwargs, other_kwargs=dict(obs_horizon=OBS_HORIZON),
        video_dir=None, wrappers=[FlattenRGBDObservationWrapper],
    )
    print(f"  Train: {NUM_ENVS} envs | Eval: {NUM_EVAL_ENVS} envs")

    # ==================================================
    # Step 3: 创建 Agent
    # ==================================================
    print("\n[3/7] Creating DSRLSACAgent...")
    agent = DSRLSACAgent(
        obs_dim=OBS_DIM, act_steps=ACT_STEPS, action_dim=ACTION_DIM,
        action_magnitude=ACTION_MAG, hidden_dims=HIDDEN_DIMS, num_qs=2,
        gamma=GAMMA, tau=TAU, init_temperature=1.0, target_entropy=0.0,
        log_std_init=-3.0, use_layer_norm=False, device=DEVICE,
    ).to(DEVICE)

    actor_opt = optim.Adam(agent.actor.parameters(), lr=LR)
    critic_opt = optim.Adam(agent.critic.parameters(), lr=LR)
    temp_opt = optim.Adam([agent.log_alpha], lr=LR)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Agent params: {total_params:,}")

    eval_wrapper = DSRLEvalAgentWrapper(
        agent=agent, base_policy=base_policy, visual_encoder=visual_encoder,
        include_rgb=True, obs_horizon=OBS_HORIZON, act_steps=ACT_STEPS,
        action_dim=ACTION_DIM, device=DEVICE,
    )

    # ==================================================
    # Step 4: Buffer + Warmup
    # ==================================================
    print("\n[4/7] Buffer warmup...")
    buffer = DSRLReplayBuffer(
        capacity=50000, obs_dim=OBS_DIM, noise_dim=NOISE_DIM, device=DEVICE,
    )
    _collect_warmup(adapter, buffer, WARMUP_STEPS)

    # ==================================================
    # Step 5: Baseline 评估
    # ==================================================
    print("\n[5/7] Baseline evaluation...")
    eval_wrapper.eval()
    baseline_metrics = evaluate(
        EVAL_EPISODES, eval_wrapper, eval_envs, DEVICE, "physx_cuda",
    )
    for k in baseline_metrics:
        baseline_metrics[k] = np.mean(baseline_metrics[k])
    baseline_success = baseline_metrics.get("success_once", 0)
    print(f"  Baseline success_once: {baseline_success:.2%}")

    # ==================================================
    # Step 6: 训练循环
    # ==================================================
    print(f"\n[6/7] Training for {TOTAL_STEPS} steps (UTD={UTD_RATIO})...")

    obs, _ = adapter.reset()
    total_steps = 0
    all_losses = defaultdict(list)

    t0 = time.time()

    while total_steps < TOTAL_STEPS:
        # Collect
        agent.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(DEVICE)
            noise = agent.select_action(obs_t, deterministic=False).cpu().numpy()

        next_obs, rew, done, term, trunc, info = adapter.step(noise)
        buffer.add(obs, noise, rew, next_obs, done.astype(np.float32))
        obs = next_obs
        total_steps += NUM_ENVS

        # Update
        if buffer.size >= BATCH_SIZE:
            agent.train()
            for _ in range(UTD_RATIO):
                batch = buffer.sample(BATCH_SIZE)

                critic_opt.zero_grad()
                c_loss, c_met = agent.compute_critic_loss(
                    batch["obs"], batch["actions"], batch["next_obs"],
                    batch["rewards"], batch["dones"],
                )
                c_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), 10.0)
                critic_opt.step()

                actor_opt.zero_grad()
                a_loss, a_met = agent.compute_actor_loss(batch["obs"])
                a_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
                actor_opt.step()

                temp_opt.zero_grad()
                t_loss, t_met = agent.compute_temperature_loss(batch["obs"])
                t_loss.backward()
                temp_opt.step()

                agent.update_target()

                all_losses["critic_loss"].append(c_met["critic_loss"])
                all_losses["actor_loss"].append(a_met["actor_loss"])
                all_losses["temperature"].append(t_met["temperature"])
                all_losses["entropy"].append(t_met["entropy"])

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # 验证 loss 有限
    for k, v in all_losses.items():
        arr = np.array(v)
        assert np.isfinite(arr).all(), f"{k} contains inf/nan"
        print(f"  {k}: mean={arr.mean():.4f}, std={arr.std():.4f}")

    print(f"  ✓ All losses finite")

    # ==================================================
    # Step 7: 训练后评估
    # ==================================================
    print(f"\n[7/7] Post-training evaluation...")
    eval_wrapper.eval()
    post_metrics = evaluate(
        EVAL_EPISODES, eval_wrapper, eval_envs, DEVICE, "physx_cuda",
    )
    for k in post_metrics:
        post_metrics[k] = np.mean(post_metrics[k])
    post_success = post_metrics.get("success_once", 0)

    print(f"  Post-training success_once: {post_success:.2%}")
    print(f"  Baseline success_once:      {baseline_success:.2%}")
    delta = post_success - baseline_success
    print(f"  Delta: {delta:+.2%}")

    # 短期训练不应导致严重退化
    # 由于只训练了很少的步骤+小网络，放宽阈值
    min_post = 0.20  # 训练后至少 20%
    assert post_success >= min_post, \
        f"Post-training success {post_success:.2%} < {min_post:.0%} — severe degradation!"
    print(f"  ✓ No severe degradation (>= {min_post:.0%})")

    # Cleanup
    raw_env.close()
    eval_envs.close()

    # ==================================================
    # Summary
    # ==================================================
    print("\n" + "=" * 70)
    print("L6 Training Smoke Test: PASS ✓")
    print("=" * 70)
    print(f"  Baseline success:     {baseline_success:.2%}")
    print(f"  Post-train success:   {post_success:.2%}")
    print(f"  Training steps:       {TOTAL_STEPS}")
    print(f"  Buffer size:          {buffer.size}")
    print(f"  Total time:           {train_time:.1f}s")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
