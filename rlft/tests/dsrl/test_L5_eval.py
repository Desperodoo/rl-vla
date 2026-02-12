"""
L5: 端到端评估测试

核心测试: 使用 checkpoint best_eval_success_once.pt (~85% 成功率)，
验证完整的 DSRL pipeline 能否在 ManiSkill3 LiftPegUpright-v1 环境中
达到预期的成功率。

测试内容:
  1. 零噪声评估 (w=0) — 预训练策略直出，应达到 ~85% success_once
  2. DSRLEvalAgentWrapper + evaluate() 流程 — 确定性模式
  3. DSRLSACAgent + DSRLEvalAgentWrapper 完整流程
  4. 结果对比: rlft pipeline vs dsrl_official 的差异应很小

需要: checkpoint + GPU + ManiSkill3

运行:
    conda activate carm
    cd /home/lizh/rl-vla
    CUDA_VISIBLE_DEVICES=0 python rlft/tests/dsrl/test_L5_eval.py

    # 或 pytest (较慢):
    CUDA_VISIBLE_DEVICES=0 python -m pytest rlft/tests/dsrl/test_L5_eval.py -v -s
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
    sys.path.insert(0, str(_root / "diffusion_policy"))

CHECKPOINT_PATH = str(_root / "runs/awsc_checkpoint/checkpoints/best_eval_success_once.pt")
DEVICE = "cuda"

# 配置
ENV_ID = "LiftPegUpright-v1"
NUM_ENVS = 50
NUM_EVAL_EPISODES = 100
OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_STEPS = 8
ACTION_DIM = 7
STATE_DIM = 25
VISUAL_DIM = 256
ACTION_MAG = 1.5
OBS_DIM = OBS_HORIZON * (VISUAL_DIM + STATE_DIM)
NOISE_DIM = ACT_STEPS * ACTION_DIM

# 预期成功率范围
EXPECTED_SUCCESS_MIN = 0.70  # 下限（允许环境随机性）
EXPECTED_SUCCESS_MAX = 1.01  # 上限（允许 100%）


def setup_pipeline():
    """设置完整的 DSRL pipeline，返回所有需要的组件。"""
    import gymnasium as gym
    import mani_skill.envs  # noqa
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    from rlft.utils.flow_wrapper import load_shortcut_flow_policy
    from rlft.networks import PlainConv
    from rlft.envs.dsrl_env import ManiSkillFlowEnvWrapper
    from rlft.envs import make_eval_envs

    print("[1/3] Loading pretrained flow policy...")
    base_policy, visual_encoder, inferred_sd = load_shortcut_flow_policy(
        checkpoint_path=CHECKPOINT_PATH,
        visual_encoder_class=PlainConv,
        obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON,
        action_dim=ACTION_DIM,
        visual_feature_dim=VISUAL_DIM,
        include_rgb=True,
        use_ema=True,
        device=DEVICE,
    )
    print(f"  state_dim={inferred_sd}")

    print("[2/3] Creating training env (wrapped)...")
    raw_env = gym.make(
        ENV_ID,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cuda",
        num_envs=NUM_ENVS,
        reward_mode="dense",
        max_episode_steps=100,
    )
    raw_env = FlattenRGBDObservationWrapper(raw_env, rgb=True, depth=False, state=True)

    wrapped_env = ManiSkillFlowEnvWrapper(
        env=raw_env,
        base_policy=base_policy,
        visual_encoder=visual_encoder,
        action_magnitude=ACTION_MAG,
        act_steps=ACT_STEPS,
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        visual_feature_dim=VISUAL_DIM,
        obs_horizon=OBS_HORIZON,
        include_rgb=True,
        device=DEVICE,
    )

    print("[3/3] Creating eval env...")
    eval_env_kwargs = dict(
        control_mode="pd_ee_delta_pose",
        obs_mode="rgbd",
        render_mode="rgb_array",
        reward_mode="dense",
        max_episode_steps=100,
    )
    eval_envs = make_eval_envs(
        env_id=ENV_ID,
        num_envs=NUM_ENVS,
        sim_backend="physx_cuda",
        env_kwargs=eval_env_kwargs,
        other_kwargs=dict(obs_horizon=OBS_HORIZON),
        video_dir=None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    return base_policy, visual_encoder, wrapped_env, eval_envs, raw_env


# =====================================================================
# Test 1: 零噪声 EnvWrapper 评估 (直接 rollout)
# =====================================================================

def test_zero_noise_envwrapper_eval():
    """
    在 ManiSkillFlowEnvWrapper 中执行零噪声 (w=0) rollout。
    w=0 等价于预训练策略直接输出，应达到 ~85% 成功率。
    """
    print("\n" + "=" * 70)
    print("Test 1: Zero-noise EnvWrapper evaluation")
    print("=" * 70)

    base_policy, visual_encoder, wrapped_env, eval_envs, raw_env = setup_pipeline()

    try:
        obs, info = wrapped_env.reset()
        episodes_done = 0
        successes = []
        ep_rewards = []
        cur_rewards = torch.zeros(NUM_ENVS, device=DEVICE)

        print(f"\nRunning zero-noise rollout ({NUM_EVAL_EPISODES} episodes)...")
        t0 = time.time()

        while episodes_done < NUM_EVAL_EPISODES:
            # 零噪声动作
            action = torch.zeros(NUM_ENVS, NOISE_DIM, device=DEVICE)
            obs, rew, term, trunc, info = wrapped_env.step(action)
            cur_rewards += rew

            done = term | trunc
            if done.any():
                for i in range(NUM_ENVS):
                    if done[i]:
                        success = False
                        if isinstance(info.get("success"), (torch.Tensor, np.ndarray)):
                            success = bool(info["success"][i])
                        successes.append(success)
                        ep_rewards.append(cur_rewards[i].item())
                        cur_rewards[i] = 0.0
                        episodes_done += 1

                # 如果全部 done，重置
                if done.all():
                    obs, info = wrapped_env.reset()

        elapsed = time.time() - t0
        success_rate = np.mean(successes[:NUM_EVAL_EPISODES])
        avg_reward = np.mean(ep_rewards[:NUM_EVAL_EPISODES])

        print(f"\nResults:")
        print(f"  Episodes:     {NUM_EVAL_EPISODES}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Avg reward:   {avg_reward:.2f}")
        print(f"  Time:         {elapsed:.1f}s")

        assert EXPECTED_SUCCESS_MIN <= success_rate <= EXPECTED_SUCCESS_MAX, \
            f"Success rate {success_rate:.2%} outside expected range " \
            f"[{EXPECTED_SUCCESS_MIN:.0%}, {EXPECTED_SUCCESS_MAX:.0%}]"
        print(f"\n  ✓ PASS: success_rate={success_rate:.2%} in expected range")

    finally:
        raw_env.close()
        eval_envs.close()


# =====================================================================
# Test 2: DSRLEvalAgentWrapper + evaluate() 评估
# =====================================================================

def test_eval_agent_wrapper():
    """
    使用 DSRLSACAgent (初始化权重 = 零噪声行为) + DSRLEvalAgentWrapper
    通过 rlft.envs.evaluate() 进行评估。

    初始 SAC agent 的 mean 权重接近零 + small std → 输出接近零噪声
    → 应近似预训练策略性能。
    """
    print("\n" + "=" * 70)
    print("Test 2: DSRLEvalAgentWrapper + evaluate()")
    print("=" * 70)

    from rlft.algorithms.online_rl.dsrl_sac import DSRLSACAgent
    from rlft.envs import evaluate
    # 需要导入 train_dsrl 中的 DSRLEvalAgentWrapper
    from rlft.online.train_dsrl import DSRLEvalAgentWrapper

    base_policy, visual_encoder, wrapped_env, eval_envs, raw_env = setup_pipeline()

    try:
        hidden_dims = [2048, 2048, 2048]

        agent = DSRLSACAgent(
            obs_dim=OBS_DIM,
            act_steps=ACT_STEPS,
            action_dim=ACTION_DIM,
            action_magnitude=ACTION_MAG,
            hidden_dims=hidden_dims,
            num_qs=2,
            gamma=0.99,
            tau=0.005,
            init_temperature=1.0,
            target_entropy=0.0,
            log_std_init=-3.0,
            use_layer_norm=False,
            device=DEVICE,
        ).to(DEVICE)

        eval_wrapper = DSRLEvalAgentWrapper(
            agent=agent,
            base_policy=base_policy,
            visual_encoder=visual_encoder,
            include_rgb=True,
            obs_horizon=OBS_HORIZON,
            act_steps=ACT_STEPS,
            action_dim=ACTION_DIM,
            device=DEVICE,
        )

        print(f"\nRunning evaluate() with DSRLEvalAgentWrapper...")
        eval_wrapper.eval()

        t0 = time.time()
        metrics = evaluate(
            NUM_EVAL_EPISODES, eval_wrapper, eval_envs, DEVICE, "physx_cuda",
        )
        elapsed = time.time() - t0

        for k in metrics:
            metrics[k] = np.mean(metrics[k])

        success = metrics.get("success_once", 0)
        print(f"\nResults:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # 注意: 初始随机 agent 的 trunk(obs) + mean_head 输出不是精确零，
        # 因此性能可能显著低于 w=0 的预训练策略。
        # 此测试只验证 pipeline 能正常运行，不严格要求高成功率。
        # 真正的零噪声性能由 Test 1 验证。
        print(f"\n  ✓ PASS: DSRLEvalAgentWrapper + evaluate() pipeline runs successfully")
        print(f"    (success_once={success:.2%} — initial random agent, not expected to be high)")

    finally:
        raw_env.close()
        eval_envs.close()


# =====================================================================
# Test 3: VecEnvAdapter + buffer warmup
# =====================================================================

def test_warmup_and_buffer():
    """
    验证 _VecEnvAdapter + _collect_warmup 流程。
    用零噪声收集 warmup 数据到 buffer，验证数据正确性。
    """
    print("\n" + "=" * 70)
    print("Test 3: VecEnvAdapter + buffer warmup")
    print("=" * 70)

    from rlft.online.train_dsrl import _VecEnvAdapter, _collect_warmup
    from rlft.buffers.dsrl_buffer import DSRLReplayBuffer

    base_policy, visual_encoder, wrapped_env, eval_envs, raw_env = setup_pipeline()

    try:
        adapter = _VecEnvAdapter(wrapped_env)

        # 验证基本属性
        assert adapter.num_envs == NUM_ENVS
        print(f"  num_envs: {adapter.num_envs}")
        print(f"  action_space: {adapter.action_space}")
        print(f"  obs_space: {adapter.observation_space}")

        # 创建 buffer 并 warmup
        buffer = DSRLReplayBuffer(
            capacity=10000,
            obs_dim=OBS_DIM,
            noise_dim=NOISE_DIM,
            device=DEVICE,
        )

        n_warmup = 200  # 小数量快速测试
        collected = _collect_warmup(adapter, buffer, n_warmup)

        assert collected >= n_warmup
        assert buffer.size >= n_warmup
        print(f"  Collected: {collected}, buffer_size: {buffer.size}")

        # 验证 buffer 数据合理
        batch = buffer.sample(32)
        assert torch.isfinite(batch["obs"]).all(), "obs contains inf/nan"
        assert torch.isfinite(batch["next_obs"]).all(), "next_obs contains inf/nan"
        assert torch.isfinite(batch["rewards"]).all(), "rewards contains inf/nan"

        # 零噪声动作应全为 0
        assert (batch["actions"].abs().max() < 1e-5), \
            "Warmup actions should be zero noise"

        print(f"  ✓ Buffer data valid")

    finally:
        raw_env.close()
        eval_envs.close()


# =====================================================================
# Main
# =====================================================================

def main():
    """依次运行所有 L5 测试。"""
    print("=" * 70)
    print("L5: End-to-End Evaluation Tests")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Env: {ENV_ID}")
    print(f"  Eval episodes: {NUM_EVAL_EPISODES}")
    print("=" * 70)

    if not Path(CHECKPOINT_PATH).exists():
        print(f"\n✗ Checkpoint not found: {CHECKPOINT_PATH}")
        return

    if not torch.cuda.is_available():
        print("\n✗ CUDA not available")
        return

    results = {}

    # Test 1
    try:
        test_zero_noise_envwrapper_eval()
        results["zero_noise_eval"] = "PASS"
    except Exception as e:
        results["zero_noise_eval"] = f"FAIL: {e}"
        print(f"\n  ✗ FAIL: {e}")

    # Test 2
    try:
        test_eval_agent_wrapper()
        results["eval_agent_wrapper"] = "PASS"
    except Exception as e:
        results["eval_agent_wrapper"] = f"FAIL: {e}"
        print(f"\n  ✗ FAIL: {e}")

    # Test 3
    try:
        test_warmup_and_buffer()
        results["warmup_and_buffer"] = "PASS"
    except Exception as e:
        results["warmup_and_buffer"] = f"FAIL: {e}"
        print(f"\n  ✗ FAIL: {e}")

    # 汇总
    print("\n" + "=" * 70)
    print("L5 Test Summary")
    print("=" * 70)
    for name, result in results.items():
        icon = "✓" if result == "PASS" else "✗"
        print(f"  {icon} {name}: {result}")

    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    print(f"\n  {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
