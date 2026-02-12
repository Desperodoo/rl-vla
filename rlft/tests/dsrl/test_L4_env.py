"""
L4: 环境集成测试

验证 ManiSkillFlowEnvWrapper 与 ManiSkill3 实际环境的集成：
  - 环境创建成功
  - action_space / observation_space 维度正确
  - reset() 返回正确形状
  - step() 使用零噪声正常运行
  - 观察历史管理正常

需要: checkpoint + GPU + ManiSkill3

运行:
    conda activate carm
    cd /home/lizh/rl-vla
    CUDA_VISIBLE_DEVICES=0 python -m pytest rlft/tests/dsrl/test_L4_env.py -v -s
"""

import pytest
import sys
from pathlib import Path

import torch
import numpy as np

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
    sys.path.insert(0, str(_root / "diffusion_policy"))

CHECKPOINT_PATH = str(_root / "runs/awsc_checkpoint/checkpoints/best_eval_success_once.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 环境配置
ENV_ID = "LiftPegUpright-v1"
NUM_ENVS = 4  # 小数量用于快速测试
OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_STEPS = 8
ACTION_DIM = 7
STATE_DIM = 25
VISUAL_DIM = 256
ACTION_MAG = 1.5
NOISE_DIM = ACT_STEPS * ACTION_DIM  # 56
OBS_DIM = OBS_HORIZON * (VISUAL_DIM + STATE_DIM)  # 562


def _skip_if_no_checkpoint():
    if not Path(CHECKPOINT_PATH).exists():
        pytest.skip("Checkpoint not found")


def _skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def wrapped_env():
    """模块级 fixture: 创建完整的 wrapped env。"""
    _skip_if_no_checkpoint()
    _skip_if_no_gpu()

    import gymnasium as gym
    import mani_skill.envs  # noqa
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    from rlft.utils.flow_wrapper import load_shortcut_flow_policy
    from rlft.networks import PlainConv
    from rlft.envs.dsrl_env import ManiSkillFlowEnvWrapper

    # 加载 flow policy
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

    # 创建 ManiSkill3 base env
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

    # 包装
    env = ManiSkillFlowEnvWrapper(
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

    yield env

    raw_env.close()


# =====================================================================
# L4-1: 空间维度
# =====================================================================

class TestEnvSpaces:
    """验证环境空间维度正确。"""

    def test_action_space_shape(self, wrapped_env):
        expected = (NOISE_DIM,)
        assert wrapped_env.action_space.shape == expected, \
            f"Action space {wrapped_env.action_space.shape} != {expected}"

    def test_action_space_bounds(self, wrapped_env):
        low = wrapped_env.action_space.low
        high = wrapped_env.action_space.high
        np.testing.assert_allclose(low, -ACTION_MAG, atol=1e-5)
        np.testing.assert_allclose(high, ACTION_MAG, atol=1e-5)

    def test_obs_space_shape(self, wrapped_env):
        expected = (OBS_DIM,)
        assert wrapped_env.observation_space.shape == expected, \
            f"Obs space {wrapped_env.observation_space.shape} != {expected}"

    def test_num_envs(self, wrapped_env):
        assert wrapped_env.num_envs == NUM_ENVS


# =====================================================================
# L4-2: reset / step 基础行为
# =====================================================================

class TestEnvResetStep:
    """验证 reset 和 step 的基本行为。"""

    def test_reset_returns_obs(self, wrapped_env):
        obs, info = wrapped_env.reset()
        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (NUM_ENVS, OBS_DIM), f"obs shape: {obs.shape}"

    def test_step_zero_noise(self, wrapped_env):
        """零噪声 step 应正常返回。"""
        wrapped_env.reset()
        action = torch.zeros(NUM_ENVS, NOISE_DIM, device=DEVICE)
        obs, rew, term, trunc, info = wrapped_env.step(action)

        assert obs.shape == (NUM_ENVS, OBS_DIM)
        assert rew.shape == (NUM_ENVS,)
        assert term.shape == (NUM_ENVS,)
        assert trunc.shape == (NUM_ENVS,)

    def test_step_random_noise(self, wrapped_env):
        """随机噪声 step 不应崩溃。"""
        wrapped_env.reset()
        action = torch.randn(NUM_ENVS, NOISE_DIM, device=DEVICE) * ACTION_MAG * 0.5
        obs, rew, term, trunc, info = wrapped_env.step(action)
        assert obs.shape == (NUM_ENVS, OBS_DIM)

    def test_step_numpy_input(self, wrapped_env):
        """step 应支持 numpy 输入。"""
        wrapped_env.reset()
        action = np.zeros((NUM_ENVS, NOISE_DIM), dtype=np.float32)
        obs, rew, term, trunc, info = wrapped_env.step(action)
        assert obs.shape == (NUM_ENVS, OBS_DIM)

    def test_obs_finite(self, wrapped_env):
        """观察值应有限（无 inf/nan）。"""
        obs, _ = wrapped_env.reset()
        assert torch.isfinite(obs).all(), "Obs contains inf/nan after reset"

        action = torch.zeros(NUM_ENVS, NOISE_DIM, device=DEVICE)
        obs, _, _, _, _ = wrapped_env.step(action)
        assert torch.isfinite(obs).all(), "Obs contains inf/nan after step"


# =====================================================================
# L4-3: 观察历史管理
# =====================================================================

class TestObsHistory:
    """验证观察历史的 rolling buffer 逻辑。"""

    def test_obs_dim_covers_history(self, wrapped_env):
        """obs_dim 应等于 obs_horizon * single_obs_dim。"""
        assert wrapped_env.obs_dim == OBS_DIM

    def test_multiple_steps_update_history(self, wrapped_env):
        """连续 step 后观察应更新（不全为零）。"""
        wrapped_env.reset()
        action = torch.zeros(NUM_ENVS, NOISE_DIM, device=DEVICE)

        prev_obs = None
        for _ in range(3):
            obs, _, _, _, _ = wrapped_env.step(action)
            if prev_obs is not None:
                # 至少某些维度应该变化
                diff = (obs - prev_obs).abs().sum().item()
                assert diff > 0, "Observations should change across steps"
            prev_obs = obs.clone()


# =====================================================================
# L4-4: 多轮 episode rollout 稳定性
# =====================================================================

class TestEnvRolloutStability:
    """验证多轮 rollout 不会崩溃。"""

    def test_multi_episode_rollout(self, wrapped_env):
        """运行 3 个完整 episode，确保无 crash。"""
        for ep in range(3):
            obs, info = wrapped_env.reset()
            total_rew = torch.zeros(NUM_ENVS, device=DEVICE)
            done = torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE)

            for step in range(100):  # max_episode_steps=100
                action = torch.zeros(NUM_ENVS, NOISE_DIM, device=DEVICE)
                obs, rew, term, trunc, info = wrapped_env.step(action)
                total_rew += rew
                done = done | term | trunc
                if done.all():
                    break

            print(f"  Episode {ep}: total_reward={total_rew.mean():.2f}, "
                  f"steps={step+1}, done_ratio={done.float().mean():.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
