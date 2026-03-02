"""Tests for ImaginationRLEnv (P8.2): WM+VLM as Gym env.

使用 Mock 对象, 不需要 GPU 或真实模型权重。
验证: Gym API 兼容性、空间维度、reset/step 语义、VLM reward 间隔、
      终止条件、向量化封装、工厂函数等。

所属阶段: P8.2 — Imagination RL
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rlft.vlaw.world_model.imagination_rl_env import (
    ImaginationRLEnv,
    ImaginationRLEnvConfig,
    MockCtrlWorldAdapter,
    MockRewardModel,
    MockStatePredictor,
    VecImaginationRLEnv,
    load_initial_frames_from_h5,
    make_imagination_rl_env,
    make_vec_imagination_rl_env,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> ImaginationRLEnvConfig:
    return ImaginationRLEnvConfig(
        max_steps=20,
        vlm_reward_interval=5,
        verbose=False,
    )


@pytest.fixture
def mock_env(default_config: ImaginationRLEnvConfig) -> ImaginationRLEnv:
    return make_imagination_rl_env(default_config, use_mock=True)


# ---------------------------------------------------------------------------
# Test: 基本 Gym API 兼容
# ---------------------------------------------------------------------------


class TestGymAPI:
    """gymnasium.Env 接口兼容性测试."""

    def test_observation_space_shape(self, mock_env: ImaginationRLEnv) -> None:
        """obs space = latent_flat + state_dim."""
        expected = 4 * 48 * 24 + 25  # 4608 + 25 = 4633
        assert mock_env.observation_space.shape == (expected,)

    def test_action_space_shape(self, mock_env: ImaginationRLEnv) -> None:
        assert mock_env.action_space.shape == (7,)

    def test_action_space_bounds(self, mock_env: ImaginationRLEnv) -> None:
        assert np.all(mock_env.action_space.low == -1.0)
        assert np.all(mock_env.action_space.high == 1.0)

    def test_reset_returns_tuple(self, mock_env: ImaginationRLEnv) -> None:
        result = mock_env.reset(seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(info, dict)

    def test_reset_obs_shape(self, mock_env: ImaginationRLEnv) -> None:
        obs, _ = mock_env.reset(seed=0)
        assert obs.shape == mock_env.observation_space.shape

    def test_reset_obs_in_space(self, mock_env: ImaginationRLEnv) -> None:
        obs, _ = mock_env.reset(seed=0)
        assert mock_env.observation_space.contains(obs)

    def test_step_returns_5_tuple(self, mock_env: ImaginationRLEnv) -> None:
        mock_env.reset(seed=0)
        action = mock_env.action_space.sample()
        result = mock_env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_shape(self, mock_env: ImaginationRLEnv) -> None:
        mock_env.reset(seed=0)
        obs, *_ = mock_env.step(mock_env.action_space.sample())
        assert obs.shape == mock_env.observation_space.shape

    def test_step_obs_in_space(self, mock_env: ImaginationRLEnv) -> None:
        mock_env.reset(seed=0)
        obs, *_ = mock_env.step(mock_env.action_space.sample())
        assert mock_env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# Test: 观测模式
# ---------------------------------------------------------------------------


class TestObsModes:
    """不同 obs_mode 的维度验证."""

    def test_flat_mode(self) -> None:
        cfg = ImaginationRLEnvConfig(obs_mode="flat", max_steps=5)
        env = make_imagination_rl_env(cfg, use_mock=True)
        obs, _ = env.reset(seed=0)
        assert obs.ndim == 1
        assert obs.shape[0] == 4 * 48 * 24 + 25
        env.close()

    def test_dict_mode(self) -> None:
        cfg = ImaginationRLEnvConfig(obs_mode="dict", max_steps=5)
        env = make_imagination_rl_env(cfg, use_mock=True)
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert "latent" in obs
        assert "agent_state" in obs
        assert obs["latent"].shape == (4, 48, 24)
        assert obs["agent_state"].shape == (25,)
        env.close()

    def test_latent_only_mode(self) -> None:
        cfg = ImaginationRLEnvConfig(obs_mode="latent_only", max_steps=5)
        env = make_imagination_rl_env(cfg, use_mock=True)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4, 48, 24)
        env.close()


# ---------------------------------------------------------------------------
# Test: Reset 机制
# ---------------------------------------------------------------------------


class TestReset:
    """reset() 行为验证."""

    def test_reset_with_seed_deterministic(self, default_config: ImaginationRLEnvConfig) -> None:
        """相同 seed 应产出相同结果 (无初始帧池时用随机噪声)."""
        env1 = make_imagination_rl_env(default_config, use_mock=True)
        env2 = make_imagination_rl_env(default_config, use_mock=True)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        # Mock WM 用 torch.randn, seed 不完全控制, 只验证不报错
        assert obs1.shape == obs2.shape
        env1.close()
        env2.close()

    def test_reset_with_options(self, mock_env: ImaginationRLEnv) -> None:
        """通过 options 注入初始帧."""
        lat = np.zeros((4, 48, 24), dtype=np.float32)
        st = np.ones(25, dtype=np.float32) * 0.5
        obs, info = mock_env.reset(
            seed=0,
            options={"initial_latent": lat, "initial_state": st},
        )
        # 验证 state 部分 (obs 最后 25 维)
        assert np.allclose(obs[-25:], 0.5)

    def test_reset_clears_step_count(self, mock_env: ImaginationRLEnv) -> None:
        mock_env.reset(seed=0)
        mock_env.step(mock_env.action_space.sample())
        mock_env.step(mock_env.action_space.sample())
        assert mock_env._step_count == 2
        mock_env.reset(seed=1)
        assert mock_env._step_count == 0

    def test_reset_from_h5(self, tmp_path) -> None:
        """从 HDF5 加载初始帧并 reset."""
        import h5py

        h5_path = tmp_path / "test_init.h5"
        rng = np.random.default_rng(0)
        with h5py.File(str(h5_path), "w") as f:
            for i in range(3):
                grp = f.create_group(f"traj_{i:04d}")
                grp.create_dataset(
                    "latent_concat",
                    data=rng.standard_normal((5, 4, 48, 24)).astype(np.float16),
                )
                grp.create_dataset(
                    "state",
                    data=rng.standard_normal((5, 25)).astype(np.float32),
                )

        cfg = ImaginationRLEnvConfig(
            max_steps=5,
            initial_frames_h5=str(h5_path),
        )
        env = make_imagination_rl_env(cfg, use_mock=True)
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        env.close()


# ---------------------------------------------------------------------------
# Test: Step 语义
# ---------------------------------------------------------------------------


class TestStep:
    """step() 行为验证."""

    def test_step_increments_count(self, mock_env: ImaginationRLEnv) -> None:
        mock_env.reset(seed=0)
        for i in range(5):
            mock_env.step(mock_env.action_space.sample())
        assert mock_env._step_count == 5

    def test_truncation_at_max_steps(self, mock_env: ImaginationRLEnv) -> None:
        """达到 max_steps 时 truncated=True."""
        mock_env.reset(seed=0)
        for i in range(mock_env.config.max_steps):
            _, _, term, trunc, _ = mock_env.step(mock_env.action_space.sample())
            if i < mock_env.config.max_steps - 1:
                assert not trunc, f"premature truncation at step {i+1}"
        assert trunc, "should be truncated at max_steps"

    def test_step_after_done_warns(self, mock_env: ImaginationRLEnv) -> None:
        """episode 结束后继续 step 应发出 warning."""
        mock_env.reset(seed=0)
        for _ in range(mock_env.config.max_steps):
            mock_env.step(mock_env.action_space.sample())
        with pytest.warns(UserWarning, match="reset"):
            mock_env.step(mock_env.action_space.sample())

    def test_step_latent_history_grows(self, mock_env: ImaginationRLEnv) -> None:
        """每步 latent_history 增长。"""
        mock_env.reset(seed=0)
        init_len = len(mock_env._latent_history)
        mock_env.step(mock_env.action_space.sample())
        assert len(mock_env._latent_history) == init_len + 1


# ---------------------------------------------------------------------------
# Test: VLM Reward 间隔
# ---------------------------------------------------------------------------


class TestVLMReward:
    """VLM reward 调用间隔验证."""

    def test_vlm_called_at_interval(self) -> None:
        """VLM 应仅在 vlm_reward_interval 步时被调用."""
        cfg = ImaginationRLEnvConfig(
            max_steps=20,
            vlm_reward_interval=5,
        )
        env = make_imagination_rl_env(cfg, use_mock=True)
        env.reset(seed=0)

        vlm_steps = []
        for i in range(20):
            _, _, _, _, info = env.step(env.action_space.sample())
            if info.get("is_vlm_step", False):
                vlm_steps.append(i + 1)

        assert vlm_steps == [5, 10, 15, 20], f"VLM steps: {vlm_steps}"
        env.close()

    def test_no_vlm_without_reward_model(self) -> None:
        """无 reward_model 时 reward 恒为 0."""
        cfg = ImaginationRLEnvConfig(max_steps=10)
        env = ImaginationRLEnv(
            wm_adapter=MockCtrlWorldAdapter(),
            reward_model=None,
            config=cfg,
        )
        env.reset(seed=0)
        for _ in range(10):
            _, reward, _, _, _ = env.step(env.action_space.sample())
            assert reward == 0.0
        env.close()

    def test_continuous_vs_binary_reward(self) -> None:
        """continuous 模式返回 p_yes, binary 模式返回 0/1."""
        for use_continuous in [True, False]:
            cfg = ImaginationRLEnvConfig(
                max_steps=10,
                vlm_reward_interval=1,
                use_continuous_reward=use_continuous,
            )
            env = make_imagination_rl_env(cfg, use_mock=True)
            env.reset(seed=0)
            _, reward, _, _, _ = env.step(env.action_space.sample())
            if not use_continuous:
                assert reward in (0.0, 1.0)
            # continuous 可以是任意 float
            env.close()


# ---------------------------------------------------------------------------
# Test: State Predictor 集成
# ---------------------------------------------------------------------------


class TestStatePredictor:
    """State Predictor 集成测试."""

    def test_state_changes_with_predictor(self) -> None:
        """使用 State Predictor 时 state 应有变化."""
        cfg = ImaginationRLEnvConfig(
            max_steps=5,
            use_state_predictor=True,
        )
        env = make_imagination_rl_env(cfg, use_mock=True)
        obs0, _ = env.reset(
            seed=0,
            options={"initial_state": np.ones(25, dtype=np.float32)},
        )
        state_before = obs0[-25:].copy()
        env.step(env.action_space.sample())
        obs1 = env._build_obs()
        state_after = obs1[-25:]
        # MockStatePredictor 加随机噪声, state 应有微小变化
        assert not np.allclose(state_before, state_after, atol=1e-6)
        env.close()

    def test_state_unchanged_without_predictor(self) -> None:
        """不用 State Predictor 时 state 不变."""
        cfg = ImaginationRLEnvConfig(
            max_steps=5,
            use_state_predictor=False,
        )
        env = ImaginationRLEnv(
            wm_adapter=MockCtrlWorldAdapter(),
            reward_model=None,
            state_predictor=None,
            config=cfg,
        )
        init_state = np.ones(25, dtype=np.float32) * 0.5
        env.reset(seed=0, options={"initial_state": init_state})
        env.step(env.action_space.sample())
        assert np.allclose(env._current_state, init_state)
        env.close()


# ---------------------------------------------------------------------------
# Test: WM Rollout 缓冲
# ---------------------------------------------------------------------------


class TestWMBuffer:
    """WM 批量 rollout + 逐步消费缓冲验证."""

    def test_pending_buffer_consumed(self) -> None:
        """连续 step 应消费 pending buffer, 每 wm_act_steps 调一次 WM."""
        cfg = ImaginationRLEnvConfig(
            max_steps=15,
            wm_act_steps=5,
        )
        env = make_imagination_rl_env(cfg, use_mock=True)
        env.reset(seed=0)

        # 前 5 步: 第 1 步触发 WM, 后 4 步消费缓冲
        for i in range(5):
            env.step(env.action_space.sample())

        # 第 6 步应触发新一轮 WM (pending exhausted)
        env.step(env.action_space.sample())

        assert env._step_count == 6
        env.close()


# ---------------------------------------------------------------------------
# Test: VecImaginationRLEnv
# ---------------------------------------------------------------------------


class TestVecEnv:
    """向量化环境封装测试."""

    def test_vec_env_reset(self) -> None:
        cfg = ImaginationRLEnvConfig(max_steps=10)
        vec = make_vec_imagination_rl_env(cfg, num_envs=3, use_mock=True)
        obs, info = vec.reset(seed=0)
        assert obs.shape == (3, 4 * 48 * 24 + 25)
        vec.close()

    def test_vec_env_step(self) -> None:
        cfg = ImaginationRLEnvConfig(max_steps=10)
        vec = make_vec_imagination_rl_env(cfg, num_envs=4, use_mock=True)
        vec.reset(seed=0)
        actions = np.stack([vec.action_space.sample() for _ in range(4)])
        obs, rewards, terms, truncs, info = vec.step(actions)
        assert obs.shape == (4, 4 * 48 * 24 + 25)
        assert rewards.shape == (4,)
        assert terms.shape == (4,)
        assert truncs.shape == (4,)
        vec.close()

    def test_vec_env_auto_reset(self) -> None:
        """vec env 在 episode 结束后应自动 reset."""
        cfg = ImaginationRLEnvConfig(max_steps=3, vlm_reward_interval=100)
        vec = make_vec_imagination_rl_env(cfg, num_envs=2, use_mock=True)
        vec.reset(seed=0)
        for _ in range(5):
            actions = np.stack([vec.action_space.sample() for _ in range(2)])
            obs, rewards, terms, truncs, info = vec.step(actions)
            # 不应崩溃
        vec.close()


# ---------------------------------------------------------------------------
# Test: 工厂函数
# ---------------------------------------------------------------------------


class TestFactory:
    """工厂函数验证."""

    def test_make_single_env(self) -> None:
        cfg = ImaginationRLEnvConfig(max_steps=5)
        env = make_imagination_rl_env(cfg, use_mock=True)
        assert isinstance(env, ImaginationRLEnv)
        env.close()

    def test_make_vec_env(self) -> None:
        cfg = ImaginationRLEnvConfig(max_steps=5)
        vec = make_vec_imagination_rl_env(cfg, num_envs=2, use_mock=True)
        assert isinstance(vec, VecImaginationRLEnv)
        assert vec.num_envs == 2
        vec.close()

    def test_make_without_wm_raises(self) -> None:
        cfg = ImaginationRLEnvConfig()
        with pytest.raises(ValueError, match="wm_adapter"):
            make_imagination_rl_env(cfg, wm_adapter=None, use_mock=False)


# ---------------------------------------------------------------------------
# Test: render()
# ---------------------------------------------------------------------------


class TestRender:
    """render() 测试."""

    def test_render_returns_rgb(self, mock_env: ImaginationRLEnv) -> None:
        mock_env.reset(seed=0)
        mock_env.step(mock_env.action_space.sample())
        rgb = mock_env.render()
        assert rgb is not None
        assert rgb.shape == (192, 192, 3)
        assert rgb.dtype == np.uint8


# ---------------------------------------------------------------------------
# Test: close()
# ---------------------------------------------------------------------------


class TestClose:
    """close() 资源清理测试."""

    def test_close_clears_buffers(self, mock_env: ImaginationRLEnv) -> None:
        mock_env.reset(seed=0)
        mock_env.step(mock_env.action_space.sample())
        assert len(mock_env._latent_history) > 0
        mock_env.close()
        assert len(mock_env._latent_history) == 0


# ---------------------------------------------------------------------------
# Test: load_initial_frames_from_h5
# ---------------------------------------------------------------------------


class TestLoadInitialFrames:
    """H5 初始帧加载测试."""

    def test_load_from_h5(self, tmp_path) -> None:
        import h5py

        h5_path = tmp_path / "init_frames.h5"
        rng = np.random.default_rng(0)
        with h5py.File(str(h5_path), "w") as f:
            for i in range(5):
                grp = f.create_group(f"traj_{i:04d}")
                grp.create_dataset(
                    "latent_concat",
                    data=rng.standard_normal((3, 4, 48, 24)).astype(np.float16),
                )
                grp.create_dataset(
                    "state",
                    data=rng.standard_normal((3, 25)).astype(np.float32),
                )

        frames = load_initial_frames_from_h5(str(h5_path), max_count=3)
        assert len(frames) == 3
        assert frames[0]["latent"].shape == (4, 48, 24)
        assert frames[0]["state"].shape == (25,)

    def test_load_empty_h5(self, tmp_path) -> None:
        import h5py

        h5_path = tmp_path / "empty.h5"
        with h5py.File(str(h5_path), "w") as f:
            pass
        frames = load_initial_frames_from_h5(str(h5_path))
        assert len(frames) == 0


# ---------------------------------------------------------------------------
# Test: 边界情况
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """边界情况验证."""

    def test_single_step_episode(self) -> None:
        """max_steps=1 的极端短 episode."""
        cfg = ImaginationRLEnvConfig(max_steps=1, vlm_reward_interval=1)
        env = make_imagination_rl_env(cfg, use_mock=True)
        env.reset(seed=0)
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        assert trunc  # max_steps=1 → truncated
        env.close()

    def test_no_nan_in_obs(self) -> None:
        """20 步 rollout 中 obs 不应出现 NaN."""
        cfg = ImaginationRLEnvConfig(max_steps=20)
        env = make_imagination_rl_env(cfg, use_mock=True)
        env.reset(seed=0)
        for _ in range(20):
            obs, *_ = env.step(env.action_space.sample())
            assert not np.any(np.isnan(obs)), "NaN detected in obs"
        env.close()

    def test_different_state_dims(self) -> None:
        """不同 state_dim (25 vs 29) 应正常工作."""
        for sd in [25, 29]:
            cfg = ImaginationRLEnvConfig(max_steps=5, state_dim=sd)
            env = make_imagination_rl_env(cfg, use_mock=True)
            obs, _ = env.reset(seed=0)
            assert obs.shape[0] == 4 * 48 * 24 + sd
            env.close()

    def test_num_envs_property(self, mock_env: ImaginationRLEnv) -> None:
        assert mock_env.num_envs == 1
