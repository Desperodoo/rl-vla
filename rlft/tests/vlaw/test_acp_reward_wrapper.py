"""ACP Reward Wrapper 单元测试 — ACPRewardConfig, ACPRewardComputer, DualCameraRewardWrapper

测试规范：无 GPU、无真实权重、无 ManiSkill 环境。使用 mock 对象。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from rlft.envs.acp_reward_wrapper import (
    ACPRewardComputer,
    ACPRewardConfig,
    DualCameraRewardWrapper,
    _extract_base_camera,
    _get_render_frame,
)


# =========================================================================
# Helpers / Mocks
# =========================================================================


class MockValueModel:
    """Deterministic mock for ManiSkillValueModel.

    Returns linearly spaced values based on frame index. Each call increments
    the internal counter, so successive calls return different values, enabling
    TD reward testing.
    """

    def __init__(self, base_value: float = -0.5, step_delta: float = 0.01):
        self.base_value = base_value
        self.step_delta = step_delta
        self._call_count = 0

    def predict_values(
        self, images: torch.Tensor, image_mask: torch.Tensor
    ) -> torch.Tensor:
        B = images.shape[0]
        vals = np.full(B, self.base_value + self._call_count * self.step_delta,
                       dtype=np.float32)
        self._call_count += 1
        return torch.from_numpy(vals)

    def load(self, path: str) -> None:
        pass


class MockEnv(gym.Env):
    """Minimal mock ManiSkill environment for wrapper testing.

    Provides sensor_data-style observations and deterministic render output.
    """

    def __init__(self, num_envs: int = 4, H: int = 128, W: int = 128):
        super().__init__()
        self.num_envs = num_envs
        self.H = H
        self.W = W
        self._step_count = 0
        self._done_at_step: Optional[int] = None  # force done at this step

        self.observation_space = gym.spaces.Dict({
            "sensor_data": gym.spaces.Dict({}),
            "agent": gym.spaces.Dict({}),
        })
        self.action_space = gym.spaces.Box(-1, 1, shape=(7,))

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        obs = self._make_obs()
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = self._make_obs()
        reward = np.ones(self.num_envs, dtype=np.float32) * 0.5  # sim reward
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        if self._done_at_step is not None and self._step_count >= self._done_at_step:
            terminated[0] = True  # env 0 terminates

        info = {"success": np.zeros(self.num_envs, dtype=bool)}
        return obs, reward, terminated, truncated, info

    def render(self):
        # Return deterministic render frames
        rgb = np.full((self.num_envs, self.H, self.W, 3), 128, dtype=np.uint8)
        return rgb

    def _make_obs(self):
        rgb = np.full((self.num_envs, self.H, self.W, 3), 100, dtype=np.uint8)
        return {
            "sensor_data": {
                "base_camera": {
                    "rgb": rgb,
                }
            },
            "agent": {
                "qpos": np.zeros((self.num_envs, 7), dtype=np.float32),
            },
        }


# =========================================================================
# B1: ACPRewardConfig tests
# =========================================================================


class TestACPRewardConfig:
    def test_defaults(self) -> None:
        cfg = ACPRewardConfig()
        assert cfg.checkpoint_path == "checkpoints/vlaw/acp/v3_so/best.safetensors"
        assert cfg.camera_height == 128
        assert cfg.camera_width == 128
        assert cfg.reward_scale == 100.0
        assert cfg.device == "cuda:1"
        assert cfg.dtype == "bfloat16"
        assert cfg.warmup_steps == 0
        assert cfg.use_sim_reward_bonus is False
        assert cfg.sim_reward_weight == 0.0

    def test_custom_values(self) -> None:
        cfg = ACPRewardConfig(
            checkpoint_path="/tmp/test.safetensors",
            reward_scale=50.0,
            device="cpu",
            warmup_steps=1000,
        )
        assert cfg.checkpoint_path == "/tmp/test.safetensors"
        assert cfg.reward_scale == 50.0
        assert cfg.device == "cpu"
        assert cfg.warmup_steps == 1000


# =========================================================================
# B2: ACPRewardComputer tests (with mock value model)
# =========================================================================


class TestACPRewardComputer:
    """Tests ACPRewardComputer with a mocked value model."""

    def _make_computer(
        self, base_value: float = -0.5, step_delta: float = 0.01,
        reward_scale: float = 100.0,
    ) -> ACPRewardComputer:
        """Create ACPRewardComputer with a mock value model injected."""
        config = ACPRewardConfig(reward_scale=reward_scale, device="cpu")
        computer = ACPRewardComputer(config)
        # Inject mock value model to skip loading real weights
        computer._value_model = MockValueModel(base_value, step_delta)
        computer._loaded = True
        return computer

    def _make_images(self, N: int = 4, H: int = 128, W: int = 128):
        """Create dummy dual-camera images."""
        rgb_base = np.random.randint(0, 255, (N, H, W, 3), dtype=np.uint8)
        rgb_render = np.random.randint(0, 255, (N, H, W, 3), dtype=np.uint8)
        return rgb_base, rgb_render

    def test_first_call_returns_zeros(self) -> None:
        """First compute_reward() should return zeros (priming the cache)."""
        computer = self._make_computer()
        computer.reset(4)
        rgb_base, rgb_render = self._make_images(4)
        reward = computer.compute_reward(rgb_base, rgb_render)

        assert reward.shape == (4,)
        np.testing.assert_array_equal(reward, np.zeros(4, dtype=np.float32))

    def test_td_reward_computation(self) -> None:
        """r = (V(s') - V(s)) * reward_scale. Second call should return non-zero."""
        computer = self._make_computer(base_value=-0.5, step_delta=0.02, reward_scale=100.0)
        computer.reset(4)
        rgb_base, rgb_render = self._make_images(4)

        # First call: primes cache with V = -0.5
        reward1 = computer.compute_reward(rgb_base, rgb_render)
        np.testing.assert_array_equal(reward1, np.zeros(4))

        # Second call: V = -0.5 + 0.02 = -0.48
        # TD reward = (-0.48 - (-0.5)) * 100 = 0.02 * 100 = 2.0
        reward2 = computer.compute_reward(rgb_base, rgb_render)
        np.testing.assert_allclose(reward2, np.full(4, 2.0), rtol=1e-5)

    def test_reward_scale(self) -> None:
        """Verify different reward_scale values are applied correctly."""
        for scale in [1.0, 50.0, 200.0]:
            computer = self._make_computer(
                base_value=-0.5, step_delta=0.01, reward_scale=scale
            )
            computer.reset(2)
            rgb_base, rgb_render = self._make_images(2)

            computer.compute_reward(rgb_base, rgb_render)  # prime
            reward = computer.compute_reward(rgb_base, rgb_render)

            expected = 0.01 * scale  # step_delta * reward_scale
            np.testing.assert_allclose(reward, np.full(2, expected), rtol=1e-5)

    def test_reset_env_clears_cache(self) -> None:
        """After reset_env(), next call returns 0 reward for reset envs."""
        computer = self._make_computer(base_value=-0.5, step_delta=0.01, reward_scale=100.0)
        computer.reset(4)
        rgb_base, rgb_render = self._make_images(4)

        # Prime + one normal step
        computer.compute_reward(rgb_base, rgb_render)
        reward1 = computer.compute_reward(rgb_base, rgb_render)
        assert np.all(reward1 != 0)

        # Reset envs 0 and 2
        computer.reset_env(np.array([0, 2]))

        # Next call: envs 0, 2 should get 0 reward (re-primed)
        reward2 = computer.compute_reward(rgb_base, rgb_render)
        assert reward2[0] == 0.0
        assert reward2[2] == 0.0
        # Envs 1, 3 should get normal reward
        assert reward2[1] != 0.0
        assert reward2[3] != 0.0

    def test_multiple_envs_independent(self) -> None:
        """Each env maintains independent cached value."""
        computer = self._make_computer()
        computer.reset(3)
        rgb_base, rgb_render = self._make_images(3)

        # Prime
        computer.compute_reward(rgb_base, rgb_render)

        # Reset only env 1
        computer.reset_env(np.array([1]))

        reward = computer.compute_reward(rgb_base, rgb_render)
        # Env 1 was reset, should be 0; envs 0,2 should be same (from mock)
        assert reward[1] == 0.0
        assert reward[0] == reward[2]

    def test_reset_full(self) -> None:
        """reset() clears all state."""
        computer = self._make_computer()
        computer.reset(4)
        rgb_base, rgb_render = self._make_images(4)

        computer.compute_reward(rgb_base, rgb_render)  # prime
        computer.compute_reward(rgb_base, rgb_render)  # normal

        # Full reset
        computer.reset(4)
        reward = computer.compute_reward(rgb_base, rgb_render)
        np.testing.assert_array_equal(reward, np.zeros(4))


# =========================================================================
# B3: Helper function tests
# =========================================================================


class TestHelperFunctions:
    def test_extract_base_camera(self) -> None:
        """_extract_base_camera extracts from sensor_data."""
        N, H, W = 4, 128, 128
        rgb = np.random.randint(0, 255, (N, H, W, 3), dtype=np.uint8)
        obs = {"sensor_data": {"base_camera": {"rgb": rgb}}}

        result = _extract_base_camera(obs, N, H, W)
        np.testing.assert_array_equal(result, rgb)

    def test_extract_base_camera_missing(self) -> None:
        """_extract_base_camera returns zeros when base_camera is missing."""
        obs = {"sensor_data": {}}
        result = _extract_base_camera(obs, 4, 128, 128)
        assert result.shape == (4, 128, 128, 3)
        np.testing.assert_array_equal(result, 0)

    def test_extract_base_camera_torch_tensor(self) -> None:
        """_extract_base_camera handles torch.Tensor input."""
        N, H, W = 2, 64, 64
        rgb_np = np.random.randint(0, 255, (N, H, W, 3), dtype=np.uint8)
        rgb_torch = torch.from_numpy(rgb_np)
        obs = {"sensor_data": {"base_camera": {"rgb": rgb_torch}}}

        result = _extract_base_camera(obs, N, H, W)
        np.testing.assert_array_equal(result, rgb_np)

    def test_extract_base_camera_resize(self) -> None:
        """_extract_base_camera resizes when source resolution differs."""
        N, src_H, src_W = 2, 256, 256
        tgt_H, tgt_W = 128, 128
        rgb = np.random.randint(0, 255, (N, src_H, src_W, 3), dtype=np.uint8)
        obs = {"sensor_data": {"base_camera": {"rgb": rgb}}}

        result = _extract_base_camera(obs, N, tgt_H, tgt_W)
        assert result.shape == (N, tgt_H, tgt_W, 3)

    def test_get_render_frame_success(self) -> None:
        """_get_render_frame returns properly shaped array from mock env."""
        env = MockEnv(num_envs=4, H=128, W=128)
        result = _get_render_frame(env, 4, 128, 128)
        assert result.shape == (4, 128, 128, 3)
        assert result.dtype == np.uint8

    def test_get_render_frame_failure(self) -> None:
        """_get_render_frame returns zeros when render() raises."""
        env = MagicMock()
        env.render.side_effect = RuntimeError("no renderer")
        result = _get_render_frame(env, 4, 128, 128)
        assert result.shape == (4, 128, 128, 3)
        np.testing.assert_array_equal(result, 0)

    def test_get_render_frame_resize(self) -> None:
        """_get_render_frame resizes when render resolution differs."""
        env = MagicMock()
        # render() returns 256x256 images
        env.render.return_value = np.full((4, 256, 256, 3), 200, dtype=np.uint8)
        result = _get_render_frame(env, 4, 128, 128)
        assert result.shape == (4, 128, 128, 3)

    def test_get_render_frame_torch_tensor(self) -> None:
        """_get_render_frame handles torch.Tensor render output."""
        env = MagicMock()
        env.render.return_value = torch.full((2, 128, 128, 3), 150, dtype=torch.uint8)
        result = _get_render_frame(env, 2, 128, 128)
        assert result.shape == (2, 128, 128, 3)
        assert result.dtype == np.uint8


# =========================================================================
# B4: DualCameraRewardWrapper tests
# =========================================================================


class TestDualCameraRewardWrapper:
    """Tests DualCameraRewardWrapper with mock env and mock value model."""

    def _make_wrapper(
        self,
        num_envs: int = 4,
        reward_scale: float = 100.0,
        warmup_steps: int = 0,
        use_sim_reward_bonus: bool = False,
        sim_reward_weight: float = 0.0,
    ) -> DualCameraRewardWrapper:
        env = MockEnv(num_envs=num_envs)
        config = ACPRewardConfig(
            reward_scale=reward_scale,
            warmup_steps=warmup_steps,
            device="cpu",
            use_sim_reward_bonus=use_sim_reward_bonus,
            sim_reward_weight=sim_reward_weight,
        )
        wrapper = DualCameraRewardWrapper(env, config)
        # Inject mock value model
        wrapper.acp_computer._value_model = MockValueModel(
            base_value=-0.5, step_delta=0.01
        )
        wrapper.acp_computer._loaded = True
        return wrapper

    def test_reset_returns_obs(self) -> None:
        """reset() should return observation dict unchanged."""
        wrapper = self._make_wrapper()
        obs, info = wrapper.reset()
        assert "sensor_data" in obs
        assert "base_camera" in obs["sensor_data"]

    def test_reward_replaced(self) -> None:
        """Returned reward should be ACP TD reward, not sim reward."""
        wrapper = self._make_wrapper(reward_scale=100.0)
        wrapper.reset()

        _, reward, _, _, info = wrapper.step(np.zeros((4, 7)))
        # Sim reward is 0.5 per env (from MockEnv)
        # ACP reward should be step_delta * reward_scale = 0.01 * 100 = 1.0
        assert isinstance(reward, np.ndarray)
        np.testing.assert_allclose(reward, np.full(4, 1.0), rtol=1e-5)

    def test_sim_reward_in_info(self) -> None:
        """Original sim reward should be preserved in info['sim_reward']."""
        wrapper = self._make_wrapper()
        wrapper.reset()

        _, _, _, _, info = wrapper.step(np.zeros((4, 7)))
        assert "sim_reward" in info
        np.testing.assert_allclose(info["sim_reward"], np.full(4, 0.5), rtol=1e-5)

    def test_warmup_uses_sim_reward(self) -> None:
        """During warmup_steps, sim reward should be used unchanged."""
        wrapper = self._make_wrapper(warmup_steps=5)
        wrapper.reset()

        # Steps 1-5 are warmup
        for _ in range(5):
            _, reward, _, _, info = wrapper.step(np.zeros((4, 7)))
            # During warmup, reward should be the original sim reward (0.5)
            np.testing.assert_allclose(reward, np.full(4, 0.5), rtol=1e-5)

        # Step 6: should use ACP reward
        _, reward, _, _, _ = wrapper.step(np.zeros((4, 7)))
        # ACP reward should not be 0.5 (sim reward)
        assert not np.allclose(reward, np.full(4, 0.5))

    def test_auto_reset_handling(self) -> None:
        """When done=True, ACP reward should be zero for that env."""
        env = MockEnv(num_envs=4)
        env._done_at_step = 2  # env 0 terminates at step 2

        config = ACPRewardConfig(reward_scale=100.0, device="cpu")
        wrapper = DualCameraRewardWrapper(env, config)
        wrapper.acp_computer._value_model = MockValueModel(-0.5, 0.01)
        wrapper.acp_computer._loaded = True
        wrapper.reset()

        # Step 1: normal
        _, reward1, _, _, _ = wrapper.step(np.zeros((4, 7)))
        assert reward1[0] != 0.0  # env 0 has normal reward

        # Step 2: env 0 terminates
        _, reward2, terminated, _, _ = wrapper.step(np.zeros((4, 7)))
        assert terminated[0] == True
        # Env 0 should have 0 reward (reset handling)
        assert reward2[0] == 0.0
        # Other envs should have normal reward
        assert reward2[1] != 0.0

    def test_blend_mode(self) -> None:
        """acp_blend mode adds weighted sim reward to ACP reward."""
        wrapper = self._make_wrapper(
            reward_scale=100.0,
            use_sim_reward_bonus=True,
            sim_reward_weight=0.5,  # add 50% of sim reward
        )
        wrapper.reset()

        _, reward, _, _, info = wrapper.step(np.zeros((4, 7)))
        # ACP TD reward = 0.01 * 100 = 1.0
        # Sim reward bonus = 0.5 * 0.5 = 0.25
        # Total = 1.0 + 0.25 = 1.25
        np.testing.assert_allclose(reward, np.full(4, 1.25), rtol=1e-5)

    def test_obs_structure_preserved(self) -> None:
        """Observation dict should pass through unchanged."""
        wrapper = self._make_wrapper()
        obs_reset, _ = wrapper.reset()
        obs_step, _, _, _, _ = wrapper.step(np.zeros((4, 7)))

        # Both should have sensor_data with base_camera
        for obs in [obs_reset, obs_step]:
            assert "sensor_data" in obs
            assert "base_camera" in obs["sensor_data"]
            assert "rgb" in obs["sensor_data"]["base_camera"]
            assert "agent" in obs

    def test_num_envs_inferred(self) -> None:
        """Wrapper should correctly infer num_envs from obs."""
        for n in [1, 4, 16]:
            wrapper = self._make_wrapper(num_envs=n)
            wrapper.reset()
            assert wrapper._num_envs == n


# =========================================================================
# B5: Integration test — reward sign convention
# =========================================================================


class TestRewardSignConvention:
    """Verify that ACP TD reward has correct sign semantics.

    A state with higher value (closer to 0) should give positive reward
    when transitioning to it from a lower-value state. This is the expected
    behavior for potential-based reward shaping.
    """

    def test_positive_progress_gives_positive_reward(self) -> None:
        """Moving to a higher-value state should give positive reward."""
        config = ACPRewardConfig(reward_scale=1.0, device="cpu")
        computer = ACPRewardComputer(config)
        # Mock: values increase (get closer to 0) each step
        computer._value_model = MockValueModel(base_value=-0.8, step_delta=0.1)
        computer._loaded = True
        computer.reset(1)

        rgb = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        computer.compute_reward(rgb, rgb)  # prime: V = -0.8
        reward = computer.compute_reward(rgb, rgb)  # V = -0.7
        assert reward[0] > 0  # positive progress -> positive reward

    def test_negative_progress_gives_negative_reward(self) -> None:
        """Moving to a lower-value state should give negative reward."""
        config = ACPRewardConfig(reward_scale=1.0, device="cpu")
        computer = ACPRewardComputer(config)
        # Mock: values decrease (get further from 0) each step
        computer._value_model = MockValueModel(base_value=-0.2, step_delta=-0.1)
        computer._loaded = True
        computer.reset(1)

        rgb = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        computer.compute_reward(rgb, rgb)  # prime: V = -0.2
        reward = computer.compute_reward(rgb, rgb)  # V = -0.3
        assert reward[0] < 0  # negative progress -> negative reward
