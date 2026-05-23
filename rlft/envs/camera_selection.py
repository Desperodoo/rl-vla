"""Camera-channel selection for flattened ManiSkill RGBD observations."""

from __future__ import annotations

import os
from typing import Sequence

import gymnasium as gym
from gymnasium import spaces


def selected_camera_names() -> list[str]:
    """Return requested ManiSkill camera names from the process environment."""
    return [
        name.strip()
        for name in os.environ.get("RLFT_MANISKILL_CAMERA_NAMES", "").split(",")
        if name.strip()
    ]


class SelectManiSkillCamerasWrapper(gym.ObservationWrapper):
    """Keep only selected camera channels after FlattenRGBDObservationWrapper.

    ManiSkill's flatten wrapper concatenates cameras along the final channel
    dimension in raw sensor-data order. This wrapper preserves the HDF5 file and
    narrows only the observation stream used by training/evaluation.
    """

    def __init__(self, env: gym.Env, camera_names: Sequence[str] | None = None):
        super().__init__(env)
        self.camera_names = list(camera_names) if camera_names is not None else selected_camera_names()

        base_env = getattr(self.env, "base_env", None) or getattr(self.env, "unwrapped", None)
        raw_obs = getattr(base_env, "_init_raw_obs", {})
        sensor_data = raw_obs.get("sensor_data", {}) if isinstance(raw_obs, dict) else {}
        if sensor_data:
            self.camera_order = list(sensor_data.keys())
        else:
            sensors = getattr(base_env, "_sensors", None) or getattr(base_env, "_sensor_configs", None) or {}
            self.camera_order = list(sensors.keys()) if hasattr(sensors, "keys") else []

        self.rgb_indices = self._build_indices(channels_per_camera=3)
        self.depth_indices = self._build_indices(channels_per_camera=1)
        self.observation_space = self._build_observation_space(env.observation_space)
        if self.camera_names and hasattr(base_env, "update_obs_space") and isinstance(raw_obs, dict):
            base_env.update_obs_space(self.observation(dict(raw_obs)))

    def _build_indices(self, channels_per_camera: int) -> list[int] | None:
        if not self.camera_names:
            return None
        missing = [name for name in self.camera_names if name not in self.camera_order]
        if missing:
            raise ValueError(
                f"Requested ManiSkill camera(s) {missing} not found; available={self.camera_order}"
            )

        indices: list[int] = []
        for name in self.camera_names:
            camera_idx = self.camera_order.index(name)
            start = camera_idx * channels_per_camera
            indices.extend(range(start, start + channels_per_camera))
        return indices

    def _slice_space(self, space: spaces.Box, indices: list[int] | None) -> spaces.Box:
        if indices is None:
            return space
        shape = (*space.shape[:-1], len(indices))
        return spaces.Box(low=space.low.min(), high=space.high.max(), shape=shape, dtype=space.dtype)

    def _build_observation_space(self, observation_space: spaces.Space) -> spaces.Space:
        if not self.camera_names or not isinstance(observation_space, spaces.Dict):
            return observation_space

        new_spaces = dict(observation_space.spaces)
        if "rgb" in new_spaces:
            new_spaces["rgb"] = self._slice_space(new_spaces["rgb"], self.rgb_indices)
        if "depth" in new_spaces:
            new_spaces["depth"] = self._slice_space(new_spaces["depth"], self.depth_indices)
        return spaces.Dict(new_spaces)

    def observation(self, observation):
        if not self.camera_names:
            return observation
        if "rgb" in observation and self.rgb_indices is not None:
            observation["rgb"] = observation["rgb"][..., self.rgb_indices]
        if "depth" in observation and self.depth_indices is not None:
            observation["depth"] = observation["depth"][..., self.depth_indices]
        return observation
