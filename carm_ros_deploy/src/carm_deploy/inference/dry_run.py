#!/usr/bin/env python3
"""
DryRunEnvironment — replay HDF5 data or synthetic observations for offline testing.

Provides the same interface as core.env_ros.RealEnvironment without any hardware
dependencies (no CARM SDK, no ROS, no cameras).

Usage:
    # From HDF5 recording
    env = DryRunEnvironment.from_hdf5("data/test_gap_fix/episode_0001.hdf5")

    # Synthetic observations
    env = DryRunEnvironment.synthetic(num_frames=100)

    # Plug into InferenceNode (replacing RealEnvironment)
    with mock.patch("inference.inference_node.RealEnvironment", return_value=env):
        node = InferenceNode(config)
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Any

from utils.log_compat import log_info, log_warn


class DryRunEnvironment:
    """Drop-in replacement for RealEnvironment that replays recorded data
    or generates synthetic observations.

    Interface contract (matches RealEnvironment):
        - get_observation() -> dict | None
        - end_control_nostep(action)
        - init_status()
        - shutdown()
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        images: np.ndarray,
        qpos_joint: np.ndarray,
        qpos_end: np.ndarray,
        loop: bool = True,
    ):
        """
        Args:
            timestamps: [T] float64 — epoch seconds
            images: [T, H, W, 3] uint8 — RGB images
            qpos_joint: [T, 7] float64 — joint angles + gripper
            qpos_end: [T, 8] float64 — EE pose + gripper
            loop: if True, wrap around when exhausting frames
        """
        assert len(timestamps) == len(images) == len(qpos_joint) == len(qpos_end)
        self._timestamps = timestamps
        self._images = images
        self._qpos_joint = qpos_joint
        self._qpos_end = qpos_end
        self._loop = loop
        self._num_frames = len(timestamps)

        self._step = 0
        self._exhausted = False
        self._actions_sent: List[np.ndarray] = []

        log_info(
            f"DryRunEnvironment: {self._num_frames} frames, "
            f"img={images.shape[1:3]}, loop={loop}"
        )

    # ---- Factory methods ----

    @classmethod
    def from_hdf5(cls, path: str, loop: bool = True) -> 'DryRunEnvironment':
        """Load from a recorded HDF5 episode (v1 or v2 format)."""
        import h5py

        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        with h5py.File(path, 'r') as f:
            obs = f['observations']
            timestamps = obs['timestamps'][:]
            images = obs['images'][:]
            qpos_joint = obs['qpos_joint'][:]
            qpos_end = obs['qpos_end'][:]

        log_info(f"Loaded HDF5: {path} ({len(timestamps)} steps)")
        return cls(timestamps, images, qpos_joint, qpos_end, loop=loop)

    @classmethod
    def synthetic(
        cls,
        num_frames: int = 100,
        image_size: tuple = (240, 320),
        seed: int = 42,
        loop: bool = True,
    ) -> 'DryRunEnvironment':
        """Generate synthetic observations for unit testing.

        Produces a gentle sinusoidal arm motion around a realistic rest pose.
        """
        rng = np.random.default_rng(seed)

        t0 = time.time()
        timestamps = np.array([t0 + i / 30.0 for i in range(num_frames)])

        h, w = image_size
        images = rng.integers(0, 256, (num_frames, h, w, 3), dtype=np.uint8)

        # Realistic rest pose (from real robot)
        base_joint = np.array([-0.05, 1.577, -0.989, 0.024, 0.893, 0.006, 0.073])
        base_ee = np.array([0.257, -0.011, 0.331, 0.998, -0.035, 0.045, -0.009, 0.073])

        # Small sinusoidal perturbations
        t = np.linspace(0, 2 * np.pi, num_frames)
        qpos_joint = np.tile(base_joint, (num_frames, 1))
        qpos_joint[:, 0] += 0.01 * np.sin(t)
        qpos_joint[:, 1] += 0.005 * np.sin(t * 2)

        qpos_end = np.tile(base_ee, (num_frames, 1))
        qpos_end[:, 0] += 0.005 * np.sin(t)  # X oscillation
        qpos_end[:, 1] += 0.003 * np.cos(t)  # Y oscillation
        qpos_end[:, 2] += 0.002 * np.sin(t * 0.5)  # Z drift

        log_info(f"Synthetic DryRunEnv: {num_frames} frames, img={image_size}")
        return cls(timestamps, images, qpos_joint, qpos_end, loop=loop)

    # ---- RealEnvironment interface ----

    def get_observation(self) -> Optional[Dict[str, Any]]:
        """Return the next observation frame."""
        if self._exhausted:
            return None

        i = self._step
        obs = {
            "stamp": float(self._timestamps[i]),
            "images": [self._images[i]],
            "qpos_joint": self._qpos_joint[i].tolist(),
            "qpos_end": self._qpos_end[i].tolist(),
            "gripper": float(self._qpos_joint[i, -1]),
            "qpos": np.concatenate([self._qpos_joint[i], self._qpos_end[i]]),
        }

        self._step += 1
        if self._step >= self._num_frames:
            if self._loop:
                self._step = 0
            else:
                self._exhausted = True

        return obs

    def end_control_nostep(self, action) -> None:
        """Record the action without executing on hardware."""
        self._actions_sent.append(np.asarray(action, dtype=np.float64).copy())

    def init_status(self) -> None:
        """No-op for dry run."""
        log_info("DryRunEnvironment: init_status (no-op)")

    def shutdown(self) -> None:
        """No-op for dry run."""
        log_info(
            f"DryRunEnvironment: shutdown ({len(self._actions_sent)} actions recorded)"
        )

    # ---- Dry-run specific accessors ----

    @property
    def actions_sent(self) -> List[np.ndarray]:
        """All actions that were 'sent' via end_control_nostep."""
        return self._actions_sent

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def current_step(self) -> int:
        return self._step

    def reset(self) -> None:
        """Reset playback to the beginning."""
        self._step = 0
        self._exhausted = False
        self._actions_sent.clear()
