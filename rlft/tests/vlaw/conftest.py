"""pytest 公共 fixtures for VLAW tests.

所有 fixture 使用随机/mock 数据，不依赖真实模型权重或 GPU。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# RGB 帧 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rgb_frames(tmp_path: Path) -> np.ndarray:
    """提供 10 帧 192×192 RGB 随机图像，uint8，shape=(10, 192, 192, 3)。"""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, (10, 192, 192, 3), dtype=np.uint8)


@pytest.fixture
def mock_pil_frames() -> List[Image.Image]:
    """返回 8 个 PIL.Image 对象的列表，尺寸 192×192 RGB。"""
    rng = np.random.default_rng(seed=7)
    frames = []
    for _ in range(8):
        arr = rng.integers(0, 255, (192, 192, 3), dtype=np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


# ---------------------------------------------------------------------------
# Latent fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_latent_single_cam() -> torch.Tensor:
    """单相机 latent，shape=(10, 4, 24, 24) float16。

    对应 192×192 → 192/8=24 的 VAE 下采样结果。
    """
    return torch.randn(10, 4, 24, 24, dtype=torch.float16)


@pytest.fixture
def mock_latent_concat() -> torch.Tensor:
    """双相机垂直拼接后 latent，shape=(10, 4, 48, 24) float16.

    对应 384×192（垂直拼接）→ (4, 384/8=48, 192/8=24) 的 VAE 下采样结果。
    """
    return torch.randn(10, 4, 48, 24, dtype=torch.float16)


# ---------------------------------------------------------------------------
# Action fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_actions() -> np.ndarray:
    """7-DOF delta pose 动作序列，shape=(10, 7) float32，值域 [-0.05, 0.05]。"""
    rng = np.random.default_rng(seed=123)
    return rng.uniform(-0.05, 0.05, (10, 7)).astype(np.float32)


# ---------------------------------------------------------------------------
# HDF5 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_hdf5_file(tmp_path: Path) -> Path:
    """创建一个含 3 条轨迹的 HDF5 临时文件，返回文件路径。

    每条轨迹包含字段：
      - rgb_base: (T, 192, 192, 3) uint8
      - rgb_render: (T, 192, 192, 3) uint8
      - actions: (T, 7) float32
      - env_success: (T,) bool
      - latent_concat: (T, 4, 48, 24) float16
    其中 T=5（3 帧有意义数据 + 2 帧边界）。
    """
    h5_path = tmp_path / "mock_rollouts.h5"
    rng = np.random.default_rng(seed=0)
    T = 5

    with h5py.File(str(h5_path), "w") as f:
        for i in range(3):
            grp = f.create_group(f"traj_{i}")
            grp.create_dataset(
                "rgb_base",
                data=rng.integers(0, 255, (T, 192, 192, 3), dtype=np.uint8),
            )
            grp.create_dataset(
                "rgb_render",
                data=rng.integers(0, 255, (T, 192, 192, 3), dtype=np.uint8),
            )
            grp.create_dataset(
                "actions",
                data=rng.uniform(-0.05, 0.05, (T, 7)).astype(np.float32),
            )
            success = np.zeros(T, dtype=bool)
            success[-1] = True  # 最后一帧成功
            grp.create_dataset("env_success", data=success)
            grp.create_dataset(
                "latent_concat",
                data=rng.standard_normal((T, 4, 48, 24)).astype(np.float16),
            )

    return h5_path


# ---------------------------------------------------------------------------
# mock_hdf5_rollout fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_hdf5_rollout(tmp_path: Path) -> Path:
    """创建 mock HDF5 rollout 文件，含 3 条轨迹，每条 T=10 帧。

    每条轨迹字段：
      - rgb_base:    (10, 192, 192, 3) uint8
      - rgb_render:  (10, 192, 192, 3) uint8
      - state:       (10, 25) float32
      - actions:     (10, 7) float32
      - env_success: (10,) bool

    第 0 条：vlm_reward=1,  success attr=True
    第 1 条：vlm_reward=0,  success attr=False
    第 2 条：vlm_reward=1,  success attr=True
    """
    h5_path = tmp_path / "mock_rollout.h5"
    rng = np.random.default_rng(seed=42)
    T = 10

    with h5py.File(str(h5_path), "w") as f:
        # meta info
        meta = f.create_group("meta")
        meta.attrs["num_trajectories"] = 3
        meta.attrs["env_id"] = "PickCube-v1"

        vlm_rewards = [1, 0, 1]
        for i in range(3):
            grp = f.create_group(f"traj_{i:04d}")
            grp.create_dataset(
                "rgb_base",
                data=rng.integers(0, 255, (T, 192, 192, 3), dtype=np.uint8),
            )
            grp.create_dataset(
                "rgb_render",
                data=rng.integers(0, 255, (T, 192, 192, 3), dtype=np.uint8),
            )
            grp.create_dataset(
                "state",
                data=rng.standard_normal((T, 25)).astype(np.float32),
            )
            grp.create_dataset(
                "actions",
                data=rng.uniform(-0.05, 0.05, (T, 7)).astype(np.float32),
            )
            env_success = np.zeros(T, dtype=bool)
            env_success[-1] = bool(vlm_rewards[i])
            grp.create_dataset("env_success", data=env_success)
            grp.attrs["vlm_reward"] = vlm_rewards[i]
            grp.attrs["success"] = bool(vlm_rewards[i])

    return h5_path


# ---------------------------------------------------------------------------
# stat.json fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_stat_json(tmp_path: Path) -> Path:
    """创建 action_stats.json，包含 p01/p99 字段，返回文件路径。"""
    stat = {
        "p01": [-0.05] * 7,
        "p99": [0.05] * 7,
        "mean": [0.0] * 7,
        "std": [0.02] * 7,
        "action_dim": 7,
    }
    stat_path = tmp_path / "action_stats.json"
    stat_path.write_text(json.dumps(stat, indent=2))
    return stat_path
