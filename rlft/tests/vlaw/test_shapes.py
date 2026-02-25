"""集成形状一致性测试.

验证 VLAW 各模块间数据形状的数学正确性。
所有测试仅使用 CPU + 随机数据，不加载真实模型。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest
import torch

# 确保项目根目录在 sys.path 中
_ROOT = str(Path(__file__).parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Latent 形状一致性测试
# ---------------------------------------------------------------------------

class TestLatentShapes:
    """验证端到端 latent 形状的数学正确性。"""

    def test_latent_shape_after_concat(self, mock_latent_concat: torch.Tensor) -> None:
        """验证 2 相机垂直拼接后 latent shape=(T, 4, 48, 24)。

        数学推导：
            单相机分辨率: 192×192
            垂直拼接后: 384×192
            VAE 8× 下采样: 384/8=48, 192/8=24
            latent: (T, 4, 48, 24)
        """
        T = 10
        assert mock_latent_concat.shape == (T, 4, 48, 24), (
            f"latent_concat shape 错误: 期望 (10, 4, 48, 24)，实际 {mock_latent_concat.shape}"
        )
        assert mock_latent_concat.dtype == torch.float16, (
            f"latent dtype 应为 float16，实际 {mock_latent_concat.dtype}"
        )

    def test_latent_shape_math_consistency(self) -> None:
        """用纯数学验证 192×192 双相机垂直拼接后的 latent 形状。

        不依赖任何模块导入，纯计算验证。
        """
        H, W = 192, 192  # 单相机分辨率
        n_cams = 2
        vae_stride = 8  # VAE 下采样倍率
        latent_channels = 4  # VAE latent 通道数

        # 垂直拼接
        concat_h = H * n_cams  # 384
        concat_w = W            # 192

        assert concat_h % vae_stride == 0, f"{concat_h} 不能被 {vae_stride} 整除"
        assert concat_w % vae_stride == 0, f"{concat_w} 不能被 {vae_stride} 整除"

        lat_h = concat_h // vae_stride  # 48
        lat_w = concat_w // vae_stride  # 24

        assert lat_h == 48, f"lat_h 期望 48，得到 {lat_h}"
        assert lat_w == 24, f"lat_w 期望 24，得到 {lat_w}"

    def test_rearrange_n_cams_2(self, mock_latent_concat: torch.Tensor) -> None:
        """验证 m=2 时拆分 latent_concat 的 rearrange 操作正确性。

        (T, 4, 48, 24) rearrange n_cams=2 → (2, T, 4, 24, 24)
        上半 [0:24] 对应相机 0，下半 [24:48] 对应相机 1。
        """
        T, C, H_concat, W = mock_latent_concat.shape  # (10, 4, 48, 24)
        n_cams = 2
        lat_h_per_cam = H_concat // n_cams  # 24

        # rearrange: 沿 height 维度拆分
        lat_split = mock_latent_concat.reshape(T, C, n_cams, lat_h_per_cam, W)
        lat_split = lat_split.permute(2, 0, 1, 3, 4)  # (2, T, 4, 24, 24)

        assert lat_split.shape == (n_cams, T, C, lat_h_per_cam, W), (
            f"rearrange 结果 shape 错误: 期望 {(n_cams, T, C, lat_h_per_cam, W)}，实际 {lat_split.shape}"
        )
        # 验证数值不变（重排后数据应与原始一致）
        assert torch.allclose(
            lat_split[0], mock_latent_concat[:, :, :lat_h_per_cam, :]
        ), "相机 0 的数值在 rearrange 后发生变化"
        assert torch.allclose(
            lat_split[1], mock_latent_concat[:, :, lat_h_per_cam:, :]
        ), "相机 1 的数值在 rearrange 后发生变化"

    def test_rearrange_batch_dim(self) -> None:
        """验证 batch=1 维度下的 rearrange，形状为 (1, T, 4, 48, 24) → (2, T, 4, 24, 24)。"""
        T, C, H_concat, W = 5, 4, 48, 24
        n_cams = 2
        lat = torch.randn(1, T, C, H_concat, W)  # batch=1

        # 先去掉 batch 维度
        lat_squeezed = lat.squeeze(0)  # (T, 4, 48, 24)
        lat_split = lat_squeezed.reshape(T, C, n_cams, H_concat // n_cams, W)
        lat_split = lat_split.permute(2, 0, 1, 3, 4)  # (2, T, 4, 24, 24)

        assert lat_split.shape == (n_cams, T, C, H_concat // n_cams, W), (
            f"batch=1 rearrange shape 错误: {lat_split.shape}"
        )


# ---------------------------------------------------------------------------
# Action 归一化测试
# ---------------------------------------------------------------------------

class TestActionNormalization:
    """测试动作归一化/反归一化的数学正确性。"""

    def test_action_normalize_range(self, mock_actions: np.ndarray) -> None:
        """给定 p01/p99，归一化后值域应在 [-1, 1]。"""
        p01 = np.array([-0.05] * 7, dtype=np.float32)
        p99 = np.array([0.05] * 7, dtype=np.float32)
        eps = 1e-8

        # 与 ctrl_world_adapter._normalize_action 相同的公式
        normalized = 2.0 * (mock_actions - p01) / (p99 - p01 + eps) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)

        assert normalized.min() >= -1.0 - 1e-6, f"归一化后最小值超出 -1: {normalized.min()}"
        assert normalized.max() <= 1.0 + 1e-6, f"归一化后最大值超出 1: {normalized.max()}"
        assert normalized.dtype == np.float32, f"归一化后 dtype 应为 float32，实际 {normalized.dtype}"

    def test_action_denormalize_roundtrip(self, mock_actions: np.ndarray) -> None:
        """归一化后反归一化，误差 < 1e-5（忽略 clip 边缘情况）。"""
        p01 = np.array([-0.05] * 7, dtype=np.float32)
        p99 = np.array([0.05] * 7, dtype=np.float32)
        eps = 1e-8

        # 归一化（不 clip，以确保可逆）
        normalized = 2.0 * (mock_actions - p01) / (p99 - p01 + eps) - 1.0

        # 反归一化（与 ctrl_world_adapter.denormalize_action 相同公式）
        reconstructed = (normalized + 1.0) / 2.0 * (p99 - p01) + p01

        # 检查精度（float32 精度有限，放宽到 1e-5）
        max_err = np.abs(mock_actions - reconstructed).max()
        assert max_err < 1e-5, (
            f"归一化→反归一化后最大误差 {max_err:.2e} 超过 1e-5"
        )

    def test_action_normalize_clipping_boundary(self) -> None:
        """超出 p01-p99 范围的动作值应被 clip 到 [-1, 1]。"""
        p01 = np.array([-0.05] * 7, dtype=np.float32)
        p99 = np.array([0.05] * 7, dtype=np.float32)
        eps = 1e-8

        # 超出范围的动作
        extreme = np.array([[0.1, -0.1, 0.2, -0.2, 0.5, -0.5, 1.0]], dtype=np.float32)
        normalized = 2.0 * (extreme - p01) / (p99 - p01 + eps) - 1.0
        clipped = np.clip(normalized, -1.0, 1.0)

        assert clipped.max() == 1.0, "超出上界的值应被 clip 到 1.0"
        assert clipped.min() == -1.0, "超出下界的值应被 clip 到 -1.0"


# ---------------------------------------------------------------------------
# uniform_sample_frames 测试
# ---------------------------------------------------------------------------

class TestUniformSampleFrames:
    """测试 reward_model.uniform_sample_frames 函数。"""

    def test_uniform_sample_frames_count(self) -> None:
        """采样结果长度应等于 min(num_frames, total_frames)。"""
        from rlft.vlaw.reward_model import uniform_sample_frames

        arr = np.random.randint(0, 255, (20, 192, 192, 3), dtype=np.uint8)
        frames = uniform_sample_frames(arr, num_frames=8)
        assert len(frames) == 8, f"期望 8 帧，实际 {len(frames)} 帧"

    def test_uniform_sample_frames_count_less_than_total(self) -> None:
        """num_frames > total 时，应返回 total 帧（不重复采样）。"""
        from rlft.vlaw.reward_model import uniform_sample_frames

        arr = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        frames = uniform_sample_frames(arr, num_frames=20)
        assert len(frames) == 5, f"超出范围时期望 5 帧，实际 {len(frames)} 帧"

    def test_uniform_sample_frames_numpy_input(self) -> None:
        """numpy 数组输入也应返回 PIL.Image 列表。"""
        from PIL import Image
        from rlft.vlaw.reward_model import uniform_sample_frames

        arr = np.random.randint(0, 255, (10, 192, 192, 3), dtype=np.uint8)
        frames = uniform_sample_frames(arr, num_frames=4)

        assert isinstance(frames, list), "返回类型应为 list"
        assert all(isinstance(f, Image.Image) for f in frames), (
            "列表中每个元素应为 PIL.Image"
        )

    def test_uniform_sample_frames_pil_input(self, mock_pil_frames: list) -> None:
        """PIL 列表输入也应正常工作，返回 PIL.Image 列表。"""
        from PIL import Image
        from rlft.vlaw.reward_model import uniform_sample_frames

        frames = uniform_sample_frames(mock_pil_frames, num_frames=4)
        assert len(frames) == 4, f"期望 4 帧，实际 {len(frames)} 帧"
        assert all(isinstance(f, Image.Image) for f in frames), (
            "PIL 输入时返回结果也应为 PIL.Image 列表"
        )

    def test_uniform_sample_frames_empty_input_raises(self) -> None:
        """空输入应抛出 ValueError。"""
        from rlft.vlaw.reward_model import uniform_sample_frames

        arr = np.zeros((0, 192, 192, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="不能为空"):
            uniform_sample_frames(arr, num_frames=8)


# ---------------------------------------------------------------------------
# HDF5 结构测试
# ---------------------------------------------------------------------------

class TestHDF5Structure:
    """基于 mock HDF5 文件验证数据结构。"""

    def test_mock_hdf5_traj_count(self, mock_hdf5_file: Path) -> None:
        """mock HDF5 文件应包含 3 条轨迹。"""
        with h5py.File(str(mock_hdf5_file), "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj_")]
            assert len(traj_keys) == 3, f"期望 3 条轨迹，实际 {len(traj_keys)} 条"

    def test_mock_hdf5_required_fields(self, mock_hdf5_file: Path) -> None:
        """每条轨迹应包含所有必要字段。"""
        required_fields = ["rgb_base", "rgb_render", "actions", "env_success", "latent_concat"]
        with h5py.File(str(mock_hdf5_file), "r") as f:
            for tkey in f.keys():
                for field in required_fields:
                    assert field in f[tkey], (
                        f"轨迹 {tkey} 缺少字段 '{field}'"
                    )

    def test_mock_hdf5_latent_shape(self, mock_hdf5_file: Path) -> None:
        """latent_concat 的形状应为 (T, 4, 48, 24)，dtype=float16。"""
        with h5py.File(str(mock_hdf5_file), "r") as f:
            grp = f["traj_0"]
            lat = grp["latent_concat"][:]
            assert lat.shape[1:] == (4, 48, 24), (
                f"latent_concat shape 错误: 期望 (..., 4, 48, 24)，实际 {lat.shape}"
            )
            assert lat.dtype == np.float16, (
                f"latent dtype 应为 float16，实际 {lat.dtype}"
            )

    def test_mock_hdf5_rgb_shape(self, mock_hdf5_file: Path) -> None:
        """rgb_base / rgb_render 形状应为 (T, 192, 192, 3)，dtype=uint8。"""
        with h5py.File(str(mock_hdf5_file), "r") as f:
            grp = f["traj_0"]
            rgb = grp["rgb_base"][:]
            assert rgb.shape[1:] == (192, 192, 3), (
                f"rgb_base shape 错误: {rgb.shape}"
            )
            assert rgb.dtype == np.uint8, (
                f"rgb_base dtype 应为 uint8，实际 {rgb.dtype}"
            )
