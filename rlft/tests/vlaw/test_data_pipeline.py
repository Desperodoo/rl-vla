"""data_pipeline 模块单元测试.

测试 rlft/vlaw/data_pipeline.py 中的核心函数。
所有测试在 CPU 上运行，不需要 GPU 或真实 VAE。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 确保项目根目录在 sys.path 中
_ROOT = str(Path(__file__).parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rlft.vlaw.data_pipeline import PipelineConfig, concat_cameras


# ---------------------------------------------------------------------------
# concat_cameras 测试
# ---------------------------------------------------------------------------

class TestConcatCameras:
    """测试 concat_cameras() 函数。"""

    def test_concat_cameras_vertical(self, mock_rgb_frames: np.ndarray) -> None:
        """验证垂直拼接后输出 shape=(T, 2H, W, 3)。"""
        T, H, W, C = mock_rgb_frames.shape
        out = concat_cameras(mock_rgb_frames, mock_rgb_frames, mode="vertical")
        assert out.shape == (T, 2 * H, W, C), (
            f"垂直拼接 shape 错误: 期望 {(T, 2*H, W, C)}，实际 {out.shape}"
        )

    def test_concat_cameras_horizontal(self, mock_rgb_frames: np.ndarray) -> None:
        """验证水平拼接后输出 shape=(T, H, 2W, 3)。"""
        T, H, W, C = mock_rgb_frames.shape
        out = concat_cameras(mock_rgb_frames, mock_rgb_frames, mode="horizontal")
        assert out.shape == (T, H, 2 * W, C), (
            f"水平拼接 shape 错误: 期望 {(T, H, 2*W, C)}，实际 {out.shape}"
        )

    def test_concat_cameras_dtype_preserved(self, mock_rgb_frames: np.ndarray) -> None:
        """验证拼接后 dtype 保持为 uint8，不被隐式转换。"""
        out = concat_cameras(mock_rgb_frames, mock_rgb_frames, mode="vertical")
        assert out.dtype == np.uint8, (
            f"dtype 被意外转换: 期望 uint8，实际 {out.dtype}"
        )

    def test_concat_cameras_invalid_mode(self, mock_rgb_frames: np.ndarray) -> None:
        """传入非法 mode 时，应抛出 ValueError。"""
        with pytest.raises(ValueError, match="Unknown concat_mode"):
            concat_cameras(mock_rgb_frames, mock_rgb_frames, mode="diagonal")

    def test_concat_cameras_values_preserved(self) -> None:
        """验证拼接后数值内容正确，上半部分来自 base，下半部分来自 render。"""
        rng = np.random.default_rng(seed=99)
        base = rng.integers(0, 128, (3, 16, 16, 3), dtype=np.uint8)
        render = rng.integers(128, 255, (3, 16, 16, 3), dtype=np.uint8)
        out = concat_cameras(base, render, mode="vertical")
        np.testing.assert_array_equal(out[:, :16, :, :], base, err_msg="上半部分应来自 base")
        np.testing.assert_array_equal(out[:, 16:, :, :], render, err_msg="下半部分应来自 render")

    def test_concat_cameras_single_frame(self) -> None:
        """验证 T=1 的单帧边界情况。"""
        rng = np.random.default_rng(seed=0)
        a = rng.integers(0, 255, (1, 192, 192, 3), dtype=np.uint8)
        out = concat_cameras(a, a, mode="vertical")
        assert out.shape == (1, 384, 192, 3), f"单帧拼接 shape 错误: {out.shape}"


# ---------------------------------------------------------------------------
# PipelineConfig 配置测试
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    """测试 PipelineConfig dataclass 默认值。"""

    def test_pipeline_config_defaults(self) -> None:
        """验证 PipelineConfig 默认值，特别是 vae_local_path 为空字符串。"""
        cfg = PipelineConfig()
        assert cfg.vae_local_path == "", (
            f"vae_local_path 应为空字符串，实际为: '{cfg.vae_local_path}'"
        )
        assert cfg.concat_mode == "vertical", (
            f"默认 concat_mode 应为 'vertical'，实际为: '{cfg.concat_mode}'"
        )
        assert cfg.camera_height == 192, f"默认 camera_height 应为 192，实际: {cfg.camera_height}"
        assert cfg.camera_width == 192, f"默认 camera_width 应为 192，实际: {cfg.camera_width}"
        assert cfg.batch_size == 16, f"默认 batch_size 应为 16，实际: {cfg.batch_size}"

    def test_pipeline_config_vae_local_path_not_hardcoded(self) -> None:
        """vae_local_path 不应包含硬编码的用户目录（如 /home/wjz）。

        这是一个回归测试，确保 VLAW MODIFICATION 修改后不再有硬编码路径。
        """
        cfg = PipelineConfig()
        assert "/home/wjz" not in cfg.vae_local_path, (
            f"vae_local_path 包含硬编码用户路径: '{cfg.vae_local_path}'"
        )
        assert "/root/" not in cfg.vae_local_path, (
            f"vae_local_path 包含硬编码用户路径: '{cfg.vae_local_path}'"
        )

    def test_pipeline_config_vae_model_id_correct(self) -> None:
        """验证默认 vae_model_id 为 stabilityai 官方 ID。"""
        cfg = PipelineConfig()
        assert "stabilityai" in cfg.vae_model_id, (
            f"vae_model_id 不包含 'stabilityai': '{cfg.vae_model_id}'"
        )

    def test_pipeline_config_latent_shape_math(self) -> None:
        """验证默认分辨率配置下 latent 形状数学一致性。

        concat_mode=vertical:
            tgt_h = camera_height * 2 = 384
            tgt_w = camera_width = 192
            lat_h = tgt_h / 8 = 48
            lat_w = tgt_w / 8 = 24
        """
        cfg = PipelineConfig()
        assert cfg.concat_mode == "vertical"
        tgt_h = cfg.camera_height * 2
        tgt_w = cfg.camera_width
        assert tgt_h % 8 == 0, f"tgt_h={tgt_h} 不能被 8 整除"
        assert tgt_w % 8 == 0, f"tgt_w={tgt_w} 不能被 8 整除"
        lat_h = tgt_h // 8
        lat_w = tgt_w // 8
        assert lat_h == 48, f"lat_h 期望 48，实际 {lat_h}"
        assert lat_w == 24, f"lat_w 期望 24，实际 {lat_w}"
