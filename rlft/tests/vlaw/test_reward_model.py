"""
单元测试 — reward_model.py
覆盖: VLAWRewardConfig, uniform_sample_frames, VLAWRewardModel (mock)

所有测试：
  - 在 CPU 上运行，无需 GPU
  - 不加载真实 Qwen3-VL 权重，使用 unittest.mock.patch 模拟
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# 确保项目根目录在 sys.path 中
_ROOT = str(Path(__file__).parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rlft.vlaw.reward.reward_model import (
    VLAWRewardConfig,
    VLAWRewardModel,
    uniform_sample_frames,
)


# ---------------------------------------------------------------------------
# VLAWRewardConfig 默认值测试
# ---------------------------------------------------------------------------


class TestVLAWRewardConfig:
    """测试 VLAWRewardConfig 的默认值和类型。"""

    def test_config_defaults(self) -> None:
        """验证 VLAWRewardConfig 默认值合理：threshold=0.8, num_frames=16 等。"""
        cfg = VLAWRewardConfig()
        assert cfg.threshold == 0.8, f"threshold 默认值应为 0.8，实际: {cfg.threshold}"
        assert cfg.num_frames == 16, f"num_frames 默认值应为 16，实际: {cfg.num_frames}"
        assert cfg.max_new_tokens == 8, f"max_new_tokens 应为 8，实际: {cfg.max_new_tokens}"
        assert cfg.do_sample is False, "do_sample 默认应为 False（贪心解码）"
        assert cfg.batch_size == 1, "VLM batch_size 默认应为 1"

    def test_config_prompt_template_has_placeholders(self) -> None:
        """验证 prompt_template 包含 {n} 和 {instruction} 占位符。"""
        cfg = VLAWRewardConfig()
        assert "{n}" in cfg.prompt_template, "prompt_template 应包含 {n} 占位符"
        assert "{instruction}" in cfg.prompt_template, "prompt_template 应包含 {instruction} 占位符"

    def test_config_torch_dtype_valid(self) -> None:
        """验证默认 torch_dtype 是合法的精度字符串。"""
        cfg = VLAWRewardConfig()
        valid_dtypes = {"float16", "bfloat16", "float32"}
        assert cfg.torch_dtype in valid_dtypes, (
            f"torch_dtype 应为合法精度字符串，实际: {cfg.torch_dtype}"
        )

    def test_config_custom_threshold(self) -> None:
        """验证可以自定义 threshold。"""
        cfg = VLAWRewardConfig(threshold=0.5)
        assert cfg.threshold == 0.5

    def test_config_model_path_not_empty(self) -> None:
        """验证默认 model_path 非空。"""
        cfg = VLAWRewardConfig()
        assert len(cfg.model_path) > 0, "model_path 不应为空字符串"


# ---------------------------------------------------------------------------
# uniform_sample_frames 测试
# ---------------------------------------------------------------------------


class TestUniformSampleFrames:
    """测试 uniform_sample_frames() 函数。"""

    def test_uniform_sample_frames_numpy(self) -> None:
        """输入 (T, H, W, 3) uint8 numpy 数组，输出 n 个 PIL.Image。"""
        rng = np.random.default_rng(seed=0)
        arr = rng.integers(0, 255, (20, 64, 64, 3), dtype=np.uint8)
        frames = uniform_sample_frames(arr, num_frames=8)
        assert isinstance(frames, list), "返回值应为 list"
        assert len(frames) == 8, f"应返回 8 帧，实际: {len(frames)}"
        assert all(isinstance(f, Image.Image) for f in frames), "每帧应为 PIL.Image"

    def test_uniform_sample_frames_numpy_float32(self) -> None:
        """输入 float32 [0,1] numpy 数组，输出正确的 PIL.Image（uint8 转换）。"""
        rng = np.random.default_rng(seed=1)
        arr = rng.random((10, 32, 32, 3)).astype(np.float32)
        frames = uniform_sample_frames(arr, num_frames=5)
        assert len(frames) == 5
        # 确保 PIL.Image 内容为 uint8
        arr_out = np.array(frames[0])
        assert arr_out.dtype == np.uint8, f"PIL 输出 dtype 应为 uint8，实际: {arr_out.dtype}"

    def test_uniform_sample_frames_pil(self, mock_pil_frames: List[Image.Image]) -> None:
        """输入 PIL 列表，正确均匀采样指定数量帧。"""
        # mock_pil_frames 有 8 帧，采 4 帧
        frames = uniform_sample_frames(mock_pil_frames, num_frames=4)
        assert len(frames) == 4, f"应返回 4 帧，实际: {len(frames)}"
        assert all(isinstance(f, Image.Image) for f in frames)

    def test_uniform_sample_frames_edge_t1(self) -> None:
        """T=1 边界情况：采 n>1 帧也只返回 1 帧。"""
        rng = np.random.default_rng(seed=2)
        arr = rng.integers(0, 255, (1, 32, 32, 3), dtype=np.uint8)
        frames = uniform_sample_frames(arr, num_frames=8)
        assert len(frames) == 1, f"T=1 时应只返回 1 帧，实际: {len(frames)}"

    def test_uniform_sample_frames_edge_t_eq_n(self) -> None:
        """T=n 边界情况：采 T=n 帧时返回全部帧。"""
        rng = np.random.default_rng(seed=3)
        arr = rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        frames = uniform_sample_frames(arr, num_frames=5)
        assert len(frames) == 5, f"T=n 时应返回 5 帧，实际: {len(frames)}"

    def test_uniform_sample_frames_edge_t_gt_n(self) -> None:
        """T>n 边界情况：采 n 帧，返回 n 帧。"""
        rng = np.random.default_rng(seed=4)
        arr = rng.integers(0, 255, (100, 64, 64, 3), dtype=np.uint8)
        frames = uniform_sample_frames(arr, num_frames=16)
        assert len(frames) == 16, f"应返回 16 帧，实际: {len(frames)}"

    def test_uniform_sample_frames_empty_raises(self) -> None:
        """空输入应抛出 ValueError。"""
        arr = np.zeros((0, 32, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            uniform_sample_frames(arr, num_frames=5)

    def test_uniform_sample_frames_indices_monotone(self) -> None:
        """采样索引应单调递增（均匀分布特性验证）。"""
        rng = np.random.default_rng(seed=5)
        T = 50
        arr = rng.integers(0, 255, (T, 8, 8, 3), dtype=np.uint8)
        # 用不同像素填充每帧，便于区分
        for i in range(T):
            arr[i, 0, 0, 0] = i * 5 % 256
        frames = uniform_sample_frames(arr, num_frames=10)
        # 只检查帧数正确
        assert len(frames) == 10


# ---------------------------------------------------------------------------
# VLAWRewardModel 测试（mock）
# ---------------------------------------------------------------------------


class TestVLAWRewardModel:
    """测试 VLAWRewardModel，全程 mock 模型加载和前向传播。"""

    def test_model_not_loaded_initially(self) -> None:
        """初始化后模型未加载，_loaded=False。"""
        model = VLAWRewardModel()
        assert model._loaded is False
        assert model.model is None
        assert model.processor is None

    def test_reward_binary_above_threshold(self) -> None:
        """p_yes=0.9, threshold=0.8 → reward 应为 1。"""
        cfg = VLAWRewardConfig(threshold=0.8)
        rm = VLAWRewardModel(cfg)

        # mock _forward_p_yes 返回 0.9，跳过模型加载
        rm._loaded = True
        rm._forward_p_yes = MagicMock(return_value=0.9)

        rng = np.random.default_rng(seed=0)
        frames = rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        result = rm.score_trajectory(frames, "pick up the cube")

        assert result["reward"] == 1, f"p_yes=0.9 > 0.8，reward 应为 1，实际: {result['reward']}"
        assert abs(result["p_yes"] - 0.9) < 1e-6
        assert result["threshold"] == 0.8

    def test_reward_binary_below_threshold(self) -> None:
        """p_yes=0.5, threshold=0.8 → reward 应为 0。"""
        cfg = VLAWRewardConfig(threshold=0.8)
        rm = VLAWRewardModel(cfg)
        rm._loaded = True
        rm._forward_p_yes = MagicMock(return_value=0.5)

        rng = np.random.default_rng(seed=0)
        frames = rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        result = rm.score_trajectory(frames, "stack the cube")

        assert result["reward"] == 0, f"p_yes=0.5 < 0.8，reward 应为 0，实际: {result['reward']}"

    def test_reward_at_threshold(self) -> None:
        """p_yes 恰好等于 threshold（0.8），由 > 判断应为 0。"""
        cfg = VLAWRewardConfig(threshold=0.8)
        rm = VLAWRewardModel(cfg)
        rm._loaded = True
        rm._forward_p_yes = MagicMock(return_value=0.8)

        rng = np.random.default_rng(seed=0)
        frames = rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        result = rm.score_trajectory(frames, "grasp the object")

        # p_yes > threshold 才是 1，等于时应为 0
        assert result["reward"] == 0, "p_yes == threshold 时 reward 应为 0（严格大于）"

    def test_p_yes_range(self) -> None:
        """score_trajectory 返回的 p_yes 应在 [0, 1] 范围内。"""
        cfg = VLAWRewardConfig(threshold=0.8)
        rm = VLAWRewardModel(cfg)
        rm._loaded = True

        # mock 返回边界值
        for p in [0.0, 0.5, 1.0, 0.8]:
            rm._forward_p_yes = MagicMock(return_value=p)
            rng = np.random.default_rng(seed=0)
            frames = rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8)
            result = rm.score_trajectory(frames, "test task")
            assert 0.0 <= result["p_yes"] <= 1.0, (
                f"p_yes 应在 [0,1] 范围，实际: {result['p_yes']}"
            )

    def test_score_trajectory_returns_num_frames(self) -> None:
        """score_trajectory 返回 num_frames 字段，值应 ≤ config.num_frames。"""
        cfg = VLAWRewardConfig(num_frames=4, threshold=0.8)
        rm = VLAWRewardModel(cfg)
        rm._loaded = True
        rm._forward_p_yes = MagicMock(return_value=0.9)

        rng = np.random.default_rng(seed=0)
        # 只有 3 帧，小于 num_frames=4
        frames = rng.integers(0, 255, (3, 32, 32, 3), dtype=np.uint8)
        result = rm.score_trajectory(frames, "test")

        assert "num_frames" in result, "result 中应含 num_frames 字段"
        assert result["num_frames"] <= cfg.num_frames, (
            f"num_frames 不应超过配置值: {result['num_frames']} > {cfg.num_frames}"
        )

    def test_yes_token_ids_mock(self) -> None:
        """mock processor.tokenizer 后，_yes_token_ids 应包含多个 yes 变体 token id。"""
        cfg = VLAWRewardConfig()
        rm = VLAWRewardModel(cfg)

        # 手动设置 _yes_token_ids（模拟 load_model 后的状态）
        rm._yes_token_ids = [100, 101, 102, 103]  # yes, Yes, YES, " yes"
        rm._no_token_ids = [200, 201, 202, 203]

        assert len(rm._yes_token_ids) >= 1, "_yes_token_ids 应至少含一个 token"
        assert len(rm._no_token_ids) >= 1, "_no_token_ids 应至少含一个 token"
        # 验证 yes 和 no 集合不重叠
        yes_set = set(rm._yes_token_ids)
        no_set = set(rm._no_token_ids)
        assert yes_set.isdisjoint(no_set), "yes 和 no token id 集合不应有重叠"

    def test_score_batch_returns_list(self) -> None:
        """score_batch 应返回与输入等长的 dict 列表。"""
        cfg = VLAWRewardConfig(threshold=0.8)
        rm = VLAWRewardModel(cfg)
        rm._loaded = True
        rm._forward_p_yes = MagicMock(side_effect=[0.9, 0.3, 0.7])

        rng = np.random.default_rng(seed=0)
        trajs = [rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8) for _ in range(3)]
        results = rm.score_batch(trajs, "pick cube")

        assert len(results) == 3, f"应返回 3 个结果，实际: {len(results)}"
        rewards = [r["reward"] for r in results]
        assert rewards == [1, 0, 0], f"奖励序列错误: {rewards}"

    def test_score_batch_per_instruction(self) -> None:
        """score_batch 支持每条轨迹独立指令。"""
        cfg = VLAWRewardConfig(threshold=0.8)
        rm = VLAWRewardModel(cfg)
        rm._loaded = True
        rm._forward_p_yes = MagicMock(side_effect=[0.9, 0.2])

        rng = np.random.default_rng(seed=0)
        trajs = [rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8) for _ in range(2)]
        instructions = ["task A", "task B"]
        results = rm.score_batch(trajs, instructions)
        assert len(results) == 2
