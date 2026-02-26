"""
单元测试 — state_predictor.py
覆盖: StatePredictorConfig, StatePredictor forward/predict_sequence

所有测试：
  - 在 CPU 上运行，不依赖 GPU
  - 不加载真实 checkpoint，只测试模型结构和数值
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# 确保项目根目录在 sys.path 中
_ROOT = str(Path(__file__).parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rlft.vlaw.policy.state_predictor import StatePredictor, StatePredictorConfig


# ---------------------------------------------------------------------------
# StatePredictorConfig 测试
# ---------------------------------------------------------------------------


class TestStatePredictorConfig:
    """测试 StatePredictorConfig 默认值和字段类型。"""

    def test_config_defaults(self) -> None:
        """验证 StatePredictorConfig 默认值：state_dim=25, hidden_dim=256 等。"""
        cfg = StatePredictorConfig()
        assert cfg.state_dim == 25, f"state_dim 默认应为 25，实际: {cfg.state_dim}"
        assert cfg.action_dim == 7, f"action_dim 默认应为 7，实际: {cfg.action_dim}"
        assert cfg.hidden_dim == 256, f"hidden_dim 默认应为 256，实际: {cfg.hidden_dim}"
        assert cfg.num_layers == 2, f"num_layers 默认应为 2，实际: {cfg.num_layers}"
        assert cfg.lr == 1e-3, f"lr 默认应为 1e-3，实际: {cfg.lr}"
        assert cfg.max_steps == 5000, f"max_steps 默认应为 5000，实际: {cfg.max_steps}"
        assert cfg.batch_size == 256, f"batch_size 默认应为 256，实际: {cfg.batch_size}"

    def test_config_custom_values(self) -> None:
        """验证 StatePredictorConfig 可以自定义字段。"""
        cfg = StatePredictorConfig(state_dim=29, hidden_dim=512, lr=5e-4)
        assert cfg.state_dim == 29
        assert cfg.hidden_dim == 512
        assert cfg.lr == 5e-4

    def test_config_checkpoint_dir_not_empty(self) -> None:
        """验证默认 checkpoint_dir 非空。"""
        cfg = StatePredictorConfig()
        assert len(cfg.checkpoint_dir) > 0, "checkpoint_dir 不应为空字符串"


# ---------------------------------------------------------------------------
# StatePredictor 结构测试
# ---------------------------------------------------------------------------


class TestStatePredictorInit:
    """测试 StatePredictor 初始化和网络结构。"""

    def test_init_default(self) -> None:
        """验证默认参数下可以正常初始化。"""
        model = StatePredictor()
        assert model.state_dim == 25
        assert model.action_dim == 7
        assert model.net is not None

    def test_init_custom_dims(self) -> None:
        """验证自定义维度可以正常初始化。"""
        model = StatePredictor(state_dim=29, action_dim=6, hidden_dim=128)
        assert model.state_dim == 29
        assert model.action_dim == 6

    def test_different_state_dims_25(self) -> None:
        """state_dim=25 能正常初始化（标准 ManiSkill qpos+qvel）。"""
        model = StatePredictor(state_dim=25)
        assert model.state_dim == 25

    def test_different_state_dims_29(self) -> None:
        """state_dim=29 能正常初始化（扩展状态向量）。"""
        model = StatePredictor(state_dim=29)
        assert model.state_dim == 29

    def test_is_nn_module(self) -> None:
        """StatePredictor 应继承自 torch.nn.Module。"""
        model = StatePredictor()
        assert isinstance(model, torch.nn.Module)


# ---------------------------------------------------------------------------
# StatePredictor.forward 测试
# ---------------------------------------------------------------------------


class TestStatePredictorForward:
    """测试 StatePredictor.forward() 形状和残差特性。"""

    def setup_method(self) -> None:
        """每个测试前创建 CPU 模型。"""
        self.model = StatePredictor(state_dim=25, action_dim=7, hidden_dim=128)
        self.model.eval()

    def test_forward_shape(self) -> None:
        """输入 (B, state_dim) 和 (B, action_dim)，输出 (B, state_dim)。"""
        B = 4
        state = torch.randn(B, 25)
        action = torch.randn(B, 7)
        with torch.no_grad():
            out = self.model(state, action)
        assert out.shape == (B, 25), f"forward 输出 shape 错误: {out.shape}"

    def test_forward_batch_size_1(self) -> None:
        """batch_size=1 的边界情况。"""
        state = torch.randn(1, 25)
        action = torch.randn(1, 7)
        with torch.no_grad():
            out = self.model(state, action)
        assert out.shape == (1, 25), f"batch_size=1 输出 shape 错误: {out.shape}"

    def test_forward_residual(self) -> None:
        """验证残差连接：output 不等于 MLP delta（output = state + delta）。"""
        state = torch.randn(2, 25)
        action = torch.randn(2, 7)
        with torch.no_grad():
            out = self.model(state, action)
        # 在随机初始化下，output ≠ state（有 delta 贡献）
        # 也 ≠ 纯 MLP 输出（有残差），只验证形状和非空
        assert out.shape == (2, 25)
        # 验证残差：state + delta 与 state 不完全相同（极小概率相同）
        assert not torch.allclose(out, state, atol=1e-6), (
            "output 应 = state + delta，不应与 state 完全相同"
        )

    def test_forward_deterministic(self) -> None:
        """相同输入，两次调用应给出完全相同的输出（eval 模式，无 dropout）。"""
        state = torch.randn(3, 25)
        action = torch.randn(3, 7)
        with torch.no_grad():
            out1 = self.model(state, action)
            out2 = self.model(state, action)
        assert torch.allclose(out1, out2), "eval 模式下 forward 应是确定性的"

    def test_forward_dtype_float32(self) -> None:
        """输入 float32，输出也应为 float32。"""
        state = torch.randn(2, 25, dtype=torch.float32)
        action = torch.randn(2, 7, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(state, action)
        assert out.dtype == torch.float32, f"输出 dtype 应为 float32，实际: {out.dtype}"

    def test_forward_large_batch(self) -> None:
        """大 batch（B=256）下应正常运行。"""
        B = 256
        state = torch.randn(B, 25)
        action = torch.randn(B, 7)
        with torch.no_grad():
            out = self.model(state, action)
        assert out.shape == (B, 25), f"大 batch 输出 shape 错误: {out.shape}"


# ---------------------------------------------------------------------------
# StatePredictor.predict_sequence 测试
# ---------------------------------------------------------------------------


class TestStatePredictorPredictSequence:
    """测试 predict_sequence() 序列递推功能。"""

    def setup_method(self) -> None:
        """每个测试前创建 CPU 模型。"""
        self.model = StatePredictor(state_dim=25, action_dim=7, hidden_dim=128)
        self.model.eval()

    def test_predict_sequence_shape(self) -> None:
        """给定 state_0 (25,) + actions (T=10, 7)，输出应为 (T+1, 25)。"""
        T = 10
        state_0 = np.random.randn(25).astype(np.float32)
        actions = np.random.randn(T, 7).astype(np.float32)
        states = self.model.predict_sequence(state_0, actions)
        assert states.shape == (T + 1, 25), (
            f"predict_sequence 输出 shape 错误: 期望 ({T+1}, 25)，实际: {states.shape}"
        )

    def test_predict_sequence_length(self) -> None:
        """序列长度应为 T+1（含初始状态）。"""
        T = 5
        state_0 = np.zeros(25, dtype=np.float32)
        actions = np.ones((T, 7), dtype=np.float32) * 0.01
        states = self.model.predict_sequence(state_0, actions)
        assert states.shape[0] == T + 1, (
            f"序列长度应为 {T+1}，实际: {states.shape[0]}"
        )

    def test_predict_sequence_first_state_matches_input(self) -> None:
        """序列第一帧应等于输入的 state_0。"""
        state_0 = np.random.randn(25).astype(np.float32)
        actions = np.random.randn(3, 7).astype(np.float32)
        states = self.model.predict_sequence(state_0, actions)
        np.testing.assert_array_almost_equal(
            states[0], state_0, decimal=5,
            err_msg="序列第一帧应等于输入初始状态"
        )

    def test_predict_sequence_dtype_numpy(self) -> None:
        """predict_sequence 应返回 numpy 数组，dtype=float32。"""
        state_0 = np.random.randn(25).astype(np.float32)
        actions = np.random.randn(4, 7).astype(np.float32)
        states = self.model.predict_sequence(state_0, actions)
        assert isinstance(states, np.ndarray), "predict_sequence 应返回 numpy 数组"
        assert states.dtype == np.float32, f"dtype 应为 float32，实际: {states.dtype}"

    def test_predict_sequence_t1(self) -> None:
        """T=1 边界情况：只有一个动作，输出形状 (2, 25)。"""
        state_0 = np.random.randn(25).astype(np.float32)
        actions = np.random.randn(1, 7).astype(np.float32)
        states = self.model.predict_sequence(state_0, actions)
        assert states.shape == (2, 25), f"T=1 时输出应为 (2,25)，实际: {states.shape}"

    def test_predict_sequence_state_dim_29(self) -> None:
        """state_dim=29 时 predict_sequence 应正常工作。"""
        model_29 = StatePredictor(state_dim=29, action_dim=7)
        model_29.eval()
        T = 6
        state_0 = np.random.randn(29).astype(np.float32)
        actions = np.random.randn(T, 7).astype(np.float32)
        states = model_29.predict_sequence(state_0, actions)
        assert states.shape == (T + 1, 29), (
            f"state_dim=29 时输出 shape 错误: {states.shape}"
        )
