"""
单元测试 — policy_updater.py
覆盖: PolicyUpdaterConfig, VLAWSuccessDataset

所有测试：
  - 在 CPU 上运行，不依赖 GPU
  - 使用 tmp_path fixture 创建临时 HDF5 文件
  - 不加载真实 checkpoint
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

# 确保项目根目录在 sys.path 中
_ROOT = str(Path(__file__).parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rlft.vlaw.policy.policy_updater import PolicyUpdaterConfig, VLAWSuccessDataset


# ---------------------------------------------------------------------------
# PolicyUpdaterConfig 测试
# ---------------------------------------------------------------------------


class TestPolicyUpdaterConfig:
    """测试 PolicyUpdaterConfig 默认值和字段类型。"""

    def test_config_defaults(self) -> None:
        """验证 PolicyUpdaterConfig 默认 num_steps, batch_size, learning_rate 等。"""
        cfg = PolicyUpdaterConfig()
        assert cfg.num_steps == 2000, f"num_steps 默认应为 2000，实际: {cfg.num_steps}"
        assert cfg.batch_size == 64, f"batch_size 默认应为 64，实际: {cfg.batch_size}"
        assert cfg.learning_rate == 1e-5, (
            f"learning_rate 默认应为 1e-5，实际: {cfg.learning_rate}"
        )
        assert cfg.warmup_steps == 100, f"warmup_steps 默认应为 100，实际: {cfg.warmup_steps}"

    def test_config_obs_action_horizon(self) -> None:
        """验证观测/动作窗口默认值。"""
        cfg = PolicyUpdaterConfig()
        assert cfg.obs_horizon == 2, f"obs_horizon 默认应为 2，实际: {cfg.obs_horizon}"
        assert cfg.action_horizon == 8, (
            f"action_horizon 默认应为 8，实际: {cfg.action_horizon}"
        )

    def test_config_data_mix_ratio_valid(self) -> None:
        """data_mix_ratio 应在 [0, 1] 范围内。"""
        cfg = PolicyUpdaterConfig()
        assert 0.0 <= cfg.data_mix_ratio <= 1.0, (
            f"data_mix_ratio 应在 [0,1]，实际: {cfg.data_mix_ratio}"
        )

    def test_config_dry_run_default_false(self) -> None:
        """dry_run 默认应为 False。"""
        cfg = PolicyUpdaterConfig()
        assert cfg.dry_run is False

    def test_config_checkpoint_path_is_string(self) -> None:
        """checkpoint_path 类型应为字符串。"""
        cfg = PolicyUpdaterConfig()
        assert isinstance(cfg.checkpoint_path, str)

    def test_config_custom_values(self) -> None:
        """验证 PolicyUpdaterConfig 可以自定义字段。"""
        cfg = PolicyUpdaterConfig(
            num_steps=500,
            batch_size=32,
            learning_rate=1e-4,
            dry_run=True,
        )
        assert cfg.num_steps == 500
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 1e-4
        assert cfg.dry_run is True

    def test_config_gpu_id(self) -> None:
        """gpu_id 类型应为 int。"""
        cfg = PolicyUpdaterConfig()
        assert isinstance(cfg.gpu_id, int)


# ---------------------------------------------------------------------------
# VLAWSuccessDataset 辅助函数
# ---------------------------------------------------------------------------


def _make_hdf5(
    tmp_path: Path,
    n_trajs: int = 3,
    T: int = 20,
    state_dim: int = 25,
    action_dim: int = 7,
    vlm_reward_values: list | None = None,
    success_values: list | None = None,
    use_env_success: bool = False,
    filename: str = "rollouts.h5",
) -> Path:
    """创建 mock HDF5 文件的辅助函数。

    Args:
        vlm_reward_values: 每条轨迹的 vlm_reward attr 值（None 表示不设置）
        success_values:    每条轨迹的 success attr 值（None 表示不设置）
        use_env_success:   是否使用 env_success dataset 代替 attr
    """
    h5_path = tmp_path / filename
    rng = np.random.default_rng(seed=42)

    with h5py.File(str(h5_path), "w") as f:
        for i in range(n_trajs):
            grp = f.create_group(f"traj_{i:04d}")
            grp.create_dataset(
                "state",
                data=rng.standard_normal((T, state_dim)).astype(np.float32),
            )
            grp.create_dataset(
                "actions",
                data=rng.uniform(-0.05, 0.05, (T, action_dim)).astype(np.float32),
            )
            if vlm_reward_values is not None:
                grp.attrs["vlm_reward"] = vlm_reward_values[i]
            elif success_values is not None:
                grp.attrs["success"] = success_values[i]
            elif use_env_success:
                env_s = np.zeros(T, dtype=bool)
                env_s[-1] = (i % 2 == 0)  # 偶数轨迹成功
                grp.create_dataset("env_success", data=env_s)
    return h5_path


# ---------------------------------------------------------------------------
# VLAWSuccessDataset 测试
# ---------------------------------------------------------------------------


class TestVLAWSuccessDataset:
    """测试 VLAWSuccessDataset HDF5 加载和样本生成。"""

    def test_dataset_from_hdf5_vlm_reward(self, tmp_path: Path) -> None:
        """含 vlm_reward=1 attr 的轨迹应被正确识别为成功轨迹并加载。"""
        # 3 条轨迹：2 条 vlm_reward=1，1 条 vlm_reward=0
        h5 = _make_hdf5(
            tmp_path,
            n_trajs=3,
            T=20,
            vlm_reward_values=[1, 0, 1],
        )
        ds = VLAWSuccessDataset(
            str(h5),
            obs_horizon=2,
            action_horizon=8,
            filter_by_vlm=True,
        )
        # 2 条成功轨迹，T=20, min_len=2+8=10, 每条产生 20-10+1=11 个样本
        assert len(ds) == 22, f"期望 22 个样本，实际: {len(ds)}"

    def test_dataset_empty_no_success(self, tmp_path: Path) -> None:
        """没有成功轨迹时，数据集长度应为 0（不崩溃）。"""
        h5 = _make_hdf5(
            tmp_path,
            n_trajs=3,
            T=20,
            vlm_reward_values=[0, 0, 0],
        )
        ds = VLAWSuccessDataset(
            str(h5),
            obs_horizon=2,
            action_horizon=8,
            filter_by_vlm=True,
        )
        assert len(ds) == 0, f"全部失败时 dataset 长度应为 0，实际: {len(ds)}"

    def test_dataset_nonexistent_file(self, tmp_path: Path) -> None:
        """HDF5 文件不存在时，数据集应返回空（不崩溃）。"""
        ds = VLAWSuccessDataset(
            str(tmp_path / "nonexistent.h5"),
            obs_horizon=2,
            action_horizon=8,
        )
        assert len(ds) == 0, "文件不存在时 dataset 长度应为 0"

    def test_dataset_item_shape(self, tmp_path: Path) -> None:
        """__getitem__ 返回的 obs/action 形状应正确。"""
        obs_horizon = 2
        action_horizon = 8
        state_dim = 25
        action_dim = 7

        h5 = _make_hdf5(
            tmp_path,
            n_trajs=1,
            T=30,
            state_dim=state_dim,
            action_dim=action_dim,
            vlm_reward_values=[1],
        )
        ds = VLAWSuccessDataset(
            str(h5),
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
        )
        assert len(ds) > 0, "数据集应非空"
        sample = ds[0]

        assert "obs" in sample, "样本应含 obs 字段"
        assert "actions" in sample, "样本应含 actions 字段"
        assert "weight" in sample, "样本应含 weight 字段"
        assert "source" in sample, "样本应含 source 字段"

        obs = sample["obs"]
        actions = sample["actions"]

        assert isinstance(obs, torch.Tensor), "obs 应为 torch.Tensor"
        assert isinstance(actions, torch.Tensor), "actions 应为 torch.Tensor"
        assert obs.shape == (obs_horizon, state_dim), (
            f"obs shape 错误: 期望 ({obs_horizon}, {state_dim})，实际: {obs.shape}"
        )
        assert actions.shape == (action_horizon, action_dim), (
            f"actions shape 错误: 期望 ({action_horizon}, {action_dim})，实际: {actions.shape}"
        )

    def test_dataset_item_weight(self, tmp_path: Path) -> None:
        """默认 weight=1.0，样本的 weight 字段应为 1.0。"""
        h5 = _make_hdf5(tmp_path, n_trajs=1, T=20, vlm_reward_values=[1])
        ds = VLAWSuccessDataset(str(h5), obs_horizon=2, action_horizon=8, weight=1.0)
        sample = ds[0]
        assert abs(float(sample["weight"]) - 1.0) < 1e-6, (
            f"weight 应为 1.0，实际: {sample['weight']}"
        )

    def test_dataset_source_tag(self, tmp_path: Path) -> None:
        """source_tag 应正确传递到样本的 source 字段。"""
        h5 = _make_hdf5(tmp_path, n_trajs=1, T=20, vlm_reward_values=[1])

        ds_real = VLAWSuccessDataset(
            str(h5), obs_horizon=2, action_horizon=8, source_tag="real"
        )
        ds_syn = VLAWSuccessDataset(
            str(h5), obs_horizon=2, action_horizon=8, source_tag="synthetic"
        )

        assert ds_real[0]["source"] == "real"
        assert ds_syn[0]["source"] == "synthetic"

    def test_dataset_state_dim_property(self, tmp_path: Path) -> None:
        """state_dim 属性应返回正确值。"""
        h5 = _make_hdf5(tmp_path, n_trajs=1, T=20, state_dim=25, vlm_reward_values=[1])
        ds = VLAWSuccessDataset(str(h5), obs_horizon=2, action_horizon=8)
        assert ds.state_dim == 25, f"state_dim 属性应为 25，实际: {ds.state_dim}"

    def test_dataset_action_dim_property(self, tmp_path: Path) -> None:
        """action_dim 属性应返回正确值。"""
        h5 = _make_hdf5(
            tmp_path, n_trajs=1, T=20, action_dim=7, vlm_reward_values=[1]
        )
        ds = VLAWSuccessDataset(str(h5), obs_horizon=2, action_horizon=8)
        assert ds.action_dim == 7, f"action_dim 属性应为 7，实际: {ds.action_dim}"

    def test_dataset_fallback_env_success(self, tmp_path: Path) -> None:
        """filter_by_vlm=False 时，应按 env_success 筛选轨迹。"""
        # 用 env_success dataset（偶数轨迹成功）
        h5 = _make_hdf5(
            tmp_path,
            n_trajs=4,
            T=20,
            use_env_success=True,
        )
        ds = VLAWSuccessDataset(
            str(h5),
            obs_horizon=2,
            action_horizon=8,
            filter_by_vlm=False,
        )
        # 4 条中 2 条成功（idx 0, 2），每条 11 样本
        assert len(ds) == 22, f"按 env_success 过滤后期望 22 样本，实际: {len(ds)}"

    def test_dataset_traj_too_short_skipped(self, tmp_path: Path) -> None:
        """轨迹长度 T < obs_horizon + action_horizon 时应被跳过。"""
        obs_horizon = 2
        action_horizon = 8
        # T=5 < 10=min_len，应被跳过
        h5 = _make_hdf5(
            tmp_path, n_trajs=2, T=5, vlm_reward_values=[1, 1]
        )
        ds = VLAWSuccessDataset(
            str(h5),
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
        )
        assert len(ds) == 0, "太短的轨迹应被跳过，dataset 长度应为 0"
