"""ACP 单元测试 — config, value_targets, advantage, hdf5_dataset

测试规范：无 GPU、无真实权重、无训练循环。使用随机/mock 数据。
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from rlft.acp.config import (
    ACPInferConfig,
    ACPTrainConfig,
    AdvantageConfig,
    ValueModelConfig,
    ValueTargetConfig,
)
from rlft.acp.value_targets import (
    compute_value_targets,
    compute_value_targets_batch,
)
from rlft.acp.advantage import (
    binarize_advantages,
    compute_dense_rewards,
    compute_n_step_advantage,
    compute_task_threshold,
    compute_trajectory_weights,
    normalize_advantages_to_weights,
)


# =========================================================================
# A1: Config tests
# =========================================================================


class TestConfig:
    def test_value_model_config_defaults(self) -> None:
        cfg = ValueModelConfig()
        assert cfg.num_bins == 201
        assert cfg.bin_min == -1.0
        assert cfg.bin_max == 0.0
        assert cfg.freeze_vision_encoder is True
        assert cfg.freeze_language_model is True
        assert cfg.image_size == (128, 128)
        assert len(cfg.camera_keys) == 2

    def test_advantage_config_defaults(self) -> None:
        cfg = AdvantageConfig()
        assert cfg.n_step == 4
        assert cfg.positive_ratio == 0.3
        assert cfg.use_continuous_weights is True

    def test_train_config_defaults(self) -> None:
        cfg = ACPTrainConfig()
        assert cfg.num_steps == 8000
        assert cfg.batch_size == 32
        assert isinstance(cfg.value_model, ValueModelConfig)
        assert isinstance(cfg.value_target, ValueTargetConfig)

    def test_infer_config_defaults(self) -> None:
        cfg = ACPInferConfig()
        assert isinstance(cfg.advantage, AdvantageConfig)
        assert cfg.write_back is True


# =========================================================================
# A2: Value target tests
# =========================================================================


class TestValueTargets:
    def test_successful_trajectory(self) -> None:
        """成功轨迹：target 应从负到 0 单调递增。"""
        env_success = np.zeros(10, dtype=bool)
        env_success[-1] = True  # 最后一帧成功
        cfg = ValueTargetConfig()

        targets = compute_value_targets(
            env_success=env_success,
            episode_length=10,
            max_episode_length=10,
            cfg=cfg,
        )

        assert targets.shape == (10,)
        assert targets.dtype == np.float32
        # 最后一帧 remaining=0，target = 0 / (10+10) = 0
        assert targets[-1] == pytest.approx(0.0)
        # 单调递增（从最负到 0）
        for t in range(len(targets) - 1):
            assert targets[t] <= targets[t + 1]

    def test_failed_trajectory(self) -> None:
        """失败轨迹：target 应更负（因为有 c_fail 惩罚）。"""
        env_success_ok = np.zeros(10, dtype=bool)
        env_success_ok[-1] = True
        env_success_fail = np.zeros(10, dtype=bool)
        cfg = ValueTargetConfig()

        targets_ok = compute_value_targets(
            env_success=env_success_ok,
            episode_length=10,
            max_episode_length=10,
            cfg=cfg,
        )
        targets_fail = compute_value_targets(
            env_success=env_success_fail,
            episode_length=10,
            max_episode_length=10,
            cfg=cfg,
        )

        # 失败轨迹每帧 target 都 <= 成功轨迹
        assert np.all(targets_fail <= targets_ok)
        # 最后一帧：失败 target 应 < 0
        assert targets_fail[-1] < 0.0

    def test_targets_in_range(self) -> None:
        """所有 target 应在 [clip_min, clip_max] 范围内。"""
        env_success = np.zeros(20, dtype=bool)
        cfg = ValueTargetConfig(clip_min=-1.0, clip_max=0.0)

        targets = compute_value_targets(
            env_success=env_success,
            episode_length=20,
            max_episode_length=20,
            cfg=cfg,
        )

        assert np.all(targets >= -1.0)
        assert np.all(targets <= 0.0)

    def test_length_mismatch_raises(self) -> None:
        """env_success 长度与 episode_length 不匹配应报错。"""
        with pytest.raises(ValueError, match="不匹配"):
            compute_value_targets(
                env_success=np.zeros(5, dtype=bool),
                episode_length=10,
                max_episode_length=10,
                cfg=ValueTargetConfig(),
            )

    def test_batch_computation(self) -> None:
        """批量计算应与单条等价。"""
        cfg = ValueTargetConfig()
        trajs = [
            {"env_success": np.zeros(10, dtype=bool), "length": 10},
            {"env_success": np.ones(5, dtype=bool), "length": 5},
        ]

        batch_results = compute_value_targets_batch(trajs, max_episode_length=10, cfg=cfg)

        assert len(batch_results) == 2
        assert batch_results[0].shape == (10,)
        assert batch_results[1].shape == (5,)

        # 与单条计算结果一致
        single_0 = compute_value_targets(
            env_success=trajs[0]["env_success"],
            episode_length=10,
            max_episode_length=10,
            cfg=cfg,
        )
        np.testing.assert_array_almost_equal(batch_results[0], single_0)


# =========================================================================
# A3: Advantage tests
# =========================================================================


class TestDenseRewards:
    def test_basic(self) -> None:
        targets = np.array([-0.5, -0.3, -0.1, 0.0], dtype=np.float32)
        rewards = compute_dense_rewards(targets)

        assert rewards.shape == (4,)
        assert rewards.dtype == np.float32
        # r[0] = -0.5 - (-0.3) = -0.2
        assert rewards[0] == pytest.approx(-0.2)
        # r[1] = -0.3 - (-0.1) = -0.2
        assert rewards[1] == pytest.approx(-0.2)
        # r[2] = -0.1 - 0.0 = -0.1
        assert rewards[2] == pytest.approx(-0.1)
        # r[3] = target[3] = 0.0
        assert rewards[3] == pytest.approx(0.0)


class TestNStepAdvantage:
    def test_n_step_1(self) -> None:
        """1-step advantage = r[t] + V[t+1] - V[t]。"""
        rewards = np.array([0.1, 0.2, 0.3, 0.0], dtype=np.float32)
        values = np.array([-0.5, -0.3, -0.1, 0.0], dtype=np.float32)

        adv = compute_n_step_advantage(rewards, values, n_step=1)

        assert adv.shape == (4,)
        # A[0] = r[0] + V[1] - V[0] = 0.1 + (-0.3) - (-0.5) = 0.3
        assert adv[0] == pytest.approx(0.3)
        # A[3] = r[3] + bootstrap(0) - V[3] = 0.0 + 0 - 0.0 = 0.0
        assert adv[3] == pytest.approx(0.0)

    def test_n_step_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="n_step"):
            compute_n_step_advantage(
                np.zeros(5, dtype=np.float32),
                np.zeros(5, dtype=np.float32),
                n_step=0,
            )


class TestThreshold:
    def test_positive_ratio_30(self) -> None:
        """positive_ratio=0.3 → quantile 0.7。"""
        rng = np.random.default_rng(42)
        advantages = rng.standard_normal(1000).astype(np.float32)

        threshold = compute_task_threshold(advantages, positive_ratio=0.3)
        positive_count = np.sum(advantages >= threshold)
        actual_ratio = positive_count / len(advantages)

        # 实际 positive ratio 应接近 0.3
        assert abs(actual_ratio - 0.3) < 0.05

    def test_empty_returns_inf(self) -> None:
        threshold = compute_task_threshold(np.array([], dtype=np.float32), 0.3)
        assert threshold == float("inf")


class TestBinarize:
    def test_basic(self) -> None:
        advantages = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)
        indicators = binarize_advantages(advantages, threshold=0.5)

        assert indicators.dtype == np.int32
        np.testing.assert_array_equal(indicators, [0, 0, 1, 1])


class TestNormalizeWeights:
    def test_range(self) -> None:
        advantages = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        cfg = AdvantageConfig(weight_clip_min=0.0, weight_clip_max=5.0)
        weights = normalize_advantages_to_weights(advantages, cfg)

        assert weights.dtype == np.float32
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 5.0)
        # min advantage → weight=0, max advantage → weight=1
        assert weights[0] == pytest.approx(0.0)
        assert weights[-1] == pytest.approx(1.0)

    def test_constant_advantages(self) -> None:
        """所有 advantage 相同时返回全 1.0。"""
        advantages = np.full(10, 0.5, dtype=np.float32)
        cfg = AdvantageConfig()
        weights = normalize_advantages_to_weights(advantages, cfg)
        np.testing.assert_array_almost_equal(weights, np.ones(10, dtype=np.float32))


class TestTrajectoryWeights:
    def test_pipeline_shapes(self) -> None:
        """完整流水线 shape / dtype 检查。"""
        T = 20
        value_targets = np.linspace(-1.0, 0.0, T, dtype=np.float32)
        predicted_values = value_targets + np.random.default_rng(0).normal(0, 0.05, T).astype(np.float32)
        cfg = AdvantageConfig(n_step=4, positive_ratio=0.3)

        result = compute_trajectory_weights(value_targets, predicted_values, cfg)

        assert result["rewards"].shape == (T,)
        assert result["advantages"].shape == (T,)
        assert result["indicators"].shape == (T,)
        assert result["weights"].shape == (T,)
        assert result["rewards"].dtype == np.float32
        assert result["indicators"].dtype == np.int32
        assert result["weights"].dtype == np.float32

    def test_binary_weights_mode(self) -> None:
        """use_continuous_weights=False 时 weights 应等于 indicators（float）。"""
        T = 20
        value_targets = np.linspace(-1.0, 0.0, T, dtype=np.float32)
        predicted_values = value_targets.copy()
        cfg = AdvantageConfig(n_step=2, positive_ratio=0.5, use_continuous_weights=False)

        result = compute_trajectory_weights(value_targets, predicted_values, cfg)
        np.testing.assert_array_equal(
            result["weights"],
            result["indicators"].astype(np.float32),
        )


# =========================================================================
# B2: HDF5 Dataset tests
# =========================================================================


@pytest.fixture
def mock_acp_hdf5(tmp_path: Path) -> Path:
    """创建 ACP 兼容的 HDF5 文件（双相机 128x128）。"""
    h5_path = tmp_path / "acp_test.h5"
    rng = np.random.default_rng(seed=99)

    with h5py.File(str(h5_path), "w") as f:
        for i in range(3):
            grp = f.create_group(f"traj_{i:04d}")
            T = 8
            grp.create_dataset(
                "rgb_base",
                data=rng.integers(0, 255, (T, 128, 128, 3), dtype=np.uint8),
            )
            grp.create_dataset(
                "rgb_render",
                data=rng.integers(0, 255, (T, 128, 128, 3), dtype=np.uint8),
            )
            success = np.zeros(T, dtype=bool)
            if i != 1:  # traj 0, 2 成功
                success[-1] = True
            grp.create_dataset("env_success", data=success)
            grp.create_dataset(
                "actions",
                data=rng.uniform(-0.05, 0.05, (T, 7)).astype(np.float32),
            )

    return h5_path


class TestACPValueDataset:
    def test_dataset_length(self, mock_acp_hdf5: Path) -> None:
        from rlft.acp.hdf5_dataset import ACPValueDataset

        ds = ACPValueDataset(
            hdf5_paths=[mock_acp_hdf5],
            camera_keys=["rgb_base", "rgb_render"],
            value_target_cfg=ValueTargetConfig(),
        )
        # 3 trajectories * 8 frames each = 24
        assert len(ds) == 24

    def test_sample_shapes(self, mock_acp_hdf5: Path) -> None:
        from rlft.acp.hdf5_dataset import ACPValueDataset

        ds = ACPValueDataset(
            hdf5_paths=[mock_acp_hdf5],
            camera_keys=["rgb_base", "rgb_render"],
            value_target_cfg=ValueTargetConfig(),
        )
        sample = ds[0]

        assert sample["images"].shape == (2, 3, 128, 128)
        assert sample["images"].dtype == torch.uint8
        assert sample["image_mask"].shape == (2,)
        assert sample["image_mask"].dtype == torch.bool
        assert torch.all(sample["image_mask"])  # 两个相机都存在
        assert isinstance(sample["value_target"].item(), float)
        assert isinstance(sample["traj_key"], str)
        assert isinstance(sample["frame_idx"], int)

    def test_value_targets_computed(self, mock_acp_hdf5: Path) -> None:
        from rlft.acp.hdf5_dataset import ACPValueDataset

        ds = ACPValueDataset(
            hdf5_paths=[mock_acp_hdf5],
            camera_keys=["rgb_base", "rgb_render"],
            value_target_cfg=ValueTargetConfig(),
        )
        # Check that targets are in valid range
        for i in range(min(5, len(ds))):
            target = ds[i]["value_target"].item()
            assert -1.0 <= target <= 0.0

    def test_missing_camera_graceful(self, mock_acp_hdf5: Path) -> None:
        """缺失相机应被零填充且 mask=False。"""
        from rlft.acp.hdf5_dataset import ACPValueDataset

        ds = ACPValueDataset(
            hdf5_paths=[mock_acp_hdf5],
            camera_keys=["rgb_base", "nonexistent_cam"],
            value_target_cfg=ValueTargetConfig(),
        )
        sample = ds[0]
        assert sample["image_mask"][0] is True or sample["image_mask"][0].item() is True
        assert sample["image_mask"][1] is False or sample["image_mask"][1].item() is False

    def test_collate(self, mock_acp_hdf5: Path) -> None:
        from rlft.acp.hdf5_dataset import ACPValueDataset, collate_acp

        ds = ACPValueDataset(
            hdf5_paths=[mock_acp_hdf5],
            camera_keys=["rgb_base", "rgb_render"],
            value_target_cfg=ValueTargetConfig(),
        )
        batch_list = [ds[0], ds[1], ds[2]]
        batch = collate_acp(batch_list)

        assert batch["images"].shape == (3, 2, 3, 128, 128)
        assert batch["image_mask"].shape == (3, 2)
        assert batch["value_target"].shape == (3,)
        assert len(batch["traj_keys"]) == 3
        assert len(batch["frame_idxs"]) == 3


# =========================================================================
# B3: VLM label mode tests (success_key="vlm_success")
# =========================================================================


@pytest.fixture
def mock_vlm_labeled_hdf5(tmp_path: Path) -> Path:
    """创建 VLM 标注风格的 HDF5 文件。

    与 mock_acp_hdf5 区别：
    - 无 env_success dataset
    - 有 vlm_success scalar attribute (int 0/1)
    """
    h5_path = tmp_path / "vlm_labeled.h5"
    rng = np.random.default_rng(seed=77)

    with h5py.File(str(h5_path), "w") as f:
        for i in range(3):
            grp = f.create_group(f"traj_{i:04d}")
            T = 8
            grp.create_dataset(
                "rgb_base",
                data=rng.integers(0, 255, (T, 128, 128, 3), dtype=np.uint8),
            )
            grp.create_dataset(
                "rgb_render",
                data=rng.integers(0, 255, (T, 128, 128, 3), dtype=np.uint8),
            )
            grp.create_dataset(
                "actions",
                data=rng.uniform(-0.05, 0.05, (T, 7)).astype(np.float32),
            )
            # VLM scalar attr: traj 0, 2 成功; traj 1 失败
            grp.attrs["vlm_success"] = int(i != 1)

    return h5_path


class TestVLMLabelMode:
    def test_vlm_dataset_length(self, mock_vlm_labeled_hdf5: Path) -> None:
        """success_key='vlm_success' 模式正确加载。"""
        from rlft.acp.hdf5_dataset import ACPValueDataset

        cfg = ValueTargetConfig(success_key="vlm_success")
        ds = ACPValueDataset(
            hdf5_paths=[mock_vlm_labeled_hdf5],
            camera_keys=["rgb_base", "rgb_render"],
            value_target_cfg=cfg,
        )
        # 3 trajectories * 8 frames each = 24
        assert len(ds) == 24

    def test_vlm_targets_in_range(self, mock_vlm_labeled_hdf5: Path) -> None:
        """VLM 模式下 value targets 应在 [-1, 0] 范围内。"""
        from rlft.acp.hdf5_dataset import ACPValueDataset

        cfg = ValueTargetConfig(success_key="vlm_success")
        ds = ACPValueDataset(
            hdf5_paths=[mock_vlm_labeled_hdf5],
            camera_keys=["rgb_base", "rgb_render"],
            value_target_cfg=cfg,
        )
        for i in range(len(ds)):
            target = ds[i]["value_target"].item()
            assert -1.0 <= target <= 0.0

    def test_vlm_success_vs_fail_targets(self, mock_vlm_labeled_hdf5: Path) -> None:
        """VLM 成功轨迹的 target 应优于失败轨迹。"""
        from rlft.acp.hdf5_dataset import ACPValueDataset

        cfg = ValueTargetConfig(success_key="vlm_success")
        ds = ACPValueDataset(
            hdf5_paths=[mock_vlm_labeled_hdf5],
            camera_keys=["rgb_base", "rgb_render"],
            value_target_cfg=cfg,
        )
        # traj_0000 是成功(vlm_success=1), traj_0001 是失败(vlm_success=0)
        # 各 8 帧, 最后一帧 target 比较
        success_last_target = ds[7]["value_target"].item()   # traj_0000 frame 7
        fail_last_target = ds[15]["value_target"].item()     # traj_0001 frame 7
        assert success_last_target > fail_last_target

    def test_vlm_consistent_with_env_success(self, mock_acp_hdf5: Path) -> None:
        """env_success 模式默认行为应不受 success_key 改动影响。"""
        from rlft.acp.hdf5_dataset import ACPValueDataset

        cfg_default = ValueTargetConfig()  # success_key="env_success"
        ds = ACPValueDataset(
            hdf5_paths=[mock_acp_hdf5],
            camera_keys=["rgb_base", "rgb_render"],
            value_target_cfg=cfg_default,
        )
        # 已有测试验证此路径, 这里只确认 success_key 默认值不影响
        assert len(ds) == 24
        assert ds[0]["value_target"].item() >= -1.0
