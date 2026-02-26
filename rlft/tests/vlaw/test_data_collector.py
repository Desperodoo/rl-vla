"""
单元测试 — data_collector.py
覆盖: CollectorConfig, HDF5 写入功能

所有测试：
  - 不依赖 GPU 或 ManiSkill 环境
  - 使用 tmp_path fixture 创建临时 HDF5 文件
  - 不加载真实 checkpoint
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch

# 确保项目根目录在 sys.path 中
_ROOT = str(Path(__file__).parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rlft.vlaw.data.collector import CollectorConfig


# ---------------------------------------------------------------------------
# CollectorConfig 测试
# ---------------------------------------------------------------------------


class TestCollectorConfig:
    """测试 CollectorConfig 默认值和字段类型。"""

    def test_config_defaults(self) -> None:
        """验证 CollectorConfig 默认 env_id, num_envs, num_episodes 等字段。"""
        cfg = CollectorConfig()
        assert isinstance(cfg.env_id, str), "env_id 应为字符串"
        assert len(cfg.env_id) > 0, "env_id 默认值不应为空"
        assert cfg.num_envs == 64, f"num_envs 默认应为 64，实际: {cfg.num_envs}"
        assert cfg.num_episodes == 50, f"num_episodes 默认应为 50，实际: {cfg.num_episodes}"

    def test_config_camera_resolution(self) -> None:
        """验证相机分辨率默认值为 192×192。"""
        cfg = CollectorConfig()
        assert cfg.camera_width == 192, f"camera_width 默认应为 192，实际: {cfg.camera_width}"
        assert cfg.camera_height == 192, f"camera_height 默认应为 192，实际: {cfg.camera_height}"

    def test_config_max_episode_steps(self) -> None:
        """max_episode_steps 默认值应为 200。"""
        cfg = CollectorConfig()
        assert cfg.max_episode_steps == 200, (
            f"max_episode_steps 默认应为 200，实际: {cfg.max_episode_steps}"
        )

    def test_config_obs_action_dims(self) -> None:
        """obs_horizon 和 act_steps 默认值应合理。"""
        cfg = CollectorConfig()
        assert cfg.obs_horizon == 2, f"obs_horizon 默认应为 2，实际: {cfg.obs_horizon}"
        assert cfg.act_steps == 8, f"act_steps 默认应为 8，实际: {cfg.act_steps}"

    def test_config_checkpoint_path_is_string(self) -> None:
        """checkpoint_path 类型应为字符串（允许空字符串，use_random_policy=True 时不需要）。"""
        cfg = CollectorConfig()
        assert isinstance(cfg.checkpoint_path, str), "checkpoint_path 应为字符串类型"

    def test_config_use_random_policy_default(self) -> None:
        """use_random_policy 默认应为 False（使用 ShortCut Flow）。"""
        cfg = CollectorConfig()
        assert cfg.use_random_policy is False

    def test_config_dry_run_default(self) -> None:
        """dry_run 默认应为 False。"""
        cfg = CollectorConfig()
        assert cfg.dry_run is False

    def test_config_sim_backend(self) -> None:
        """sim_backend 默认应为 'physx_cuda'。"""
        cfg = CollectorConfig()
        assert cfg.sim_backend == "physx_cuda", (
            f"sim_backend 默认应为 'physx_cuda'，实际: {cfg.sim_backend}"
        )

    def test_config_control_mode(self) -> None:
        """control_mode 默认应为 'pd_ee_delta_pose'。"""
        cfg = CollectorConfig()
        assert cfg.control_mode == "pd_ee_delta_pose", (
            f"control_mode 默认应为 'pd_ee_delta_pose'，实际: {cfg.control_mode}"
        )

    def test_config_visual_feature_dim(self) -> None:
        """visual_feature_dim 默认应为 256（PlainConv 输出维度）。"""
        cfg = CollectorConfig()
        assert cfg.visual_feature_dim == 256, (
            f"visual_feature_dim 默认应为 256，实际: {cfg.visual_feature_dim}"
        )

    def test_config_output_dir_not_empty(self) -> None:
        """output_dir 不应为空字符串。"""
        cfg = CollectorConfig()
        assert len(cfg.output_dir) > 0, "output_dir 不应为空字符串"

    def test_config_source_tag_default(self) -> None:
        """source_tag 默认应为 'real'（真实 rollout）。"""
        cfg = CollectorConfig()
        assert cfg.source_tag == "real", (
            f"source_tag 默认应为 'real'，实际: {cfg.source_tag}"
        )

    def test_config_frame_skip_positive(self) -> None:
        """frame_skip 应为正整数（用于控制保存频率）。"""
        cfg = CollectorConfig()
        assert cfg.frame_skip > 0, f"frame_skip 应为正整数，实际: {cfg.frame_skip}"

    def test_config_gpu_id_int(self) -> None:
        """gpu_id 应为 int 类型。"""
        cfg = CollectorConfig()
        assert isinstance(cfg.gpu_id, int)

    def test_config_custom_env_id(self) -> None:
        """CollectorConfig 可以自定义 env_id。"""
        cfg = CollectorConfig(env_id="LiftPegUpright-v1")
        assert cfg.env_id == "LiftPegUpright-v1"

    def test_config_include_rgb_default(self) -> None:
        """include_rgb 默认应为 True。"""
        cfg = CollectorConfig()
        assert cfg.include_rgb is True


# ---------------------------------------------------------------------------
# HDF5 写入功能测试
# ---------------------------------------------------------------------------


class TestHDF5Writer:
    """测试 VLAWDataCollector 的 HDF5 写入功能（不需要真实环境）。"""

    def _make_mock_trajectories(
        self,
        n: int = 3,
        T: int = 10,
        camera_h: int = 192,
        camera_w: int = 192,
    ) -> list[dict]:
        """生成 mock 轨迹数据。"""
        rng = np.random.default_rng(seed=0)
        trajs = []
        for i in range(n):
            env_success = np.zeros(T, dtype=bool)
            env_success[-1] = True  # 末帧成功
            traj = {
                "rgb_base": rng.integers(0, 255, (T, camera_h, camera_w, 3), dtype=np.uint8),
                "rgb_render": rng.integers(0, 255, (T, camera_h, camera_w, 3), dtype=np.uint8),
                "state": rng.standard_normal((T, 25)).astype(np.float32),
                "actions": rng.uniform(-0.05, 0.05, (T, 7)).astype(np.float32),
                "env_success": env_success,
                "task_instruction": "pick up the cube",
                "source": "real",
            }
            trajs.append(traj)
        return trajs

    def test_hdf5_writer_creates_file(self, tmp_path: Path) -> None:
        """save_hdf5 应创建 HDF5 文件（使用 mock collector）。"""
        from rlft.vlaw.data.collector import VLAWDataCollector

        cfg = CollectorConfig(
            env_id="PickCube-v1",
            output_dir=str(tmp_path / "output"),
            use_random_policy=True,
        )

        # 创建 collector 但不初始化 GPU/env（直接测试 save_hdf5）
        with patch.object(VLAWDataCollector, "__init__", lambda self, c: None):
            collector = VLAWDataCollector.__new__(VLAWDataCollector)
            collector.cfg = cfg

        trajs = self._make_mock_trajectories(n=3, T=10)
        out_path = tmp_path / "output" / "test_rollouts.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        saved_path = collector.save_hdf5(trajs, output_path=str(out_path))
        assert saved_path.exists(), f"HDF5 文件应已创建: {saved_path}"

    def test_hdf5_writer_trajectory_count(self, tmp_path: Path) -> None:
        """写入 3 条轨迹，读回后轨迹数量应为 3。"""
        from rlft.vlaw.data.collector import VLAWDataCollector

        cfg = CollectorConfig(env_id="PickCube-v1", use_random_policy=True)
        with patch.object(VLAWDataCollector, "__init__", lambda self, c: None):
            collector = VLAWDataCollector.__new__(VLAWDataCollector)
            collector.cfg = cfg

        trajs = self._make_mock_trajectories(n=3, T=10)
        out_path = tmp_path / "rollouts.h5"
        collector.save_hdf5(trajs, output_path=str(out_path))

        with h5py.File(str(out_path), "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj_")]
            assert len(traj_keys) == 3, (
                f"应有 3 条轨迹，实际: {len(traj_keys)}"
            )

    def test_hdf5_writer_meta_attributes(self, tmp_path: Path) -> None:
        """写入的 HDF5 文件应含 meta 组，包含 num_trajectories 等属性。"""
        from rlft.vlaw.data.collector import VLAWDataCollector

        cfg = CollectorConfig(env_id="PickCube-v1", use_random_policy=True)
        with patch.object(VLAWDataCollector, "__init__", lambda self, c: None):
            collector = VLAWDataCollector.__new__(VLAWDataCollector)
            collector.cfg = cfg

        trajs = self._make_mock_trajectories(n=2, T=10)
        out_path = tmp_path / "meta_test.h5"
        collector.save_hdf5(trajs, output_path=str(out_path))

        with h5py.File(str(out_path), "r") as f:
            assert "meta" in f, "应含 meta 组"
            meta = f["meta"]
            assert "num_trajectories" in meta.attrs, "meta 应含 num_trajectories"
            assert meta.attrs["num_trajectories"] == 2

    def test_hdf5_writer_trajectory_datasets(self, tmp_path: Path) -> None:
        """每条轨迹应含 rgb_base, state, actions, env_success 等字段。"""
        from rlft.vlaw.data.collector import VLAWDataCollector

        cfg = CollectorConfig(env_id="PickCube-v1", use_random_policy=True)
        with patch.object(VLAWDataCollector, "__init__", lambda self, c: None):
            collector = VLAWDataCollector.__new__(VLAWDataCollector)
            collector.cfg = cfg

        trajs = self._make_mock_trajectories(n=1, T=10)
        out_path = tmp_path / "traj_test.h5"
        collector.save_hdf5(trajs, output_path=str(out_path))

        with h5py.File(str(out_path), "r") as f:
            traj = f["traj_0000"]
            assert "rgb_base" in traj, "轨迹应含 rgb_base"
            assert "state" in traj, "轨迹应含 state"
            assert "actions" in traj, "轨迹应含 actions"
            assert "env_success" in traj, "轨迹应含 env_success"

    def test_hdf5_writer_data_shapes(self, tmp_path: Path) -> None:
        """写入的 dataset shape 应与原始数据一致（T, H, W, 3）。"""
        from rlft.vlaw.data.collector import VLAWDataCollector

        T = 10
        H = W = 192

        cfg = CollectorConfig(
            env_id="PickCube-v1", camera_height=H, camera_width=W, use_random_policy=True
        )
        with patch.object(VLAWDataCollector, "__init__", lambda self, c: None):
            collector = VLAWDataCollector.__new__(VLAWDataCollector)
            collector.cfg = cfg

        trajs = self._make_mock_trajectories(n=1, T=T, camera_h=H, camera_w=W)
        out_path = tmp_path / "shape_test.h5"
        collector.save_hdf5(trajs, output_path=str(out_path))

        with h5py.File(str(out_path), "r") as f:
            rgb = f["traj_0000"]["rgb_base"]
            assert rgb.shape == (T, H, W, 3), (
                f"rgb_base shape 错误: 期望 ({T},{H},{W},3)，实际: {rgb.shape}"
            )
            state = f["traj_0000"]["state"]
            assert state.shape == (T, 25), (
                f"state shape 错误: 期望 ({T},25)，实际: {state.shape}"
            )
            actions = f["traj_0000"]["actions"]
            assert actions.shape == (T, 7), (
                f"actions shape 错误: 期望 ({T},7)，实际: {actions.shape}"
            )

    def test_hdf5_writer_success_attr(self, tmp_path: Path) -> None:
        """每条轨迹的 success attr 应正确反映 env_success.any() 的结果。"""
        from rlft.vlaw.data.collector import VLAWDataCollector

        cfg = CollectorConfig(env_id="PickCube-v1", use_random_policy=True)
        with patch.object(VLAWDataCollector, "__init__", lambda self, c: None):
            collector = VLAWDataCollector.__new__(VLAWDataCollector)
            collector.cfg = cfg

        # 第 0 条成功（末帧 True），第 1 条失败（全 False）
        rng = np.random.default_rng(seed=1)
        trajs = []
        for success_val in [True, False]:
            env_success = np.zeros(10, dtype=bool)
            if success_val:
                env_success[-1] = True
            trajs.append({
                "rgb_base": rng.integers(0, 255, (10, 192, 192, 3), dtype=np.uint8),
                "rgb_render": rng.integers(0, 255, (10, 192, 192, 3), dtype=np.uint8),
                "state": rng.standard_normal((10, 25)).astype(np.float32),
                "actions": rng.uniform(-0.05, 0.05, (10, 7)).astype(np.float32),
                "env_success": env_success,
                "task_instruction": "pick cube",
                "source": "real",
            })

        out_path = tmp_path / "success_test.h5"
        collector.save_hdf5(trajs, output_path=str(out_path))

        with h5py.File(str(out_path), "r") as f:
            assert f["traj_0000"].attrs["success"] is True or \
                   bool(f["traj_0000"].attrs["success"]) is True, \
                "第 0 条轨迹 success attr 应为 True"
            assert f["traj_0001"].attrs["success"] is False or \
                   bool(f["traj_0001"].attrs["success"]) is False, \
                "第 1 条轨迹 success attr 应为 False"

    def test_hdf5_writer_success_rate_in_meta(self, tmp_path: Path) -> None:
        """meta.success_rate 应正确计算（1/2 成功 = 0.5）。"""
        from rlft.vlaw.data.collector import VLAWDataCollector

        cfg = CollectorConfig(env_id="PickCube-v1", use_random_policy=True)
        with patch.object(VLAWDataCollector, "__init__", lambda self, c: None):
            collector = VLAWDataCollector.__new__(VLAWDataCollector)
            collector.cfg = cfg

        rng = np.random.default_rng(seed=2)
        trajs = []
        for success_val in [True, False]:
            env_success = np.zeros(10, dtype=bool)
            if success_val:
                env_success[-1] = True
            trajs.append({
                "rgb_base": rng.integers(0, 255, (10, 192, 192, 3), dtype=np.uint8),
                "rgb_render": rng.integers(0, 255, (10, 192, 192, 3), dtype=np.uint8),
                "state": rng.standard_normal((10, 25)).astype(np.float32),
                "actions": rng.uniform(-0.05, 0.05, (10, 7)).astype(np.float32),
                "env_success": env_success,
                "task_instruction": "pick cube",
                "source": "real",
            })

        out_path = tmp_path / "sr_test.h5"
        collector.save_hdf5(trajs, output_path=str(out_path))

        with h5py.File(str(out_path), "r") as f:
            sr = float(f["meta"].attrs["success_rate"])
            assert abs(sr - 0.5) < 1e-6, f"success_rate 应为 0.5，实际: {sr}"
