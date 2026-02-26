"""
单元测试 — imagination_env.py
覆盖: ImaginationEnvConfig (配置字段验证，dry_run 模式)

所有测试：
  - 不加载 ManiSkill 或 Ctrl-World 模型权重
  - 不依赖 GPU
  - 仅测试配置结构和引擎初始化逻辑（mock 依赖项）
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# 确保项目根目录在 sys.path 中
_ROOT = str(Path(__file__).parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rlft.vlaw.world_model.imagination_env import ImaginationEnvConfig, ImaginationEnvEngine


# ---------------------------------------------------------------------------
# ImaginationEnvConfig 测试
# ---------------------------------------------------------------------------


class TestImaginationEnvConfig:
    """测试 ImaginationEnvConfig 的默认值、类型和字段完整性。"""

    def test_config_defaults(self) -> None:
        """验证 ImaginationEnvConfig 字段类型和默认值正确。"""
        cfg = ImaginationEnvConfig()
        assert isinstance(cfg.num_interact, int), "num_interact 应为 int"
        assert isinstance(cfg.act_steps, int), "act_steps 应为 int"
        assert isinstance(cfg.obs_horizon, int), "obs_horizon 应为 int"
        assert isinstance(cfg.decode_for_policy, bool), "decode_for_policy 应为 bool"
        assert isinstance(cfg.output_dir, str), "output_dir 应为 str"
        assert isinstance(cfg.gpu_id, int), "gpu_id 应为 int"
        assert isinstance(cfg.batch_size, int), "batch_size 应为 int"
        assert isinstance(cfg.dry_run, bool), "dry_run 应为 bool"

    def test_config_default_num_interact(self) -> None:
        """num_interact 默认值应为 12（VLAW 论文 K_interact）。"""
        cfg = ImaginationEnvConfig()
        assert cfg.num_interact == 12, (
            f"num_interact 默认值应为 12，实际: {cfg.num_interact}"
        )

    def test_config_tasks_is_list(self) -> None:
        """tasks 字段应为 list 类型，且默认非空。"""
        cfg = ImaginationEnvConfig()
        assert isinstance(cfg.tasks, list), "tasks 应为 list 类型"
        assert len(cfg.tasks) > 0, "tasks 默认不应为空"

    def test_config_task_id(self) -> None:
        """task_id 可以设置为任意字符串。"""
        cfg = ImaginationEnvConfig(task_id="PickCube-v1")
        assert cfg.task_id == "PickCube-v1"

        cfg2 = ImaginationEnvConfig(task_id="StackCube-v1")
        assert cfg2.task_id == "StackCube-v1"

    def test_config_dry_run_default_false(self) -> None:
        """dry_run 默认应为 False。"""
        cfg = ImaginationEnvConfig()
        assert cfg.dry_run is False

    def test_config_dry_run_can_be_true(self) -> None:
        """dry_run 可以设置为 True。"""
        cfg = ImaginationEnvConfig(dry_run=True)
        assert cfg.dry_run is True

    def test_config_num_envs(self) -> None:
        """num_envs 默认值应为 16（P4.3 并行 env 配置）。"""
        cfg = ImaginationEnvConfig()
        assert cfg.num_envs == 16, (
            f"num_envs 默认值应为 16，实际: {cfg.num_envs}"
        )

    def test_config_camera_resolution(self) -> None:
        """camera_width 和 camera_height 默认均为 192。"""
        cfg = ImaginationEnvConfig()
        assert cfg.camera_width == 192, f"camera_width 默认应为 192，实际: {cfg.camera_width}"
        assert cfg.camera_height == 192, f"camera_height 默认应为 192，实际: {cfg.camera_height}"

    def test_config_obs_mode(self) -> None:
        """obs_mode 默认应为 'rgbd'（VLAW 使用 RGB-D 观测）。"""
        cfg = ImaginationEnvConfig()
        assert cfg.obs_mode == "rgbd", f"obs_mode 默认应为 'rgbd'，实际: {cfg.obs_mode}"

    def test_config_control_mode(self) -> None:
        """control_mode 默认应为 'pd_ee_delta_pose'。"""
        cfg = ImaginationEnvConfig()
        assert cfg.control_mode == "pd_ee_delta_pose", (
            f"control_mode 应为 'pd_ee_delta_pose'，实际: {cfg.control_mode}"
        )

    def test_config_output_dir_not_empty(self) -> None:
        """output_dir 默认不应为空字符串。"""
        cfg = ImaginationEnvConfig()
        assert len(cfg.output_dir) > 0, "output_dir 不应为空字符串"

    def test_config_max_episode_steps(self) -> None:
        """max_episode_steps 默认应为 200。"""
        cfg = ImaginationEnvConfig()
        assert cfg.max_episode_steps == 200, (
            f"max_episode_steps 默认应为 200，实际: {cfg.max_episode_steps}"
        )

    def test_config_decode_for_policy_default(self) -> None:
        """decode_for_policy 默认应为 True（需要将 latent 解码给策略）。"""
        cfg = ImaginationEnvConfig()
        assert cfg.decode_for_policy is True

    def test_config_act_steps_default(self) -> None:
        """act_steps 默认为 5（世界模型单次推理步数）。"""
        cfg = ImaginationEnvConfig()
        assert cfg.act_steps == 5, f"act_steps 默认应为 5，实际: {cfg.act_steps}"


# ---------------------------------------------------------------------------
# ImaginationEnvEngine 初始化测试
# ---------------------------------------------------------------------------


class TestImaginationEnvEngineInit:
    """测试 ImaginationEnvEngine 初始化（不加载真实模型权重）。"""

    def _make_mock_engine(
        self, cfg: ImaginationEnvConfig | None = None
    ) -> ImaginationEnvEngine:
        """创建 mock ImaginationEnvEngine（不加载真实 wm_adapter/policy）。"""
        cfg = cfg or ImaginationEnvConfig()
        mock_wm = MagicMock()
        mock_policy = MagicMock()
        engine = ImaginationEnvEngine(
            wm_adapter=mock_wm,
            policy=mock_policy,
            config=cfg,
        )
        return engine

    def test_engine_init(self) -> None:
        """ImaginationEnvEngine 可以用 mock 依赖正常初始化。"""
        engine = self._make_mock_engine()
        assert engine is not None
        assert engine.config is not None
        assert engine.wm_adapter is not None
        assert engine.policy is not None

    def test_engine_has_device(self) -> None:
        """初始化后应有 device 属性。"""
        engine = self._make_mock_engine()
        assert hasattr(engine, "device"), "ImaginationEnvEngine 应有 device 属性"
        assert isinstance(engine.device, torch.device), "device 应为 torch.device"

    def test_engine_config_stored(self) -> None:
        """ImaginationEnvEngine 应存储传入的 config。"""
        cfg = ImaginationEnvConfig(task_id="PickCube-v1", dry_run=True)
        engine = self._make_mock_engine(cfg=cfg)
        assert engine.config.task_id == "PickCube-v1"
        assert engine.config.dry_run is True

    def test_engine_dry_run_config(self) -> None:
        """dry_run=True 的 ImaginationEnvConfig 可以正常传入引擎。"""
        cfg = ImaginationEnvConfig(dry_run=True)
        engine = self._make_mock_engine(cfg=cfg)
        assert engine.config.dry_run is True

    def test_engine_multiple_tasks(self) -> None:
        """含有多个任务的 config 可以正常创建引擎。"""
        cfg = ImaginationEnvConfig(
            tasks=["LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"]
        )
        engine = self._make_mock_engine(cfg=cfg)
        assert len(engine.config.tasks) == 3
