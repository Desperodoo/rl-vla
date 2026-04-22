"""Tests for standalone pi05 inference node and policy loader."""

import importlib.util
import os
import sys
import types
from unittest import mock

import numpy as np
import torch


_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RL_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CARM_DEPLOY_ROOT)))
for p in (_CARM_DEPLOY_ROOT, _RL_VLA_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


_rospy_mock = types.ModuleType("rospy")
_rospy_mock.loginfo = lambda *a, **kw: None
_rospy_mock.logwarn = lambda *a, **kw: None
_rospy_mock.logerr = lambda *a, **kw: None
_rospy_mock.loginfo_throttle = lambda *a, **kw: None
_rospy_mock.logwarn_throttle = lambda *a, **kw: None
_rospy_mock.is_shutdown = lambda: False
_rospy_mock.has_param = lambda *a, **kw: False
_rospy_mock.get_param = lambda *a, **kw: None
_rospy_mock.myargv = lambda: ["pytest"]
_rospy_mock.on_shutdown = lambda cb: None
_rospy_mock.signal_shutdown = lambda reason: None
_rospy_mock.init_node = lambda *a, **kw: None
sys.modules["rospy"] = _rospy_mock

for mod_name in (
    "std_msgs",
    "std_msgs.msg",
    "sensor_msgs",
    "sensor_msgs.msg",
    "geometry_msgs",
    "geometry_msgs.msg",
    "message_filters",
    "cv_bridge",
):
    sys.modules[mod_name] = mock.MagicMock()

if importlib.util.find_spec("lerobot") is None:
    _lerobot_root = types.ModuleType("lerobot")
    _lerobot_configs = types.ModuleType("lerobot.configs")
    _lerobot_configs_policies = types.ModuleType("lerobot.configs.policies")
    _lerobot_datasets = types.ModuleType("lerobot.datasets")
    _lerobot_dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    _lerobot_policies = types.ModuleType("lerobot.policies")
    _lerobot_factory_mod = types.ModuleType("lerobot.policies.factory")


    class _DummyPreTrainedConfig:
        @classmethod
        def from_pretrained(cls, path):
            cfg = cls()
            cfg.device = "cpu"
            cfg.pretrained_path = path
            cfg.input_features = {}
            cfg.output_features = {}
            cfg.normalization_mapping = {}
            cfg.n_action_steps = 15
            cfg.use_peft = False
            return cfg


    class _DummyDataset:
        def __init__(self, repo_id=None, root=None):
            self.repo_id = repo_id
            self.root = root
            self.meta = types.SimpleNamespace(stats={}, episodes=[])


    _lerobot_configs_policies.PreTrainedConfig = _DummyPreTrainedConfig
    _lerobot_dataset_mod.LeRobotDataset = _DummyDataset
    _lerobot_factory_mod.make_policy = lambda cfg, ds_meta=None, rename_map=None: mock.MagicMock(config=cfg)
    _lerobot_factory_mod.make_pre_post_processors = lambda **kwargs: (lambda x: x, lambda x: x)

    sys.modules["lerobot"] = _lerobot_root
    sys.modules["lerobot.configs"] = _lerobot_configs
    sys.modules["lerobot.configs.policies"] = _lerobot_configs_policies
    sys.modules["lerobot.datasets"] = _lerobot_datasets
    sys.modules["lerobot.datasets.lerobot_dataset"] = _lerobot_dataset_mod
    sys.modules["lerobot.policies"] = _lerobot_policies
    sys.modules["lerobot.policies.factory"] = _lerobot_factory_mod

if importlib.util.find_spec("peft") is None:
    _peft_mod = types.ModuleType("peft")
    _peft_mod.PeftConfig = type("PeftConfig", (), {"from_pretrained": staticmethod(lambda path: object())})
    sys.modules["peft"] = _peft_mod

_sc_path = os.path.join(_CARM_DEPLOY_ROOT, "core", "safety_controller.py")
_sc_spec = importlib.util.spec_from_file_location("core.safety_controller", _sc_path)
_sc_mod = importlib.util.module_from_spec(_sc_spec)
_sc_spec.loader.exec_module(_sc_mod)
sys.modules["core.safety_controller"] = _sc_mod

_core_mock = types.ModuleType("core")
_core_mock.SafetyController = _sc_mod.SafetyController
_core_mock.safety_controller = _sc_mod
sys.modules["core"] = _core_mock

_env_ros_mock = types.ModuleType("core.env_ros")
_env_ros_mock.RealEnvironment = mock.MagicMock()
sys.modules["core.env_ros"] = _env_ros_mock

_ki_mock = types.ModuleType("utils.keyboard_intervention")
_ki_mock.KeyboardInterventionHandler = mock.MagicMock()
sys.modules["utils.keyboard_intervention"] = _ki_mock

from inference.inference_pi05_ros import InferencePi05Node
from inference.policy_loader_pi05 import LeRobotPi05Policy


class TestLeRobotPi05Policy:
    def test_build_sample_includes_ee_pose(self):
        policy = LeRobotPi05Policy(
            {
                "device": "cpu",
                "state_mode": "joint_only",
                "dataset_root": "/tmp",
                "repo_id": "carm/pi05_local",
            }
        )
        image = np.zeros((3, 224, 224), dtype=np.float32)
        qpos = np.arange(7, dtype=np.float32)
        ee_pose = np.arange(7, dtype=np.float32) * 0.1
        sample = policy._build_sample({"image": image, "qpos": qpos, "ee_pose": ee_pose})
        assert {"observation.image", "observation.state", "observation.ee_pose"}.issubset(set(sample.keys()))
        assert tuple(sample["observation.image"].shape) == (3, 224, 224)
        assert tuple(sample["observation.state"].shape) == (7,)
        assert tuple(sample["observation.ee_pose"].shape) == (7,)

    def test_build_state_from_obs_joint_only(self):
        policy = LeRobotPi05Policy({"device": "cpu", "state_mode": "joint_only", "dataset_root": "/tmp"})
        qpos_joint = np.arange(7, dtype=np.float32)
        qpos_end = np.arange(8, dtype=np.float32)
        state = policy.build_state_from_obs(qpos_joint, qpos_end)
        np.testing.assert_allclose(state, qpos_joint)


class TestInferencePi05Node:
    def _make_node(self, overrides=None):
        config = {
            "pretrain": "/tmp/fake_pi05",
            "dataset_root": "/tmp/fake_dataset",
            "repo_id": "carm/pi05_local",
            "device": "cpu",
            "state_mode": "joint_only",
            "control_mode": "joint",
            "action_representation": "joint_absolute_gripper",
            "desire_inference_freq": 10,
            "temporal_factor_k": 0.05,
            "execution_mode": "receding_horizon",
            "control_freq": 50,
            "safety_config": "",
            "log_dir": "/tmp/pi05_infer_logs",
            "record_inference": False,
            "intervention": False,
            "max_steps": 50,
        }
        if overrides:
            config.update(overrides)

        with mock.patch("inference.inference_pi05_ros.RealEnvironment") as MockEnv, \
             mock.patch.object(InferencePi05Node, "_create_policy") as mock_policy, \
             mock.patch.object(InferencePi05Node, "_create_safety_controller") as mock_safety, \
             mock.patch.object(InferencePi05Node, "_create_logger") as mock_logger, \
             mock.patch.object(InferencePi05Node, "_setup_logger_metadata"):
            fake_policy = mock.MagicMock()
            fake_policy.pred_horizon = 15
            fake_policy.action_dim = 8
            fake_policy.action_dim_full = 8
            fake_policy.state_mode = "joint_only"
            fake_policy.control_mode = "joint"
            fake_policy.action_representation = "joint_absolute_gripper"
            fake_policy.target_image_size = (224, 224)
            fake_policy.device = torch.device("cpu")
            mock_policy.return_value = fake_policy

            fake_safety = mock.MagicMock()
            fake_safety.check_and_clip.side_effect = lambda action, current: (action, [])
            fake_safety.check_workspace.side_effect = lambda pose: (pose, [])
            fake_safety.check_joint_limits.side_effect = lambda action: (action, [])
            mock_safety.return_value = fake_safety

            fake_logger = mock.MagicMock()
            fake_logger.log_dir = config["log_dir"]
            mock_logger.return_value = fake_logger

            node = InferencePi05Node(config)
            node.running = False
            if node.inference_thread.is_alive():
                node.inference_thread.join(timeout=1.0)
        return node, MockEnv

    def test_node_uses_joint_control_mode(self):
        node, _ = self._make_node()
        assert node.policy.control_mode == "joint"
        assert node._action_dim_full == 8
        assert node.execution_mode == "receding_horizon"

    def test_apply_safety_uses_joint_observation(self):
        node, _ = self._make_node()
        node.latest_obs = {"qpos_joint": np.zeros(7, dtype=np.float32), "qpos_end": np.zeros(8, dtype=np.float32)}
        actions = np.ones((2, 8), dtype=np.float32)
        clipped, clipped_any, warnings = node._apply_safety(actions)
        np.testing.assert_allclose(clipped, actions)
        assert clipped_any is False
        assert warnings == []

    def test_append_chunk_truncates_at_act_horizon(self):
        node, _ = self._make_node({"act_horizon": 3})
        actions = np.arange(40, dtype=np.float32).reshape(5, 8)
        node._append_chunk(actions, chunk_base_time=10.0)
        with node.lock_tfs:
            assert len(node.action_manager.trajectories) == 1
            _, traj = node.action_manager.trajectories[0]
            assert len(traj.timestamps) == 3
