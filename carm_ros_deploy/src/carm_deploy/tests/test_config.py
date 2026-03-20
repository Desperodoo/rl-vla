"""Tests for inference.config.InferenceConfig."""

import sys
import os
import types
import pytest
from unittest import mock

# Path setup
_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RL_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CARM_DEPLOY_ROOT)))
for p in (_CARM_DEPLOY_ROOT, _RL_VLA_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Mock rospy before importing anything
if "rospy" not in sys.modules:
    _rospy_mock = types.ModuleType("rospy")
    _rospy_mock.loginfo = lambda *a, **kw: None
    _rospy_mock.logwarn = lambda *a, **kw: None
    _rospy_mock.logerr = lambda *a, **kw: None
    sys.modules["rospy"] = _rospy_mock

from inference.config import InferenceConfig


class TestDefaults:
    def test_default_robot_ip(self):
        cfg = InferenceConfig()
        assert cfg.robot_ip == '10.42.0.101'

    def test_default_control_freq(self):
        cfg = InferenceConfig()
        assert cfg.control_freq == 50

    def test_default_act_horizon(self):
        cfg = InferenceConfig()
        assert cfg.act_horizon == 8

    def test_default_execution_mode(self):
        cfg = InferenceConfig()
        assert cfg.execution_mode == 'receding_horizon'


class TestBUG6Fix:
    """BUG-6: timeline_enabled is now a derived property from timeline_disabled."""

    def test_timeline_enabled_default(self):
        cfg = InferenceConfig()
        assert cfg.timeline_enabled is True
        assert cfg.timeline_disabled is False

    def test_timeline_disabled(self):
        cfg = InferenceConfig(timeline_disabled=True)
        assert cfg.timeline_enabled is False

    def test_timeline_enabled_not_settable(self):
        """timeline_enabled is read-only property, cannot be set."""
        cfg = InferenceConfig()
        with pytest.raises(AttributeError):
            cfg.timeline_enabled = False


class TestBUG8Fix:
    """BUG-8: truncate_at_act_horizon defaults True, can be disabled."""

    def test_default_true(self):
        cfg = InferenceConfig()
        assert cfg.truncate_at_act_horizon is True

    def test_can_disable(self):
        cfg = InferenceConfig(truncate_at_act_horizon=False)
        assert cfg.truncate_at_act_horizon is False


class TestGAP2Fix:
    """GAP-2: teleop_scale fixed to 1.0."""

    def test_teleop_scale_fixed(self):
        cfg = InferenceConfig()
        assert cfg.teleop_scale == 1.0

    def test_teleop_scale_readonly(self):
        cfg = InferenceConfig()
        with pytest.raises(AttributeError):
            cfg.teleop_scale = 0.5


class TestSerialization:
    def test_to_dict(self):
        cfg = InferenceConfig(robot_ip='1.2.3.4', control_freq=100)
        d = cfg.to_dict()
        assert d['robot_ip'] == '1.2.3.4'
        assert d['control_freq'] == 100
        # Derived properties included
        assert d['timeline_enabled'] is True
        assert d['teleop_scale'] == 1.0

    def test_from_dict(self):
        d = {'robot_ip': '5.6.7.8', 'control_freq': 200, 'unknown_key': 42}
        cfg = InferenceConfig.from_dict(d)
        assert cfg.robot_ip == '5.6.7.8'
        assert cfg.control_freq == 200

    def test_from_dict_ignores_unknown(self):
        d = {'completely_unknown': True}
        cfg = InferenceConfig.from_dict(d)  # should not raise
        assert cfg.robot_ip == '10.42.0.101'  # default

    def test_from_argparse(self):
        ns = mock.MagicMock()
        ns.__dict__ = {'robot_ip': '9.9.9.9', 'act_horizon': 4}
        cfg = InferenceConfig.from_argparse(ns)
        assert cfg.robot_ip == '9.9.9.9'
        assert cfg.act_horizon == 4

    def test_roundtrip(self):
        cfg = InferenceConfig(act_horizon=12, num_inference_steps=5)
        d = cfg.to_dict()
        cfg2 = InferenceConfig.from_dict(d)
        assert cfg2.act_horizon == 12
        assert cfg2.num_inference_steps == 5


class TestNormalization:
    def test_normalize_camera_topics_string(self):
        cfg = InferenceConfig(camera_topics='/cam1,/cam2')
        cfg.normalize_camera_topics()
        assert cfg.camera_topics == ['/cam1', '/cam2']

    def test_normalize_camera_topics_list(self):
        cfg = InferenceConfig(camera_topics=['/cam1'])
        cfg.normalize_camera_topics()
        assert cfg.camera_topics == ['/cam1']

    def test_normalize_arm_init_pose_string(self):
        cfg = InferenceConfig(arm_init_pose='0.1 0.2 0.3 0.4 0.5 0.6 0.7')
        cfg.normalize_arm_init_pose()
        assert cfg.arm_init_pose == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    def test_normalize_arm_init_gripper_string(self):
        cfg = InferenceConfig(arm_init_gripper='0.05')
        cfg.normalize_arm_init_pose()
        assert cfg.arm_init_gripper == 0.05


class TestSafetyConfig:
    def test_resolve_default(self, tmp_path):
        cfg = InferenceConfig(safety_config='')
        # Create a fake safety config
        fake_root = str(tmp_path)
        safety_file = tmp_path / 'safety_config.json'
        safety_file.write_text('{}')
        result = cfg.resolve_safety_config(fake_root)
        assert result == str(safety_file)

    def test_resolve_explicit(self, tmp_path):
        safety_file = tmp_path / 'custom_safety.json'
        safety_file.write_text('{}')
        cfg = InferenceConfig(safety_config=str(safety_file))
        result = cfg.resolve_safety_config('/unused')
        assert result == str(safety_file)
