"""Tests for core.safety_controller — SafetyController."""

import json
import importlib
import numpy as np
import pytest

import sys, os
_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CARM_DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, _CARM_DEPLOY_ROOT)

# Import safety_controller directly to avoid core/__init__.py pulling in env_ros
# which depends on the CARM SDK hardware library.
import importlib.util
_sc_path = os.path.join(_CARM_DEPLOY_ROOT, "core", "safety_controller.py")
_spec = importlib.util.spec_from_file_location("safety_controller", _sc_path)
_sc_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sc_mod)

SafetyController = _sc_mod.SafetyController
JointLimits = _sc_mod.JointLimits
WorkspaceLimits = _sc_mod.WorkspaceLimits
SafetyParams = _sc_mod.SafetyParams
get_safe_joint_limits = _sc_mod.get_safe_joint_limits
get_safe_gripper_limits = _sc_mod.get_safe_gripper_limits


class TestJointLimits:
    def test_default_limits_have_correct_length(self):
        jl = JointLimits()
        assert len(jl.joint_min) == 6
        assert len(jl.joint_max) == 6

    def test_safe_limits_within_physical(self):
        jmin, jmax = get_safe_joint_limits(0.1)
        # safe range should be tighter
        for i in range(6):
            assert jmin[i] >= -3.5  # physical limits are around ±3.14
            assert jmax[i] <= 3.5

    def test_gripper_limits_positive(self):
        gmin, gmax = get_safe_gripper_limits(0.1)
        assert gmin > 0
        assert gmax < 0.08
        assert gmin < gmax


class TestCheckJointLimits:
    def test_safe_action_unchanged(self):
        sc = SafetyController()
        action = np.array([0.0, 1.0, -1.0, 0.0, 0.5, 0.0, 0.04])
        clipped, warnings = sc.check_joint_limits(action)
        np.testing.assert_array_equal(clipped, action)
        assert len(warnings) == 0

    def test_clipping(self):
        sc = SafetyController()
        # Joint 0 outside limits
        action = np.array([10.0, 1.0, -1.0, 0.0, 0.5, 0.0, 0.04])
        clipped, warnings = sc.check_joint_limits(action)
        assert clipped[0] == sc.joint_limits.joint_max[0]
        assert len(warnings) > 0

    def test_gripper_clip_low(self):
        sc = SafetyController()
        action = np.array([0.0] * 6 + [-0.1])
        clipped, warnings = sc.check_joint_limits(action)
        assert clipped[6] >= sc.joint_limits.gripper_min


class TestClipActionDelta:
    def test_no_clip_small_delta(self):
        sc = SafetyController(safety_params=SafetyParams(max_joint_delta=0.2, max_gripper_delta=0.05))
        action = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.03])
        current = np.zeros(7)
        clipped, warnings = sc.clip_action_delta(action, current)
        np.testing.assert_array_almost_equal(clipped, action)

    def test_clips_large_delta(self):
        sc = SafetyController(safety_params=SafetyParams(max_joint_delta=0.05))
        action = np.array([1.0] + [0.0] * 6)
        current = np.zeros(7)
        clipped, warnings = sc.clip_action_delta(action, current)
        assert abs(clipped[0] - current[0]) <= 0.05 + 1e-8
        assert len(warnings) > 0

    def test_none_current_state(self):
        sc = SafetyController()
        action = np.array([10.0] * 7)
        clipped, warnings = sc.clip_action_delta(action, None)
        np.testing.assert_array_equal(clipped, action)
        assert len(warnings) == 0


class TestCheckWorkspace:
    def test_inside_workspace(self):
        sc = SafetyController()
        pose = np.array([0.3, 0.0, 0.2, 0, 0, 0, 1.0])
        clipped, warnings = sc.check_workspace(pose)
        np.testing.assert_array_equal(clipped, pose)

    def test_clips_xyz(self):
        sc = SafetyController(workspace_limits=WorkspaceLimits(
            x_min=0.1, x_max=0.5, y_min=-0.3, y_max=0.3, z_min=0.05, z_max=0.4,
        ))
        pose = np.array([0.0, 0.5, 0.6, 0, 0, 0, 1.0])  # x below, y above, z above
        clipped, warnings = sc.check_workspace(pose)
        assert clipped[0] == 0.1
        assert clipped[1] == 0.3
        assert clipped[2] == 0.4
        assert len(warnings) == 3


class TestLowPassFilter:
    def test_constant_input_converges(self):
        sc = SafetyController(safety_params=SafetyParams(filter_alpha=0.3))
        for _ in range(50):
            out = sc.apply_low_pass_filter(np.ones(7))
        np.testing.assert_allclose(out, np.ones(7), atol=1e-6)

    def test_first_call_passthrough(self):
        sc = SafetyController()
        action = np.array([1.0, 2.0, 3.0])
        out = sc.apply_low_pass_filter(action)
        np.testing.assert_array_equal(out, action)

    def test_step_response_damped(self):
        sc = SafetyController(safety_params=SafetyParams(filter_alpha=0.3))
        sc.apply_low_pass_filter(np.zeros(3))  # init
        out = sc.apply_low_pass_filter(np.ones(3))
        # Should be < 1 due to filtering
        assert np.all(out < 1.0)


class TestCheckAndClip:
    def test_full_pipeline(self):
        sc = SafetyController()
        action = np.array([0.0, 1.0, -0.5, 0.0, 0.5, 0.0, 0.04])
        current = np.zeros(7)
        safe, warnings = sc.check_and_clip(action, current)
        assert safe is not None
        assert sc.stats['total_checks'] == 1

    def test_stats_increment(self):
        sc = SafetyController()
        for _ in range(3):
            sc.check_and_clip(np.zeros(7), np.zeros(7))
        assert sc.stats['total_checks'] == 3


class TestReset:
    def test_reset_clears_filter(self):
        sc = SafetyController()
        sc.apply_low_pass_filter(np.ones(7))
        sc.reset()
        assert sc.prev_action is None


class TestSerializationRoundtrip:
    def test_save_load(self, tmp_path):
        sc = SafetyController(
            joint_limits=JointLimits(),
            workspace_limits=WorkspaceLimits(x_min=0.15, x_max=0.45),
            safety_params=SafetyParams(max_joint_delta=0.08, filter_alpha=0.25),
        )
        path = str(tmp_path / "safety.json")
        sc.save_config(path)

        sc2 = SafetyController.from_config(path)
        np.testing.assert_allclose(sc2.joint_limits.joint_min, sc.joint_limits.joint_min)
        assert sc2.workspace_limits.x_min == 0.15
        assert sc2.params.max_joint_delta == 0.08
        assert sc2.params.filter_alpha == 0.25
