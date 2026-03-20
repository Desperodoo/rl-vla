"""Tests for inference.action_processor."""

import sys
import os
import types
import numpy as np
import pytest
from unittest import mock

# Path setup
_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RL_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CARM_DEPLOY_ROOT)))
for p in (_CARM_DEPLOY_ROOT, _RL_VLA_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Mock rospy
if "rospy" not in sys.modules:
    _rospy_mock = types.ModuleType("rospy")
    _rospy_mock.loginfo = lambda *a, **kw: None
    _rospy_mock.logwarn = lambda *a, **kw: None
    _rospy_mock.logerr = lambda *a, **kw: None
    sys.modules["rospy"] = _rospy_mock

from inference.action_processor import ActionIndices, ActionProcessor, SafetyResult


# ═══════════════════════════════════════════════════════════════════════════
# ActionIndices
# ═══════════════════════════════════════════════════════════════════════════

class TestActionIndices:
    def test_full_mode_15d(self):
        idx = ActionIndices.from_action_dim(15)
        assert idx.is_full_mode is True
        assert idx.rel_pose_start == 7
        assert idx.rel_pose_end == 14
        assert idx.gripper_idx == 14

    def test_ee_mode_8d(self):
        idx = ActionIndices.from_action_dim(8)
        assert idx.is_full_mode is False
        assert idx.rel_pose_start == 0
        assert idx.rel_pose_end == 7
        assert idx.gripper_idx == 7

    def test_frozen(self):
        idx = ActionIndices.from_action_dim(15)
        with pytest.raises(Exception):
            idx.is_full_mode = False


# ═══════════════════════════════════════════════════════════════════════════
# ActionProcessor — speed scaling
# ═══════════════════════════════════════════════════════════════════════════

def _make_processor(
    action_dim_full=8,
    speed_scale=1.0,
    check_workspace=False,
    max_trans=0.1,
):
    """Create an ActionProcessor with a mocked safety controller."""
    safety = mock.MagicMock()
    safety.joint_limits = mock.MagicMock()
    safety.joint_limits.gripper_min = 0.0
    safety.joint_limits.gripper_max = 0.08
    safety.check_workspace = mock.MagicMock(return_value=(None, []))
    return ActionProcessor(
        action_dim_full=action_dim_full,
        safety_controller=safety,
        inference_speed_scale=speed_scale,
        check_workspace=check_workspace,
        max_relative_translation=max_trans,
    )


class TestSpeedScale:
    def test_no_scale(self):
        proc = _make_processor(speed_scale=1.0)
        actions = np.random.randn(4, 8)
        result = proc.apply_speed_scale(actions)
        np.testing.assert_array_equal(result, actions)

    def test_scale_modifies_rel_pose(self):
        proc = _make_processor(speed_scale=0.5)
        actions = np.ones((2, 8))
        result = proc.apply_speed_scale(actions)
        # Rel pose columns (0:7 for 8D mode) should be scaled, gripper (7) untouched
        # apply_teleop_scale is called on each row's rel_pose, which applies the scale
        # We just verify the result differs from input
        assert not np.array_equal(result[:, :7], actions[:, :7])


# ═══════════════════════════════════════════════════════════════════════════
# ActionProcessor — safety checks (BUG-3 fix)
# ═══════════════════════════════════════════════════════════════════════════

class TestSafetyChecks:
    def test_gripper_clip_direct(self):
        """BUG-3 fix: gripper limits checked directly, not via dummy joints."""
        proc = _make_processor(check_workspace=False)
        # Gripper value exceeds max (0.08)
        actions = np.zeros((1, 8))
        actions[0, 7] = 0.15  # exceeds gripper_max=0.08

        # Mock apply_relative_transform to return a valid pose
        with mock.patch(
            'inference.action_processor.apply_relative_transform',
            return_value=[0.25, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0, 0.08],
        ):
            result = proc.apply_safety_checks(actions, [0.25, 0.0, 0.33, 0, 0, 0, 1, 0.05])

        assert result.clipped is True
        assert result.actions[0, 7] == pytest.approx(0.08)
        assert any('Gripper clipped' in e for e in result.events)

    def test_gripper_below_min(self):
        proc = _make_processor(check_workspace=False)
        actions = np.zeros((1, 8))
        actions[0, 7] = -0.01  # below gripper_min=0.0

        with mock.patch(
            'inference.action_processor.apply_relative_transform',
            return_value=[0.25, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0, 0.0],
        ):
            result = proc.apply_safety_checks(actions, [0.25, 0.0, 0.33, 0, 0, 0, 1, 0.05])

        assert result.clipped is True
        assert result.actions[0, 7] == pytest.approx(0.0)

    def test_translation_clamped(self):
        proc = _make_processor(check_workspace=False, max_trans=0.05)
        actions = np.zeros((1, 8))
        actions[0, 0:3] = [0.1, 0.0, 0.0]  # 0.1m > max 0.05m

        with mock.patch(
            'inference.action_processor.apply_relative_transform',
            return_value=[0.25, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0, 0.0],
        ):
            result = proc.apply_safety_checks(actions, [0.25, 0.0, 0.33, 0, 0, 0, 1, 0.05])

        assert result.clipped is True
        trans_norm = np.linalg.norm(result.actions[0, 0:3])
        assert trans_norm <= 0.05 + 1e-6
        assert any('Translation scaled' in e for e in result.events)

    def test_no_clip_normal(self):
        proc = _make_processor(check_workspace=False, max_trans=1.0)
        actions = np.zeros((1, 8))
        actions[0, 0:3] = [0.01, 0.01, 0.01]  # small
        actions[0, 7] = 0.04  # normal gripper

        with mock.patch(
            'inference.action_processor.apply_relative_transform',
            return_value=[0.25, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0, 0.04],
        ):
            result = proc.apply_safety_checks(actions, [0.25, 0.0, 0.33, 0, 0, 0, 1, 0.04])

        assert result.clipped is False
        assert len(result.events) == 0


# ═══════════════════════════════════════════════════════════════════════════
# ActionProcessor — convert_to_absolute
# ═══════════════════════════════════════════════════════════════════════════

class TestConvertToAbsolute:
    def test_output_shape(self):
        proc = _make_processor(action_dim_full=8)
        actions = np.random.randn(4, 8)
        qpos_end = [0.25, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0, 0.05]

        with mock.patch(
            'inference.action_processor.apply_relative_transform',
            return_value=[0.25, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0, 0.05],
        ):
            result = proc.convert_to_absolute(actions, qpos_end)

        assert result.shape == (4, 8)

    def test_calls_apply_relative_transform(self):
        proc = _make_processor(action_dim_full=8)
        actions = np.random.randn(2, 8)
        qpos_end = [0.25, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0, 0.05]

        with mock.patch(
            'inference.action_processor.apply_relative_transform',
            return_value=[0.26, 0.01, 0.34, 0.0, 0.0, 0.0, 1.0, 0.04],
        ) as mock_art:
            proc.convert_to_absolute(actions, qpos_end)

        assert mock_art.call_count == 2  # once per action step
