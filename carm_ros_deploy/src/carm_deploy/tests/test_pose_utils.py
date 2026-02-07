"""Tests for rlft.utils.pose_utils — SE(3) transformation utilities."""

import numpy as np
import pytest


from rlft.utils.pose_utils import (
    pose_to_transform_matrix,
    transform_matrix_to_pose,
    compute_relative_pose_transform,
    apply_relative_transform,
    quaternion_slerp,
    apply_teleop_scale,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _random_pose():
    """Return a random valid pose [x, y, z, qx, qy, qz, qw]."""
    pos = np.random.randn(3)
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    return np.concatenate([pos, q])


# ── pose ↔ matrix roundtrip ──────────────────────────────────────────────────

class TestPoseMatrixRoundtrip:
    def test_identity(self):
        pos = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        T = pose_to_transform_matrix(pos, quat)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-7)

    def test_roundtrip(self):
        for _ in range(20):
            pose = _random_pose()
            T = pose_to_transform_matrix(pose[:3], pose[3:])
            pos2, q2 = transform_matrix_to_pose(T)
            # position exact
            np.testing.assert_allclose(pos2, pose[:3], atol=1e-7)
            # quaternion may flip sign
            if np.dot(q2, pose[3:]) < 0:
                q2 = -q2
            np.testing.assert_allclose(q2, pose[3:], atol=1e-7)

    def test_matrix_shape(self):
        T = pose_to_transform_matrix(np.zeros(3), np.array([0, 0, 0, 1.0]))
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])


# ── relative transform roundtrip ────────────────────────────────────────────

class TestRelativeTransform:
    def test_roundtrip_identity(self):
        """relative(A, A) should be identity."""
        pose = _random_pose()
        rel = compute_relative_pose_transform(pose, pose)
        np.testing.assert_allclose(rel[:3], 0, atol=1e-7)
        # quaternion ≈ identity (0,0,0,1) or its negative
        assert abs(abs(rel[6]) - 1.0) < 1e-6

    def test_apply_recovers_target(self):
        """apply_relative_transform(relative(A,B), A) ≈ B"""
        for _ in range(20):
            A = _random_pose()
            B = _random_pose()
            rel = compute_relative_pose_transform(A, B)
            recovered = apply_relative_transform(rel, A)
            np.testing.assert_allclose(recovered[:3], B[:3], atol=1e-6)
            if np.dot(recovered[3:], B[3:]) < 0:
                recovered[3:] = -recovered[3:]
            np.testing.assert_allclose(recovered[3:], B[3:], atol=1e-6)

    def test_apply_with_gripper(self):
        A = _random_pose()
        B = _random_pose()
        rel = compute_relative_pose_transform(A, B)
        result = apply_relative_transform(rel, A, gripper=0.05)
        assert result.shape == (8,)
        assert result[7] == pytest.approx(0.05)


# ── quaternion_slerp ─────────────────────────────────────────────────────────

class TestQuaternionSlerp:
    def test_endpoints(self):
        q0 = np.array([0, 0, 0, 1.0])
        q1 = np.array([0, 0, 1, 0.0])
        np.testing.assert_allclose(quaternion_slerp(q0, q1, 0.0), q0, atol=1e-7)
        np.testing.assert_allclose(quaternion_slerp(q0, q1, 1.0), q1, atol=1e-7)

    def test_midpoint_unit(self):
        q0 = np.array([0, 0, 0, 1.0])
        q1 = np.array([0, 0, 1, 0.0])
        mid = quaternion_slerp(q0, q1, 0.5)
        assert np.linalg.norm(mid) == pytest.approx(1.0, abs=1e-7)

    def test_identical_quaternions(self):
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        result = quaternion_slerp(q, q, 0.5)
        np.testing.assert_allclose(result, q, atol=1e-7)


# ── apply_teleop_scale ───────────────────────────────────────────────────────

class TestApplyTeleopScale:
    def test_scale_one(self):
        delta = _random_pose()
        delta[3:] /= np.linalg.norm(delta[3:])
        scaled = apply_teleop_scale(delta, 1.0)
        np.testing.assert_allclose(scaled, delta, atol=1e-7)

    def test_scale_zero_position(self):
        delta = _random_pose()
        delta[3:] /= np.linalg.norm(delta[3:])
        scaled = apply_teleop_scale(delta, 0.0)
        np.testing.assert_allclose(scaled[:3], 0, atol=1e-7)

    def test_scale_shrinks_translation(self):
        delta = np.array([1.0, 2.0, 3.0, 0, 0, 0, 1.0])
        scaled = apply_teleop_scale(delta, 0.5)
        np.testing.assert_allclose(scaled[:3], [0.5, 1.0, 1.5])

    def test_does_not_modify_input(self):
        delta = np.array([1.0, 0.0, 0.0, 0, 0, 0, 1.0])
        original = delta.copy()
        apply_teleop_scale(delta, 0.5)
        np.testing.assert_array_equal(delta, original)
