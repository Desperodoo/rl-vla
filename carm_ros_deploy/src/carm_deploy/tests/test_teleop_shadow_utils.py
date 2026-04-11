"""Tests for teleop shadow utility functions."""

import numpy as np

from data.teleop_shadow_utils import (
    build_hold_target_chunk,
    compute_learning_level_chunk,
    extract_processed_target_abs,
    quaternion_angle_distance,
)


class TestTeleopShadowUtils:
    def test_build_hold_target_chunk(self):
        target = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.05], dtype=np.float64)
        chunk = build_hold_target_chunk(target, pred_horizon=4)
        assert chunk.shape == (4, 8)
        np.testing.assert_allclose(chunk[0], target)
        np.testing.assert_allclose(chunk[-1], target)

    def test_extract_processed_target_abs_active(self):
        state = {
            "processed": {
                "target_pose_abs": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
                "gripper_pose": 0.04,
                "active": True,
            }
        }
        target = extract_processed_target_abs(state)
        np.testing.assert_allclose(target, [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.04])

    def test_extract_processed_target_abs_inactive(self):
        state = {"processed": {"target_pose_abs": [1, 2, 3, 0, 0, 0, 1], "active": False}}
        assert extract_processed_target_abs(state) is None

    def test_compute_learning_level_chunk_reconstructs(self):
        ref_pose = np.array([0.25, -0.01, 0.32, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        target_pose_abs = np.array([0.3, 0.02, 0.28, 0.0, 0.0, 0.0, 1.0, 0.05], dtype=np.float64)
        result = compute_learning_level_chunk(ref_pose, target_pose_abs, pred_horizon=6)

        assert result["human_chunk_abs"].shape == (6, 8)
        assert result["human_chunk_rel"].shape == (6, 7)
        assert result["reconstructed_target_abs"].shape == (6, 8)
        np.testing.assert_allclose(result["human_chunk_abs"][0], target_pose_abs)
        np.testing.assert_allclose(result["reconstructed_target_abs"][0], target_pose_abs, atol=1e-6)
        assert result["abs_reconstruction_pos_error"] < 1e-8
        assert result["abs_reconstruction_rot_error"] < 1e-8

    def test_quaternion_angle_distance_identity(self):
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        assert quaternion_angle_distance(q, q) == 0.0
