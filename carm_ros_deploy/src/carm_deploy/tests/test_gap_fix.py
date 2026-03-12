"""Tests for GAP-1/2/3 fix: teleop → record → dataset → inference data flow alignment.

Verifies mathematical consistency of the full pipeline WITHOUT hardware dependencies.
All tests use synthetic data to mock the teleop → training → inference chain.
"""

import numpy as np
import pytest
import tempfile
import os
import h5py

from scipy.spatial.transform import Rotation as R

from rlft.utils.pose_utils import (
    pose_to_transform_matrix,
    transform_matrix_to_pose,
    compute_relative_pose_transform,
    apply_relative_transform,
    apply_teleop_scale,
    quaternion_slerp,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _random_pose():
    """Return a random valid pose [x, y, z, qx, qy, qz, qw]."""
    pos = np.random.uniform(-0.5, 0.5, 3)
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    return np.concatenate([pos, q])


def _pose_near(a, b, pos_atol=1e-6, rot_atol=1e-6):
    """Check if two poses are approximately equal (handles quaternion sign)."""
    np.testing.assert_allclose(a[:3], b[:3], atol=pos_atol)
    # Quaternion may have flipped sign
    q_a, q_b = a[3:7], b[3:7]
    if np.dot(q_a, q_b) < 0:
        q_b = -q_b
    np.testing.assert_allclose(q_a, q_b, atol=rot_atol)


def _simulate_backend_compute_control(
    init_vr_pose_4x4: np.ndarray,
    curr_vr_pose_4x4: np.ndarray,
    init_arm_pose_7: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Simulate backend's compute_control() to produce target_end_arm_pose.

    Replicates the backend logic:
        delta_pos, delta_quat = homogeneous_diff_with_scale(init_vr, curr_vr, scale)
        target = apply_transform_to_pose(init_arm, delta_pos, delta_quat)

    Uses LEFT-multiply rotation convention (world frame) as in the backend.
    """
    # Position delta (scaled)
    pos_diff = (curr_vr_pose_4x4[:3, 3] - init_vr_pose_4x4[:3, 3]) * scale

    # Rotation delta (world frame left-multiply, then slerp for scaling)
    R1 = init_vr_pose_4x4[:3, :3]
    R2 = curr_vr_pose_4x4[:3, :3]
    R_diff = R2 @ R1.T
    identity_q = np.array([0, 0, 0, 1.0])
    full_q = R.from_matrix(R_diff).as_quat()
    scaled_q = quaternion_slerp(identity_q, full_q, scale)

    # Apply to arm pose (LEFT-multiply rotation, add position)
    new_pos = init_arm_pose_7[:3] + pos_diff
    diff_rot = R.from_quat(scaled_q)
    cur_rot = R.from_quat(init_arm_pose_7[3:7])
    new_rot = diff_rot * cur_rot  # scipy: left-multiply
    new_quat = new_rot.as_quat()

    return np.concatenate([new_pos, new_quat])


def _pose_to_4x4(pose_7: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,qx,qy,qz,qw] to 4x4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, 3] = pose_7[:3]
    T[:3, :3] = R.from_quat(pose_7[3:7]).as_matrix()
    return T


# ── Test: Full data flow roundtrip ───────────────────────────────────────────


class TestDataFlowAlignment:
    """End-to-end data flow: teleop → record → dataset → inference → execute."""

    def test_identity_roundtrip(self):
        """When target_pose == obs_pose, relative_pose should be identity."""
        obs_pose = _random_pose()
        target_pose = obs_pose.copy()

        # Dataset: compute relative
        rel = compute_relative_pose_transform(obs_pose, target_pose)

        # rel should be identity: [0,0,0, 0,0,0,1]
        np.testing.assert_allclose(rel[:3], 0, atol=1e-7)
        assert abs(abs(rel[6]) - 1.0) < 1e-6

        # Inference: apply relative
        recovered = apply_relative_transform(rel, obs_pose)
        _pose_near(recovered, target_pose)

    def test_known_translation_roundtrip(self):
        """Known translation: full roundtrip should recover target."""
        obs_pose = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
        target_pose = np.array([0.35, 0.02, 0.22, 0.0, 0.0, 0.0, 1.0])

        # Dataset: compute relative
        rel = compute_relative_pose_transform(obs_pose, target_pose)

        # Inference: apply relative
        recovered = apply_relative_transform(rel, obs_pose)
        _pose_near(recovered, target_pose)

    def test_known_rotation_roundtrip(self):
        """Known rotation: full roundtrip should recover target."""
        obs_pose = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
        # Target: 30 degree rotation around Z
        rot_z = R.from_euler('z', 30, degrees=True).as_quat()
        target_pose = np.concatenate([[0.3, 0.0, 0.2], rot_z])

        # Dataset: compute relative
        rel = compute_relative_pose_transform(obs_pose, target_pose)

        # Inference: apply relative
        recovered = apply_relative_transform(rel, obs_pose)
        _pose_near(recovered, target_pose)

    def test_non_identity_orientation_roundtrip(self):
        """When obs has non-identity orientation, roundtrip still works."""
        for _ in range(50):
            obs_pose = _random_pose()
            target_pose = _random_pose()

            rel = compute_relative_pose_transform(obs_pose, target_pose)
            recovered = apply_relative_transform(rel, obs_pose)
            _pose_near(recovered, target_pose)

    def test_scale_not_double_applied(self):
        """Scale is embedded in target_pose from backend, so inference should NOT re-scale.

        Pipeline:
        1. Backend: scale=0.4 → target_end_arm_pose (already scaled)
        2. Record: action = target_end_arm_pose
        3. Dataset: rel = inv(T_obs) @ T_target
        4. Inference: T_target_recovered = T_obs @ T_rel  (no extra scale)
        5. Verify: T_target_recovered ≈ target_end_arm_pose
        """
        # Simulate: VR moves 10cm in x
        init_vr = np.eye(4)
        init_vr[:3, 3] = [0, 0, 0]

        curr_vr = np.eye(4)
        curr_vr[:3, 3] = [0.1, 0, 0]  # 10cm x-movement

        init_arm_pose = np.array([0.3, 0.0, 0.2, 0, 0, 0, 1.0])
        scale = 0.4

        # Backend produces target (scaled 0.4 → 4cm movement)
        target_pose = _simulate_backend_compute_control(
            init_vr, curr_vr, init_arm_pose, scale
        )
        # Verify: arm moved ~4cm in x
        np.testing.assert_allclose(target_pose[:3], [0.34, 0.0, 0.2], atol=1e-7)

        # Record: action = target_pose (8D with gripper)
        action = np.concatenate([target_pose, [0.05]])  # 8D

        # Dataset: obs at current actual pose (assume = init_arm for simplicity)
        obs_pose = init_arm_pose
        rel = compute_relative_pose_transform(obs_pose, action[:7])

        # Inference: apply relative with NO extra scaling (teleop_scale=1.0)
        recovered = apply_relative_transform(rel, obs_pose)
        _pose_near(recovered, target_pose)

        # WRONG: if we applied scale=0.4 AGAIN, target would be different
        rel_scaled = apply_teleop_scale(rel, 0.4)
        recovered_wrong = apply_relative_transform(rel_scaled, obs_pose)
        # This should NOT match the original target (double-scaled)
        with pytest.raises(AssertionError):
            _pose_near(recovered_wrong, target_pose)

    def test_backend_left_multiply_absorbed(self):
        """GAP-3: Backend uses left-multiply rotation, but recording absolute
        target_pose and computing relative via SE(3) inverse absorbs this difference.

        The key insight: we don't need to know HOW the target was computed
        (left-multiply vs right-multiply), because we directly record the
        resulting absolute target pose and compute the relative transformation
        using the standard SE(3) convention.
        """
        for _ in range(20):
            # Random initial arm orientation (non-identity)
            init_arm = _random_pose()
            # Random VR poses
            init_vr = _pose_to_4x4(_random_pose())
            curr_vr = _pose_to_4x4(_random_pose())

            # Backend computes target via LEFT-multiply convention
            target = _simulate_backend_compute_control(
                init_vr, curr_vr, init_arm, scale=0.4
            )

            # At recording time, obs = actual arm state (may differ from init due to tracking)
            # For this test, assume obs = init_arm (worst case for convention difference)
            obs = init_arm.copy()

            # Dataset: compute relative using SE(3) RIGHT-multiply convention
            rel = compute_relative_pose_transform(obs, target)

            # Inference: apply using same RIGHT-multiply convention
            recovered = apply_relative_transform(rel, obs)

            # Should match the original target
            _pose_near(recovered, target)


# ── Test: HDF5 v2 format ─────────────────────────────────────────────────────


class TestHDF5FormatV2:
    """Verify new HDF5 format (v2) write/read roundtrip."""

    def _create_v2_hdf5(self, filepath: str, num_steps: int = 10):
        """Create a v2 format HDF5 file with synthetic data."""
        with h5py.File(filepath, 'w') as f:
            obs = f.create_group('observations')

            # Images: [T, H, W, C]
            images = np.random.randint(0, 255, (num_steps, 240, 320, 3), dtype=np.uint8)
            obs.create_dataset('images', data=images, compression='gzip')

            # qpos_joint: [T, 7] (6 joints + 1 gripper)
            qpos_joint = np.random.randn(num_steps, 7).astype(np.float64)
            obs.create_dataset('qpos_joint', data=qpos_joint)

            # qpos_end: [T, 8] (7 ee_pose + 1 gripper)
            qpos_end = np.random.randn(num_steps, 8).astype(np.float64)
            # Normalize quaternions
            for t in range(num_steps):
                q = qpos_end[t, 3:7]
                qpos_end[t, 3:7] = q / np.linalg.norm(q)
            obs.create_dataset('qpos_end', data=qpos_end)

            obs.create_dataset('qpos', data=np.zeros((num_steps, 15)))
            obs.create_dataset('gripper', data=qpos_joint[:, -1])
            obs.create_dataset('timestamps', data=np.arange(num_steps, dtype=np.float64))

            # Action: [T, 8] = target_pose(7) + gripper(1)
            action = np.random.randn(num_steps, 8).astype(np.float64)
            for t in range(num_steps):
                q = action[t, 3:7]
                action[t, 3:7] = q / np.linalg.norm(q)
            f.create_dataset('action', data=action)

            # Teleop scale: [T]
            teleop_scale = np.full(num_steps, 0.4)
            f.create_dataset('teleop_scale', data=teleop_scale)

            # Metadata
            f.attrs['num_steps'] = num_steps
            f.attrs['record_freq'] = 30
            f.attrs['image_width'] = 320
            f.attrs['image_height'] = 240
            f.attrs['data_version'] = 'v2'

        return qpos_end, action

    def test_write_read_roundtrip(self):
        """Write v2 format, read back, verify fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'episode_0001_test.hdf5')
            qpos_end_orig, action_orig = self._create_v2_hdf5(filepath)

            with h5py.File(filepath, 'r') as f:
                assert f.attrs['data_version'] == 'v2'
                assert f['action'].shape == (10, 8)
                assert f['teleop_scale'].shape == (10,)

                action_loaded = np.array(f['action'])
                np.testing.assert_allclose(action_loaded, action_orig)

    def test_data_version_tag(self):
        """Verify data_version='v2' attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'episode_0001_test.hdf5')
            self._create_v2_hdf5(filepath)

            with h5py.File(filepath, 'r') as f:
                assert f.attrs.get('data_version', 'v1') == 'v2'

    def test_action_dim_is_8(self):
        """Action should be 8D in v2 format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'episode_0001_test.hdf5')
            self._create_v2_hdf5(filepath)

            with h5py.File(filepath, 'r') as f:
                assert f['action'].shape[-1] == 8

    def test_relative_pose_from_v2_action(self):
        """Compute relative pose from v2 action and verify roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'episode_0001_test.hdf5')
            qpos_end, action = self._create_v2_hdf5(filepath)

            for t in range(len(action)):
                obs_pose = qpos_end[t, :7]
                target_pose = action[t, :7]  # v2: first 7 elements

                rel = compute_relative_pose_transform(obs_pose, target_pose)
                recovered = apply_relative_transform(rel, obs_pose)
                _pose_near(recovered, target_pose)


# ── Test: Backend API mock format ─────────────────────────────────────────────


class TestBackendAPIMock:
    """Verify expected format of backend /api/joystick/teleop_target response."""

    def _mock_active_response(self) -> dict:
        """Mock an active teleop response."""
        return {
            'target_pose': [0.3, 0.01, 0.22, 0.0, 0.0, 0.0, 1.0],
            'gripper_pose': 0.05,
            'gripper_tau': 10.0,
            'scale': 0.4,
            'active': True,
        }

    def _mock_inactive_response(self) -> dict:
        """Mock an inactive teleop response (clutch released)."""
        return {
            'target_pose': None,
            'gripper_pose': None,
            'gripper_tau': None,
            'scale': 0.4,
            'active': False,
        }

    def test_active_response_format(self):
        """Active response should have valid target_pose."""
        resp = self._mock_active_response()
        assert resp['active'] is True
        assert len(resp['target_pose']) == 7
        assert resp['gripper_pose'] is not None
        assert 0 < resp['scale'] <= 1.0

    def test_inactive_response_format(self):
        """Inactive response should have target_pose=None."""
        resp = self._mock_inactive_response()
        assert resp['active'] is False
        assert resp['target_pose'] is None

    def test_action_from_active_response(self):
        """Construct 8D action from active response."""
        resp = self._mock_active_response()
        target = resp['target_pose']
        gripper = resp['gripper_pose']

        action = np.array(target + [gripper], dtype=np.float64)
        assert action.shape == (8,)
        np.testing.assert_allclose(action[:7], target)
        assert action[7] == gripper


# ── Test: Teleop scale semantics ──────────────────────────────────────────────


class TestTeleopScaleSemantics:
    """Verify correct scale semantics across the pipeline."""

    def test_apply_teleop_scale_identity(self):
        """scale=1.0 should be a no-op."""
        delta = np.array([0.05, 0.03, 0.01, 0, 0, 0, 1.0])
        scaled = apply_teleop_scale(delta, 1.0)
        np.testing.assert_allclose(scaled, delta, atol=1e-7)

    def test_scale_halves_translation(self):
        """scale=0.5 should halve translation."""
        delta = np.array([0.1, 0.2, 0.3, 0, 0, 0, 1.0])
        scaled = apply_teleop_scale(delta, 0.5)
        np.testing.assert_allclose(scaled[:3], [0.05, 0.1, 0.15])

    def test_scale_consistency_between_backend_and_pose_utils(self):
        """Backend's homogeneous_diff_with_scale and pose_utils.apply_teleop_scale
        should produce equivalent scaling for pure translations."""
        # Pure translation delta
        init = np.eye(4)
        curr = np.eye(4)
        curr[:3, 3] = [0.1, 0.2, 0.3]
        scale = 0.4

        # Backend-style: compute diff with scale
        pos_diff = (curr[:3, 3] - init[:3, 3]) * scale
        # Result: [0.04, 0.08, 0.12]

        # pose_utils-style: compute full diff then scale
        full_diff = curr[:3, 3] - init[:3, 3]
        delta_pose = np.concatenate([full_diff, [0, 0, 0, 1.0]])
        scaled = apply_teleop_scale(delta_pose, scale)

        np.testing.assert_allclose(scaled[:3], pos_diff, atol=1e-7)
