"""Tests for teleop bridge shared components."""

import time

import numpy as np

from data.teleop_bridge import TeleopShadowTransformer, TeleopUpperControlBridge


class FakeEnv:
    def __init__(self):
        self.commands = []
        self.qpos_end = np.array([0.25, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0, 0.04], dtype=np.float64)

    def get_state_observation(self):
        return {"qpos_end": self.qpos_end.copy()}

    def end_control_nostep(self, action):
        self.commands.append(np.asarray(action, dtype=np.float64))


class FakeSignalClient:
    def __init__(self, fail_on_upper=False, fail_lower_attempts=0):
        self.calls = []
        self.fail_on_upper = fail_on_upper
        self.fail_lower_attempts = int(fail_lower_attempts)

    def set_control_state(self, local_control_enabled, control_owner, timeout_s=None):
        if self.fail_on_upper and (not bool(local_control_enabled)) and control_owner == "upper_machine":
            raise RuntimeError("failed to acquire upper owner")
        if bool(local_control_enabled) and control_owner == "lower_machine" and self.fail_lower_attempts > 0:
            self.fail_lower_attempts -= 1
            raise RuntimeError("temporary lower owner restore failure")
        self.calls.append((bool(local_control_enabled), control_owner))
        return {
            "local_control_enabled": bool(local_control_enabled),
            "control_owner": control_owner,
            "updated_at": 1.0,
        }


class TestTeleopShadowTransformer:
    def test_build_active_target(self):
        transformer = TeleopShadowTransformer(pred_horizon=4)
        ref_pose = np.array([0.25, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        teleop_state_v2 = {
            "processed": {
                "target_pose_abs": [0.31, 0.02, 0.28, 0.0, 0.0, 0.0, 1.0],
                "gripper_pose": 0.05,
                "active": True,
            }
        }

        result = transformer.build(ref_pose, teleop_state_v2)

        assert result["teleop_valid"] is True
        assert result["processed_target_abs"].shape == (8,)
        assert result["human_chunk_abs"].shape == (4, 8)
        assert result["human_chunk_rel"].shape == (4, 7)
        assert result["reconstructed_target_abs"].shape == (4, 8)
        np.testing.assert_allclose(result["processed_target_abs"], [0.31, 0.02, 0.28, 0.0, 0.0, 0.0, 1.0, 0.05])

    def test_build_inactive_target_uses_zero_placeholders(self):
        transformer = TeleopShadowTransformer(pred_horizon=3)
        ref_pose = np.array([0.25, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        teleop_state_v2 = {
            "processed": {
                "target_pose_abs": [0.31, 0.02, 0.28, 0.0, 0.0, 0.0, 1.0],
                "gripper_pose": 0.05,
                "active": False,
            }
        }

        result = transformer.build(ref_pose, teleop_state_v2)

        assert result["teleop_valid"] is False
        np.testing.assert_allclose(result["processed_target_abs"], np.zeros(8))
        np.testing.assert_allclose(result["human_chunk_abs"], np.zeros((3, 8)))
        np.testing.assert_allclose(result["human_chunk_rel"], np.zeros((3, 7)))


class TestTeleopUpperControlBridge:
    def test_candidate_only_mode_tracks_without_sending(self):
        env = FakeEnv()
        client = FakeSignalClient()
        bridge = TeleopUpperControlBridge(env, client, control_freq=100.0, signal_timeout_ms=150.0, live_enabled=False)
        bridge.start()
        try:
            bridge.update_signal(
                processed_target_abs=np.array([0.30, 0.01, 0.29, 0.0, 0.0, 0.0, 1.0, 0.05]),
                signal_age_ms=20.0,
                teleop_active=True,
                processed_sequence=7,
            )
            bridge.activate_for_recording()
            time.sleep(0.05)
            snapshot = bridge.snapshot()
            assert snapshot["teleop_candidate_applied"] is False
            assert snapshot["teleop_candidate_stale"] is False
            assert len(env.commands) == 0
            np.testing.assert_allclose(snapshot["upper_candidate_target_abs"], [0.30, 0.01, 0.29, 0.0, 0.0, 0.0, 1.0, 0.05])
        finally:
            bridge.stop()

    def test_live_mode_switches_owner_and_sends_commands(self):
        env = FakeEnv()
        client = FakeSignalClient()
        bridge = TeleopUpperControlBridge(env, client, control_freq=100.0, signal_timeout_ms=150.0, live_enabled=True)
        bridge.start()
        try:
            bridge.update_signal(
                processed_target_abs=np.array([0.30, 0.01, 0.29, 0.0, 0.0, 0.0, 1.0, 0.05]),
                signal_age_ms=20.0,
                teleop_active=True,
                processed_sequence=8,
            )
            bridge.activate_for_recording()
            time.sleep(0.05)
            bridge.deactivate_owner()
            assert client.calls[0] == (False, "upper_machine")
            assert client.calls[-1] == (True, "lower_machine")
            assert len(env.commands) > 0
        finally:
            bridge.stop()

    def test_live_mode_stop_restores_owner(self):
        env = FakeEnv()
        client = FakeSignalClient()
        bridge = TeleopUpperControlBridge(env, client, control_freq=100.0, signal_timeout_ms=150.0, live_enabled=True)
        bridge.start()
        bridge.update_signal(
            processed_target_abs=np.array([0.30, 0.01, 0.29, 0.0, 0.0, 0.0, 1.0, 0.05]),
            signal_age_ms=20.0,
            teleop_active=True,
            processed_sequence=10,
        )
        bridge.activate_for_recording()
        time.sleep(0.05)
        bridge.stop()

        assert client.calls[0] == (False, "upper_machine")
        assert client.calls[-1] == (True, "lower_machine")

    def test_live_mode_restore_retries_after_temporary_failure(self):
        env = FakeEnv()
        client = FakeSignalClient(fail_lower_attempts=1)
        bridge = TeleopUpperControlBridge(env, client, control_freq=100.0, signal_timeout_ms=150.0, live_enabled=True)
        bridge.start()
        bridge.update_signal(
            processed_target_abs=np.array([0.30, 0.01, 0.29, 0.0, 0.0, 0.0, 1.0, 0.05]),
            signal_age_ms=20.0,
            teleop_active=True,
            processed_sequence=10,
        )
        bridge.activate_for_recording()
        time.sleep(0.05)
        bridge.stop()

        assert client.calls[0] == (False, "upper_machine")
        assert client.calls[-1] == (True, "lower_machine")
        assert bridge.snapshot()["last_error"] is None

    def test_live_mode_owner_acquire_failure_keeps_bridge_safe(self):
        env = FakeEnv()
        client = FakeSignalClient(fail_on_upper=True)
        bridge = TeleopUpperControlBridge(env, client, control_freq=100.0, signal_timeout_ms=150.0, live_enabled=True)
        bridge.start()
        try:
            bridge.update_signal(
                processed_target_abs=np.array([0.30, 0.01, 0.29, 0.0, 0.0, 0.0, 1.0, 0.05]),
                signal_age_ms=20.0,
                teleop_active=True,
                processed_sequence=11,
            )
            bridge.activate_for_recording()
            time.sleep(0.05)
            snapshot = bridge.snapshot()
            assert snapshot["owner_active"] is False
            assert snapshot["teleop_candidate_applied"] is False
            assert "failed to acquire upper owner" in snapshot["last_error"]
            assert len(env.commands) == 0
        finally:
            bridge.stop()

    def test_stale_signal_blocks_live_send(self):
        env = FakeEnv()
        client = FakeSignalClient()
        bridge = TeleopUpperControlBridge(env, client, control_freq=100.0, signal_timeout_ms=30.0, live_enabled=True)
        bridge.start()
        try:
            bridge.update_signal(
                processed_target_abs=np.array([0.30, 0.01, 0.29, 0.0, 0.0, 0.0, 1.0, 0.05]),
                signal_age_ms=50.0,
                teleop_active=True,
                processed_sequence=9,
            )
            bridge.activate_for_recording()
            time.sleep(0.05)
            snapshot = bridge.snapshot()
            assert snapshot["teleop_candidate_stale"] is True
            assert snapshot["teleop_candidate_applied"] is False
            assert len(env.commands) == 0
        finally:
            bridge.stop()

    def test_inactive_signal_keeps_candidate_errors_finite(self):
        env = FakeEnv()
        client = FakeSignalClient()
        bridge = TeleopUpperControlBridge(env, client, control_freq=100.0, signal_timeout_ms=150.0, live_enabled=False)
        bridge.start()
        try:
            bridge.update_signal(
                processed_target_abs=None,
                signal_age_ms=None,
                teleop_active=False,
                processed_sequence=0,
            )
            time.sleep(0.05)
            snapshot = bridge.snapshot()
            assert snapshot["teleop_candidate_stale"] is True
            assert snapshot["teleop_candidate_applied"] is False
            assert np.isfinite(snapshot["upper_candidate_pos_error"])
            assert np.isfinite(snapshot["upper_candidate_rot_error"])
            assert snapshot["upper_candidate_rot_error"] == 0.0
        finally:
            bridge.stop()
