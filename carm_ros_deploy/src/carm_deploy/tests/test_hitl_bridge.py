import os
import sys
import types
import numpy as np

_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CARM_DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, _CARM_DEPLOY_ROOT)

_rospy_mock = types.ModuleType("rospy")
_rospy_mock.loginfo = lambda *a, **kw: None
_rospy_mock.logwarn = lambda *a, **kw: None
_rospy_mock.logerr = lambda *a, **kw: None
_rospy_mock.logwarn_throttle = lambda *a, **kw: None
_rospy_mock.is_shutdown = lambda: False
sys.modules.setdefault("rospy", _rospy_mock)

from inference.hitl_bridge import HumanChunkProposalBuilder, HitlArbitrationBridge


def _teleop_snapshot(active=True, signal_age_ms=20.0, t_recv_sys=1.0, seq=12, target_pose_abs=None, gripper_pose=0.05):
    if target_pose_abs is None:
        target_pose_abs = [0.31, 0.02, 0.28, 0.0, 0.0, 0.0, 1.0]
    return {
        "teleop_active": active,
        "signal_age_ms": signal_age_ms,
        "processed_sequence": seq,
        "raw_sequence": seq,
        "t_recv_sys": t_recv_sys,
        "teleop_state_v2": {
            "processed": {
                "target_pose_abs": target_pose_abs,
                "gripper_pose": gripper_pose,
                "active": active,
            }
        },
    }


class TestHumanChunkProposalBuilder:
    def test_active_snapshot_builds_valid_chunk(self):
        builder = HumanChunkProposalBuilder(pred_horizon=4, act_horizon=2, control_freq=50.0, stale_timeout_ms=150.0, require_active=True)
        qpos_end = np.array([0.25, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0, 0.04], dtype=np.float64)

        result = builder.build(qpos_end, _teleop_snapshot(active=True, signal_age_ms=15.0, t_recv_sys=1.0))

        assert result["human_valid"] is True
        assert result["human_active"] is True
        assert result["human_stale"] is False
        assert result["human_chunk_proposal"].shape == (4, 8)
        assert result["history_count"] == 1
        assert result["history_usable"] is False
        np.testing.assert_allclose(result["processed_target_abs"], [0.31, 0.02, 0.28, 0.0, 0.0, 0.0, 1.0, 0.05])
        np.testing.assert_allclose(result["human_chunk_proposal"][0], result["processed_target_abs"])

    def test_inactive_snapshot_becomes_unavailable(self):
        builder = HumanChunkProposalBuilder(pred_horizon=4, stale_timeout_ms=150.0, require_active=True)
        qpos_end = np.array([0.25, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0, 0.04], dtype=np.float64)

        result = builder.build(qpos_end, _teleop_snapshot(active=False, signal_age_ms=15.0))

        assert result["human_valid"] is False
        assert result["human_active"] is False

    def test_stale_snapshot_becomes_unavailable(self):
        builder = HumanChunkProposalBuilder(pred_horizon=4, stale_timeout_ms=30.0, require_active=True)
        qpos_end = np.array([0.25, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0, 0.04], dtype=np.float64)

        result = builder.build(qpos_end, _teleop_snapshot(active=True, signal_age_ms=50.0))

        assert result["human_valid"] is False
        assert result["human_stale"] is True

    def test_history_rollout_uses_short_horizon_then_hold(self):
        builder = HumanChunkProposalBuilder(
            pred_horizon=5,
            act_horizon=3,
            control_freq=50.0,
            stale_timeout_ms=150.0,
            require_active=True,
            history_window_ms=200.0,
            min_history_span_ms=20.0,
        )
        qpos_end = np.array([0.25, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0, 0.04], dtype=np.float64)

        builder.build(qpos_end, _teleop_snapshot(t_recv_sys=1.00, seq=10, target_pose_abs=[0.30, 0.00, 0.28, 0.0, 0.0, 0.0, 1.0]))
        result = builder.build(qpos_end, _teleop_snapshot(t_recv_sys=1.10, seq=11, target_pose_abs=[0.32, 0.00, 0.28, 0.0, 0.0, 0.0, 1.0]))

        assert result["history_usable"] is True
        assert result["history_count"] == 2
        assert result["rollout_step_count"] == 3
        assert result["history_span_ms"] >= 100.0 - 1e-6
        np.testing.assert_allclose(result["human_chunk_proposal"][0], result["processed_target_abs"])
        assert result["human_chunk_proposal"][1, 0] > result["human_chunk_proposal"][0, 0]
        assert result["human_chunk_proposal"][2, 0] > result["human_chunk_proposal"][1, 0]
        np.testing.assert_allclose(result["human_chunk_proposal"][3], result["human_chunk_proposal"][2])
        np.testing.assert_allclose(result["human_chunk_proposal"][4], result["human_chunk_proposal"][2])
        assert result["linear_velocity"][0] > 0.0


class TestHitlArbitrationBridge:
    def test_valid_human_chunk_is_selected(self):
        bridge = HitlArbitrationBridge()
        policy_chunk = np.ones((4, 8), dtype=np.float64)
        human_chunk = np.full((4, 8), 2.0, dtype=np.float64)

        result = bridge.arbitrate(policy_chunk, {
            "human_chunk_proposal": human_chunk,
            "human_valid": True,
            "human_active": True,
            "human_stale": False,
            "teleop_valid": True,
        })

        assert result["shared_source"] == "human"
        np.testing.assert_allclose(result["shared_chunk"], human_chunk)

    def test_invalid_human_chunk_falls_back_to_policy(self):
        bridge = HitlArbitrationBridge()
        policy_chunk = np.ones((4, 8), dtype=np.float64)
        human_chunk = np.full((4, 8), 2.0, dtype=np.float64)

        result = bridge.arbitrate(policy_chunk, {
            "human_chunk_proposal": human_chunk,
            "human_valid": False,
            "human_active": False,
            "human_stale": False,
            "teleop_valid": False,
        })

        assert result["shared_source"] == "policy_fallback"
        assert result["fallback_reason"] == "human_inactive"
        np.testing.assert_allclose(result["shared_chunk"], policy_chunk)
