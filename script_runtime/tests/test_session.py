from pathlib import Path

from script_runtime.adapters import MockSDKBridge
from script_runtime.core import SkillStatus
from script_runtime.session import build_pick_place_session


def test_build_pick_place_session_runs_with_mock(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    config = {
        "runtime": {"task_id": "pick-place-session-test", "trace_path": str(trace_path)},
        "execution": {"active_source": "policy", "control_owner": "script_runtime"},
        "task_goal": {"task_name": "pick_place"},
        "scene": {
            "workspace_ready": True,
            "tracking_lost": False,
            "depth_anomaly": False,
            "detection_confidence": 1.0,
            "calibration_version": "test",
            "object_pose": [0.42, -0.04, 0.11, 0.0, 0.0, 0.0, 1.0],
            "grasp_candidates": [{"pose": [0.42, -0.04, 0.11, 0.0, 0.0, 0.0, 1.0]}],
        },
        "gripper": {"open_width": 0.08, "close_width": 0.01, "tau": 10.0},
        "poses": {
            "home_joints": [0.0, -0.6, 1.1, 0.0, 1.45, 0.0],
            "reset_joints": [0.0, -0.6, 1.1, 0.0, 1.45, 0.0],
            "pregrasp_pose": [0.42, -0.04, 0.21, 0.0, 0.0, 0.0, 1.0],
            "lift_pose": [0.42, -0.04, 0.28, 0.0, 0.0, 0.0, 1.0],
            "place_pose": [0.25, 0.22, 0.18, 0.0, 0.0, 0.0, 1.0],
            "retreat_pose": [0.25, 0.22, 0.28, 0.0, 0.0, 0.0, 1.0],
        },
    }
    session = build_pick_place_session(config=config, sdk_bridge=MockSDKBridge())
    result = session.run()

    assert result.status == SkillStatus.SUCCESS
    assert trace_path.exists()
    assert session.blackboard.world_state.robot.eef_pose == [0.25, 0.22, 0.28, 0.0, 0.0, 0.0, 1.0]
    assert session.blackboard.world_state.scene.grasped is False
