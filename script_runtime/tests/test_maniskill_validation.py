from pathlib import Path

from script_runtime.validation.maniskill_rollout import run_maniskill_validation, save_validation_summary
from script_runtime.tests.test_maniskill_oracle_integration import StubManiSkillBridge


def test_run_maniskill_validation_can_summarize_stub_bridge(monkeypatch, tmp_path: Path):
    from script_runtime import validation as validation_pkg  # noqa: F401
    from script_runtime.validation import maniskill_rollout

    monkeypatch.setattr(maniskill_rollout, "ManiSkillBridge", StubManiSkillBridge)
    config = {
        "runtime": {},
        "validation": {
            "num_episodes": 2,
            "output_dir": str(tmp_path / "traces"),
            "summary_path": str(tmp_path / "summary.json"),
        },
        "execution": {"active_source": "policy", "control_owner": "script_runtime"},
        "task_goal": {"task_name": "pick_cube_validation"},
        "scene": {
            "workspace_ready": True,
            "tracking_lost": False,
            "depth_anomaly": False,
            "detection_confidence": 1.0,
            "calibration_version": "oracle_state",
        },
        "poses": {
            "home_joints": [0.0] * 6,
            "pregrasp_pose": [0.1, 0.0, 0.13, 0.0, 0.0, 0.0, 1.0],
            "lift_pose": [0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
            "place_pose": [0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0],
            "retreat_pose": [0.2, 0.1, 0.18, 0.0, 0.0, 0.0, 1.0],
        },
        "gripper": {"open_width": 0.04, "close_width": 0.0},
        "maniskill": {},
    }
    summary = run_maniskill_validation(config)
    summary_path = save_validation_summary(summary, config["validation"]["summary_path"])

    assert summary.num_episodes == 2
    assert summary.runtime_successes == 2
    assert summary.failure_code_counts["NONE"] == 2
    assert summary_path.exists()
