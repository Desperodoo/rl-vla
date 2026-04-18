"""Validate script-runtime pick-place logic in ManiSkill before real hardware."""

from __future__ import annotations

import argparse
from pathlib import Path

from script_runtime.adapters import ManiSkillBridge
from script_runtime.session import build_pick_place_session, load_runtime_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run script_runtime pick-place in ManiSkill PickCube-v1.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "tasks" / "pick_cube_maniskill.yaml"),
        help="Path to YAML/JSON runtime config.",
    )
    parser.add_argument(
        "--trace-out",
        default="",
        help="Optional trace output override. Writes JSONL trace after execution.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    config = load_runtime_config(args.config)
    if args.trace_out:
        config.setdefault("runtime", {})["trace_path"] = args.trace_out

    sim_cfg = config.get("maniskill", {})
    bridge = ManiSkillBridge(
        env_id=str(sim_cfg.get("env_id", "PickCube-v1")),
        obs_mode=str(sim_cfg.get("obs_mode", "state")),
        control_mode=str(sim_cfg.get("control_mode", "pd_ee_pose")),
        render_mode=sim_cfg.get("render_mode"),
        sim_backend=str(sim_cfg.get("sim_backend", "physx_cpu")),
        num_envs=int(sim_cfg.get("num_envs", 1)),
        max_episode_steps=sim_cfg.get("max_episode_steps"),
        cached_reset_options=dict(sim_cfg.get("reset_options", {})),
        grasp_height_offset=float(sim_cfg.get("grasp_height_offset", 0.0)),
        pregrasp_height_offset=float(sim_cfg.get("pregrasp_height_offset", 0.08)),
        place_height_offset=float(sim_cfg.get("place_height_offset", 0.06)),
        place_release_height_offset=float(sim_cfg.get("place_release_height_offset", 0.015)),
        retreat_height_offset=float(sim_cfg.get("retreat_height_offset", 0.08)),
        pose_substeps=int(sim_cfg.get("pose_substeps", 60)),
        gripper_substeps=int(sim_cfg.get("gripper_substeps", 15)),
        settle_substeps=int(sim_cfg.get("settle_substeps", 25)),
        position_tolerance=float(sim_cfg.get("position_tolerance", 0.01)),
        guarded_close_descend_delta=float(sim_cfg.get("guarded_close_descend_delta", 0.02)),
        guarded_close_descent_steps=int(sim_cfg.get("guarded_close_descent_steps", 6)),
        guarded_close_hold_steps=int(sim_cfg.get("guarded_close_hold_steps", 10)),
        grasp_phase_descend_segments=int(sim_cfg.get("grasp_phase_descend_segments", 5)),
        grasp_phase_close_hold_steps=int(sim_cfg.get("grasp_phase_close_hold_steps", 12)),
        grasp_phase_lift_delta=float(sim_cfg.get("grasp_phase_lift_delta", 0.03)),
        grasp_phase_lift_steps=int(sim_cfg.get("grasp_phase_lift_steps", 6)),
        lock_orientation=bool(sim_cfg.get("lock_orientation", True)),
        delta_translation_limit=float(sim_cfg.get("delta_translation_limit", 0.02)),
        delta_rotation_limit=float(sim_cfg.get("delta_rotation_limit", 0.05)),
        capture_video=bool(sim_cfg.get("capture_video", False)),
        capture_every_n_steps=int(sim_cfg.get("capture_every_n_steps", 2)),
        artifact_dir=sim_cfg.get("artifact_dir"),
        episode_name=str(sim_cfg.get("episode_name", "single_run")),
        env_kwargs=dict(sim_cfg.get("env_kwargs", {})),
    )

    session = build_pick_place_session(config, sdk_bridge=bridge)
    session.adapters["maniskill"] = bridge
    try:
        result = session.run()
        status = bridge.get_status()
    finally:
        session.shutdown()

    print(f"task={session.task.name}")
    print(f"status={result.status.value}")
    print(f"failure_code={result.failure_code.value}")
    print(f"message={result.message}")
    print(f"sim_success={status.get('success', False)}")
    print(f"is_grasped={status.get('is_grasped', False)}")
    print(f"trace_events={len(session.trace_recorder.events)}")
    if session.trace_path is not None:
        print(f"trace_path={session.trace_path}")
    for key, value in sorted((session.runtime_artifacts or {}).items()):
        print(f"{key}={value}")
    return 0 if result.status.value == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
