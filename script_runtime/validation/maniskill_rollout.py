"""Batch validation utilities for ManiSkill-backed script-runtime sessions."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from script_runtime.adapters import ManiSkillBridge
from script_runtime.session import build_pick_place_session


@dataclass
class EpisodeValidationResult:
    episode_index: int
    task_status: str
    failure_code: str
    sim_success: bool
    trace_events: int
    failed_skills: List[str]
    trace_path: Optional[str] = None
    rollout_gif: Optional[str] = None
    grounding_json: Optional[str] = None
    grounding_topdown_png: Optional[str] = None


@dataclass
class ValidationSummary:
    num_episodes: int
    runtime_successes: int
    sim_successes: int
    failure_code_counts: Dict[str, int]
    failed_skill_counts: Dict[str, int]
    episode_results: List[EpisodeValidationResult]


def run_maniskill_validation(config: Dict[str, Any]) -> ValidationSummary:
    validation_cfg = config.get("validation", {})
    num_episodes = int(validation_cfg.get("num_episodes", 3))
    output_dir = validation_cfg.get("output_dir")
    base_seed = validation_cfg.get("seed")

    episode_results: List[EpisodeValidationResult] = []
    failure_counter: Counter[str] = Counter()
    failed_skill_counter: Counter[str] = Counter()
    runtime_successes = 0
    sim_successes = 0

    for episode_index in range(num_episodes):
        episode_config = _clone_config(config)
        if output_dir:
            trace_name = f"episode_{episode_index:03d}.jsonl"
            episode_config.setdefault("runtime", {})["trace_path"] = str(Path(output_dir) / trace_name)

        sim_cfg = episode_config.get("maniskill", {})
        bridge = ManiSkillBridge(
            env_id=str(sim_cfg.get("env_id", "PickCube-v1")),
            obs_mode=str(sim_cfg.get("obs_mode", "state")),
            control_mode=str(sim_cfg.get("control_mode", "pd_ee_pose")),
            render_mode=sim_cfg.get("render_mode"),
            sim_backend=str(sim_cfg.get("sim_backend", "physx_cpu")),
            num_envs=int(sim_cfg.get("num_envs", 1)),
            max_episode_steps=sim_cfg.get("max_episode_steps"),
            cached_reset_options=dict(sim_cfg.get("reset_options", {})),
            reset_seed=(None if base_seed is None else int(base_seed) + episode_index),
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
            artifact_dir=(str(Path(output_dir) / "media") if output_dir else None),
            episode_name=f"episode_{episode_index:03d}",
            env_kwargs=dict(sim_cfg.get("env_kwargs", {})),
        )

        session = build_pick_place_session(episode_config, sdk_bridge=bridge)
        session.adapters["maniskill"] = bridge

        try:
            result = session.run()
            status = bridge.get_status()
        finally:
            session.shutdown()

        failed_skills = [
            event.skill_name
            for event in session.trace_recorder.events
            if event.failure_code not in ("NONE", "", None)
        ]
        episode_result = EpisodeValidationResult(
            episode_index=episode_index,
            task_status=result.status.value,
            failure_code=result.failure_code.value,
            sim_success=bool(status.get("success", False)),
            trace_events=len(session.trace_recorder.events),
            failed_skills=failed_skills,
            trace_path=(str(session.trace_path) if session.trace_path is not None else None),
            rollout_gif=session.runtime_artifacts.get("rollout_gif"),
            grounding_json=session.runtime_artifacts.get("grounding_json"),
            grounding_topdown_png=session.runtime_artifacts.get("grounding_topdown_png"),
        )
        episode_results.append(episode_result)

        if result.status.value == "SUCCESS":
            runtime_successes += 1
        if episode_result.sim_success:
            sim_successes += 1
        failure_counter[episode_result.failure_code] += 1
        failed_skill_counter.update(failed_skills)

    return ValidationSummary(
        num_episodes=num_episodes,
        runtime_successes=runtime_successes,
        sim_successes=sim_successes,
        failure_code_counts=dict(failure_counter),
        failed_skill_counts=dict(failed_skill_counter),
        episode_results=episode_results,
    )


def save_validation_summary(summary: ValidationSummary, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def _clone_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(config))
