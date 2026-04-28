from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from rlft.offline.pi05_bridge.subtask_annotations import (
    build_episode_annotation,
    compute_wrist_blue_cup_score,
    detect_rule_based_subtask_boundary,
    load_subtask_annotation_sidecar,
    parse_vlm_boundary_frame,
    validate_annotation_record,
    write_annotation_sidecar,
)
from rlft.offline.pi05_bridge.task_semantics import load_pi05_task_semantics


def _write_tiny_carm_episode(path: Path, *, num_frames: int = 100, fps: float = 20.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["num_steps"] = num_frames
        handle.attrs["record_freq"] = fps
        handle.create_dataset("action", data=np.zeros((num_frames, 8), dtype=np.float32))
        obs = handle.create_group("observations")
        obs.create_dataset("qpos_end", data=np.zeros((num_frames, 8), dtype=np.float32))
        obs.create_dataset("gripper", data=np.zeros((num_frames,), dtype=np.float32))


def _write_tiny_carm_episode_with_images(path: Path, *, num_frames: int = 80, fps: float = 20.0) -> None:
    _write_tiny_carm_episode(path, num_frames=num_frames, fps=fps)
    with h5py.File(path, "r+") as handle:
        action = handle["action"][:]
        action[18:26, 7] = np.linspace(0.0, 1.0, 8)
        del handle["action"]
        handle.create_dataset("action", data=action)
        handle["observations/gripper"][18:26] = np.linspace(0.0, 1.0, 8)
        handle["observations/qpos_end"][18:26, 2] = np.linspace(0.0, 1.0, 8)
        frames = np.zeros((num_frames, 32, 32, 3), dtype=np.uint8)
        frames[45:, 8:24, 10:26, 1] = 120
        frames[45:, 8:24, 10:26, 2] = 220
        by_camera = handle["observations"].create_group("images_by_camera")
        by_camera.create_dataset("wrist", data=frames)
        by_camera.create_dataset("third_person", data=frames)


def test_task_semantics_prompt_contains_current_subtask() -> None:
    semantics = load_pi05_task_semantics()

    assert semantics.subtask_names == ("pick_tape", "place_tape_in_cup")
    assert semantics.prompt_for("pick_tape") == (
        "Pick up the black tape roll and place it into the blue cup. "
        "Current subtask: Pick up the black tape roll."
    )


def test_parse_vlm_boundary_frame_closed_vocabulary() -> None:
    raw = {
        "subtasks": [
            {"name": "pick_tape", "timestamps": {"start": "00:00", "end": "00:05"}},
            {"name": "place_tape_in_cup", "timestamps": {"start": "00:05", "end": "00:10"}},
        ]
    }

    boundary, parsed = parse_vlm_boundary_frame(
        raw,
        fps=20.0,
        expected_subtasks=("pick_tape", "place_tape_in_cup"),
    )

    assert boundary == 100
    assert parsed == raw


def test_parse_vlm_boundary_frame_accepts_frame_index_output() -> None:
    raw = {
        "boundary_frame": 123,
        "subtasks": [
            {"name": "pick_tape"},
            {"name": "place_tape_in_cup"},
        ],
    }

    boundary, parsed = parse_vlm_boundary_frame(
        raw,
        fps=20.0,
        expected_subtasks=("pick_tape", "place_tape_in_cup"),
    )

    assert boundary == 123
    assert parsed == raw


def test_build_annotation_keeps_visual_boundary_when_robot_signal_disagrees(tmp_path: Path) -> None:
    recorded_root = tmp_path / "recorded_data"
    episode_path = recorded_root / "fixed_dual_light" / "episode_0001.hdf5"
    _write_tiny_carm_episode(episode_path, num_frames=100, fps=10.0)
    with h5py.File(episode_path, "r+") as handle:
        action = handle["action"][:]
        action[31:45, 7] = np.linspace(0.0, 1.0, 14)
        del handle["action"]
        handle.create_dataset("action", data=action)
        handle["observations/gripper"][31:45] = np.linspace(0.0, 1.0, 14)
        handle["observations/qpos_end"][31:45, 0] = np.linspace(0.0, 1.0, 14)
    semantics = load_pi05_task_semantics()

    record = build_episode_annotation(
        episode_path,
        semantics,
        vlm_boundary_frame=60,
        vlm_raw_output={"boundary_frame": 60},
        recorded_root=recorded_root,
        refine=True,
    )

    assert record["boundary_frame"] == 60
    assert record["segments"][0]["end_frame"] == 60
    assert record["refinement"]["policy"] == "validate_visual_boundary_only"
    assert "needs_review_boundary_signal_disagreement" in record["flags"]


def test_blue_cup_score_detects_synthetic_blue_patch() -> None:
    frames = np.zeros((20, 16, 16, 3), dtype=np.uint8)
    frames[10:, 4:12, 4:12, 1] = 120
    frames[10:, 4:12, 4:12, 2] = 220

    score = compute_wrist_blue_cup_score(frames, fps=20.0)["score"]

    assert float(score[10]) > float(score[8])
    assert int(np.argmax(score)) >= 10


def test_rule_detector_finds_blue_cup_after_robot_lower_bound(tmp_path: Path) -> None:
    recorded_root = tmp_path / "recorded_data"
    episode_path = recorded_root / "fixed_dual_light" / "episode_0001.hdf5"
    _write_tiny_carm_episode_with_images(episode_path)

    result = detect_rule_based_subtask_boundary(
        episode_path,
        min_blue_threshold=0.02,
        stable_seconds=0.10,
        lower_bound_margin_seconds=0.20,
    )

    assert 42 <= result["boundary_frame"] <= 48
    assert result["boundary_source"] == "rule_detector"
    assert result["grasp_lift_frame"] < result["boundary_frame"]


def test_build_write_load_annotation_sidecar(tmp_path: Path) -> None:
    recorded_root = tmp_path / "recorded_data"
    episode_path = recorded_root / "fixed_dual_light" / "episode_0001.hdf5"
    _write_tiny_carm_episode(episode_path)
    semantics = load_pi05_task_semantics()

    record = build_episode_annotation(
        episode_path,
        semantics,
        vlm_boundary_frame=50,
        vlm_raw_output={"subtasks": []},
        recorded_root=recorded_root,
        refine=True,
    )
    errors = validate_annotation_record(record, semantics)
    assert errors == []
    assert record["episode_id"] == "fixed_dual_light/episode_0001.hdf5"
    assert record["segments"][0]["task_prompt"].endswith("Pick up the black tape roll.")
    assert record["segments"][1]["task_prompt"].endswith("Place the tape roll into the blue cup.")

    sidecar_path = tmp_path / "annotations.json"
    write_annotation_sidecar(
        sidecar_path,
        semantics=semantics,
        records=[record],
        recorded_root=recorded_root,
    )

    payload = json.loads(sidecar_path.read_text())
    assert payload["schema_version"] == "pi05_subtask_sidecar_v1"
    loaded = load_subtask_annotation_sidecar(sidecar_path, semantics)
    annotation = loaded["fixed_dual_light/episode_0001.hdf5"]
    assert annotation.boundary_frame == 50
    assert annotation.segment_for_frame(49).name == "pick_tape"
    assert annotation.segment_for_frame(50).name == "place_tape_in_cup"
