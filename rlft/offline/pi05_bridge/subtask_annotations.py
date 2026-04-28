from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .task_semantics import Pi05TaskSemantics


ANNOTATION_SCHEMA_VERSION = "pi05_subtask_sidecar_v1"
DEFAULT_SUBSETS = ("fixed_dual_light", "fixed_left_light", "fixed_no_light", "random_no_light")


@dataclass(frozen=True)
class Pi05SubtaskSegment:
    name: str
    instruction: str
    start_frame: int
    end_frame: int
    task_prompt: str

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index < self.end_frame


@dataclass(frozen=True)
class Pi05EpisodeAnnotation:
    episode_id: str
    source_path: str
    source_subset: str
    fps: float
    num_frames: int
    boundary_frame: int
    segments: tuple[Pi05SubtaskSegment, ...]
    flags: tuple[str, ...]
    confidence: float
    review_status: str

    def segment_for_frame(self, frame_index: int) -> Pi05SubtaskSegment:
        for segment in self.segments:
            if segment.contains(frame_index):
                return segment
        raise IndexError(f"Frame {frame_index} is outside annotated range for {self.episode_id}")


def make_episode_id(path: str | Path, recorded_root: str | Path | None = None) -> str:
    episode_path = Path(path).expanduser().resolve()
    if recorded_root is not None:
        try:
            rel = episode_path.relative_to(Path(recorded_root).expanduser().resolve())
            return rel.as_posix()
        except ValueError:
            pass
    return f"{episode_path.parent.name}/{episode_path.name}"


def iter_recorded_episode_paths(
    recorded_root: str | Path,
    subsets: Iterable[str] = DEFAULT_SUBSETS,
) -> list[Path]:
    root = Path(recorded_root).expanduser().resolve()
    paths: list[Path] = []
    for subset in subsets:
        subset_dir = root / subset
        paths.extend(sorted(subset_dir.glob("episode_*.hdf5")))
    return paths


def read_episode_info(path: str | Path) -> dict[str, Any]:
    episode_path = Path(path).expanduser().resolve()
    with h5py.File(episode_path, "r") as handle:
        num_frames = int(handle.attrs.get("num_steps", handle["action"].shape[0]))
        fps = float(handle.attrs.get("record_freq", 30.0))
        by_camera = handle.get("observations/images_by_camera")
        if by_camera is not None:
            cameras = list(by_camera.keys())
        elif "observations/images" in handle:
            cameras = ["single_view"]
        else:
            cameras = []
    return {
        "source_path": str(episode_path),
        "source_subset": episode_path.parent.name,
        "episode_filename": episode_path.name,
        "episode_id": make_episode_id(episode_path),
        "fps": fps,
        "num_frames": num_frames,
        "cameras": cameras,
    }


def _timestamp_to_seconds(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip()
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    raise ValueError(f"Unsupported timestamp format: {value!r}")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if "```json" in stripped:
        stripped = stripped.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in stripped:
        stripped = stripped.split("```", 1)[1].split("```", 1)[0]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group())


def parse_vlm_boundary_frame(
    raw_output: str | dict[str, Any],
    *,
    fps: float,
    expected_subtasks: tuple[str, str],
) -> tuple[int, dict[str, Any]]:
    data = _extract_json_object(raw_output) if isinstance(raw_output, str) else raw_output
    frame_boundary = data.get("boundary_frame")
    if frame_boundary is None and isinstance(data.get("boundary"), dict):
        frame_boundary = data["boundary"].get("frame")

    subtasks = data.get("subtasks")
    if not isinstance(subtasks, list) or len(subtasks) != 2:
        if frame_boundary is not None:
            return int(round(float(frame_boundary))), data
        raise ValueError("VLM output must contain exactly two subtasks")

    names = tuple(str(item.get("name", "")).strip() for item in subtasks)
    if names != expected_subtasks:
        raise ValueError(f"Unexpected subtask order {names}; expected {expected_subtasks}")

    if frame_boundary is not None:
        return int(round(float(frame_boundary))), data

    first_end = subtasks[0].get("timestamps", {}).get("end")
    second_start = subtasks[1].get("timestamps", {}).get("start")
    if first_end is None or second_start is None:
        raise ValueError("VLM output needs first end and second start timestamps")

    first_end_s = _timestamp_to_seconds(first_end)
    second_start_s = _timestamp_to_seconds(second_start)
    if abs(first_end_s - second_start_s) > 1.0:
        raise ValueError(f"Subtask boundary is not continuous: {first_end!r} vs {second_start!r}")

    return int(round(((first_end_s + second_start_s) / 2.0) * fps)), data


def refine_boundary_from_robot_signals(
    episode_path: str | Path,
    predicted_boundary_frame: int,
    *,
    window_seconds: float = 2.0,
) -> dict[str, Any]:
    with h5py.File(Path(episode_path).expanduser(), "r") as handle:
        fps = float(handle.attrs.get("record_freq", 30.0))
        actions = np.asarray(handle["action"])
        qpos_end = np.asarray(handle["observations/qpos_end"])
        obs_gripper = np.asarray(handle["observations/gripper"])

    num_frames = actions.shape[0]
    window = max(1, int(round(window_seconds * fps)))
    start = max(1, predicted_boundary_frame - window)
    end = min(num_frames - 1, predicted_boundary_frame + window)
    if end <= start:
        return {
            "frame": int(np.clip(predicted_boundary_frame, 1, num_frames - 1)),
            "score": 0.0,
            "source": "clipped_vlm_boundary",
        }

    action_gripper = actions[:, 7] if actions.shape[1] > 7 else obs_gripper
    ee_xyz = qpos_end[:, :3]

    grip_signal = np.abs(np.diff(action_gripper.astype(np.float64)))
    obs_grip_signal = np.abs(np.diff(obs_gripper.astype(np.float64)))
    ee_speed = np.linalg.norm(np.diff(ee_xyz.astype(np.float64), axis=0), axis=1)

    def _norm(x: np.ndarray) -> np.ndarray:
        denom = float(np.nanpercentile(x, 95))
        if denom <= 1e-9 or not math.isfinite(denom):
            return np.zeros_like(x, dtype=np.float64)
        return np.clip(x / denom, 0.0, 2.0)

    score = 0.55 * _norm(grip_signal) + 0.30 * _norm(obs_grip_signal) + 0.15 * _norm(ee_speed)
    local = score[start:end]
    if local.size == 0 or float(np.nanmax(local)) <= 0:
        frame = int(np.clip(predicted_boundary_frame, 1, num_frames - 1))
        best_score = 0.0
        source = "vlm_boundary_no_signal_peak"
    else:
        frame = int(start + np.nanargmax(local) + 1)
        best_score = float(np.nanmax(local))
        source = "robot_signal_peak"

    return {
        "frame": frame,
        "score": best_score,
        "source": source,
        "search_window": [int(start), int(end)],
    }


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64)
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def _normalize_signal(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    denom = float(np.nanpercentile(values, 95))
    if denom <= 1e-9 or not math.isfinite(denom):
        return np.zeros_like(values, dtype=np.float64)
    return np.clip(values / denom, 0.0, 2.0)


def estimate_grasp_lift_frame_from_robot_signals(episode_path: str | Path) -> dict[str, Any]:
    with h5py.File(Path(episode_path).expanduser(), "r") as handle:
        fps = float(handle.attrs.get("record_freq", 30.0))
        actions = np.asarray(handle["action"])
        qpos_end = np.asarray(handle["observations/qpos_end"])
        obs_gripper = np.asarray(handle["observations/gripper"])

    action_gripper = actions[:, 7] if actions.shape[1] > 7 else obs_gripper
    ee_xyz = qpos_end[:, :3]
    grip_signal = np.abs(np.diff(action_gripper.astype(np.float64), prepend=action_gripper[0]))
    obs_grip_signal = np.abs(np.diff(obs_gripper.astype(np.float64), prepend=obs_gripper[0]))
    ee_speed = np.linalg.norm(np.diff(ee_xyz.astype(np.float64), axis=0, prepend=ee_xyz[:1]), axis=1)

    score = (
        0.45 * _normalize_signal(grip_signal)
        + 0.35 * _normalize_signal(obs_grip_signal)
        + 0.20 * _normalize_signal(ee_speed)
    )
    smooth = _moving_average(score, max(1, int(round(0.2 * fps))))
    num_frames = actions.shape[0]
    peak_frame = int(np.nanargmax(smooth)) if smooth.size else 0
    peak_score = float(smooth[peak_frame]) if smooth.size else 0.0
    threshold = max(0.25, float(np.nanpercentile(smooth, 80)) if smooth.size else 0.25)
    candidates = np.where(smooth >= threshold)[0]
    signal_frame = int(candidates[0]) if candidates.size else peak_frame

    z = ee_xyz[:, 2].astype(np.float64)
    z_search_start = int(round(0.10 * num_frames))
    z_min_frame = int(z_search_start + np.nanargmin(z[z_search_start:])) if z_search_start < num_frames else int(np.nanargmin(z))
    z_after = z[z_min_frame:]
    z_min = float(z[z_min_frame])
    z_peak_after = float(np.nanmax(z_after)) if z_after.size else z_min
    z_lift_level = z_min + max(0.04, 0.30 * max(0.0, z_peak_after - z_min))
    z_lift_candidates = np.where(z_after >= z_lift_level)[0]
    z_lift_frame = int(z_min_frame + z_lift_candidates[0]) if z_lift_candidates.size else z_min_frame

    frame = max(signal_frame, z_lift_frame)
    flags: list[str] = []
    if peak_score <= 1e-6:
        flags.append("needs_review_rule_no_robot_signal")
        frame = z_lift_frame

    return {
        "frame": frame,
        "signal_frame": signal_frame,
        "z_lift_frame": z_lift_frame,
        "z_min_frame": z_min_frame,
        "z_lift_level": z_lift_level,
        "peak_frame": peak_frame,
        "peak_score": peak_score,
        "threshold": threshold,
        "source": "robot_signal_lower_bound",
        "flags": flags,
    }


def compute_wrist_blue_cup_score(
    wrist_frames: np.ndarray,
    *,
    fps: float,
) -> dict[str, Any]:
    rgb = wrist_frames.astype(np.int16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    mask = (blue > 80) & (green > 55) & (blue > red + 25) & (green > red + 10)
    fraction = mask.mean(axis=(1, 2)).astype(np.float64)

    largest_component = np.zeros(mask.shape[0], dtype=np.float64)
    try:
        import cv2

        pixel_count = float(mask.shape[1] * mask.shape[2])
        for index, frame_mask in enumerate(mask):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(frame_mask.astype(np.uint8), 8)
            if num_labels > 1:
                largest_component[index] = float(np.max(stats[1:, cv2.CC_STAT_AREA])) / pixel_count
    except Exception:
        largest_component = fraction.copy()

    smooth_window = max(1, int(round(0.25 * fps)))
    smooth_fraction = _moving_average(fraction, smooth_window)
    smooth_component = _moving_average(largest_component, smooth_window)
    score = 0.70 * smooth_fraction + 0.30 * smooth_component
    return {
        "fraction": fraction,
        "largest_component": largest_component,
        "score": score,
        "smooth_window": smooth_window,
    }


def detect_rule_based_subtask_boundary(
    episode_path: str | Path,
    *,
    min_blue_threshold: float = 0.012,
    stable_seconds: float = 0.12,
    lower_bound_margin_seconds: float = 0.20,
) -> dict[str, Any]:
    episode_path = Path(episode_path).expanduser()
    info = read_episode_info(episode_path)
    fps = float(info["fps"])
    num_frames = int(info["num_frames"])
    flags: list[str] = []

    robot = estimate_grasp_lift_frame_from_robot_signals(episode_path)
    flags.extend(robot.get("flags", []))
    lower_bound = int(robot["frame"] + round(lower_bound_margin_seconds * fps))
    lower_bound = int(np.clip(lower_bound, 0, num_frames - 1))

    with h5py.File(episode_path, "r") as handle:
        by_camera = handle.get("observations/images_by_camera")
        fallback = handle.get("observations/images")
        if by_camera is not None and "wrist" in by_camera:
            wrist = np.asarray(by_camera["wrist"])
            wrist_source = "observations/images_by_camera/wrist"
        elif fallback is not None:
            wrist = np.asarray(fallback)
            wrist_source = "observations/images"
            flags.append("needs_review_missing_wrist_view")
        else:
            raise KeyError(f"No wrist or fallback image stream found in {episode_path}")

    blue = compute_wrist_blue_cup_score(wrist, fps=fps)
    score = np.asarray(blue["score"], dtype=np.float64)
    after = score[lower_bound:]
    if after.size == 0:
        candidate = num_frames - 1
        threshold = min_blue_threshold
        flags.append("needs_review_rule_no_search_window")
    else:
        baseline = float(np.nanpercentile(after, 25))
        high = float(np.nanpercentile(after, 95))
        adaptive_threshold = baseline + 0.15 * max(0.0, high - baseline)
        threshold = max(min_blue_threshold, min(0.04, adaptive_threshold))
        stable_frames = max(1, int(round(stable_seconds * fps)))
        candidate = -1
        for frame in range(lower_bound, num_frames):
            end = min(num_frames, frame + stable_frames)
            if end - frame < stable_frames:
                break
            if float(np.nanmin(score[frame:end])) >= threshold:
                candidate = frame
                break
        if candidate < 0:
            above = np.where(after >= threshold)[0]
            if above.size:
                candidate = int(lower_bound + above[0])
                flags.append("needs_review_rule_unstable_blue_score")
            else:
                candidate = int(lower_bound + np.nanargmax(after))
                flags.append("needs_review_rule_no_blue_cup")

    lower = int(round(num_frames * 0.05))
    upper = int(round(num_frames * 0.95))
    if candidate <= lower or candidate >= upper:
        flags.append("needs_review_rule_boundary_near_episode_edge")
    if candidate <= lower_bound:
        flags.append("needs_review_rule_boundary_not_after_grasp_lift")

    pre_start = max(lower_bound, candidate - int(round(1.0 * fps)))
    pre_median = float(np.nanmedian(score[pre_start:candidate])) if candidate > pre_start else 0.0
    candidate_score = float(score[candidate]) if 0 <= candidate < len(score) else 0.0
    peak_after = float(np.nanmax(after)) if after.size else 0.0
    margin = candidate_score - pre_median
    confidence = 0.85
    if margin < 0.006:
        flags.append("needs_review_rule_low_blue_margin")
        confidence = 0.55
    if peak_after > 1e-9 and candidate_score < 0.45 * peak_after:
        flags.append("needs_review_rule_early_weak_blue_score")
        confidence = min(confidence, 0.50)
    if any(flag.startswith("needs_review") for flag in flags):
        confidence = min(confidence, 0.45)

    summary_indices = sorted(
        {
            0,
            lower_bound,
            int(robot["frame"]),
            int(candidate),
            min(num_frames - 1, int(candidate + round(0.5 * fps))),
            num_frames - 1,
        }
    )
    score_samples = [
        {
            "frame": int(index),
            "time_s": float(index / fps),
            "score": float(score[index]),
            "fraction": float(blue["fraction"][index]),
            "largest_component": float(blue["largest_component"][index]),
        }
        for index in summary_indices
        if 0 <= index < num_frames
    ]

    return {
        "boundary_frame": int(np.clip(candidate, 1, num_frames - 1)),
        "boundary_source": "rule_detector",
        "grasp_lift_frame": int(robot["frame"]),
        "grasp_lift_lower_bound_frame": int(lower_bound),
        "cup_visible_frame": int(candidate),
        "confidence": float(confidence),
        "flags": sorted(set(flags)),
        "blue_score_trace_summary": {
            "wrist_source": wrist_source,
            "threshold": float(threshold),
            "min_blue_threshold": float(min_blue_threshold),
            "stable_seconds": float(stable_seconds),
            "stable_frames": int(max(1, int(round(stable_seconds * fps)))),
            "candidate_score": candidate_score,
            "pre_window_median": pre_median,
            "margin": float(margin),
            "peak_after_lower_bound": peak_after,
            "max_score": float(np.nanmax(score)) if score.size else 0.0,
            "score_samples": score_samples,
        },
        "robot_signal": robot,
    }


def build_episode_annotation(
    episode_path: str | Path,
    semantics: Pi05TaskSemantics,
    *,
    vlm_boundary_frame: int,
    vlm_raw_output: dict[str, Any] | str | None,
    recorded_root: str | Path | None = None,
    refine: bool = True,
    boundary_source: str = "vlm",
    detector_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = read_episode_info(episode_path)
    num_frames = int(info["num_frames"])
    fps = float(info["fps"])
    flags: list[str] = []

    clipped_vlm_boundary = int(np.clip(vlm_boundary_frame, 1, num_frames - 1))
    if clipped_vlm_boundary != vlm_boundary_frame:
        flags.append("vlm_boundary_clipped")

    refinement = (
        refine_boundary_from_robot_signals(episode_path, clipped_vlm_boundary)
        if refine
        else {"frame": clipped_vlm_boundary, "score": 0.0, "source": "vlm_boundary"}
    )
    boundary_frame = clipped_vlm_boundary
    max_allowed_delta = int(round(fps))
    signal_frame = int(np.clip(refinement["frame"], 1, num_frames - 1))
    signal_delta = abs(signal_frame - clipped_vlm_boundary)
    refinement["delta_from_vlm_frame"] = int(signal_frame - clipped_vlm_boundary)
    refinement["policy"] = "validate_visual_boundary_only"
    if signal_delta > max_allowed_delta:
        flags.append("needs_review_boundary_signal_disagreement")

    lower = int(round(num_frames * 0.05))
    upper = int(round(num_frames * 0.95))
    if boundary_frame <= lower or boundary_frame >= upper:
        flags.append("needs_review_boundary_near_episode_edge")
    if detector_output is not None:
        flags.extend(str(flag) for flag in detector_output.get("flags", []))

    subtask_names = semantics.subtask_names
    segments = [
        {
            "name": subtask_names[0],
            "instruction": semantics.instruction_for(subtask_names[0]),
            "start_frame": 0,
            "end_frame": boundary_frame,
            "task_prompt": semantics.prompt_for(subtask_names[0]),
        },
        {
            "name": subtask_names[1],
            "instruction": semantics.instruction_for(subtask_names[1]),
            "start_frame": boundary_frame,
            "end_frame": num_frames,
            "task_prompt": semantics.prompt_for(subtask_names[1]),
        },
    ]

    confidence = 0.85
    if detector_output is not None:
        confidence = float(detector_output.get("confidence", confidence))
    if any(flag.startswith("needs_review") for flag in flags):
        confidence = min(confidence, 0.45)

    return {
        "episode_id": make_episode_id(episode_path, recorded_root=recorded_root),
        "source_path": str(Path(episode_path).expanduser().resolve()),
        "source_subset": info["source_subset"],
        "episode_filename": info["episode_filename"],
        "fps": fps,
        "num_frames": num_frames,
        "vlm": {
            "boundary_frame": clipped_vlm_boundary,
            "raw_output": vlm_raw_output,
        },
        "refinement": refinement,
        "boundary_frame": boundary_frame,
        "boundary_source": boundary_source,
        "rule_detector": detector_output,
        "segments": segments,
        "flags": sorted(set(flags)),
        "confidence": confidence,
        "review_status": "needs_review" if any(flag.startswith("needs_review") for flag in flags) else "auto",
    }


def validate_annotation_record(record: dict[str, Any], semantics: Pi05TaskSemantics) -> list[str]:
    errors: list[str] = []
    segments = record.get("segments", [])
    if len(segments) != 2:
        return ["annotation must contain exactly two segments"]
    expected_names = semantics.subtask_names
    names = tuple(segment.get("name") for segment in segments)
    if names != expected_names:
        errors.append(f"segment names/order {names} != {expected_names}")
    num_frames = int(record.get("num_frames", -1))
    if segments[0].get("start_frame") != 0:
        errors.append("first segment must start at frame 0")
    if segments[0].get("end_frame") != segments[1].get("start_frame"):
        errors.append("segments must be continuous")
    if segments[1].get("end_frame") != num_frames:
        errors.append("last segment must end at num_frames")
    boundary = int(record.get("boundary_frame", -1))
    if boundary != segments[0].get("end_frame"):
        errors.append("boundary_frame must equal first segment end_frame")
    if not (0 < boundary < num_frames):
        errors.append("boundary_frame must be inside the episode")
    return errors


def load_subtask_annotation_sidecar(
    path: str | Path,
    semantics: Pi05TaskSemantics,
) -> dict[str, Pi05EpisodeAnnotation]:
    sidecar_path = Path(path).expanduser().resolve()
    with open(sidecar_path) as handle:
        data = json.load(handle)
    records = data.get("episodes", data)
    if not isinstance(records, list):
        raise ValueError("Subtask annotation sidecar must contain an 'episodes' list")

    annotations: dict[str, Pi05EpisodeAnnotation] = {}
    for record in records:
        errors = validate_annotation_record(record, semantics)
        if errors:
            raise ValueError(f"Invalid annotation for {record.get('episode_id')}: {errors}")
        segments = tuple(
            Pi05SubtaskSegment(
                name=str(segment["name"]),
                instruction=str(segment["instruction"]),
                start_frame=int(segment["start_frame"]),
                end_frame=int(segment["end_frame"]),
                task_prompt=str(segment["task_prompt"]),
            )
            for segment in record["segments"]
        )
        annotation = Pi05EpisodeAnnotation(
            episode_id=str(record["episode_id"]),
            source_path=str(record["source_path"]),
            source_subset=str(record["source_subset"]),
            fps=float(record["fps"]),
            num_frames=int(record["num_frames"]),
            boundary_frame=int(record["boundary_frame"]),
            segments=segments,
            flags=tuple(record.get("flags", [])),
            confidence=float(record.get("confidence", 0.0)),
            review_status=str(record.get("review_status", "unknown")),
        )
        annotations[annotation.episode_id] = annotation
    return annotations


def write_annotation_sidecar(
    output_path: str | Path,
    *,
    semantics: Pi05TaskSemantics,
    records: list[dict[str, Any]],
    recorded_root: str | Path,
) -> None:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "task_semantics": semantics.as_dict(),
        "recorded_root": str(Path(recorded_root).expanduser().resolve()),
        "num_episodes": len(records),
        "episodes": records,
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def export_review_video(
    episode_path: str | Path,
    output_path: str | Path,
    *,
    target_fps: float = 2.0,
    height: int = 240,
) -> Path:
    import cv2

    episode_path = Path(episode_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(episode_path, "r") as handle:
        fps = float(handle.attrs.get("record_freq", 30.0))
        by_camera = handle.get("observations/images_by_camera")
        fallback = handle.get("observations/images")
        if by_camera is not None and "third_person" in by_camera:
            third = np.asarray(by_camera["third_person"])
        elif fallback is not None:
            third = np.asarray(fallback)
        else:
            raise KeyError(f"No third_person or observations/images frames in {episode_path}")

        if by_camera is not None and "wrist" in by_camera:
            wrist = np.asarray(by_camera["wrist"])
        elif fallback is not None:
            wrist = np.asarray(fallback)
        else:
            wrist = third

    stride = max(1, int(round(fps / target_fps)))
    frames = []
    for idx in range(0, min(len(third), len(wrist)), stride):
        left = third[idx]
        right = wrist[idx]
        if left.shape[0] != height:
            scale = height / left.shape[0]
            left = cv2.resize(left, (int(left.shape[1] * scale), height), interpolation=cv2.INTER_AREA)
        if right.shape[0] != height:
            scale = height / right.shape[0]
            right = cv2.resize(right, (int(right.shape[1] * scale), height), interpolation=cv2.INTER_AREA)
        frames.append(np.concatenate([left, right], axis=1))

    if not frames:
        raise ValueError(f"No frames available in {episode_path}")

    width = frames[0].shape[1]
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(float(target_fps)),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            assert proc.stdin is not None
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame).astype(np.uint8).tobytes())
            proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg exited with code {ret} for {output_path}")
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            if proc.poll() is None:
                proc.kill()
    else:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(target_fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Neither ffmpeg nor OpenCV VideoWriter could create review video")
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(np.ascontiguousarray(frame).astype(np.uint8), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    return output_path


def write_boundary_contact_sheet(
    episode_path: str | Path,
    output_path: str | Path,
    *,
    markers: dict[str, int],
    target_fps: float = 2.0,
    max_tiles: int = 36,
    columns: int = 3,
    tile_height: int = 160,
) -> Path:
    import cv2
    from PIL import Image, ImageDraw

    episode_path = Path(episode_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(episode_path, "r") as handle:
        fps = float(handle.attrs.get("record_freq", 30.0))
        by_camera = handle.get("observations/images_by_camera")
        fallback = handle.get("observations/images")
        if by_camera is not None and "third_person" in by_camera:
            third = by_camera["third_person"]
        elif fallback is not None:
            third = fallback
        else:
            raise KeyError(f"No third_person or observations/images frames in {episode_path}")

        if by_camera is not None and "wrist" in by_camera:
            wrist = by_camera["wrist"]
        elif fallback is not None:
            wrist = fallback
        else:
            wrist = third

        num_frames = min(len(third), len(wrist))
        stride = max(1, int(round(fps / target_fps)))
        sampled = set(range(0, num_frames, stride))
        for frame in markers.values():
            if 0 <= int(frame) < num_frames:
                sampled.add(int(frame))
                for offset in (-stride, stride):
                    nearby = int(frame) + offset
                    if 0 <= nearby < num_frames:
                        sampled.add(nearby)
        indices = sorted(sampled)
        if len(indices) > max_tiles:
            must_keep = {0, num_frames - 1}
            must_keep.update(int(frame) for frame in markers.values() if 0 <= int(frame) < num_frames)
            remaining = [idx for idx in indices if idx not in must_keep]
            keep_budget = max(0, max_tiles - len(must_keep))
            if keep_budget and remaining:
                positions = np.linspace(0, len(remaining) - 1, keep_budget).round().astype(int)
                must_keep.update(remaining[int(pos)] for pos in positions)
            indices = sorted(idx for idx in must_keep if 0 <= idx < num_frames)

        tiles = []
        for idx in indices:
            left = np.asarray(third[idx])
            right = np.asarray(wrist[idx])
            if left.shape[0] != tile_height:
                scale = tile_height / left.shape[0]
                left = cv2.resize(left, (int(left.shape[1] * scale), tile_height), interpolation=cv2.INTER_AREA)
            if right.shape[0] != tile_height:
                scale = tile_height / right.shape[0]
                right = cv2.resize(right, (int(right.shape[1] * scale), tile_height), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(np.concatenate([left, right], axis=1))
            header = 28
            tile = Image.new("RGB", (image.width, image.height + header), (245, 245, 245))
            tile.paste(image, (0, header))
            draw = ImageDraw.Draw(tile)
            tags = [name for name, frame in markers.items() if int(frame) == idx]
            draw.rectangle([0, 0, tile.width - 1, header - 1], fill=(18, 18, 18))
            tag_text = f" [{' | '.join(tags)}]" if tags else ""
            draw.text((8, 7), f"frame {idx} t={idx / fps:.2f}s third_person | wrist{tag_text}", fill=(255, 255, 255))
            border = (160, 160, 160)
            if tags:
                border = (20, 150, 80) if "rule" in tags else (60, 110, 220)
            draw.rectangle([0, 0, tile.width - 1, tile.height - 1], outline=border, width=4)
            tiles.append(tile)

    if not tiles:
        raise ValueError(f"No frames available in {episode_path}")

    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    rows = int(math.ceil(len(tiles) / columns))
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (230, 230, 230))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))
    sheet.save(output_path, quality=92)
    return output_path


def annotate_video_with_qwen(
    video_path: str | Path,
    semantics: Pi05TaskSemantics,
    *,
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_new_tokens: int = 768,
    local_files_only: bool = False,
) -> str:
    return QwenVideoAnnotator(
        semantics=semantics,
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        local_files_only=local_files_only,
    ).annotate(video_path)


class QwenVideoAnnotator:
    def __init__(
        self,
        *,
        semantics: Pi05TaskSemantics,
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 768,
        local_files_only: bool = False,
    ) -> None:
        self.semantics = semantics
        self.device = device
        self.max_new_tokens = max_new_tokens

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose one of {sorted(dtype_map)}")

        device_map = "auto" if device.startswith("cuda") else device
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype_map[dtype],
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is not None and hasattr(video_processor, "do_sample_frames"):
            # qwen_vl_utils has already decoded the 2 FPS review video into frames.
            # Re-sampling here requires video metadata that the decoded tensor does not carry.
            video_processor.do_sample_frames = False

    def _prompt(self) -> str:
        subtask_lines = "\n".join(
            f"- {name}: {self.semantics.instruction_for(name)}"
            for name in self.semantics.subtask_names
        )
        return (
            "You are segmenting a robot manipulation demonstration into exactly two subtasks.\n"
            "Use only this closed vocabulary:\n"
            f"{subtask_lines}\n"
            "The task is: "
            f"{self.semantics.description}\n"
            "Use this precise boundary rule:\n"
            "- The pick_tape subtask continues while the robot gripper is grasping the black tape roll and lifting it.\n"
            "- Do not end pick_tape at the first contact or first stable grasp.\n"
            "- End pick_tape only when the gripper is holding the black tape roll, the object has been lifted, "
            "and the blue cup first becomes visible in the wrist/first-person camera view.\n"
            "- The place_tape_in_cup subtask starts at that moment, when the lifted tape roll begins moving toward "
            "the visible blue cup.\n"
            "Return only valid JSON with exactly this schema: "
            "{\"subtasks\":[{\"name\":\"pick_tape\",\"timestamps\":{\"start\":\"00:00\",\"end\":\"<boundary>\"}},"
            "{\"name\":\"place_tape_in_cup\",\"timestamps\":{\"start\":\"<boundary>\",\"end\":\"<video_end>\"}}]}.\n"
            "Use concrete MM:SS timestamps from the video, not placeholders. "
            "The first end timestamp must equal the second start timestamp. Do not add extra keys."
        )

    def annotate(self, video_path: str | Path) -> str:
        import torch
        import cv2

        from qwen_vl_utils import process_vision_info

        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 2.0)
        cap.release()
        duration_s = frame_count / fps if fps > 0 else 0.0
        video_hint = (
            f"This review video is sampled at {fps:.2f} FPS and contains {frame_count} frames "
            f"({duration_s:.1f} seconds). Segment the whole video."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self._prompt()}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": video_hint},
                    {"type": "video", "video": str(video_path), "fps": 2.0},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(
            [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids, strict=True)],
            skip_special_tokens=True,
        )[0].strip()


def write_review_html(output_path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        video = record.get("review_video", "")
        flags = ", ".join(record.get("flags", []))
        rows.append(
            "<tr>"
            f"<td>{record.get('episode_id')}</td>"
            f"<td>{record.get('boundary_frame')}</td>"
            f"<td>{record.get('confidence')}</td>"
            f"<td>{record.get('review_status')}</td>"
            f"<td>{flags}</td>"
            f"<td><video src='{video}' controls width='480'></video></td>"
            "</tr>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>PI05 Subtask Review</title>"
        "<style>body{font-family:sans-serif}td,th{padding:6px;border:1px solid #ccc}</style>"
        "</head><body><h1>PI05 Subtask Review</h1><table>"
        "<tr><th>episode</th><th>boundary</th><th>confidence</th><th>status</th><th>flags</th><th>video</th></tr>"
        + "\n".join(rows)
        + "</table></body></html>\n"
    )
    path.write_text(html)
