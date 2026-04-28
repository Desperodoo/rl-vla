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
    subtasks = data.get("subtasks")
    if not isinstance(subtasks, list) or len(subtasks) != 2:
        raise ValueError("VLM output must contain exactly two subtasks")

    names = tuple(str(item.get("name", "")).strip() for item in subtasks)
    if names != expected_subtasks:
        raise ValueError(f"Unexpected subtask order {names}; expected {expected_subtasks}")

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


def build_episode_annotation(
    episode_path: str | Path,
    semantics: Pi05TaskSemantics,
    *,
    vlm_boundary_frame: int,
    vlm_raw_output: dict[str, Any] | str | None,
    recorded_root: str | Path | None = None,
    refine: bool = True,
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
    boundary_frame = int(np.clip(refinement["frame"], 1, num_frames - 1))
    max_allowed_delta = int(round(fps))
    if abs(boundary_frame - clipped_vlm_boundary) > max_allowed_delta:
        flags.append("needs_review_boundary_signal_disagreement")
        boundary_frame = clipped_vlm_boundary

    lower = int(round(num_frames * 0.05))
    upper = int(round(num_frames * 0.95))
    if boundary_frame <= lower or boundary_frame >= upper:
        flags.append("needs_review_boundary_near_episode_edge")

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
    if any(flag.startswith("needs_review") for flag in flags):
        confidence = 0.45
    elif refinement.get("source") == "robot_signal_peak":
        confidence = 0.75

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
        "segments": segments,
        "flags": flags,
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
