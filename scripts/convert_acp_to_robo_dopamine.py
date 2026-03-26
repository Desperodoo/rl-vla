#!/usr/bin/env python3
"""Convert ACP HDF5 rollouts into Robo-Dopamine raw episode videos.

Layout produced:
  <output_root>/
    task_instruction.json
    episode_0000/
      annotated_keyframes.json
      cam_high.mp4
      cam_left_wrist.mp4
      cam_right_wrist.mp4

Mapping:
- cam_high.mp4     <- rgb_base, one ACP traj per Robo-Dopamine episode
- cam_left_wrist   <- rgb_render
- cam_right_wrist  <- rgb_render (duplicate, because ACP only stores two views)

The output is intentionally raw-video-only so Robo-Dopamine's own dataset scripts
can do preprocessing, sampling, and finetune JSON generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np


DEFAULT_INPUT_ROOT = Path("/home/wjz/rl-vla/data/vlaw/rollouts/mixed/LiftPegUpright-v1")
DEFAULT_OUTPUT_ROOT = Path("/home/wjz/rl-vla/data/robo_dopamine/acp_data_raw")
DEFAULT_FPS = 10


def _iter_h5_files(input_root: Path) -> Iterable[Path]:
    if input_root.is_file() and input_root.suffix.lower() in {".h5", ".hdf5"}:
        yield input_root
        return
    yield from sorted(p for p in input_root.rglob("*.h5") if p.is_file())


def _safe_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _pick_task_instruction(f: h5py.File) -> str:
    for key in sorted(f.keys()):
        if not key.startswith("traj_"):
            continue
        grp = f[key]
        if "task_instruction" in grp.attrs:
            return _safe_text(grp.attrs["task_instruction"])
    return "Pick up the peg and lift it upright."


def _video_writer(path: Path, fps: int, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def _write_mp4(frames: np.ndarray, out_path: Path, fps: int) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected (T, H, W, 3) frames, got {frames.shape}")
    h, w = int(frames.shape[1]), int(frames.shape[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _video_writer(out_path, fps=fps, frame_size=(w, h))
    try:
        for frame in frames:
            bgr = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def _write_episode_annotation(episode_dir: Path, num_frames: int) -> None:
    annotation = [
        {
            "anotation": "acp_episode",
            "start_frame_id": 0,
            "end_frame_id": max(0, num_frames - 1),
        }
    ]
    (episode_dir / "annotated_keyframes.json").write_text(
        json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _traj_groups(f: h5py.File) -> Iterable[h5py.Group]:
    for key in sorted(f.keys()):
        if not key.startswith("traj_"):
            continue
        grp = f[key]
        if "rgb_base" not in grp or "rgb_render" not in grp:
            continue
        yield grp


def convert_acp_to_raw(input_root: Path, output_root: Path, fps: int, limit: int | None) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    task_instructions: set[str] = set()
    episode_idx = 0

    for h5_path in _iter_h5_files(input_root):
        with h5py.File(str(h5_path), "r") as f:
            traj_groups = list(_traj_groups(f))
            if not traj_groups:
                continue

            task_instructions.add(_pick_task_instruction(f))

            for traj_group in traj_groups:
                episode_dir = output_root / f"episode_{episode_idx:04d}"
                episode_dir.mkdir(parents=True, exist_ok=True)

                rgb_base = np.asarray(traj_group["rgb_base"], dtype=np.uint8)
                rgb_render = np.asarray(traj_group["rgb_render"], dtype=np.uint8)

                if rgb_base.shape[0] != rgb_render.shape[0]:
                    raise RuntimeError(
                        f"Frame count mismatch in {h5_path}:{traj_group.name}: rgb_base={rgb_base.shape[0]} rgb_render={rgb_render.shape[0]}"
                    )

                _write_mp4(rgb_base, episode_dir / "cam_high.mp4", fps=fps)
                _write_mp4(rgb_render, episode_dir / "cam_left_wrist.mp4", fps=fps)
                _write_mp4(rgb_render, episode_dir / "cam_right_wrist.mp4", fps=fps)
                _write_episode_annotation(episode_dir, num_frames=int(rgb_base.shape[0]))

                episode_idx += 1
                if limit is not None and episode_idx >= limit:
                    break

        if limit is not None and episode_idx >= limit:
            break

    if not task_instructions:
        task_instructions.add("Pick up the peg and lift it upright.")

    task_path = output_root / "task_instruction.json"
    task_path.write_text(json.dumps(sorted(task_instructions), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[acp-convert] wrote {episode_idx} episodes to {output_root}")
    print(f"[acp-convert] wrote task instructions to {task_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ACP HDF5 rollouts into Robo-Dopamine raw videos.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of trajectories to convert")
    args = parser.parse_args()

    convert_acp_to_raw(args.input_root, args.output_root, fps=args.fps, limit=args.limit)


if __name__ == "__main__":
    main()
