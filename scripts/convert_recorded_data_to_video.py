#!/usr/bin/env python3
"""Convert one episode from each recorded_data subfolder into a video.

For every immediate subdirectory under the input root, this script picks one
HDF5 episode file (default: the first sorted episode_*.hdf5), selects a single
camera view (default: primary_camera if available), and writes an mp4 video.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


def iter_episode_dirs(root_dir: Path) -> Iterable[Path]:
    for path in sorted(root_dir.iterdir()):
        if path.is_dir() and list(path.glob("episode_*.hdf5")):
            yield path


def pick_episode_file(folder: Path, episode_index: int) -> Path:
    episodes = sorted(folder.glob("episode_*.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {folder}")

    if episode_index < 0:
        episode_index = len(episodes) + episode_index
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(
            f"episode_index={episode_index} out of range for {folder} ({len(episodes)} episodes)"
        )
    return episodes[episode_index]


def select_camera_frames(h5_file: Any, camera_name: Optional[str]) -> tuple[np.ndarray, str]:
    obs = h5_file["observations"]

    if "images_by_camera" in obs:
        cameras = obs["images_by_camera"]
        available = sorted(cameras.keys())
        if not available:
            raise ValueError("observations/images_by_camera exists but contains no cameras")

        chosen = camera_name
        if chosen in (None, "", "primary"):
            chosen = h5_file.attrs.get("primary_camera", "")
            if isinstance(chosen, bytes):
                chosen = chosen.decode("utf-8")
            if not chosen or chosen not in cameras:
                chosen = available[0]
        elif chosen not in cameras:
            raise KeyError(f"camera '{chosen}' not found; available={available}")

        return np.asarray(cameras[chosen]), chosen

    if "images" not in obs:
        raise KeyError("Neither observations/images_by_camera nor observations/images exists")

    return np.asarray(obs["images"]), "images"


def write_mp4(frames: np.ndarray, output_path: Path, fps: float) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape [T, H, W, 3], got {frames.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames.shape[1]), int(frames.shape[2])

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
        str(float(fps)),
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
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Unexpected frame size {frame.shape[:2]}, expected {(height, width)}")
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret} for {output_path}")
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        if proc.poll() is None:
            proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick one episode from each recorded_data folder and convert a single camera view to mp4"
    )
    parser.add_argument("--root-dir", type=Path, default=Path("recorded_data"), help="Dataset root directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recorded_data/video_exports"),
        help="Where to save generated videos",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="primary",
        help="Camera name to export; use 'primary' to prefer primary_camera",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Which episode to pick in each folder after sorting episode_*.hdf5",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video FPS override; defaults to record_freq stored in the file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not root_dir.exists():
        print(f"[ERROR] root dir not found: {root_dir}")
        return 1

    folders = list(iter_episode_dirs(root_dir))
    if not folders:
        print(f"[WARN] No episode folders found under {root_dir}")
        return 0

    print(f"[INFO] root_dir={root_dir}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] camera_name={args.camera_name}")
    print(f"[INFO] episode_index={args.episode_index}\n")

    converted = 0
    for folder in folders:
        try:
            episode_path = pick_episode_file(folder, args.episode_index)
            import h5py  # lazy import so --help works even if dependencies are missing

            with h5py.File(episode_path, "r") as f:
                frames, camera_used = select_camera_frames(f, args.camera_name)
                fps = float(args.fps if args.fps is not None else f.attrs.get("record_freq", 30))

            out_path = output_dir / folder.name / f"{episode_path.stem}_{camera_used}.mp4"
            write_mp4(frames, out_path, fps=fps)
            print(f"[OK] {folder.name}: {episode_path.name} -> {out_path} (camera={camera_used}, fps={fps:g})")
            converted += 1
        except Exception as exc:
            print(f"[WARN] {folder.name}: skipped ({exc})")

    print(f"\n[INFO] converted {converted}/{len(folders)} folders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
