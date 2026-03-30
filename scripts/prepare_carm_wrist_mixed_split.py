#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def collect_episode_files(directory: Path):
    return sorted(directory.glob("episode_*.hdf5"))


def split_stratified(files, train_ratio, seed):
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    if n >= 2:
        n_train = max(1, min(n - 1, n_train))
    return files[:n_train], files[n_train:]


def infer_position_type(source_dir: Path):
    name = source_dir.name.lower()
    if name.startswith("fixed_"):
        return "fixed"
    if name.startswith("random_"):
        return "random"
    return "unknown"


def infer_lighting(source_dir: Path):
    name = source_dir.name.lower()
    if "dual_light" in name:
        return "dual_light"
    if "left_light" in name:
        return "left_light"
    if "no_light" in name:
        return "no_light"
    return "unknown"


def read_wrist_images(obs_group: h5py.Group):
    if "images_by_camera" in obs_group and "wrist" in obs_group["images_by_camera"]:
        return np.asarray(obs_group["images_by_camera"]["wrist"])
    return np.asarray(obs_group["images"])


def copy_episode_wrist_only(src_path: Path, dst_path: Path, source_name: str):
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key in ["action", "teleop_scale"]:
            if key in src:
                src.copy(key, dst)

        obs_src = src["observations"]
        obs_dst = dst.create_group("observations")

        wrist_images = read_wrist_images(obs_src)
        obs_dst.create_dataset("images", data=wrist_images, compression="gzip")

        images_by_camera = obs_dst.create_group("images_by_camera")
        images_by_camera.create_dataset("wrist", data=wrist_images, compression="gzip")

        for key in ["gripper", "qpos", "qpos_end", "qpos_joint", "timestamps"]:
            if key in obs_src:
                src.copy(f"observations/{key}", obs_dst)

        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["mixed_source"] = source_name
        dst.attrs["camera_mode"] = "wrist_only"
        dst.attrs["primary_camera"] = "wrist"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="recorded_data")
    parser.add_argument("--source_patterns", nargs="+", default=["fixed_*", "random_*"])
    parser.add_argument("--output_dir", default="recorded_data/mixed_wrist_poslight_split")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    source_dirs = []
    for pattern in args.source_patterns:
        source_dirs.extend(sorted(data_root.glob(pattern)))
    source_dirs = sorted(set([directory for directory in source_dirs if directory.is_dir()]))

    if len(source_dirs) == 0:
        raise ValueError(f"No source directories found under {data_root} with patterns: {args.source_patterns}")

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    source_summaries = {}

    for source_dir in source_dirs:
        files = collect_episode_files(source_dir)
        if len(files) == 0:
            continue

        position_type = infer_position_type(source_dir)
        lighting = infer_lighting(source_dir)
        stratum = f"{position_type}|{lighting}"

        for file_path in files:
            grouped[stratum].append(
                {
                    "file": file_path,
                    "source_dir": source_dir.name,
                    "position_type": position_type,
                    "lighting": lighting,
                    "stratum": stratum,
                }
            )

        source_summaries[source_dir.name] = {
            "count": len(files),
            "position_type": position_type,
            "lighting": lighting,
            "stratum": stratum,
        }

    if len(grouped) == 0:
        raise ValueError(f"No episode files found in matched source directories: {source_dirs}")

    train_items = []
    test_items = []

    for stratum, items in sorted(grouped.items()):
        files = [item["file"] for item in items]
        train_files, _ = split_stratified(files, args.train_ratio, args.seed)
        train_set = set(train_files)
        for item in items:
            if item["file"] in train_set:
                train_items.append(item)
            else:
                test_items.append(item)

    manifest = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "stratify_by": ["position_type", "lighting"],
        "source_patterns": args.source_patterns,
        "sources": source_summaries,
        "splits": {
            "train": [],
            "test": [],
        },
        "summary": {},
        "stratum_summary": {},
    }

    counters = {"train": 0, "test": 0}
    running_idx = {"train": 0, "test": 0}
    stratum_counters = defaultdict(lambda: {"train": 0, "test": 0, "total": 0})

    def export_split(items, split_name):
        for item in items:
            src_file = item["file"]
            dst_name = f"episode_{running_idx[split_name]:04d}.hdf5"
            dst_file = (train_dir if split_name == "train" else test_dir) / dst_name
            running_idx[split_name] += 1

            copy_episode_wrist_only(src_file, dst_file, source_name=item["source_dir"])

            manifest["splits"][split_name].append(
                {
                    "source": item["source_dir"],
                    "position_type": item["position_type"],
                    "lighting": item["lighting"],
                    "stratum": item["stratum"],
                    "src": str(src_file),
                    "dst": str(dst_file),
                }
            )

            counters[split_name] += 1
            stratum_counters[item["stratum"]][split_name] += 1
            stratum_counters[item["stratum"]]["total"] += 1

    export_split(train_items, "train")
    export_split(test_items, "test")

    manifest["summary"] = {
        "train_total": counters["train"],
        "test_total": counters["test"],
        "total": counters["train"] + counters["test"],
    }
    manifest["stratum_summary"] = dict(stratum_counters)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Done")
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))
    print(json.dumps(manifest["stratum_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
