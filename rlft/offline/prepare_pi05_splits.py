from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import tyro


@dataclass
class Args:
    source_root: str = "./recorded_data"
    output_root: str = "./recorded_data_splits"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 1
    link_mode: Literal["symlink", "copy"] = "symlink"
    force: bool = False


def _get_data_version(path: Path) -> str:
    with h5py.File(path, "r") as handle:
        value = handle.attrs.get("data_version", "unknown")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _split_counts(num_items: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    test_count = round(num_items * test_ratio)
    val_count = round(num_items * val_ratio)
    if num_items >= 10:
        test_count = max(1, test_count)
        val_count = max(1, val_count)
    while test_count + val_count >= num_items:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break
    train_count = num_items - val_count - test_count
    return train_count, val_count, test_count


def _materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        dst.write_bytes(src.read_bytes())


def main() -> None:
    args = tyro.cli(Args)
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output root already exists: {output_root}. Pass --force to overwrite.")
        for child in output_root.iterdir():
            if child.is_dir() and not child.is_symlink():
                for sub in child.rglob("*"):
                    if sub.is_file() or sub.is_symlink():
                        sub.unlink()
                for sub in sorted(child.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
                child.rmdir()
            else:
                child.unlink()

    episode_records: list[dict] = []
    for subset_dir in sorted(source_root.iterdir()):
        if not subset_dir.is_dir() or subset_dir.name == "video_exports":
            continue
        files = sorted(subset_dir.glob("episode_*.hdf5"))
        for file_path in files:
            episode_records.append(
                {
                    "source_subset": subset_dir.name,
                    "source_path": str(file_path.resolve()),
                    "source_filename": file_path.name,
                    "data_version": _get_data_version(file_path),
                }
            )

    if not episode_records:
        raise ValueError(f"No episode_*.hdf5 files found under {source_root}")

    rng = random.Random(args.seed)
    buckets: dict[tuple[str, str], list[dict]] = {}
    for record in episode_records:
        key = (record["source_subset"], record["data_version"])
        buckets.setdefault(key, []).append(record)

    split_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for key, records in sorted(buckets.items()):
        records = records.copy()
        rng.shuffle(records)
        train_count, val_count, test_count = _split_counts(len(records), args.val_ratio, args.test_ratio)
        split_records["train"].extend(records[:train_count])
        split_records["val"].extend(records[train_count: train_count + val_count])
        split_records["test"].extend(records[train_count + val_count: train_count + val_count + test_count])

    for split_name in split_records:
        split_records[split_name].sort(key=lambda item: (item["source_subset"], item["source_filename"]))

    summary = {
        "seed": args.seed,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "link_mode": args.link_mode,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "total_episodes": len(episode_records),
        "splits": {},
    }

    for split_name, records in split_records.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(records):
            destination = split_dir / f"episode_{index:06d}.hdf5"
            _materialize(Path(record["source_path"]), destination, args.link_mode)
            record["staged_path"] = str(destination)
        subset_counts: dict[str, int] = {}
        version_counts: dict[str, int] = {}
        for record in records:
            subset_counts[record["source_subset"]] = subset_counts.get(record["source_subset"], 0) + 1
            version_counts[record["data_version"]] = version_counts.get(record["data_version"], 0) + 1
        summary["splits"][split_name] = {
            "num_episodes": len(records),
            "subset_counts": subset_counts,
            "data_version_counts": version_counts,
        }
        manifest_path = output_root / f"{split_name}_manifest.json"
        manifest_path.write_text(json.dumps(records, indent=2) + "\n")

    summary_path = output_root / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
