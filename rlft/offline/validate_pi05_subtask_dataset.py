from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from rlft.offline.pi05_bridge.subtask_annotations import load_subtask_annotation_sidecar
from rlft.offline.pi05_bridge.task_semantics import load_pi05_task_semantics


@dataclass
class Args:
    dataset_root: str
    subtask_annotations_path: str
    task_semantics_path: str = "configs/pi05_task_semantics/pick_and_place_tape_into_cup.json"
    repo_id: str = "carm/pi05_local"
    policy_pretrained_path: Optional[str] = None
    tokenizer_path_override: Optional[str] = os.environ.get("PI05_TOKENIZER_PATH")
    device: str = "cpu"


def _load_data_index(dataset_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((dataset_root / "data").glob("*/*.parquet")):
        frames.append(pd.read_parquet(path, columns=["episode_index", "frame_index", "task_index"]))
    if not frames:
        raise FileNotFoundError(f"No data parquet files found under {dataset_root / 'data'}")
    return pd.concat(frames, ignore_index=True)


def _load_export_episode_map(dataset_root: Path) -> dict[int, str]:
    metadata_path = dataset_root / "pi05_bridge_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    return {int(record["episode_index"]): str(record["episode_id"]) for record in metadata["episodes"]}


def _validate_policy_preprocessor(args: Args, dataset_root: Path, expected_prompt: str) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from rlft.offline.eval_pi05 import Args as EvalArgs
    from rlft.offline.eval_pi05 import _load_policy_and_processors

    dataset = LeRobotDataset(args.repo_id, root=dataset_root)
    sample = dataset[0]
    if "Current subtask:" not in sample.get("task", ""):
        raise AssertionError("Raw dataset sample task does not contain Current subtask")

    eval_args = EvalArgs(
        dataset_root=str(dataset_root),
        policy_pretrained_path=str(Path(args.policy_pretrained_path).expanduser().resolve()),
        tokenizer_path_override=args.tokenizer_path_override,
        device=args.device,
    )
    _, preprocessor, _ = _load_policy_and_processors(eval_args, dataset)
    processed = preprocessor(sample)
    observation = processed.get("observation", processed)
    language_keys = [key for key in observation if "language" in str(key)]
    if not language_keys:
        raise AssertionError("PI05 preprocessor did not produce language token keys")
    if expected_prompt not in sample["task"]:
        raise AssertionError("Dataset prompt does not match expected subtask prompt")


def main() -> None:
    args = tyro.cli(Args)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    semantics = load_pi05_task_semantics(args.task_semantics_path)
    annotations = load_subtask_annotation_sidecar(args.subtask_annotations_path, semantics)
    expected_prompts = [semantics.prompt_for(name) for name in semantics.subtask_names]

    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    task_strings = list(tasks_df.index)
    missing_prompts = [prompt for prompt in expected_prompts if prompt not in task_strings]
    if missing_prompts:
        raise AssertionError(f"Missing expected task prompts in tasks.parquet: {missing_prompts}")

    frame_df = _load_data_index(dataset_root)
    episode_map = _load_export_episode_map(dataset_root)
    task_by_index = {int(row.task_index): str(index) for index, row in tasks_df.iterrows()}
    failures: list[str] = []

    for episode_index, group in frame_df.groupby("episode_index"):
        group = group.sort_values("frame_index")
        switches = int((group["task_index"].diff().fillna(0) != 0).sum())
        episode_id = episode_map[int(episode_index)]
        annotation = annotations.get(episode_id)
        if annotation is None:
            failures.append(f"missing annotation for exported episode {episode_index}: {episode_id}")
            continue
        if switches != 1:
            failures.append(f"{episode_id}: expected exactly one task_index switch, got {switches}")
        boundary = annotation.boundary_frame
        before = max(0, boundary - 10)
        after = min(annotation.num_frames - 1, boundary + 10)
        before_rows = group[group["frame_index"] == before]
        after_rows = group[group["frame_index"] == after]
        if before_rows.empty or after_rows.empty:
            failures.append(f"{episode_id}: missing boundary probe rows")
            continue
        before_prompt = task_by_index[int(before_rows.iloc[0]["task_index"])]
        after_prompt = task_by_index[int(after_rows.iloc[0]["task_index"])]
        if before_prompt != expected_prompts[0]:
            failures.append(f"{episode_id}: frame {before} prompt mismatch before boundary")
        if after_prompt != expected_prompts[1]:
            failures.append(f"{episode_id}: frame {after} prompt mismatch after boundary")

    if failures:
        raise AssertionError("\n".join(failures[:20]))

    if args.policy_pretrained_path:
        _validate_policy_preprocessor(args, dataset_root, expected_prompts[0])

    print(
        json.dumps(
            {
                "ok": True,
                "dataset_root": str(dataset_root),
                "num_exported_episodes": len(episode_map),
                "task_prompts": expected_prompts,
                "policy_preprocessor_smoke": bool(args.policy_pretrained_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
