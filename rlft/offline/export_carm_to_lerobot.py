from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import tyro

from rlft.offline.pi05_bridge import (
    Pi05BridgeContract,
    Pi05ActionContract,
    Pi05ObservationContract,
    export_carm_to_lerobot_dataset,
    validate_lerobot_dataset_path,
)


@dataclass
class Args:
    demo_path: str = "~/rl-vla/recorded_data/mix"
    output_dir: str = "~/rl-vla/runs/pi05_lerobot_export"
    num_demos: Optional[int] = None
    state_mode: Literal["joint_only", "ee_only", "both"] = "ee_only"
    action_representation: Literal["ee_delta_pose_gripper", "absolute_pose_gripper"] = "ee_delta_pose_gripper"
    obs_horizon: int = 2
    action_horizon: int = 16
    window_stride: int = 1
    episode_manifest: Optional[str] = None
    task_semantics_path: Optional[str] = None
    subtask_annotations_path: Optional[str] = None
    recorded_root: Optional[str] = None
    allow_needs_review_annotations: bool = False


def _load_manifest_episode_paths(path: str | None) -> list[str] | None:
    if path is None:
        return None
    manifest_path = Path(path).expanduser().resolve()
    with open(manifest_path) as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Episode manifest must be a JSON list: {manifest_path}")
    paths: list[str] = []
    for record in records:
        if not isinstance(record, dict) or "source_path" not in record:
            raise ValueError(f"Manifest records must contain source_path: {record!r}")
        paths.append(str(Path(record["source_path"]).expanduser().resolve()))
    return paths


def main() -> None:
    args = tyro.cli(Args)
    base_contract = Pi05BridgeContract(
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        window_stride=args.window_stride,
    )
    contract = Pi05BridgeContract(
        obs_horizon=base_contract.obs_horizon,
        action_horizon=base_contract.action_horizon,
        window_stride=base_contract.window_stride,
        observation=Pi05ObservationContract(
            image_key=base_contract.observation.image_key,
            state_key=base_contract.observation.state_key,
            ee_pose_key=base_contract.observation.ee_pose_key,
            image_layout=base_contract.observation.image_layout,
            state_mode=args.state_mode,
            image_size=base_contract.observation.image_size,
            normalize_images=base_contract.observation.normalize_images,
            include_depth=base_contract.observation.include_depth,
        ),
        action=Pi05ActionContract(representation=args.action_representation),
    )
    episode_paths = _load_manifest_episode_paths(args.episode_manifest)

    export_result = export_carm_to_lerobot_dataset(
        demo_path=args.demo_path,
        output_dir=args.output_dir,
        contract=contract,
        num_episodes=args.num_demos,
        episode_paths=episode_paths,
        task_semantics_path=args.task_semantics_path,
        subtask_annotations_path=args.subtask_annotations_path,
        recorded_root=args.recorded_root,
        allow_needs_review_annotations=args.allow_needs_review_annotations,
    )
    validation_result = validate_lerobot_dataset_path(export_result["dataset_path"])

    print(json.dumps({"export": export_result, "validation": validation_result}, indent=2))
    if not validation_result["summary"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
