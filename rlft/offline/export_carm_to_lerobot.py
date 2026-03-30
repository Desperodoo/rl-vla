from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Literal

import tyro

from rlft.offline.pi05_bridge import (
    Pi05BridgeContract,
    Pi05ObservationContract,
    export_carm_to_lerobot_dataset,
    validate_lerobot_dataset_path,
)


@dataclass
class Args:
    demo_path: str = "~/rl-vla/recorded_data/mix"
    output_dir: str = "~/rl-vla/runs/pi05_lerobot_export"
    num_demos: Optional[int] = None
    state_mode: Literal["joint_only", "ee_only", "both"] = "joint_only"
    obs_horizon: int = 2
    action_horizon: int = 16
    window_stride: int = 1


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
        action=base_contract.action,
    )

    export_result = export_carm_to_lerobot_dataset(
        demo_path=args.demo_path,
        output_dir=args.output_dir,
        contract=contract,
        num_episodes=args.num_demos,
    )
    validation_result = validate_lerobot_dataset_path(export_result["dataset_path"])

    print(json.dumps({"export": export_result, "validation": validation_result}, indent=2))
    if not validation_result["summary"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
