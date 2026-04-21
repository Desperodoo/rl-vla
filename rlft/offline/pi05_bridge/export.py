from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rlft.datasets import create_carm_obs_process_fn, get_carm_data_info, load_carm_dataset

from .action_transform import transform_carm_raw_action_sequence
from .contract import Pi05BridgeContract


def _require_lerobot_dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "LeRobotDataset is unavailable. Install Hugging Face LeRobot first, e.g. 'pip install lerobot'."
        ) from exc
    return LeRobotDataset


def _build_lerobot_features(image_shape: tuple[int, int, int], state_dim: int, action_dim: int) -> dict[str, dict[str, Any]]:
    height, width, channels = image_shape
    return {
        "observation.image": {
            "dtype": "image",
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None,
        },
        "observation.ee_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None,
        },
    }


def export_carm_to_lerobot_dataset(
    demo_path: str,
    output_dir: str,
    contract: Pi05BridgeContract,
    num_episodes: Optional[int] = None,
) -> dict[str, Any]:
    LeRobotDataset = _require_lerobot_dataset_class()

    demo_path = str(Path(demo_path).expanduser())
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        shutil.rmtree(output_root)

    data_info = get_carm_data_info(demo_path, state_mode=contract.observation.state_mode)
    raw_dataset = load_carm_dataset(demo_path, num_episodes=num_episodes, verbose=True)
    process_obs = create_carm_obs_process_fn(
        output_format="NHWC",
        target_size=contract.observation.image_size,
        normalize_images=contract.observation.normalize_images,
        state_mode=contract.observation.state_mode,
    )

    first_images = raw_dataset["images"][0]
    processed_first = process_obs(first_images, raw_dataset["qpos_joint"][0], raw_dataset["qpos_end"][0])
    features = _build_lerobot_features(
        image_shape=tuple(processed_first["rgb"].shape[1:]),
        state_dim=processed_first["state"].shape[-1],
        action_dim=contract.action.target_dim,
    )

    repo_id = "carm/pi05_local"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=features,
        root=output_root,
        robot_type="carm",
        use_videos=False,
    )

    exported_episodes: list[dict[str, Any]] = []
    for episode_index, (images, qpos_joint, qpos_end, actions, timestamps) in enumerate(
        zip(
            raw_dataset["images"],
            raw_dataset["qpos_joint"],
            raw_dataset["qpos_end"],
            raw_dataset["action"],
            raw_dataset["timestamps"],
        )
    ):
        obs = process_obs(images, qpos_joint, qpos_end)
        bridge_actions = transform_carm_raw_action_sequence(
            actions,
            obs["ee_pose"],
            contract.action.representation,
        )
        for frame_index in range(actions.shape[0]):
            dataset.add_frame(
                {
                    "task": "carm_fixed_dual_light",
                    "observation.image": obs["rgb"][frame_index],
                    "observation.state": obs["state"][frame_index].astype(np.float32),
                    "observation.ee_pose": obs["ee_pose"][frame_index].astype(np.float32),
                    "action": bridge_actions[frame_index].astype(np.float32),
                }
            )
        dataset.save_episode()
        exported_episodes.append(
            {
                "episode_index": episode_index,
                "num_steps": int(actions.shape[0]),
                "timestamp_min": float(np.min(timestamps)),
                "timestamp_max": float(np.max(timestamps)),
            }
        )

    metadata = {
        "format": "lerobot_native_local_v1",
        "source": {
            "demo_path": demo_path,
            "num_episodes": len(exported_episodes),
            "data_info": data_info,
            "raw_action_dim": int(raw_dataset["action"][0].shape[-1]),
        },
        "bridge_contract": contract.as_metadata(),
        "exported_action_dim": contract.action.target_dim,
        "episodes": exported_episodes,
        "repo_id": repo_id,
    }

    with open(output_root / "pi05_bridge_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "dataset_path": str(output_root),
        "metadata_path": str(output_root / "pi05_bridge_metadata.json"),
        "num_episodes": len(exported_episodes),
        "data_info": data_info,
        "repo_id": repo_id,
    }
