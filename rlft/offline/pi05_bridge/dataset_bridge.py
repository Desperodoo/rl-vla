from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from rlft.datasets import ActionNormalizer, load_carm_dataset, create_carm_obs_process_fn

from .contract import Pi05BridgeContract


@dataclass
class Pi05EpisodeWindow:
    episode_index: int
    start: int
    end: int


class Pi05LeRobotDatasetBridge(Dataset):
    """Thin dataset bridge from CARM HDF5 episodes to a LeRobot-style sample dict.

    Current scope:
    - single RGB stream
    - state from create_carm_obs_process_fn(...)
    - 8D absolute teleop action passthrough for v2 datasets
    - optional action normalization reused from rlft.datasets.ActionNormalizer
    """

    def __init__(
        self,
        data_path: str,
        contract: Pi05BridgeContract,
        num_episodes: Optional[int] = None,
        action_normalizer: Optional[ActionNormalizer] = None,
    ) -> None:
        self.contract = contract
        self.action_normalizer = action_normalizer
        self.raw_data = load_carm_dataset(data_path, num_episodes=num_episodes)
        self.obs_process_fn = create_carm_obs_process_fn(
            output_format=contract.observation.image_layout,
            target_size=contract.observation.image_size,
            normalize_images=contract.observation.normalize_images,
            state_mode=contract.observation.state_mode,
        )

        self.episodes = []
        self.windows: list[Pi05EpisodeWindow] = []
        all_actions = []

        for ep_idx in range(len(self.raw_data["images"])):
            images = self.raw_data["images"][ep_idx]
            qpos_joint = self.raw_data["qpos_joint"][ep_idx]
            qpos_end = self.raw_data["qpos_end"][ep_idx]
            actions = self.raw_data["action"][ep_idx].astype(np.float32)

            if actions.shape[-1] != contract.action.target_dim:
                raise ValueError(
                    f"Expected CARM v2 action dim {contract.action.target_dim}, got {actions.shape[-1]}"
                )

            obs = self.obs_process_fn(images, qpos_joint, qpos_end)
            episode = {
                "rgb": obs["rgb"],
                "state": obs["state"],
                "ee_pose": obs["ee_pose"],
                "action": actions,
            }
            self.episodes.append(episode)
            all_actions.append(actions)

            episode_len = actions.shape[0]
            horizon = contract.action_horizon
            stride = contract.window_stride
            for start in range(0, max(episode_len - horizon + 1, 1), stride):
                end = min(start + horizon, episode_len)
                self.windows.append(Pi05EpisodeWindow(ep_idx, start, end))

        if self.action_normalizer is not None and all_actions:
            self.action_normalizer.fit(np.concatenate(all_actions, axis=0))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        window = self.windows[index]
        episode = self.episodes[window.episode_index]
        start, end = window.start, window.end
        horizon = self.contract.action_horizon

        rgb = episode["rgb"][start:end]
        state = episode["state"][start:end]
        ee_pose = episode["ee_pose"][start:end]
        action = episode["action"][start:end]

        if end - start < horizon:
            pad = horizon - (end - start)
            rgb = np.concatenate([rgb, np.repeat(rgb[-1:], pad, axis=0)], axis=0)
            state = np.concatenate([state, np.repeat(state[-1:], pad, axis=0)], axis=0)
            ee_pose = np.concatenate([ee_pose, np.repeat(ee_pose[-1:], pad, axis=0)], axis=0)
            action = np.concatenate([action, np.repeat(action[-1:], pad, axis=0)], axis=0)

        normalized_action = action
        if self.action_normalizer is not None:
            normalized_action = self.action_normalizer.transform(action)

        sample = {
            self.contract.observation.image_key: torch.from_numpy(rgb).float(),
            self.contract.observation.state_key: torch.from_numpy(state).float(),
            self.contract.observation.ee_pose_key: torch.from_numpy(ee_pose).float(),
            self.contract.action.action_key: torch.from_numpy(normalized_action).float(),
            "action_unnormalized": torch.from_numpy(action).float(),
            "episode_index": torch.tensor(window.episode_index, dtype=torch.long),
            "start_index": torch.tensor(start, dtype=torch.long),
        }
        return sample

    def get_action_stats(self) -> Optional[dict]:
        if self.action_normalizer is None or self.action_normalizer.stats is None:
            return None
        return {
            "mode": self.action_normalizer.mode,
            "stats": {k: v.tolist() for k, v in self.action_normalizer.stats.items()},
        }


def build_pi05_dataset_bridge(
    data_path: str,
    contract: Pi05BridgeContract,
    num_episodes: Optional[int] = None,
    normalize_actions: bool = True,
    action_norm_mode: str = "standard",
) -> Pi05LeRobotDatasetBridge:
    action_normalizer = None
    if normalize_actions:
        action_normalizer = ActionNormalizer(mode=action_norm_mode)

    return Pi05LeRobotDatasetBridge(
        data_path=data_path,
        contract=contract,
        num_episodes=num_episodes,
        action_normalizer=action_normalizer,
    )
