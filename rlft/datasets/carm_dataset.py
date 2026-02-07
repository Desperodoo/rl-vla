"""
CARM Dataset for Real Robot Imitation Learning.

Handles relative pose computation aligned with inference behavior.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable, List
from tqdm import tqdm

from .data_utils import load_carm_dataset


# =============================================================================
# Pose Transformation Utilities (imported from shared module)
# =============================================================================

from rlft.utils.pose_utils import (
    pose_to_transform_matrix,
    transform_matrix_to_pose,
    compute_relative_pose_transform,
)


# =============================================================================
# Action Normalizer
# =============================================================================

class ActionNormalizer:
    """Normalize and denormalize actions for training stability.
    
    Supports two modes:
    - 'standard': (x - mean) / std
    - 'minmax': (x - min) / (max - min) * 2 - 1 (maps to [-1, 1])
    """
    
    def __init__(self, mode: str = 'standard', eps: float = 1e-6):
        self.mode = mode
        self.eps = eps
        self.stats = None
    
    def fit(self, actions: np.ndarray):
        """Compute normalization statistics from actions."""
        if self.mode == 'standard':
            self.stats = {
                'mean': np.mean(actions, axis=0),
                'std': np.std(actions, axis=0) + self.eps,
            }
        elif self.mode == 'minmax':
            self.stats = {
                'min': np.min(actions, axis=0),
                'max': np.max(actions, axis=0),
            }
    
    def transform(self, actions: np.ndarray) -> np.ndarray:
        """Normalize actions."""
        if self.stats is None:
            raise ValueError("Call fit() first")
        
        if self.mode == 'standard':
            return (actions - self.stats['mean']) / self.stats['std']
        elif self.mode == 'minmax':
            range_val = self.stats['max'] - self.stats['min'] + self.eps
            return (actions - self.stats['min']) / range_val * 2 - 1
    
    def inverse_transform(self, normalized_actions: np.ndarray) -> np.ndarray:
        """Denormalize actions."""
        if self.stats is None:
            raise ValueError("Call fit() first")
        
        if self.mode == 'standard':
            return normalized_actions * self.stats['std'] + self.stats['mean']
        elif self.mode == 'minmax':
            range_val = self.stats['max'] - self.stats['min'] + self.eps
            return (normalized_actions + 1) / 2 * range_val + self.stats['min']
    
    def save(self, filepath: str):
        """Save normalization statistics to JSON."""
        import json
        stats_serializable = {k: v.tolist() for k, v in self.stats.items()}
        with open(filepath, 'w') as f:
            json.dump({'mode': self.mode, 'stats': stats_serializable}, f, indent=2)
    
    def load(self, filepath: str):
        """Load normalization statistics from JSON."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.mode = data['mode']
        self.stats = {k: np.array(v) for k, v in data['stats'].items()}

    @classmethod
    def from_checkpoint(cls, ckpt_data: dict) -> 'ActionNormalizer':
        """Create ActionNormalizer from checkpoint data.
        
        Args:
            ckpt_data: dict with 'mode' and 'stats' keys, as saved by train_carm.py's save_ckpt().
                       stats values can be lists (from .tolist()) or np.ndarray.
        
        Returns:
            ActionNormalizer with loaded stats.
        """
        normalizer = cls(mode=ckpt_data['mode'])
        normalizer.stats = {k: np.array(v) for k, v in ckpt_data['stats'].items()}
        return normalizer


# =============================================================================
# CARM Dataset
# =============================================================================

class CARMDataset(Dataset):
    """Dataset for CARM robot demonstrations.
    
    Loads demonstrations from CARM HDF5 files and processes them for training.
    Computes relative actions ensuring all actions in a prediction horizon
    are relative to the observation frame's pose.
    
    Action modes:
        - 'full': [joint(6), relative_end_pose(7)] = 13D (gripper is discrete)
        - 'ee_only': [relative_end_pose(7)] = 7D (gripper is discrete)
    
    Args:
        data_path: Path to directory containing HDF5 files
        obs_process_fn: Function to process observations
        device: Device to store tensors on
        num_episodes: Number of episodes to load (None = all)
        obs_horizon: Observation stacking horizon
        pred_horizon: Action prediction horizon
        action_mode: 'full' or 'ee_only'
        precompute_actions: If True, precompute all relative actions at init
        action_normalizer: Optional action normalizer
        gripper_threshold: Threshold for discrete gripper classification
    """
    
    def __init__(
        self,
        data_path: str,
        obs_process_fn: Callable,
        device,
        num_episodes: Optional[int],
        obs_horizon: int,
        pred_horizon: int,
        action_mode: str = "full",
        precompute_actions: bool = False,
        action_normalizer: Optional[ActionNormalizer] = None,
        gripper_threshold: float = 0.05,
    ):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.device = device
        self.action_mode = action_mode
        self.precompute_actions = precompute_actions
        self.action_normalizer = action_normalizer
        self.gripper_threshold = gripper_threshold
        
        # Determine action dimension (continuous only, gripper is discrete)
        self.action_dim = 13 if action_mode == 'full' else 7
        
        # Load dataset
        print(f"Loading CARM dataset from {data_path}...")
        raw_data = load_carm_dataset(data_path, num_episodes=num_episodes)
        
        print("Processing trajectories...")
        
        trajectories = {
            "observations": [],
            "raw_actions": [],
            "qpos_end": [],
        }
        
        all_relative_actions = []
        
        for ep_idx in tqdm(range(len(raw_data['images'])), desc="Processing episodes"):
            images = raw_data['images'][ep_idx]
            qpos_joint = raw_data['qpos_joint'][ep_idx]
            qpos_end = raw_data['qpos_end'][ep_idx]
            
            # Process observations
            obs_dict = obs_process_fn(images, qpos_joint, qpos_end)
            
            processed_obs = {
                'rgb': torch.from_numpy(obs_dict['rgb']).to(device),
                'state': torch.from_numpy(obs_dict['state']).to(device),
            }
            
            raw_actions = raw_data['action'][ep_idx]
            
            trajectories["observations"].append(processed_obs)
            trajectories["raw_actions"].append(raw_actions)
            trajectories["qpos_end"].append(qpos_end[:, :7])
            
            # Compute sample relative actions for normalization stats
            for t in range(0, len(raw_actions) - pred_horizon, pred_horizon):
                ref_pose = qpos_end[t, :7]
                for k in range(pred_horizon):
                    target_pose = raw_actions[t + k, 7:14]
                    relative_pose = compute_relative_pose_transform(ref_pose, target_pose)
                    
                    if self.action_mode == 'full':
                        rel_action = np.zeros(self.action_dim, dtype=np.float32)
                        rel_action[:6] = raw_actions[t + k, :6]
                        rel_action[6:13] = relative_pose
                    else:
                        rel_action = relative_pose.astype(np.float32)
                    
                    all_relative_actions.append(rel_action)
        
        # Compute action normalization stats
        if self.action_normalizer is not None and len(all_relative_actions) > 0:
            all_relative_actions = np.array(all_relative_actions)
            self.action_normalizer.fit(all_relative_actions)
            print(f"Action normalization stats computed on {len(all_relative_actions)} samples")
        
        self.obs_keys = list(processed_obs.keys())
        print(f"Obs keys: {self.obs_keys}")
        
        # Compute slices
        print("Computing slice indices...")
        self.slices = []
        num_traj = len(trajectories["observations"])
        total_transitions = 0
        
        for traj_idx in range(num_traj):
            L = trajectories["raw_actions"][traj_idx].shape[0]
            total_transitions += L
            
            pad_before = obs_horizon - 1
            for start in range(-pad_before, L - pred_horizon + 1):
                self.slices.append((traj_idx, start, start + pred_horizon))
        
        print(f"Total transitions: {total_transitions}, Total sequences: {len(self.slices)}")
        self.trajectories = trajectories
        
        # Precompute all relative actions if requested
        if self.precompute_actions:
            print("Precomputing relative actions for all slices...")
            self._precompute_all_actions()
    
    def _precompute_all_actions(self):
        """Precompute relative actions and gripper labels for all slices."""
        num_slices = len(self.slices)
        self.precomputed_actions = torch.zeros(
            (num_slices, self.pred_horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.precomputed_gripper_labels = torch.zeros(
            (num_slices, self.pred_horizon),
            dtype=torch.long,
            device=self.device,
        )
        
        for idx in tqdm(range(num_slices), desc="Precomputing actions"):
            traj_idx, start, end = self.slices[idx]
            raw_actions = self.trajectories["raw_actions"][traj_idx]
            qpos_end = self.trajectories["qpos_end"][traj_idx]
            L = len(raw_actions)
            
            # Determine reference pose
            obs_frame_idx = max(0, start + self.obs_horizon - 1)
            ref_pose = qpos_end[obs_frame_idx]
            
            # Get action indices with padding
            act_indices = list(range(max(0, start), min(end, L)))
            if start < 0:
                act_indices = [0] * (-start) + act_indices
            if end > L:
                act_indices = act_indices + [L - 1] * (end - L)
            
            # Compute relative actions and gripper labels
            for k, act_idx in enumerate(act_indices):
                raw_action = raw_actions[act_idx]
                target_pose = raw_action[7:14]
                relative_pose = compute_relative_pose_transform(ref_pose, target_pose)
                gripper_val = raw_action[14]
                
                if self.action_mode == 'full':
                    self.precomputed_actions[idx, k, :6] = torch.from_numpy(raw_action[:6])
                    self.precomputed_actions[idx, k, 6:13] = torch.from_numpy(relative_pose)
                else:
                    self.precomputed_actions[idx, k, :7] = torch.from_numpy(relative_pose)
                
                self.precomputed_gripper_labels[idx, k] = 1 if gripper_val < self.gripper_threshold else 0
            
            # Apply normalization
            if self.action_normalizer is not None:
                act_np = self.precomputed_actions[idx].cpu().numpy()
                act_np = self.action_normalizer.transform(act_np)
                self.precomputed_actions[idx] = torch.from_numpy(act_np).to(self.device)
        
        # Free raw data
        del self.trajectories["raw_actions"]
        del self.trajectories["qpos_end"]
        
        print(f"Precomputed actions shape: {self.precomputed_actions.shape}")
    
    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        obs_traj = self.trajectories["observations"][traj_idx]
        
        # Get observation sequence
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start):start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
        
        # Get action sequence and gripper labels
        if self.precompute_actions:
            act_seq = self.precomputed_actions[index]
            gripper_label = self.precomputed_gripper_labels[index]
        else:
            raw_actions = self.trajectories["raw_actions"][traj_idx]
            qpos_end = self.trajectories["qpos_end"][traj_idx]
            L = len(raw_actions)
            
            obs_frame_idx = max(0, start + self.obs_horizon - 1)
            ref_pose = qpos_end[obs_frame_idx]
            
            act_indices = list(range(max(0, start), min(end, L)))
            if start < 0:
                act_indices = [0] * (-start) + act_indices
            if end > L:
                act_indices = act_indices + [L - 1] * (end - L)
            
            act_seq_list = []
            gripper_label_list = []
            for idx in act_indices:
                raw_action = raw_actions[idx]
                target_pose = raw_action[7:14]
                relative_pose = compute_relative_pose_transform(ref_pose, target_pose)
                gripper_val = raw_action[14]
                
                if self.action_mode == 'full':
                    rel_action = np.zeros(self.action_dim, dtype=np.float32)
                    rel_action[:6] = raw_action[:6]
                    rel_action[6:13] = relative_pose
                else:
                    rel_action = relative_pose.astype(np.float32)
                
                act_seq_list.append(rel_action)
                gripper_label_list.append(1 if gripper_val < self.gripper_threshold else 0)
            
            act_seq = np.stack(act_seq_list, axis=0)
            gripper_label = np.array(gripper_label_list, dtype=np.int64)
            
            if self.action_normalizer is not None:
                act_seq = self.action_normalizer.transform(act_seq)
            
            act_seq = torch.from_numpy(act_seq).float().to(self.device)
            gripper_label = torch.from_numpy(gripper_label).long().to(self.device)
        
        assert obs_seq["state"].shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon
        assert gripper_label.shape[0] == self.pred_horizon
        
        return {
            "observations": obs_seq,
            "actions_cont": act_seq,
            "gripper_label": gripper_label,
        }
    
    def __len__(self):
        return len(self.slices)
