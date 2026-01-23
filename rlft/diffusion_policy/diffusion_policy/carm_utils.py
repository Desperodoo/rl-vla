"""
CARM Robot Data Utilities

Tools for loading and processing CARM robot demonstration data.
Designed to be compatible with the action space used in infer_g3_api.py.

Data Format (CARM HDF5):
    observations/
        images          [T, H, W, 3]      # RGB images
        qpos_joint      [T, 7]            # joint positions (6) + gripper (1)
        qpos_end        [T, 8]            # end effector pose (7) + gripper (1)
        qpos            [T, 15]           # combined state
        gripper         [T]               # gripper position
        timestamps      [T]               # timestamps
    action              [T, 15]           # joint(6) + gripper(1) + end_pose(7) + gripper(1)

Action Space Design (aligned with infer_g3_api.py):
    - Policy outputs: [joint(6), gripper(1), relative_end_pose(7), gripper(1)] = 15D
    - relative_end_pose is a transformation relative to current pose
    - At inference: target_pose = current_pose @ relative_pose_transform
State Mode Options:
    - joint_only: qpos_joint [7] = 6 joints + 1 gripper
    - ee_only: qpos_end [8] = 7 ee_pose (xyz + quat) + 1 gripper  
    - both: [qpos_joint[7], qpos_end[:7]] [14] = 7 joint + 7 ee_pose (gripper from joint)"""

import os
import glob
import json
import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Any, Callable, Literal
from h5py import File
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# ============================================================================
# Pose Transformation Utilities
# ============================================================================

def pose_to_transform_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Convert pose (xyz + quaternion) to 4x4 transformation matrix.
    
    Args:
        position: Translation [x, y, z]
        quaternion: Quaternion [qx, qy, qz, qw]
        
    Returns:
        4x4 transformation matrix
    """
    rotation = R.from_quat(quaternion).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def transform_matrix_to_pose(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 4x4 transformation matrix to pose (xyz + quaternion).
    
    Args:
        transform: 4x4 transformation matrix
        
    Returns:
        position: [x, y, z]
        quaternion: [qx, qy, qz, qw]
    """
    position = transform[:3, 3]
    quaternion = R.from_matrix(transform[:3, :3]).as_quat()
    return position, quaternion


def compute_relative_pose_transform(pose_current: np.ndarray, pose_target: np.ndarray) -> np.ndarray:
    """
    Compute relative pose transformation from current to target.
    
    This is the INVERSE of compute_relative_pose() in infer_g3_api.py.
    
    In infer_g3_api.py:
        target_pose = current_pose @ relative_transform
        
    So here we compute:
        relative_transform = current_pose^{-1} @ target_pose
    
    Args:
        pose_current: Current pose [x, y, z, qx, qy, qz, qw]
        pose_target: Target pose [x, y, z, qx, qy, qz, qw]
        
    Returns:
        relative_pose: Relative transformation [x, y, z, qx, qy, qz, qw]
    """
    T_current = pose_to_transform_matrix(pose_current[:3], pose_current[3:7])
    T_target = pose_to_transform_matrix(pose_target[:3], pose_target[3:7])
    
    # relative_transform = T_current^{-1} @ T_target
    T_relative = np.linalg.inv(T_current) @ T_target
    
    position, quaternion = transform_matrix_to_pose(T_relative)
    return np.concatenate([position, quaternion])


def apply_relative_pose_transform(pose_current: np.ndarray, relative_pose: np.ndarray) -> np.ndarray:
    """
    Apply relative pose transformation to current pose (same as compute_relative_pose in infer_g3_api.py).
    
    target_pose = current_pose @ relative_transform
    
    Args:
        pose_current: Current pose [x, y, z, qx, qy, qz, qw]
        relative_pose: Relative transformation [x, y, z, qx, qy, qz, qw]
        
    Returns:
        target_pose: Target pose [x, y, z, qx, qy, qz, qw]
    """
    T_current = pose_to_transform_matrix(pose_current[:3], pose_current[3:7])
    T_relative = pose_to_transform_matrix(relative_pose[:3], relative_pose[3:7])
    
    T_target = T_current @ T_relative
    
    position, quaternion = transform_matrix_to_pose(T_target)
    return np.concatenate([position, quaternion])


# ============================================================================
# CARM Data Loading
# ============================================================================

def load_carm_episode(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load a single CARM episode from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        Dictionary with observations and actions
    """
    data = {}
    with File(filepath, 'r') as f:
        # Load observations
        obs = f['observations']
        data['images'] = np.array(obs['images'])           # [T, H, W, 3]
        data['qpos_joint'] = np.array(obs['qpos_joint'])   # [T, 7] (6 joints + gripper)
        data['qpos_end'] = np.array(obs['qpos_end'])       # [T, 8] (7 pose + gripper)
        data['gripper'] = np.array(obs['gripper'])         # [T]
        data['timestamps'] = np.array(obs['timestamps'])   # [T]
        
        # Load actions if available
        if 'action' in f:
            data['action'] = np.array(f['action'])         # [T, 15]
        
        # Load metadata
        data['num_steps'] = f.attrs.get('num_steps', len(data['timestamps']))
        
    return data


def load_carm_dataset(
    data_dir: str,
    num_episodes: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[np.ndarray]]:
    """
    Load CARM dataset from directory containing HDF5 files.
    
    Args:
        data_dir: Directory containing episode HDF5 files
        num_episodes: Maximum number of episodes to load (None = all)
        verbose: Whether to print progress
        
    Returns:
        Dictionary with lists of arrays for each data field
    """
    data_dir = os.path.expanduser(data_dir)
    
    # Find all HDF5 files
    pattern = os.path.join(data_dir, "episode_*.hdf5")
    files = sorted(glob.glob(pattern))
    
    if len(files) == 0:
        raise ValueError(f"No episode files found in {data_dir}")
    
    if num_episodes is not None:
        files = files[:num_episodes]
    
    if verbose:
        print(f"Loading {len(files)} episodes from {data_dir}")
    
    dataset = {
        'images': [],
        'qpos_joint': [],
        'qpos_end': [],
        'gripper': [],
        'timestamps': [],
        'action': [],
    }
    
    iterator = tqdm(files, desc="Loading episodes") if verbose else files
    for filepath in iterator:
        episode = load_carm_episode(filepath)
        
        dataset['images'].append(episode['images'])
        dataset['qpos_joint'].append(episode['qpos_joint'])
        dataset['qpos_end'].append(episode['qpos_end'])
        dataset['gripper'].append(episode['gripper'])
        dataset['timestamps'].append(episode['timestamps'])
        
        if 'action' in episode:
            dataset['action'].append(episode['action'])
    
    return dataset


# ============================================================================
# Observation Processing
# ============================================================================

def get_state_dim_for_mode(state_mode: str) -> int:
    """
    Get the state dimension for a given state mode.
    
    Args:
        state_mode: One of 'joint_only', 'ee_only', 'both'
        
    Returns:
        State dimension
    """
    if state_mode == 'joint_only':
        return 7  # 6 joints + 1 gripper
    elif state_mode == 'ee_only':
        return 8  # 7 ee_pose (xyz + quat) + 1 gripper
    elif state_mode == 'both':
        return 14  # 7 joint (6 joints + gripper) + 7 ee_pose (no extra gripper)
    else:
        raise ValueError(f"Unknown state_mode: {state_mode}. Must be 'joint_only', 'ee_only', or 'both'")


def create_carm_obs_process_fn(
    output_format: str = "NCHW",
    target_size: Optional[Tuple[int, int]] = None,
    normalize_images: bool = True,
    state_mode: Literal["joint_only", "ee_only", "both"] = "joint_only",
) -> Callable:
    """
    Create observation processing function for CARM data.
    
    Aligned with infer_g3_api.py:
        - Image: RGB image 
        - State: Configurable via state_mode
    
    Args:
        output_format: "NCHW" for training, "NHWC" for storage
        target_size: Optional (H, W) for resizing images
        normalize_images: Whether to normalize images to [0, 1]
        state_mode: State composition mode:
            - 'joint_only': qpos_joint [7] = 6 joints + 1 gripper (default, original behavior)
            - 'ee_only': qpos_end [8] = 7 ee_pose (xyz + quat) + 1 gripper
            - 'both': concat [14] = qpos_joint[7] + qpos_end[:7] (gripper from joint only)
        
    Returns:
        Function that processes observations
    """
    
    def process_fn(
        images: np.ndarray,
        qpos_joint: np.ndarray,
        qpos_end: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Process CARM observations.
        
        Args:
            images: RGB images [T, H, W, 3]
            qpos_joint: Joint positions [T, 7] (6 joints + gripper)
            qpos_end: End effector pose [T, 8] (7 pose + gripper)
            
        Returns:
            Dict with 'rgb', 'state', 'ee_pose' keys
        """
        # Process images
        rgb = images.copy()
        
        # Resize if needed
        if target_size is not None:
            T = rgb.shape[0]
            resized = np.zeros((T, target_size[0], target_size[1], 3), dtype=np.uint8)
            for i in range(T):
                resized[i] = cv2.resize(rgb[i], (target_size[1], target_size[0]),
                                       interpolation=cv2.INTER_LINEAR)
            rgb = resized
        
        # Convert format
        if output_format == "NCHW":
            rgb = np.transpose(rgb, (0, 3, 1, 2))  # [T, C, H, W]
        
        # State: configurable based on state_mode
        if state_mode == 'joint_only':
            # Original behavior: qpos_joint [7] = 6 joints + 1 gripper
            state = qpos_joint.astype(np.float32)  # [T, 7]
        elif state_mode == 'ee_only':
            # EE pose only: qpos_end [8] = 7 ee_pose + 1 gripper
            state = qpos_end.astype(np.float32)  # [T, 8]
        elif state_mode == 'both':
            # Both joint and EE: concat [14] = qpos_joint[7] + qpos_end[:7]
            # Note: gripper is included in qpos_joint, so we don't duplicate from qpos_end
            state = np.concatenate([
                qpos_joint.astype(np.float32),  # [T, 7] (6 joints + gripper)
                qpos_end[:, :7].astype(np.float32),  # [T, 7] (ee_pose without gripper)
            ], axis=-1)  # [T, 14]
        else:
            raise ValueError(f"Unknown state_mode: {state_mode}")
        
        # Also provide ee_pose for action computation (always needed for relative pose)
        ee_pose = qpos_end[:, :7].astype(np.float32)  # [T, 7] (without gripper)
        
        return {
            'rgb': rgb,
            'state': state,
            'ee_pose': ee_pose,
        }
    
    return process_fn


# ============================================================================
# Action Normalization
# ============================================================================

class ActionNormalizer:
    """
    Normalize and denormalize actions for training stability.
    
    Supports two modes:
    - 'standard': (x - mean) / std
    - 'minmax': (x - min) / (max - min) * 2 - 1  (maps to [-1, 1])
    """
    
    def __init__(
        self,
        mode: str = 'standard',
        eps: float = 1e-6,
    ):
        self.mode = mode
        self.eps = eps
        self.stats = None
    
    def fit(self, actions: np.ndarray):
        """
        Compute normalization statistics from actions.
        
        Args:
            actions: Actions array [N, action_dim]
        """
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
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
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
        stats_serializable = {k: v.tolist() for k, v in self.stats.items()}
        with open(filepath, 'w') as f:
            json.dump({'mode': self.mode, 'stats': stats_serializable}, f, indent=2)
    
    def load(self, filepath: str):
        """Load normalization statistics from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.mode = data['mode']
        self.stats = {k: np.array(v) for k, v in data['stats'].items()}


# ============================================================================
# Dataset Information
# ============================================================================

def get_carm_data_info(
    data_dir: str,
    state_mode: Literal["joint_only", "ee_only", "both"] = "joint_only",
) -> Dict[str, Any]:
    """
    Get information about CARM dataset.
    
    Args:
        data_dir: Directory containing episode HDF5 files
        state_mode: State composition mode for computing state_dim
        
    Returns:
        Dict with dataset information
    """
    data_dir = os.path.expanduser(data_dir)
    
    # Try to load dataset_info.json if available
    info_path = os.path.join(data_dir, 'dataset_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            saved_info = json.load(f)
    else:
        saved_info = {}
    
    # Find episode files
    pattern = os.path.join(data_dir, "episode_*.hdf5")
    files = sorted(glob.glob(pattern))
    
    if len(files) == 0:
        raise ValueError(f"No episode files found in {data_dir}")
    
    # Load first episode to get dimensions
    with File(files[0], 'r') as f:
        obs = f['observations']
        image_shape = obs['images'].shape[1:]  # [H, W, C]
        qpos_joint_dim = obs['qpos_joint'].shape[-1]  # 7
        qpos_end_dim = obs['qpos_end'].shape[-1]  # 8
        
        if 'action' in f:
            action_dim = f['action'].shape[-1]  # 15
        else:
            action_dim = 15  # default
    
    # Compute state_dim based on state_mode
    state_dim = get_state_dim_for_mode(state_mode)
    
    info = {
        'num_episodes': len(files),
        'image_shape': list(image_shape),
        'state_dim': state_dim,  # Depends on state_mode
        'state_mode': state_mode,  # Record which mode was used
        'qpos_joint_dim': qpos_joint_dim,  # 7 (raw dimension)
        'qpos_end_dim': qpos_end_dim,  # 8 (raw dimension)
        'ee_pose_dim': 7,  # end effector pose without gripper
        'action_dim': action_dim,  # 15
        'gripper_dim': 1,
    }
    
    # Add saved info
    info.update(saved_info.get('summary', {}))
    
    return info


# ============================================================================
# State Encoder (from utils.py)
# ============================================================================

import torch.nn as nn


class StateEncoder(nn.Module):
    """MLP encoder for state observations.
    
    Projects state features to a latent space to align with visual features.
    This helps with multimodal fusion when combining state and visual inputs.
    
    Args:
        state_dim: Input state dimension
        hidden_dim: Hidden layer dimension (default: 128)
        out_dim: Output feature dimension (default: 256)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.out_dim = out_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state observations."""
        return self.mlp(state)
