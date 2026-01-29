"""
Data Loading and Processing Utilities.

Unified utilities for:
- HDF5 data loading (ManiSkill and CARM formats)
- Observation processing and encoding
- DataLoader utilities
"""

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque
from typing import Dict, Optional, List, Any, Tuple, Callable, Literal
from gymnasium import spaces
from h5py import Dataset, File, Group
from torch.utils.data.sampler import Sampler
from tqdm import tqdm


# =============================================================================
# DataLoader Utilities
# =============================================================================

class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler, resampling until specified iterations.
    
    Useful for training with fixed number of iterations instead of epochs.
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def worker_init_fn(worker_id, base_seed=None):
    """Initialize worker random seed for DataLoader."""
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)


# =============================================================================
# HDF5 Loading Utilities
# =============================================================================

def load_content_from_h5_file(file):
    """Recursively load content from HDF5 file or group."""
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unsupported h5 file type: {type(file)}")


def load_hdf5(path: str) -> Dict:
    """Load entire HDF5 file into memory."""
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret


def load_traj_hdf5(path: str, num_traj: Optional[int] = None) -> Dict:
    """Load trajectory HDF5 file (ManiSkill format).
    
    Args:
        path: Path to trajectory HDF5 file
        num_traj: Maximum number of trajectories to load (None = all)
        
    Returns:
        Dict with trajectory data, keys like 'traj_0', 'traj_1', ...
    """
    path = os.path.expanduser(path)
    print("Loading HDF5 file", path)
    file = File(path, "r")
    keys = list(file.keys())
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
        keys = keys[:num_traj]
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()
    print("Loaded")
    return ret


# =============================================================================
# ManiSkill Observation Processing
# =============================================================================
# 
# ManiSkill Demo File Formats:
# ============================
# ManiSkill saves demonstrations in HDF5 format with naming convention:
#   trajectory.{obs_mode}.{control_mode}.{sim_backend}.h5
#
# Examples:
#   - trajectory.none.pd_joint_delta_pos.cpu.h5      (no observations, CPU)
#   - trajectory.state.pd_ee_delta_pose.physx_cuda.h5 (state only, GPU)
#   - trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5   (with RGB, GPU)
#   - trajectory.rgbd.pd_ee_delta_pose.physx_cuda.h5  (with RGB+Depth, GPU)
#
# IMPORTANT: Only files with obs_mode != "none" have the 'obs' key!
# If you get KeyError: 'obs', you're using a "trajectory.none.*" file.
# Use "trajectory.rgb.*" or "trajectory.rgbd.*" for visual policy training.
#
# Raw Demo File Structure (obs_mode="rgb" or "rgbd"):
# ===================================================
# Each trajectory (traj_0, traj_1, ...) contains:
#   - obs/: Nested observation dictionary
#       - agent/: Robot proprioceptive state
#           - qpos: (T, N_joints) joint positions
#           - qvel: (T, N_joints) joint velocities
#       - extra/: Task-specific state info
#           - tcp_pose: (T, 7) tool center point pose [xyz + quaternion]
#           - goal_pos: (T, 3) goal position (if applicable)
#           - is_grasped: (T, 1) grasp state boolean
#           ... (varies by task)
#       - sensor_data/: Camera observations
#           - base_camera/: 
#               - rgb: (T, H, W, 3) uint8 RGB image
#               - depth: (T, H, W, 1) float32 depth map
#           - hand_camera/: (optional second camera)
#               - rgb: (T, H, W, 3)
#               - depth: (T, H, W, 1)
#       - sensor_param/: Camera intrinsics and extrinsics (not used for training)
#   - actions: (T, action_dim) actions taken
#   - terminated: (T,) episode termination flags
#   - truncated: (T,) episode truncation flags
#
# Observation Processing Pipeline:
# ================================
# Raw ManiSkill obs -> obs_process_fn -> {"state": (T, state_dim), "rgb": (T, C, H, W)}
#
# 1. State: Concatenate all arrays in obs["agent"] and obs["extra"]
#    Example: qpos(9) + qvel(9) + tcp_pose(7) + ... = state_dim
#
# 2. RGB: Stack all camera rgb images along channel dimension
#    Example: base_camera(H,W,3) + hand_camera(H,W,3) -> (H,W,6) -> transpose to (6,H,W)
#
# =============================================================================

def build_state_obs_extractor(env_id: str) -> Callable:
    """Build state extractor for ManiSkill environments.
    
    Returns a function that extracts all agent and extra state values
    from a raw ManiSkill observation dictionary.
    
    Args:
        env_id: Environment ID (currently unused, but allows per-env customization)
        
    Returns:
        Callable that takes obs dict and returns list of state arrays to concatenate
    """
    return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())


def create_obs_process_fn(env_id: str, output_format: str = "NCHW") -> Callable:
    """Factory function to create observation processing function.
    
    Creates a function that processes raw ManiSkill observations into a
    standardized format with 'state', 'rgb', and optionally 'depth' keys.
    
    The processing pipeline:
    1. RGB Processing:
       - Extract rgb arrays from all cameras in sensor_data
       - Concatenate along channel dimension: (T, H, W, 3*num_cameras)
       - Transpose to NCHW format if requested: (T, C, H, W)
    
    2. Depth Processing (if available):
       - Extract depth arrays from all cameras in sensor_data
       - Concatenate along channel dimension: (T, H, W, 1*num_cameras)
       - Transpose to NCHW format if requested: (T, C, H, W)
    
    3. State Processing:
       - Extract all values from agent dict (qpos, qvel, etc.)
       - Extract all values from extra dict (tcp_pose, goal_pos, etc.)
       - Concatenate into single state vector: (T, state_dim)
       - Handle type conversions (bool->float, float64->float32)
    
    Args:
        env_id: Environment ID (for state extractor configuration)
        output_format: RGB output format, "NCHW" (channel-first) or "NHWC" (channel-last)
                      PyTorch models typically expect NCHW
    
    Returns:
        obs_process_fn: Function that takes raw obs dict and returns
                       {"state": (T, state_dim), "rgb": (T, C, H, W), "depth": (T, C, H, W)}
                       
    Example:
        >>> obs_fn = create_obs_process_fn("LiftPegUpright-v1", "NCHW")
        >>> raw_obs = load_trajectory_obs(...)  # From HDF5
        >>> processed = obs_fn(raw_obs)
        >>> processed["state"].shape  # (T, 25) for LiftPegUpright
        >>> processed["rgb"].shape    # (T, 3, 128, 128) with single camera
    """
    state_extractor = build_state_obs_extractor(env_id)
    
    def obs_process_fn(obs):
        # =====================================================================
        # Handle state-only data (simple array format)
        # =====================================================================
        # State-only demos have obs as simple (T, state_dim) array
        if isinstance(obs, np.ndarray):
            state = obs.astype(np.float32) if obs.dtype != np.float32 else obs
            return {"state": state, "rgb": None, "depth": None}
        
        # =====================================================================
        # RGB Processing
        # =====================================================================
        # Extract rgb from each camera in sensor_data and concatenate
        # sensor_data structure: {"camera_name": {"rgb": array, "depth": array}, ...}
        rgb = None
        depth = None
        
        if "sensor_data" in obs:
            img_dict = obs["sensor_data"]
            
            # Process RGB
            rgb_list = [v["rgb"] for v in img_dict.values() if "rgb" in v]
            if rgb_list:
                rgb_nhwc = np.concatenate(rgb_list, axis=-1)  # (T, H, W, 3*num_cameras)
                if output_format == "NCHW":
                    rgb = np.transpose(rgb_nhwc, (0, 3, 1, 2))  # (T, C, H, W)
                else:
                    rgb = rgb_nhwc
            
            # Process Depth
            depth_list = [v["depth"] for v in img_dict.values() if "depth" in v]
            if depth_list:
                depth_nhwc = np.concatenate(depth_list, axis=-1)  # (T, H, W, 1*num_cameras)
                if output_format == "NCHW":
                    depth = np.transpose(depth_nhwc, (0, 3, 1, 2))  # (T, C, H, W)
                else:
                    depth = depth_nhwc
        
        # =====================================================================
        # State Processing  
        # =====================================================================
        # Extract and concatenate all state values from agent and extra dicts
        states_to_stack = state_extractor(obs)  # List of arrays
        processed_states = []
        for s in states_to_stack:
            arr = np.array(s)
            # Ensure 2D: (T, feature) - expand dim if (T,) to (T, 1)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            # Type conversions for compatibility
            if arr.dtype == np.bool_:
                arr = arr.astype(np.float32)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            processed_states.append(arr)
        
        state = np.hstack(processed_states)  # (T, state_dim)
        return {"state": state, "rgb": rgb, "depth": depth}
    
    return obs_process_fn


# =============================================================================
# CARM Data Loading and Processing
# =============================================================================

def load_carm_episode(filepath: str) -> Dict[str, np.ndarray]:
    """Load a single CARM episode from HDF5 file."""
    data = {}
    with File(filepath, 'r') as f:
        obs = f['observations']
        data['images'] = np.array(obs['images'])
        data['qpos_joint'] = np.array(obs['qpos_joint'])
        data['qpos_end'] = np.array(obs['qpos_end'])
        data['gripper'] = np.array(obs['gripper'])
        data['timestamps'] = np.array(obs['timestamps'])
        
        if 'action' in f:
            data['action'] = np.array(f['action'])
        
        data['num_steps'] = f.attrs.get('num_steps', len(data['timestamps']))
    
    return data


def load_carm_dataset(
    data_dir: str,
    num_episodes: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[np.ndarray]]:
    """Load CARM dataset from directory containing HDF5 files.
    
    Args:
        data_dir: Directory containing episode HDF5 files
        num_episodes: Maximum number of episodes to load
        verbose: Whether to print progress
        
    Returns:
        Dictionary with lists of arrays for each data field
    """
    data_dir = os.path.expanduser(data_dir)
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


def get_state_dim_for_mode(state_mode: str) -> int:
    """Get the state dimension for a given state mode."""
    if state_mode == 'joint_only':
        return 7  # 6 joints + 1 gripper
    elif state_mode == 'ee_only':
        return 8  # 7 ee_pose + 1 gripper
    elif state_mode == 'both':
        return 14  # 7 joint + 7 ee_pose
    else:
        raise ValueError(f"Unknown state_mode: {state_mode}")


def create_carm_obs_process_fn(
    output_format: str = "NCHW",
    target_size: Optional[Tuple[int, int]] = None,
    normalize_images: bool = True,
    state_mode: Literal["joint_only", "ee_only", "both"] = "joint_only",
) -> Callable:
    """Create observation processing function for CARM data.
    
    Args:
        output_format: "NCHW" for training, "NHWC" for storage
        target_size: Optional (H, W) for resizing images
        normalize_images: Whether to normalize images to [0, 1]
        state_mode: State composition mode
        
    Returns:
        Function that processes observations
    """
    def process_fn(
        images: np.ndarray,
        qpos_joint: np.ndarray,
        qpos_end: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        rgb = images.copy()
        
        if target_size is not None:
            T = rgb.shape[0]
            resized = np.zeros((T, target_size[0], target_size[1], 3), dtype=np.uint8)
            for i in range(T):
                resized[i] = cv2.resize(rgb[i], (target_size[1], target_size[0]),
                                       interpolation=cv2.INTER_LINEAR)
            rgb = resized
        
        if output_format == "NCHW":
            rgb = np.transpose(rgb, (0, 3, 1, 2))
        
        # State based on mode
        if state_mode == 'joint_only':
            state = qpos_joint.astype(np.float32)
        elif state_mode == 'ee_only':
            state = qpos_end.astype(np.float32)
        elif state_mode == 'both':
            state = np.concatenate([
                qpos_joint.astype(np.float32),
                qpos_end[:, :7].astype(np.float32),
            ], axis=-1)
        else:
            raise ValueError(f"Unknown state_mode: {state_mode}")
        
        ee_pose = qpos_end[:, :7].astype(np.float32)
        
        return {'rgb': rgb, 'state': state, 'ee_pose': ee_pose}
    
    return process_fn


def get_carm_data_info(
    data_dir: str,
    state_mode: Literal["joint_only", "ee_only", "both"] = "joint_only",
) -> Dict[str, Any]:
    """Get information about CARM dataset."""
    data_dir = os.path.expanduser(data_dir)
    
    info_path = os.path.join(data_dir, 'dataset_info.json')
    saved_info = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            saved_info = json.load(f)
    
    pattern = os.path.join(data_dir, "episode_*.hdf5")
    files = sorted(glob.glob(pattern))
    
    if len(files) == 0:
        raise ValueError(f"No episode files found in {data_dir}")
    
    with File(files[0], 'r') as f:
        obs = f['observations']
        image_shape = obs['images'].shape[1:]
        qpos_joint_dim = obs['qpos_joint'].shape[-1]
        qpos_end_dim = obs['qpos_end'].shape[-1]
        action_dim = f['action'].shape[-1] if 'action' in f else 15
    
    state_dim = get_state_dim_for_mode(state_mode)
    
    info = {
        'num_episodes': len(files),
        'image_shape': list(image_shape),
        'state_dim': state_dim,
        'state_mode': state_mode,
        'qpos_joint_dim': qpos_joint_dim,
        'qpos_end_dim': qpos_end_dim,
        'ee_pose_dim': 7,
        'action_dim': action_dim,
        'gripper_dim': 1,
    }
    info.update(saved_info.get('summary', {}))
    
    return info


# =============================================================================
# Observation Encoding
# =============================================================================

class ObservationStacker:
    """Observation stacker for online training pipelines.
    
    Manages observation history for temporal stacking.
    Supports both dict observations and tensor observations.
    
    Args:
        obs_horizon: Number of observation frames to stack
        num_envs: Number of parallel environments
    """
    
    def __init__(self, obs_horizon: int, num_envs: int = 1):
        self.obs_horizon = obs_horizon
        self.num_envs = num_envs
        self._deque = deque(maxlen=obs_horizon)
    
    def _to_dict(self, obs):
        """Convert observation to dict format if needed."""
        if isinstance(obs, dict):
            return obs
        # Tensor or ndarray - wrap in dict with 'state' key
        return {"state": obs}
    
    def reset(self, initial_obs):
        """Reset observation history with initial observation."""
        self._deque.clear()
        obs_dict = self._to_dict(initial_obs)
        for _ in range(self.obs_horizon):
            self._deque.append(obs_dict)
    
    def append(self, obs):
        """Append a new observation to the history."""
        self._deque.append(self._to_dict(obs))
    
    def get_stacked(self) -> Dict[str, torch.Tensor]:
        """Get stacked observations.
        
        Returns:
            Dict with keys from observations, values stacked along dim=1
            Shape: {key: (B, T, *obs_shape)}
        """
        obs_list = list(self._deque)
        while len(obs_list) < self.obs_horizon:
            obs_list.insert(0, obs_list[0])
        
        result = {}
        for k in obs_list[0].keys():
            values = [o[k] for o in obs_list]
            # Handle both tensor and ndarray
            if isinstance(values[0], np.ndarray):
                result[k] = np.stack(values, axis=1)
            else:
                result[k] = torch.stack(values, dim=1)
        return result
    
    def __len__(self) -> int:
        return len(self._deque)


def encode_observations(
    obs_seq: Dict[str, torch.Tensor],
    visual_encoder: Optional[nn.Module],
    include_rgb: bool,
    device: torch.device,
) -> torch.Tensor:
    """Encode observation sequence to get conditioning features.
    
    Unified function for encoding observations across all training pipelines.
    Supports both NCHW (offline data) and NHWC (online environment) input formats.
    
    Args:
        obs_seq: Dict with 'state' and optionally 'rgb' observations
        visual_encoder: Visual encoder module (can be None for state-only)
        include_rgb: Whether to include RGB in observations
        device: Device for output tensor
        
    Returns:
        Flattened observation conditioning tensor [B, T * (visual_dim + state_dim)]
    """
    state = obs_seq["state"]
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state)
    state = state.float().to(device)
    
    B = state.shape[0]
    T = state.shape[1]
    
    features_list = []
    
    # Visual features
    if include_rgb and visual_encoder is not None and "rgb" in obs_seq:
        rgb = obs_seq["rgb"]
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        rgb = rgb.to(device)
        
        # Auto-detect format and convert to NCHW if needed
        if rgb.dim() == 5:
            if rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:  # Common channel counts
                rgb = rgb.permute(0, 1, 4, 2, 3)
        
        rgb_flat = rgb.reshape(B * T, *rgb.shape[2:]).float()
        
        if rgb_flat.max() > 1.0:
            rgb_flat = rgb_flat / 255.0
        
        visual_feat = visual_encoder(rgb_flat)
        visual_feat = visual_feat.view(B, T, -1)
        features_list.append(visual_feat)
    
    features_list.append(state)
    obs_features = torch.cat(features_list, dim=-1)
    obs_cond = obs_features.reshape(B, -1)
    
    return obs_cond
