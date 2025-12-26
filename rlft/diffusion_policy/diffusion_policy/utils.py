import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Dict, Optional, List, Any, Tuple, Callable
from gymnasium import spaces
from h5py import Dataset, File, Group
from torch.utils.data.sampler import Sampler
from tqdm import tqdm


# ============================================================================
# State Encoder MLP
# ============================================================================

class StateEncoder(nn.Module):
    """MLP encoder for state observations.
    
    Projects state features to a latent space to align with visual features.
    This helps with multimodal fusion when combining state and visual inputs.
    
    Architecture: Linear -> ReLU -> Linear -> ReLU
    
    Args:
        state_dim: Input state dimension
        hidden_dim: Hidden layer dimension (default: 128)
        out_dim: Output feature dimension (default: 256)
    
    Example:
        >>> encoder = StateEncoder(state_dim=13, hidden_dim=128, out_dim=256)
        >>> state = torch.randn(32, 2, 13)  # [B, T, state_dim]
        >>> features = encoder(state)  # [B, T, 256]
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
        """Encode state observations.
        
        Args:
            state: State tensor of shape [B, state_dim], [B, T, state_dim], 
                   or [B*T, state_dim]
        
        Returns:
            Encoded features with same batch dimensions but out_dim as last dim
        """
        return self.mlp(state)


class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler.
    Resampling from it until a specified number of iterations have been sampled
    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
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
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


TARGET_KEY_TO_SOURCE_KEY = {
    "states": "env_states",
    "observations": "obs",
    "success": "success",
    "next_observations": "obs",
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    "actions": "actions",
}


def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_hdf5(
    path,
):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret


def load_traj_hdf5(path, num_traj=None):
    import os
    path = os.path.expanduser(path)  # Expand ~ to home directory
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


def load_demo_dataset(
    path, keys=["observations", "actions"], num_traj=None, concat=True
):
    # assert num_traj is None
    raw_data = load_traj_hdf5(path, num_traj)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    _traj = raw_data["traj_0"]
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [raw_data[idx][source_key] for idx in raw_data]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ["observations", "states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[:-1] for t in dataset[target_key]], axis=0
                )
            elif target_key in ["next_observations", "next_states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[1:] for t in dataset[target_key]], axis=0
                )
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print("Load", target_key, dataset[target_key].shape)
        else:
            print(
                "Load",
                target_key,
                len(dataset[target_key]),
                type(dataset[target_key][0]),
            )
    return dataset


def convert_obs(obs, concat_fn, transpose_fn, state_obs_extractor):
    """Convert ManiSkill observations to standard format.
    
    Args:
        obs: Raw observation dict from ManiSkill environment
        concat_fn: Function to concatenate arrays (e.g., np.concatenate)
        transpose_fn: Function to transpose images (e.g., np.transpose)
        state_obs_extractor: Function to extract state observations
        
    Returns:
        Dict with 'state' and 'rgb' keys in NCHW format
    """
    img_dict = obs["sensor_data"]

    new_img_dict = {
        "rgb": transpose_fn(
            concat_fn([v["rgb"] for v in img_dict.values()])
        )  # (C, H, W) or (B, C, H, W)
    }

    # Unified version
    states_to_stack = state_obs_extractor(obs)
    for j in range(len(states_to_stack)):
        if states_to_stack[j].dtype == np.float64:
            states_to_stack[j] = states_to_stack[j].astype(np.float32)
    try:
        state = np.hstack(states_to_stack)
    except:  # dirty fix for concat trajectory of states
        state = np.column_stack(states_to_stack)
    if state.dtype == np.float64:
        for x in states_to_stack:
            print(x.shape, x.dtype)
        import pdb

        pdb.set_trace()

    out_dict = {
        "state": state,
        "rgb": new_img_dict["rgb"],
    }

    return out_dict


def build_obs_space(env, state_obs_extractor):
    """Build observation space for environment.
    
    Args:
        env: Gymnasium environment
        state_obs_extractor: Function to extract state observations
        
    Returns:
        spaces.Dict with 'state' and 'rgb' keys
    """
    obs_space = env.observation_space

    # Unified version
    state_dim = sum([v.shape[0] for v in state_obs_extractor(obs_space)])

    single_img_space = next(iter(env.observation_space["image"].values()))
    h, w, _ = single_img_space["rgb"].shape
    n_images = len(env.observation_space["image"])

    return spaces.Dict(
        {
            "state": spaces.Box(
                -float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32
            ),
            "rgb": spaces.Box(0, 255, shape=(n_images * 3, h, w), dtype=np.uint8),
        }
    )


def build_state_obs_extractor(env_id):
    # NOTE: You can tune/modify state observations specific to each environment here as you wish. By default we include all data
    # but in some use cases you might want to exclude e.g. obs["agent"]["qvel"] as qvel is not always something you query in the real world.
    return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())


# ============================================================================
# Real Robot Data Adapters (ARX5)
# ============================================================================

def build_real_robot_state_extractor():
    """Build state extractor for ARX5 real robot data.
    
    ARX5 真机数据的state结构:
        obs/joint_pos: float32, (T, 6)      # 6DoF关节位置
        obs/joint_vel: float32, (T, 6)      # 6DoF关节速度
        obs/gripper_pos: float32, (T, 1)    # 夹爪位置
    
    Returns:
        State extractor function: obs -> list of arrays to stack
    """
    def extractor(obs):
        components = []
        
        # Joint positions (6DoF)
        if "joint_pos" in obs:
            components.append(np.array(obs["joint_pos"]))
        
        # Joint velocities (6DoF)
        if "joint_vel" in obs:
            components.append(np.array(obs["joint_vel"]))
        
        # Gripper position (1DoF)
        if "gripper_pos" in obs:
            gripper = np.array(obs["gripper_pos"])
            if gripper.ndim == 1:
                gripper = gripper[:, np.newaxis]
            components.append(gripper)
        
        return components
    
    return extractor


def create_real_robot_obs_process_fn(
    output_format: str = "NCHW",
    camera_name: str = "wrist",  # "wrist" (eye-in-hand) or "external" (eye-to-hand)
    include_depth: bool = False,
    target_size: Optional[Tuple[int, int]] = None,  # (H, W) for resizing, None = no resize
):
    """Create observation processing function for ARX5 real robot data.
    
    ARX5 真机数据结构:
        obs/images/wrist/rgb: uint8, (T, 256, 256, 3)      # Eye-in-hand RGB
        obs/images/wrist/depth: uint16, (T, 256, 256)      # Eye-in-hand Depth
        obs/images/external/rgb: uint8, (T, 256, 256, 3)   # Eye-to-hand RGB
        obs/images/external/depth: uint16, (T, 256, 256)   # Eye-to-hand Depth
        obs/joint_pos: float32, (T, 6)
        obs/joint_vel: float32, (T, 6)
        obs/gripper_pos: float32, (T, 1)
    
    Args:
        output_format: "NCHW" for GPU training, "NHWC" for storage
        camera_name: Which camera to use ("wrist" or "external")
        include_depth: Whether to include depth channel
        target_size: Optional (H, W) to resize images (e.g., (128, 128) for ManiSkill compat)
    
    Returns:
        obs_process_fn: Function that takes raw obs dict and returns
                       {"state": (T, state_dim), "rgb": (T, C, H, W)}
    """
    state_extractor = build_real_robot_state_extractor()
    
    def obs_process_fn(obs):
        """Process real robot observations.
        
        Args:
            obs: Raw observation dict from HDF5 file
        
        Returns:
            Dict with "state" and "rgb" keys
        """
        # ===== Process State =====
        states_to_stack = state_extractor(obs)
        
        processed_states = []
        for s in states_to_stack:
            arr = np.array(s)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            if arr.dtype == np.bool_:
                arr = arr.astype(np.float32)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            processed_states.append(arr)
        
        state = np.hstack(processed_states)  # (T, state_dim)
        
        # ===== Process Images =====
        # Get RGB from specified camera
        if "images" in obs and camera_name in obs["images"]:
            rgb = np.array(obs["images"][camera_name]["rgb"])  # (T, H, W, 3)
        else:
            raise KeyError(f"Camera '{camera_name}' not found in obs['images']. "
                          f"Available cameras: {list(obs.get('images', {}).keys())}")
        
        # Optionally resize images
        if target_size is not None:
            import cv2
            T = rgb.shape[0]
            resized = np.zeros((T, target_size[0], target_size[1], 3), dtype=np.uint8)
            for i in range(T):
                resized[i] = cv2.resize(rgb[i], (target_size[1], target_size[0]), 
                                       interpolation=cv2.INTER_LINEAR)
            rgb = resized
        
        # Optionally include depth
        if include_depth:
            if "depth" in obs["images"][camera_name]:
                depth = np.array(obs["images"][camera_name]["depth"])  # (T, H, W), uint16
                # Normalize depth to [0, 1] range (assuming depth_scale=1000, max ~10m)
                depth_normalized = depth.astype(np.float32) / 10000.0  # Max 10m
                depth_normalized = np.clip(depth_normalized, 0, 1)
                # Convert to uint8 for channel concatenation
                depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                
                # Resize depth if needed
                if target_size is not None:
                    import cv2
                    T = depth_uint8.shape[0]
                    resized_depth = np.zeros((T, target_size[0], target_size[1]), dtype=np.uint8)
                    for i in range(T):
                        resized_depth[i] = cv2.resize(depth_uint8[i], (target_size[1], target_size[0]),
                                                     interpolation=cv2.INTER_NEAREST)
                    depth_uint8 = resized_depth
                
                # Add channel dimension: (T, H, W) -> (T, H, W, 1)
                depth_uint8 = depth_uint8[..., np.newaxis]
                
                # Concatenate RGB + D: (T, H, W, 4)
                rgb = np.concatenate([rgb, depth_uint8], axis=-1)
            else:
                print(f"Warning: Depth requested but not found for camera '{camera_name}'")
        
        # Convert to target format
        if output_format == "NCHW":
            rgb = np.transpose(rgb, (0, 3, 1, 2))  # (T, C, H, W)
        # else: keep NHWC (T, H, W, C)
        
        return {"state": state, "rgb": rgb}
    
    return obs_process_fn


def get_real_robot_data_info(data_path: str) -> Dict[str, Any]:
    """Get information about real robot dataset.
    
    Reads the config.yaml and stats.json to provide dataset metadata.
    
    Args:
        data_path: Path to trajectory.h5 file
    
    Returns:
        Dict with dataset information:
            - state_dim: Total state dimension (joint_pos + joint_vel + gripper)
            - action_dim: Action dimension
            - image_size: (H, W) tuple
            - cameras: List of available camera names
            - has_depth: Whether depth is available
            - control_mode: Control mode from trajectory.json
            - num_trajectories: Number of trajectories
            - stats: Normalization statistics (if available)
    """
    import os
    import json
    import yaml
    
    data_dir = os.path.dirname(os.path.expanduser(data_path))
    info = {}
    
    # Read config.yaml
    config_path = os.path.join(data_dir, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        info["cameras"] = list(config.get("cameras", {}).keys())
        info["has_depth"] = any(
            cam.get("enable_depth", False) 
            for cam in config.get("cameras", {}).values()
        )
        info["robot_model"] = config.get("robot", {}).get("model", "unknown")
        info["joint_dof"] = config.get("robot", {}).get("joint_dof", 6)
    
    # Read stats.json
    stats_path = os.path.join(data_dir, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            info["stats"] = json.load(f)
    
    # Read trajectory.json
    traj_json_path = os.path.join(data_dir, "trajectory.json")
    if os.path.exists(traj_json_path):
        with open(traj_json_path, 'r') as f:
            traj_info = json.load(f)
        info["control_mode"] = traj_info.get("episodes", [{}])[0].get("control_mode", "joint_pos")
        info["num_trajectories"] = len(traj_info.get("episodes", []))
    
    # Read HDF5 to get dimensions
    from h5py import File
    with File(os.path.expanduser(data_path), 'r') as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        if traj_keys:
            first_traj = f[traj_keys[0]]
            info["action_dim"] = first_traj["actions"].shape[-1]
            
            # State dimension
            state_dim = 0
            if "joint_pos" in first_traj["obs"]:
                state_dim += first_traj["obs"]["joint_pos"].shape[-1]
            if "joint_vel" in first_traj["obs"]:
                state_dim += first_traj["obs"]["joint_vel"].shape[-1]
            if "gripper_pos" in first_traj["obs"]:
                gripper_shape = first_traj["obs"]["gripper_pos"].shape
                state_dim += gripper_shape[-1] if len(gripper_shape) > 1 else 1
            info["state_dim"] = state_dim
            
            # Image size
            if "images" in first_traj["obs"]:
                for cam_name in first_traj["obs"]["images"].keys():
                    rgb_shape = first_traj["obs"]["images"][cam_name]["rgb"].shape
                    info["image_size"] = (rgb_shape[1], rgb_shape[2])  # (H, W)
                    break
    
    return info


def create_obs_process_fn(
    env_id: str,
    output_format: str = "NCHW",
):
    """Factory function to create observation processing function.
    
    Creates a function that processes raw ManiSkill observations into a
    standardized format with 'state' and 'rgb' keys.
    
    This unifies observation processing across:
    - train_offline_rl.py (offline RL dataset loading)
    - train_rlpd_online.py (online RL with offline data mixing)
    - OfflineRLDataset (dataset class)
    
    Args:
        env_id: Environment ID (for state extractor configuration)
        output_format: RGB output format, "NCHW" or "NHWC"
            - NCHW: (T, C, H, W) - for GPU training, matches PyTorch conv layers
            - NHWC: (T, H, W, C) - for CPU storage, matches raw ManiSkill output
    
    Returns:
        obs_process_fn: Function that takes raw obs dict and returns
                       {"state": (T, state_dim), "rgb": (T, C, H, W) or (T, H, W, C)}
    
    Example:
        >>> obs_process_fn = create_obs_process_fn("PegInsertionSide-v1", output_format="NCHW")
        >>> processed = obs_process_fn(raw_obs)
        >>> print(processed["state"].shape)  # (T, 25)
        >>> print(processed["rgb"].shape)    # (T, 6, 128, 128) for NCHW
    """
    state_extractor = build_state_obs_extractor(env_id)
    
    def obs_process_fn(obs):
        """Process raw ManiSkill observations.
        
        Args:
            obs: Raw observation dict with:
                - obs["sensor_data"][cam_name]["rgb"]: (T, H, W, 3) uint8
                - obs["agent"][key]: (T, dim) state components
                - obs["extra"][key]: (T, dim) extra state components
        
        Returns:
            Dict with:
                - "state": (T, state_dim) float32
                - "rgb": (T, C, H, W) or (T, H, W, C) depending on output_format
        """
        # Process RGB: concatenate cameras along channel dimension
        img_dict = obs["sensor_data"]
        rgb_list = [v["rgb"] for v in img_dict.values()]
        
        # Concatenate along channel axis (NHWC format from ManiSkill)
        rgb_nhwc = np.concatenate(rgb_list, axis=-1)  # (T, H, W, C*num_cams)
        
        # Convert to target format
        if output_format == "NCHW":
            rgb = np.transpose(rgb_nhwc, (0, 3, 1, 2))  # (T, C, H, W)
        else:
            rgb = rgb_nhwc  # Keep NHWC
        
        # Process state: extract and stack all state components
        states_to_stack = state_extractor(obs)
        
        processed_states = []
        for s in states_to_stack:
            arr = np.array(s)
            # Handle 1D arrays: expand to 2D
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            # Convert bool to float32
            if arr.dtype == np.bool_:
                arr = arr.astype(np.float32)
            # Convert float64 to float32
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            processed_states.append(arr)
        
        state = np.hstack(processed_states)
        
        return {"state": state, "rgb": rgb}
    
    return obs_process_fn


# ============================================================================
# Observation Stacking and Encoding Utilities
# ============================================================================

class ObservationStacker:
    """Observation stacker for online training pipelines.
    
    Manages a deque of observations and provides methods to:
    - Append new observations
    - Get stacked observations with proper padding
    - Reset the observation history
    
    This class unifies the obs_deque (train_online_finetune.py) and 
    obs_history (train_rlpd_online.py) implementations.
    
    Args:
        obs_horizon: Number of observation frames to stack
        num_envs: Number of parallel environments (for reset handling)
    """
    
    def __init__(self, obs_horizon: int, num_envs: int = 1):
        self.obs_horizon = obs_horizon
        self.num_envs = num_envs
        self._deque = deque(maxlen=obs_horizon)
    
    def reset(self, initial_obs: Dict[str, torch.Tensor]):
        """Reset observation history with initial observation.
        
        Fills the entire history with the initial observation for padding.
        
        Args:
            initial_obs: Initial observation dict from environment reset
        """
        self._deque.clear()
        for _ in range(self.obs_horizon):
            self._deque.append(initial_obs)
    
    def append(self, obs: Dict[str, torch.Tensor]):
        """Append a new observation to the history.
        
        Args:
            obs: New observation dict from environment step
        """
        self._deque.append(obs)
    
    def get_stacked(self) -> Dict[str, torch.Tensor]:
        """Get stacked observations with proper padding.
        
        If history is not full, pads with the first observation.
        
        Returns:
            Dict with stacked tensors: [num_envs, obs_horizon, ...]
        """
        obs_list = list(self._deque)
        
        # Pad if needed (shouldn't happen after reset, but safe to handle)
        while len(obs_list) < self.obs_horizon:
            obs_list.insert(0, obs_list[0])
        
        # Stack along time dimension (dim=1)
        stacked = {
            k: torch.stack([o[k] for o in obs_list], dim=1)
            for k in obs_list[0].keys()
        }
        
        return stacked
    
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
    
    Input formats (auto-detected):
        Offline data (NCHW): obs_seq["rgb"] shape [B, T, C, H, W]
        Online data (NHWC): obs_seq["rgb"] shape [B, T, H, W, C]
        State: obs_seq["state"] shape [B, T, state_dim]
    
    Output:
        obs_cond: [B, T * (visual_dim + state_dim)]
    
    Args:
        obs_seq: Dict with 'state' and optionally 'rgb' observations
        visual_encoder: Visual encoder module (PlainConv), can be None for state-only
        include_rgb: Whether to include RGB in observations
        device: Device for output tensor
        
    Returns:
        Flattened observation conditioning tensor
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
        # NHWC: [B, T, H, W, C] where C is typically 3 or small
        # NCHW: [B, T, C, H, W] where C can be 3*n_cameras
        if rgb.dim() == 5:
            # Check if last dim is channel (NHWC) or spatial (NCHW)
            # In NHWC, last dim is C (typically 3, 4, or small)
            # In NCHW, last dim is W (typically 64, 128, etc.)
            if rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:  # Common channel counts
                # (B, T, H, W, C) -> (B, T, C, H, W)
                rgb = rgb.permute(0, 1, 4, 2, 3)
        
        # Flatten for encoder
        rgb_flat = rgb.reshape(B * T, *rgb.shape[2:]).float()
        
        # Normalize if needed (values > 1 indicates uint8 range)
        if rgb_flat.max() > 1.0:
            rgb_flat = rgb_flat / 255.0
        
        visual_feat = visual_encoder(rgb_flat)  # [B*T, visual_dim]
        visual_feat = visual_feat.view(B, T, -1)  # [B, T, visual_dim]
        features_list.append(visual_feat)
    
    # State features
    features_list.append(state)
    
    # Concatenate features: [B, T, visual_dim + state_dim]
    obs_features = torch.cat(features_list, dim=-1)
    
    # Flatten to [B, T * (visual_dim + state_dim)]
    obs_cond = obs_features.reshape(B, -1)
    
    return obs_cond


# ============================================================================
# Agent Wrapper for Evaluation
# ============================================================================

class AgentWrapper(nn.Module):
    """Wrapper that combines visual encoder and agent for evaluation.
    
    Aligns with train_rgbd.py's Agent.encode_obs and Agent.get_action methods.
    Can be used by both offline (train_offline_rl.py) and online (train_online_finetune.py) training.
    
    Args:
        agent: The policy agent (diffusion policy, flow matching, etc.)
        visual_encoder: Visual encoder for RGB images (can be None for state-only)
        include_rgb: Whether to include RGB in observations
        obs_horizon: Number of observation frames to stack
        act_horizon: Number of action frames to predict (optional, for slicing output)
    """
    def __init__(self, agent, visual_encoder, include_rgb, obs_horizon, act_horizon=None):
        super().__init__()
        self.agent = agent
        self.visual_encoder = visual_encoder
        self.include_rgb = include_rgb
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon if act_horizon is not None else obs_horizon
        
    def encode_obs(self, obs_seq, eval_mode=True):
        """Encode observations to get obs_cond for agents.
        
        Uses the unified encode_observations function.
        
        Input from environment:
            obs_seq["rgb"]: (B, obs_horizon, H, W, C) or (B, obs_horizon, C, H, W)
            obs_seq["state"]: (B, obs_horizon, state_dim) float32
        
        Output:
            obs_cond: (B, obs_horizon * (visual_dim + state_dim))
        """
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cuda')
        return encode_observations(obs_seq, self.visual_encoder, self.include_rgb, device)
    
    def get_action(self, obs_seq, **kwargs):
        """Get action from observations (for evaluation).
        
        Mirrors train_rgbd.py Agent.get_action.
        
        Args:
            obs_seq: Dict with 'rgb', 'state' observations
            **kwargs: Additional arguments to pass to agent.get_action()
                - For ReinFlowAgent: deterministic=True, use_ema=True
        """
        with torch.no_grad():
            # encode_obs returns (B, global_cond_dim) - flattened observation features
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            
            # Pass flattened obs_cond directly to agent.get_action
            # The agent's velocity_net/noise_pred_net expects global_cond as (B, global_cond_dim)
            result = self.agent.get_action(obs_cond, **kwargs)
            
            # Handle agents that return (action_seq, chains) tuple (e.g., ReinFlowAgent)
            if isinstance(result, tuple):
                action_seq = result[0]
            else:
                action_seq = result
            
            # Only return act_horizon actions (aligned with train_rgbd.py)
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            return action_seq[:, start:end]
    
    def eval(self):
        self.agent.eval()
        if self.visual_encoder is not None:
            self.visual_encoder.eval()
        return self
    
    def train(self):
        self.agent.train()
        if self.visual_encoder is not None:
            self.visual_encoder.train()
        return self
