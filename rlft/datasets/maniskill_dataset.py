"""
ManiSkill Dataset for Offline Imitation Learning and Reinforcement Learning.

Supports both pure imitation learning and offline RL with SMDP formulation.

=============================================================================
Data Flow Overview
=============================================================================

1. Demo File Selection:
   - Use files with obs_mode != "none": trajectory.rgb.*.h5 or trajectory.rgbd.*.h5
   - Files with trajectory.none.*.h5 do NOT have observations!

2. Loading Pipeline:
   load_traj_hdf5()          : Raw HDF5 -> Dict[str, Dict] with traj_0, traj_1...
        |
        v
   obs_process_fn()          : Raw obs dict -> {"state": (T, state_dim), "rgb": (T, C, H, W)}
        |
        v
   Slice Computation         : Compute (traj_idx, start, end) indices for sampling
        |
        v
   __getitem__()             : Return {"observations": {...}, "actions": (pred_horizon, act_dim)}

3. Observation Processing (obs_process_fn):
   Raw ManiSkill obs:
   {
     "agent": {"qpos": (T, N), "qvel": (T, N)},
     "extra": {"tcp_pose": (T, 7), ...},
     "sensor_data": {"camera": {"rgb": (T, H, W, 3)}}
   }
   
   Processed obs:
   {
     "state": (T, state_dim),  # Concatenated agent + extra
     "rgb": (T, C, H, W)       # Channel-first format
   }

4. Slice Indexing for Action Chunking:
   For obs_horizon=2, pred_horizon=16:
   
   Time:    0   1   2   3   4   ...  T-1  T
   Obs:     o0  o1  o2  o3  o4  ...  oT-1 oT    (T+1 observations)
   Act:     a0  a1  a2  a3  a4  ...  aT-1       (T actions)
   
   Sample at index start=5 returns:
   - obs_seq: [o5, o6]                          (obs_horizon=2)
   - act_seq: [a5, a6, ..., a20]               (pred_horizon=16)
   
   Padding is applied at boundaries to maintain consistent shapes.

=============================================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable, TYPE_CHECKING
from gymnasium import spaces

from .data_utils import load_traj_hdf5, create_obs_process_fn

if TYPE_CHECKING:
    from .carm_dataset import ActionNormalizer


def reorder_keys(d, ref_dict):
    """Reorder dict keys to match reference dict."""
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


class ManiSkillDataset(Dataset):
    """Dataset for ManiSkill demonstrations (pure imitation learning).
    
    Loads demonstration trajectories from HDF5 files and prepares them for
    diffusion policy / flow matching training with action chunking.
    
    The dataset computes all valid sample indices upfront:
    - Each sample provides obs_horizon consecutive observations
    - Each sample provides pred_horizon consecutive actions
    - Padding is applied at trajectory boundaries
    
    Memory Layout:
        trajectories["observations"]: List of dicts, each with:
            - "state": Tensor (T+1, state_dim)
            - "rgb": Tensor (T+1, C, H, W) if include_rgb
        trajectories["actions"]: List of Tensor (T, act_dim)
        
    Slice Format:
        (traj_idx, start, end) where:
        - traj_idx: Which trajectory
        - start: Start index for obs/action (can be negative for padding)
        - end: End index for actions (can exceed T for padding)
    
    Args:
        data_path: Path to HDF5 demo file (must have obs_mode != "none")
        include_rgb: Whether to include RGB observations
        include_depth: Whether to include depth observations
        device: Device to store tensors on
        num_traj: Number of trajectories to load (None = all)
        obs_horizon: Observation stacking horizon (how many frames of obs to stack)
        pred_horizon: Action prediction horizon (how many future actions to predict)
        control_mode: Control mode for action padding at episode end
        env_id: Environment ID for state extraction
        obs_process_fn: Optional custom observation processing function
        obs_space: Optional observation space for reordering keys
        action_normalizer: Optional action normalizer for training stability
    """
    
    def __init__(
        self,
        data_path: str,
        include_rgb: bool,
        device,
        num_traj: Optional[int],
        obs_horizon: int,
        pred_horizon: int,
        control_mode: str,
        env_id: str = None,
        obs_process_fn: Optional[Callable] = None,
        obs_space=None,
        rgb_format: str = "NCHW",
        action_normalizer: Optional["ActionNormalizer"] = None,
        include_depth: bool = False,
    ):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.device = device
        self.rgb_format = rgb_format
        self.action_normalizer = action_normalizer
        
        # Load demo dataset
        raw_data = load_traj_hdf5(data_path, num_traj=num_traj)
        
        print("Raw trajectory loaded, beginning observation pre-processing...")
        
        # Create obs_process_fn if not provided
        if obs_process_fn is None:
            if env_id is None:
                raise ValueError("env_id is required when obs_process_fn is not provided")
            obs_process_fn = create_obs_process_fn(env_id, output_format=rgb_format)
            use_reorder_keys = False
        else:
            use_reorder_keys = obs_space is not None
        
        # Process trajectories
        trajectories = {
            "observations": [],
            "actions": [],
        }
        
        for traj_key in sorted(raw_data.keys(), key=lambda x: int(x.split("_")[-1])):
            traj = raw_data[traj_key]
            
            # Process observations
            if use_reorder_keys:
                obs_dict = reorder_keys(traj["obs"], obs_space)
                obs_dict = obs_process_fn(obs_dict)
            else:
                obs_dict = obs_process_fn(traj["obs"])
            
            processed_obs = {}
            if include_rgb:
                processed_obs["rgb"] = torch.from_numpy(obs_dict["rgb"]).to(device)
            if include_depth and obs_dict.get("depth") is not None:
                processed_obs["depth"] = torch.from_numpy(obs_dict["depth"]).to(device)
            processed_obs["state"] = torch.from_numpy(obs_dict["state"]).to(device)
            
            trajectories["observations"].append(processed_obs)
            trajectories["actions"].append(
                torch.Tensor(traj["actions"]).to(device=device)
            )
        
        self.obs_keys = list(processed_obs.keys())
        print("Obs/action pre-processing done, computing slice indices...")
        
        # Fit action normalizer if provided
        if self.action_normalizer is not None:
            all_actions = np.concatenate([
                traj["actions"] for traj in raw_data.values()
            ], axis=0)
            self.action_normalizer.fit(all_actions)
            print(f"Action normalizer fitted with {len(all_actions)} samples, mode: {self.action_normalizer.mode}")
            
            # Normalize stored actions
            for i in range(len(trajectories["actions"])):
                actions_np = trajectories["actions"][i].cpu().numpy()
                normalized = self.action_normalizer.transform(actions_np)
                trajectories["actions"][i] = torch.from_numpy(normalized).float().to(device)
        
        # Action padding for delta controllers
        if "delta_pos" in control_mode or control_mode == "base_pd_joint_vel_arm_pd_joint_vel":
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device
            )
        else:
            self.pad_action_arm = None
        
        # Compute slices
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L
            
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]
        
        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")
        self.trajectories = trajectories
    
    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape
        
        obs_traj = self.trajectories["observations"][traj_idx]
        
        # Get observation sequence
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start):start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
        
        # Get action sequence
        act_seq = self.trajectories["actions"][traj_idx][max(0, start):end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            if self.pad_action_arm is not None:
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            else:
                pad_action = act_seq[-1]
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
        
        assert obs_seq["state"].shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon
        
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }
    
    def __len__(self):
        return len(self.slices)


class OfflineRLDataset(Dataset):
    """Dataset for Offline RL with chunk-level transitions (SMDP formulation).
    
    Extends ManiSkillDataset to support action chunking with proper Bellman
    equation formulation for offline RL algorithms.
    
    For a chunk of length τ starting at timestep t:
    - cumulative_reward: R_t^(τ) = Σ_{i=0}^{τ-1} γ^i r_{t+i}
    - next_observations: s_{t+τ} (state after chunk execution)
    - chunk_done: 1 if episode ends within chunk, 0 otherwise
    - effective_length: τ (actual chunk length)
    - discount_factor: γ^τ (for proper SMDP Bellman target)
    
    Args:
        data_path: Path to HDF5 demo file
        include_rgb: Whether to include RGB observations
        device: Device to store tensors on
        num_traj: Number of trajectories to load (None = all)
        obs_horizon: Observation stacking horizon
        pred_horizon: Action prediction horizon
        act_horizon: Action execution horizon (for SMDP)
        control_mode: Control mode for action padding
        env_id: Environment ID for state extraction
        obs_process_fn: Optional observation processing function
        obs_space: Optional observation space for key reordering
        rgb_format: RGB output format ("NCHW" or "NHWC")
        gamma: Discount factor for SMDP
        action_normalizer: Optional action normalizer for training stability
    """
    
    def __init__(
        self,
        data_path: str,
        include_rgb: bool,
        device,
        num_traj: Optional[int],
        obs_horizon: int,
        pred_horizon: int,
        act_horizon: int,
        control_mode: str,
        env_id: str = None,
        obs_process_fn: Optional[Callable] = None,
        obs_space=None,
        rgb_format: str = "NCHW",
        gamma: float = 0.99,
        action_normalizer: Optional["ActionNormalizer"] = None,
        include_depth: bool = False,
    ):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.gamma = gamma
        self.device = device
        self.rgb_format = rgb_format
        self.action_normalizer = action_normalizer
        
        # Load demo dataset
        raw_data = load_traj_hdf5(data_path, num_traj=num_traj)
        
        print("Raw trajectory loaded, beginning observation pre-processing...")
        
        # Create obs_process_fn if not provided
        if obs_process_fn is None:
            if env_id is None:
                raise ValueError("env_id is required when obs_process_fn is not provided")
            obs_process_fn = create_obs_process_fn(env_id, output_format=rgb_format)
            use_reorder_keys = False
        else:
            use_reorder_keys = obs_space is not None
        
        # Process trajectories
        trajectories = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        for traj_key in sorted(raw_data.keys(), key=lambda x: int(x.split("_")[-1])):
            traj = raw_data[traj_key]
            
            # Process observations
            if use_reorder_keys:
                obs_dict = reorder_keys(traj["obs"], obs_space)
                obs_dict = obs_process_fn(obs_dict)
            else:
                obs_dict = obs_process_fn(traj["obs"])
            
            processed_obs = {}
            if include_rgb:
                processed_obs["rgb"] = torch.from_numpy(obs_dict["rgb"]).to(device)
            if include_depth and obs_dict.get("depth") is not None:
                processed_obs["depth"] = torch.from_numpy(obs_dict["depth"]).to(device)
            processed_obs["state"] = torch.from_numpy(obs_dict["state"]).to(device)
            
            trajectories["observations"].append(processed_obs)
            trajectories["actions"].append(
                torch.Tensor(traj["actions"]).to(device=device)
            )
            
            # Process rewards
            if "rewards" in traj:
                rewards = traj["rewards"]
            elif "reward" in traj:
                rewards = traj["reward"]
            else:
                rewards = np.zeros(len(traj["actions"]))
            trajectories["rewards"].append(torch.Tensor(rewards).to(device=device))
            
            # Process dones
            if "dones" in traj:
                dones = traj["dones"]
            elif "done" in traj:
                dones = traj["done"]
            elif "terminated" in traj:
                dones = traj["terminated"]
            else:
                dones = np.zeros(len(traj["actions"]))
                dones[-1] = 1.0
            trajectories["dones"].append(torch.Tensor(dones).to(device=device))
        
        self.obs_keys = list(processed_obs.keys())
        print("Obs/action pre-processing done, computing slice indices...")
        
        # Fit action normalizer if provided
        if self.action_normalizer is not None:
            all_actions = np.concatenate([
                traj["actions"] for traj in raw_data.values()
            ], axis=0)
            self.action_normalizer.fit(all_actions)
            print(f"Action normalizer fitted with {len(all_actions)} samples, mode: {self.action_normalizer.mode}")
            
            # Normalize stored actions
            for i in range(len(trajectories["actions"])):
                actions_np = trajectories["actions"][i].cpu().numpy()
                normalized = self.action_normalizer.transform(actions_np)
                trajectories["actions"][i] = torch.from_numpy(normalized).float().to(device)
        
        # Action padding
        if "delta_pos" in control_mode or control_mode == "base_pd_joint_vel_arm_pd_joint_vel":
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device
            )
        else:
            self.pad_action_arm = None
        
        # Compute slices
        self.slices = []
        num_traj_count = len(trajectories["actions"])
        total_transitions = 0
        
        for traj_idx in range(num_traj_count):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L
            
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]
        
        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")
        self.trajectories = trajectories
    
    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape
        
        obs_traj = self.trajectories["observations"][traj_idx]
        
        # Get observation sequence
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start):start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
        
        # ===== SMDP Chunk-Level Transition =====
        action_start = max(0, start)
        effective_length = min(self.act_horizon, L - action_start)
        effective_length = max(1, effective_length)
        
        # Compute cumulative discounted reward
        cumulative_reward = 0.0
        chunk_done = 0.0
        
        rewards_traj = self.trajectories["rewards"][traj_idx]
        dones_traj = self.trajectories["dones"][traj_idx]
        
        for i in range(effective_length):
            step_idx = action_start + i
            if step_idx < L:
                cumulative_reward += (self.gamma ** i) * rewards_traj[step_idx].item()
                if dones_traj[step_idx].item() > 0.5:
                    chunk_done = 1.0
                    effective_length = i + 1
                    break
        
        discount_factor = self.gamma ** effective_length
        
        # Get next observation sequence
        next_obs_start = action_start + effective_length
        next_obs_seq = {}
        for k, v in obs_traj.items():
            actual_start = min(next_obs_start, L)
            next_obs_seq[k] = v[actual_start:actual_start + self.obs_horizon]
            if next_obs_seq[k].shape[0] < self.obs_horizon:
                pad_len = self.obs_horizon - next_obs_seq[k].shape[0]
                if next_obs_seq[k].shape[0] > 0:
                    pad_obs_seq = torch.stack([next_obs_seq[k][-1]] * pad_len, dim=0)
                else:
                    pad_obs_seq = torch.stack([v[-1]] * self.obs_horizon, dim=0)
                    next_obs_seq[k] = pad_obs_seq
                    continue
                next_obs_seq[k] = torch.cat((next_obs_seq[k], pad_obs_seq), dim=0)
        
        # Get action sequence for policy training (pred_horizon)
        act_seq = self.trajectories["actions"][traj_idx][max(0, start):end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            if self.pad_action_arm is not None:
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            else:
                pad_action = act_seq[-1]
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
        
        # Get action sequence for Q-learning (act_horizon)
        act_start = max(0, start)
        act_effective_end = min(act_start + effective_length, L)
        act_seq_for_q = self.trajectories["actions"][traj_idx][act_start:act_effective_end]
        
        if act_seq_for_q.shape[0] == 0:
            act_seq_for_q = self.trajectories["actions"][traj_idx][L-1:L]
        
        if start < 0:
            act_seq_for_q = torch.cat([act_seq_for_q[0].repeat(-start, 1), act_seq_for_q], dim=0)
        
        if act_seq_for_q.shape[0] < self.act_horizon:
            pad_len = self.act_horizon - act_seq_for_q.shape[0]
            if self.pad_action_arm is not None:
                gripper_action = act_seq_for_q[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            else:
                pad_action = act_seq_for_q[-1]
            act_seq_for_q = torch.cat([act_seq_for_q, pad_action.repeat(pad_len, 1)], dim=0)
        
        act_seq_for_q = act_seq_for_q[:self.act_horizon]
        
        assert obs_seq["state"].shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon
        assert act_seq_for_q.shape[0] == self.act_horizon
        
        return {
            "observations": obs_seq,
            "next_observations": next_obs_seq,
            "actions": act_seq,
            "actions_for_q": act_seq_for_q,
            "cumulative_reward": torch.tensor(cumulative_reward, dtype=torch.float32, device=self.device),
            "chunk_done": torch.tensor(chunk_done, dtype=torch.float32, device=self.device),
            "effective_length": torch.tensor(effective_length, dtype=torch.float32, device=self.device),
            "discount_factor": torch.tensor(discount_factor, dtype=torch.float32, device=self.device),
            "rewards": rewards_traj[min(action_start, L - 1)],
            "dones": dones_traj[min(action_start, L - 1)],
        }
    
    def __len__(self):
        return len(self.slices)
