"""
Unified Offline RL Training Script

Supports training multiple algorithms on ManiSkill environments:
- diffusion_policy: DDPM-based Diffusion Policy (imitation learning)
- flow_matching: Flow Matching Policy (imitation learning)
- reflected_flow: Reflected Flow for bounded actions
- consistency_flow: Consistency Flow with self-consistency
- shortcut_flow: ShortCut Flow with adaptive steps
- diffusion_double_q: Diffusion Policy + Double Q-Learning (offline RL)
- cpql: Consistency Policy Q-Learning (offline RL)
- awcp: Advantage-Weighted Consistency Policy (offline RL)
- aw_shortcut_flow: Advantage-Weighted ShortCut Flow (offline RL)

Based on the official ManiSkill diffusion_policy implementation.
"""

ALGO_NAME = "OfflineRL_UNet"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Literal

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import (
    AgentWrapper,
    IterationBasedBatchSampler,
    build_state_obs_extractor,
    convert_obs,
    worker_init_fn,
)
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.algorithms import (
    DiffusionPolicyAgent,
    FlowMatchingAgent,
    ReflectedFlowAgent,
    ConsistencyFlowAgent,
    ShortCutFlowAgent,
    ShortCutVelocityUNet1D,
    DiffusionDoubleQAgent,
    CPQLAgent,
    AWCPAgent,
    AWShortCutFlowAgent,
    DPPOAgent,
    ReinFlowAgent,
)
from diffusion_policy.algorithms.networks import VelocityUNet1D, DoubleQNetwork, EnsembleQNetwork


@dataclass
class Args:
    # Experiment settings
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""

    # Environment settings
    env_id: str = "LiftPegUpright-v1"
    """the id of the environment"""
    demo_path: str = "~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"
    """the path of demo dataset"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    max_episode_steps: Optional[int] = None
    """max episode steps for evaluation"""
    control_mode: str = "pd_ee_delta_pose"
    """the control mode"""
    obs_mode: str = "rgb"
    """observation mode: state or rgb"""
    sim_backend: str = "physx_cuda"
    """simulation backend for evaluation"""

    # Training settings
    total_iters: int = 1_000_000
    """total training iterations"""
    batch_size: int = 256
    """batch size"""
    lr: float = 1e-4
    """learning rate"""
    lr_critic: float = 3e-4
    """learning rate for critic (only for offline RL)"""

    # Diffusion Policy / Flow Matching settings
    obs_horizon: int = 2
    """observation horizon"""
    act_horizon: int = 8
    """action execution horizon"""
    pred_horizon: int = 16
    """action prediction horizon"""
    diffusion_step_embed_dim: int = 64
    """timestep embedding dimension"""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """U-Net channel dimensions"""
    n_groups: int = 8
    """GroupNorm groups"""
    visual_feature_dim: int = 256
    """visual encoder output dimension"""

    # Algorithm selection
    algorithm: Literal[
        "diffusion_policy",
        "flow_matching", 
        "reflected_flow", 
        "consistency_flow", 
        "shortcut_flow",
        "diffusion_double_q", 
        "cpql",
        "awcp",
        "aw_shortcut_flow",
        "dppo",
        "reinflow"
    ] = "diffusion_policy"
    """algorithm to train"""
    
    # Diffusion Policy specific hyperparameters
    num_diffusion_iters: int = 100
    """number of diffusion iterations for DDPM"""
    
    # DPPO specific hyperparameters
    ft_denoising_steps: int = 5
    """number of denoising steps to fine-tune (for DPPO)"""
    
    # Flow variant specific hyperparameters
    reflection_mode: Literal["hard", "soft"] = "hard"
    """reflection mode for reflected_flow"""
    boundary_reg_weight: float = 0.01
    """boundary regularization weight for reflected_flow"""
    max_denoising_steps: int = 8
    """max denoising steps for shortcut_flow"""
    self_consistency_k: float = 0.25
    """fraction of batch for self-consistency in shortcut_flow"""
    ema_decay: float = 0.999
    """EMA decay rate for consistency_flow and shortcut_flow"""
    
    # Consistency Flow specific hyperparameters
    cons_use_flow_t: bool = False
    """reuse flow t for consistency branch instead of resampling"""
    cons_full_t_range: bool = False
    """sample consistency t in [0,1] instead of clipped range"""
    cons_t_min: float = 0.05
    """minimum t for consistency sampling when not using full range"""
    cons_t_max: float = 0.95
    """maximum t for consistency sampling when not using full range"""
    cons_t_upper: float = 0.95
    """upper clamp for t_plus (set 0.99 for CPQL-style)"""
    cons_delta_mode: Literal["random", "fixed"] = "random"
    """delta sampling strategy for consistency"""
    cons_delta_min: float = 0.02
    """minimum delta when using random delta"""
    cons_delta_max: float = 0.15
    """maximum delta when using random delta (static cap)"""
    cons_delta_fixed: float = 0.01
    """fixed delta when cons_delta_mode=fixed"""
    cons_delta_dynamic_max: bool = False
    """cap random delta by remaining time (e.g., 0.99 - t_cons)"""
    cons_delta_cap: float = 0.99
    """ceiling used when cons_delta_dynamic_max is enabled"""
    cons_teacher_steps: int = 2
    """teacher rollout steps to t=1"""
    cons_teacher_from: Literal["t_plus", "t_cons"] = "t_plus"
    """where teacher rollout starts"""
    cons_student_point: Literal["t_plus", "t_cons"] = "t_plus"
    """student evaluation point for consistency loss"""
    cons_loss_space: Literal["velocity", "endpoint"] = "velocity"
    """consistency loss space: velocity (v) or endpoint (x1)"""

    # ShortCut Flow specific hyperparameters
    sc_t_min: float = 0.0
    """minimum t for time sampling in shortcut_flow"""
    sc_t_max: float = 1.0
    """maximum t for time sampling in shortcut_flow"""
    sc_t_sampling_mode: Literal["uniform", "truncated"] = "uniform"
    """time sampling mode: uniform (then clamp) or truncated (direct valid range)"""
    sc_step_size_mode: Literal["power2", "uniform", "fixed"] = "fixed"
    """step size sampling mode: power2 (log-uniform), uniform, or fixed. Sweep shows fixed small step is best."""
    sc_min_step_size: float = 0.0625
    """minimum step size (1/16 by default)"""
    sc_max_step_size: float = 0.5
    """maximum step size (1/2 by default)"""
    sc_fixed_step_size: float = 0.0625
    """fixed step size when sc_step_size_mode=fixed. Sweep shows 1/16 is best for reliable teacher targets."""
    sc_target_mode: Literal["velocity", "endpoint"] = "velocity"
    """shortcut target mode: velocity (match v) or endpoint (match x1)"""
    sc_teacher_steps: int = 1
    """teacher rollout steps for shortcut target. Sweep shows single step preserves locality best."""
    sc_use_ema_teacher: bool = True
    """whether to use EMA network as teacher for shortcut target"""
    sc_inference_mode: Literal["adaptive", "uniform"] = "uniform"
    """inference mode: adaptive (variable step sizes) or uniform (fixed dt). Sweep shows uniform avoids solver mismatch."""
    sc_num_inference_steps: int = 8
    """number of inference steps (for uniform mode or as fallback)"""

    # Offline RL hyperparameters (inherited from toy_diffusion_rl)
    alpha: float = 0.01
    """Q-value weight for diffusion_double_q (0.01 for cpql)"""
    bc_weight: float = 1.0
    """BC/flow matching loss weight"""
    consistency_weight: float = 0.3
    """consistency/shortcut loss weight. Sweep shows flow-heavy (0.3-0.5) is best for ShortCut Flow."""
    gamma: float = 0.99
    """discount factor"""
    tau: float = 0.005
    """soft update coefficient"""
    reward_scale: float = 0.1
    """reward scaling for Q-value stability"""
    q_target_clip: float = 100.0
    """Q target clipping"""
    num_flow_steps: int = 10
    """ODE integration steps for flow matching"""
    
    # Diffusion Double Q specific hyperparameters
    q_grad_mode: Literal["whole_grad", "last_few", "single_step"] = "last_few"
    """Q-gradient mode: whole_grad (full chain), last_few (partial), single_step (fast approx)"""
    q_grad_steps: int = 5
    """Number of steps with gradient in last_few mode"""
    
    # AWCP (Advantage-Weighted Consistency Policy) specific hyperparameters
    beta: float = 10.0
    """Temperature for advantage weighting in AWCP (higher = more selective)"""
    weight_clip: float = 100.0
    """Maximum weight to prevent outlier dominance in AWCP"""

    # Ensemble Q-Network settings (for AWShortCutFlow -> RLPD fine-tuning compatibility)
    use_ensemble_q: bool = True
    """Use EnsembleQNetwork instead of DoubleQNetwork for aw_shortcut_flow.
    Recommended True for seamless checkpoint transfer to online RLPD training."""
    num_qs: int = 10
    """Number of Q-networks in ensemble (RLPD sample-efficient default: 10)"""
    num_min_qs: int = 2
    """Number of Q-networks for subsample + min (RLPD default: 2)"""
    q_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    """Hidden dimensions for Q-network MLP (RLPD default: [256, 256, 256])"""

    # Logging settings
    log_freq: int = 1000
    """logging frequency"""
    eval_freq: int = 5000
    """evaluation frequency"""
    save_freq: Optional[int] = None
    """checkpoint save frequency"""
    num_eval_episodes: int = 100
    """number of evaluation episodes"""
    num_eval_envs: int = 10
    """number of parallel eval environments"""
    num_dataload_workers: int = 0
    """dataloader workers"""

    # Additional tags
    demo_type: Optional[str] = None


def reorder_keys(d, ref_dict):
    """Reorder dict keys to match reference dict."""
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


class OfflineRLDataset(Dataset):
    """Dataset for Offline RL with chunk-level transitions (SMDP formulation).
    
    Extends the official diffusion_policy dataset to support action chunking
    with proper Bellman equation formulation:
    
    For a chunk of length τ starting at timestep t:
    - cumulative_reward: R_t^(τ) = Σ_{i=0}^{τ-1} γ^i r_{t+i}
    - next_observations: s_{t+τ} (state after chunk execution)
    - chunk_done: 1 if episode ends within chunk, 0 otherwise
    - effective_length: τ (actual chunk length, may be < pred_horizon at episode end)
    - discount_factor: γ^τ (for proper SMDP Bellman target)
    
    The SMDP Bellman equation is:
    Q(s_t, â_t) = R_t^(τ) + (1 - d_t^(τ)) * γ^τ * max_{â'} Q(s_{t+τ}, â')
    
    Args:
        data_path: Path to HDF5 demo file
        include_rgb: Whether to include RGB observations
        device: Device to store tensors on
        num_traj: Number of trajectories to load (None = all)
        obs_horizon: Observation stacking horizon
        pred_horizon: Action prediction horizon
        act_horizon: Action execution horizon (for SMDP)
        control_mode: Control mode for action padding detection
        env_id: Environment ID for state extraction (required if obs_process_fn not provided)
        obs_process_fn: Optional custom observation processing function
        obs_space: Optional observation space for reordering keys
        gamma: Discount factor for SMDP
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
        env_id: str = None,  # Required if obs_process_fn not provided
        obs_process_fn=None,  # If None, will use create_obs_process_fn
        obs_space=None,
        rgb_format: str = "NCHW",  # "NCHW" for GPU training, "NHWC" for online mixing
        gamma: float = 0.99,
    ):
        self.include_rgb = include_rgb
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.gamma = gamma
        self.device = device
        self.rgb_format = rgb_format
        
        # Load demo dataset with RL signals
        from diffusion_policy.utils import load_traj_hdf5, create_obs_process_fn
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
            processed_obs["state"] = torch.from_numpy(obs_dict["state"]).to(device)
            
            trajectories["observations"].append(processed_obs)
            
            # Process actions
            trajectories["actions"].append(
                torch.Tensor(traj["actions"]).to(device=device)
            )
            
            # Process rewards (handle different possible keys)
            if "rewards" in traj:
                rewards = traj["rewards"]
            elif "reward" in traj:
                rewards = traj["reward"]
            else:
                # Default to zeros if no reward found
                rewards = np.zeros(len(traj["actions"]))
            trajectories["rewards"].append(
                torch.Tensor(rewards).to(device=device)
            )
            
            # Process dones
            if "dones" in traj:
                dones = traj["dones"]
            elif "done" in traj:
                dones = traj["done"]
            elif "terminated" in traj:
                dones = traj["terminated"]
            else:
                # Default: only last step is done
                dones = np.zeros(len(traj["actions"]))
                dones[-1] = 1.0
            trajectories["dones"].append(
                torch.Tensor(dones).to(device=device)
            )
        
        self.obs_keys = list(processed_obs.keys())
        print("Obs/action pre-processing done, computing slice indices...")
        
        # Action padding for delta controllers
        if "delta_pos" in control_mode or control_mode == "base_pd_joint_vel_arm_pd_joint_vel":
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device
            )
        else:
            self.pad_action_arm = None
        
        # Compute slices (same as official implementation)
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
        
        # Get observation sequence (for policy input)
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start):start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
        
        # ===== SMDP Chunk-Level Transition =====
        # For action chunking, we need to compute:
        # 1. cumulative_reward: R_t^(τ) = Σ_{i=0}^{τ-1} γ^i r_{t+i}
        # 2. next_observations: s_{t+τ} (state after chunk execution)
        # 3. chunk_done: 1 if episode ends within chunk
        # 4. effective_length: τ (actual chunk length)
        # 5. discount_factor: γ^τ
        
        # Determine the actual action indices for this chunk
        # The chunk covers actions from action_start to action_start + act_horizon
        action_start = max(0, start)
        
        # Compute effective chunk length (may be shorter at episode end)
        # We use act_horizon (execution horizon) not pred_horizon for chunk length
        effective_length = min(self.act_horizon, L - action_start)
        effective_length = max(1, effective_length)  # At least 1 step
        
        # Compute cumulative discounted reward: R_t^(τ) = Σ_{i=0}^{τ-1} γ^i r_{t+i}
        cumulative_reward = 0.0
        chunk_done = 0.0
        
        rewards_traj = self.trajectories["rewards"][traj_idx]
        dones_traj = self.trajectories["dones"][traj_idx]
        
        for i in range(effective_length):
            step_idx = action_start + i
            if step_idx < L:
                # Add discounted reward
                cumulative_reward += (self.gamma ** i) * rewards_traj[step_idx].item()
                # Check if done within chunk
                if dones_traj[step_idx].item() > 0.5:
                    chunk_done = 1.0
                    effective_length = i + 1  # Truncate effective length
                    break
        
        # Compute discount factor: γ^τ
        discount_factor = self.gamma ** effective_length
        
        # Get next observation sequence (state after chunk: s_{t+τ})
        # This is the observation at timestep (action_start + effective_length)
        next_obs_start = action_start + effective_length
        next_obs_seq = {}
        for k, v in obs_traj.items():
            # Clamp to valid range [0, L] (observations have L+1 entries)
            actual_start = min(next_obs_start, L)
            next_obs_seq[k] = v[actual_start:actual_start + self.obs_horizon]
            # Pad if we need more observations than available
            if next_obs_seq[k].shape[0] < self.obs_horizon:
                pad_len = self.obs_horizon - next_obs_seq[k].shape[0]
                if next_obs_seq[k].shape[0] > 0:
                    pad_obs_seq = torch.stack([next_obs_seq[k][-1]] * pad_len, dim=0)
                else:
                    # Edge case: use last available observation
                    pad_obs_seq = torch.stack([v[-1]] * self.obs_horizon, dim=0)
                    next_obs_seq[k] = pad_obs_seq
                    continue
                next_obs_seq[k] = torch.cat((next_obs_seq[k], pad_obs_seq), dim=0)
        
        # Get action sequence (full pred_horizon for policy training)
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
        
        # Get act_horizon action sequence for Q-learning (matches SMDP reward horizon)
        # IMPORTANT: Use effective_length (not act_horizon) to match the reward computation
        # This ensures act_seq_for_q only contains actions that contributed to cumulative_reward
        act_start = max(0, start)
        act_effective_end = min(act_start + effective_length, L)  # Use effective_length!
        act_seq_for_q = self.trajectories["actions"][traj_idx][act_start:act_effective_end]
        
        # Handle edge case: empty sequence (start >= L)
        if act_seq_for_q.shape[0] == 0:
            # Use the last action in trajectory
            act_seq_for_q = self.trajectories["actions"][traj_idx][L-1:L]
        
        # Pad at beginning if start < 0
        if start < 0:
            act_seq_for_q = torch.cat([act_seq_for_q[0].repeat(-start, 1), act_seq_for_q], dim=0)
        
        # Pad at end to reach act_horizon (for fixed-size Q-network input)
        # These padded actions are "invalid" - they occur after episode termination
        if act_seq_for_q.shape[0] < self.act_horizon:
            pad_len = self.act_horizon - act_seq_for_q.shape[0]
            if self.pad_action_arm is not None:
                gripper_action = act_seq_for_q[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            else:
                pad_action = act_seq_for_q[-1]
            act_seq_for_q = torch.cat([act_seq_for_q, pad_action.repeat(pad_len, 1)], dim=0)
        
        # Truncate if longer than act_horizon (can happen with padding at beginning)
        act_seq_for_q = act_seq_for_q[:self.act_horizon]
        
        assert obs_seq["state"].shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon
        assert act_seq_for_q.shape[0] == self.act_horizon
        
        return {
            "observations": obs_seq,
            "next_observations": next_obs_seq,
            "actions": act_seq,  # Full pred_horizon for policy BC training
            "actions_for_q": act_seq_for_q,  # act_horizon for Q-learning (matches reward horizon)
            # SMDP chunk-level fields (on same device as other tensors)
            "cumulative_reward": torch.tensor(cumulative_reward, dtype=torch.float32, device=self.device),
            "chunk_done": torch.tensor(chunk_done, dtype=torch.float32, device=self.device),
            "effective_length": torch.tensor(effective_length, dtype=torch.float32, device=self.device),
            "discount_factor": torch.tensor(discount_factor, dtype=torch.float32, device=self.device),
            # Keep single-step reward/done for backward compatibility (IL algorithms)
            "rewards": rewards_traj[min(action_start, L - 1)],
            "dones": dones_traj[min(action_start, L - 1)],
        }
    
    def __len__(self):
        return len(self.slices)


def create_agent(algorithm: str, action_dim: int, global_cond_dim: int, args):
    """Create agent based on algorithm name with unified external network creation.
    
    Args:
        algorithm: Algorithm name
        action_dim: Action dimension from environment
        global_cond_dim: Global conditioning dimension (obs_horizon * feature_dim)
        args: Training arguments
        
    Returns:
        Initialized agent module
    """
    device = "cuda" if args.cuda else "cpu"
    
    if algorithm == "diffusion_policy":
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        
        return DiffusionPolicyAgent(
            noise_pred_net=noise_pred_net,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            num_diffusion_iters=args.num_diffusion_iters,
            device=device,
        )
    
    elif algorithm == "flow_matching":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        return FlowMatchingAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            num_flow_steps=args.num_flow_steps,
            device=device,
        )
    
    elif algorithm == "reflected_flow":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        return ReflectedFlowAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            num_flow_steps=args.num_flow_steps,
            reflection_mode=args.reflection_mode,
            boundary_reg_weight=args.boundary_reg_weight,
            device=device,
        )
    
    elif algorithm == "consistency_flow":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        return ConsistencyFlowAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            num_flow_steps=args.num_flow_steps,
            flow_weight=args.bc_weight,
            consistency_weight=args.consistency_weight,
            ema_decay=args.ema_decay,
            cons_use_flow_t=args.cons_use_flow_t,
            cons_full_t_range=args.cons_full_t_range,
            cons_t_min=args.cons_t_min,
            cons_t_max=args.cons_t_max,
            cons_t_upper=args.cons_t_upper,
            cons_delta_mode=args.cons_delta_mode,
            cons_delta_min=args.cons_delta_min,
            cons_delta_max=args.cons_delta_max,
            cons_delta_fixed=args.cons_delta_fixed,
            cons_delta_dynamic_max=args.cons_delta_dynamic_max,
            cons_delta_cap=args.cons_delta_cap,
            teacher_steps=args.cons_teacher_steps,
            teacher_from=args.cons_teacher_from,
            student_point=args.cons_student_point,
            consistency_loss_space=args.cons_loss_space,
            device=device,
        )
    
    elif algorithm == "shortcut_flow":
        shortcut_velocity_net = ShortCutVelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        return ShortCutFlowAgent(
            velocity_net=shortcut_velocity_net,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            max_denoising_steps=args.max_denoising_steps,
            self_consistency_k=args.self_consistency_k,
            flow_weight=args.bc_weight,
            shortcut_weight=args.consistency_weight,
            ema_decay=args.ema_decay,
            # Time sampling hyperparameters
            t_min=args.sc_t_min,
            t_max=args.sc_t_max,
            t_sampling_mode=args.sc_t_sampling_mode,
            # Step size hyperparameters
            step_size_mode=args.sc_step_size_mode,
            min_step_size=args.sc_min_step_size,
            max_step_size=args.sc_max_step_size,
            fixed_step_size=args.sc_fixed_step_size,
            # Target computation hyperparameters
            target_mode=args.sc_target_mode,
            teacher_steps=args.sc_teacher_steps,
            use_ema_teacher=args.sc_use_ema_teacher,
            # Inference hyperparameters
            inference_mode=args.sc_inference_mode,
            num_inference_steps=args.sc_num_inference_steps,
            device=device,
        )
    
    elif algorithm == "diffusion_double_q":
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        # Q-Network: simple MLP, uses act_horizon to match SMDP reward horizon
        q_network = DoubleQNetwork(
            action_dim=action_dim,
            obs_dim=global_cond_dim,
            action_horizon=args.act_horizon,
        )
        
        return DiffusionDoubleQAgent(
            noise_pred_net=noise_pred_net,
            q_network=q_network,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            act_horizon=args.act_horizon,  # Add act_horizon for Q-learning
            num_diffusion_iters=args.num_diffusion_iters,
            alpha=args.alpha,
            bc_weight=args.bc_weight,
            gamma=args.gamma,
            tau=args.tau,
            reward_scale=args.reward_scale,
            q_target_clip=args.q_target_clip,
            q_grad_mode=args.q_grad_mode,
            q_grad_steps=args.q_grad_steps,
            device=device,
        )
    
    elif algorithm == "cpql":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        # Q-Network: simple MLP, uses act_horizon to match SMDP reward horizon
        q_network = DoubleQNetwork(
            action_dim=action_dim,
            obs_dim=global_cond_dim,
            action_horizon=args.act_horizon,
        )
        
        return CPQLAgent(
            velocity_net=velocity_net,
            q_network=q_network,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            act_horizon=args.act_horizon,  # Add act_horizon for Q-learning
            num_flow_steps=args.num_flow_steps,
            alpha=args.alpha,  # CPQL uses smaller alpha
            bc_weight=args.bc_weight,
            consistency_weight=args.consistency_weight,
            gamma=args.gamma,
            tau=args.tau,
            reward_scale=args.reward_scale,
            q_target_clip=args.q_target_clip,
            ema_decay=args.ema_decay,
            q_grad_mode=args.q_grad_mode,
            q_grad_steps=args.q_grad_steps,
            device=device,
        )
    
    elif algorithm == "awcp":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        # Q-Network for advantage weighting
        q_network = DoubleQNetwork(
            action_dim=action_dim,
            obs_dim=global_cond_dim,
            action_horizon=args.act_horizon,
        )
        
        return AWCPAgent(
            velocity_net=velocity_net,
            q_network=q_network,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            act_horizon=args.act_horizon,
            num_flow_steps=args.num_flow_steps,
            beta=args.beta,  # AWCP temperature parameter
            bc_weight=args.bc_weight,
            consistency_weight=args.consistency_weight,
            gamma=args.gamma,
            tau=args.tau,
            reward_scale=args.reward_scale,
            q_target_clip=args.q_target_clip,
            ema_decay=args.ema_decay,
            weight_clip=args.weight_clip,
            device=device,
        )
    
    elif algorithm == "aw_shortcut_flow":
        shortcut_velocity_net = ShortCutVelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        # Q-Network for advantage weighting
        # Use EnsembleQNetwork by default for RLPD fine-tuning compatibility
        if args.use_ensemble_q:
            q_network = EnsembleQNetwork(
                action_dim=action_dim,
                obs_dim=global_cond_dim,
                action_horizon=args.act_horizon,
                hidden_dims=args.q_hidden_dims,
                num_qs=args.num_qs,
                num_min_qs=args.num_min_qs,
            )
        else:
            # Backward compatibility: use DoubleQNetwork
            q_network = DoubleQNetwork(
                action_dim=action_dim,
                obs_dim=global_cond_dim,
                action_horizon=args.act_horizon,
            )
        
        return AWShortCutFlowAgent(
            velocity_net=shortcut_velocity_net,
            q_network=q_network,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            act_horizon=args.act_horizon,
            # Offline RL hyperparameters
            beta=args.beta,
            bc_weight=args.bc_weight,
            shortcut_weight=args.consistency_weight,
            self_consistency_k=args.self_consistency_k,
            gamma=args.gamma,
            tau=args.tau,
            reward_scale=args.reward_scale,
            q_target_clip=args.q_target_clip,
            ema_decay=args.ema_decay,
            weight_clip=args.weight_clip,
            # ShortCut Flow parameters
            step_size_mode=args.sc_step_size_mode,
            fixed_step_size=args.sc_fixed_step_size,
            min_step_size=args.sc_min_step_size,
            max_step_size=args.sc_max_step_size,
            target_mode=args.sc_target_mode,
            teacher_steps=args.sc_teacher_steps,
            use_ema_teacher=args.sc_use_ema_teacher,
            t_min=args.sc_t_min,
            t_max=args.sc_t_max,
            inference_mode=args.sc_inference_mode,
            num_inference_steps=args.sc_num_inference_steps,
            device=device,
        )
    
    elif algorithm == "dppo":
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        
        return DPPOAgent(
            noise_pred_net=noise_pred_net,
            obs_dim=global_cond_dim,
            act_dim=action_dim,
            pred_horizon=args.pred_horizon,
            obs_horizon=args.obs_horizon,
            act_horizon=args.act_horizon,
            num_diffusion_iters=args.num_diffusion_iters,
            ft_denoising_steps=args.ft_denoising_steps,
            ema_decay=args.ema_decay,
            use_ema=True,
        )
    
    elif algorithm == "reinflow":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        return ReinFlowAgent(
            velocity_net=velocity_net,
            obs_dim=global_cond_dim,
            act_dim=action_dim,
            pred_horizon=args.pred_horizon,
            obs_horizon=args.obs_horizon,
            act_horizon=args.act_horizon,
            num_flow_steps=args.num_flow_steps,
            ema_decay=args.ema_decay,
            use_ema=True,
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def save_ckpt(run_name, tag, agent, ema_agent, visual_encoder=None):
    """Save checkpoint. Assumes ema_agent has already been updated with EMA params.
    
    Args:
        run_name: Run directory name
        tag: Checkpoint tag (e.g., 'best_eval_success_once')
        agent: Agent model
        ema_agent: EMA agent model
        visual_encoder: Optional visual encoder model (PlainConv)
    """
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ckpt = {
        "agent": agent.state_dict(),
        "ema_agent": ema_agent.state_dict(),
    }
    if visual_encoder is not None:
        ckpt["visual_encoder"] = visual_encoder.state_dict()
    torch.save(ckpt, f"runs/{run_name}/checkpoints/{tag}.pt")


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.exp_name is None:
        args.exp_name = f"{args.algorithm}-{args.env_id}"
        run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    
    # Validate demo path
    if args.demo_path.endswith(".h5"):
        import json
        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert control_mode == args.control_mode, \
                f"Control mode mismatch: {control_mode} vs {args.control_mode}"
    
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Create evaluation environment
    # Note: Use dense reward to match offline dataset reward mode
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="dense",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
    )
    assert args.max_episode_steps is not None, "max_episode_steps must be specified"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    
    # Wandb tracking
    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_eval_envs,
            env_id=args.env_id,
            env_horizon=args.max_episode_steps,
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group=args.algorithm,
            tags=[args.algorithm, "offline_rl"],
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Setup data processing
    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
        state_obs_extractor=build_state_obs_extractor(args.env_id),
    )
    
    # Get observation space from temp env
    tmp_env = gym.make(args.env_id, **env_kwargs)
    original_obs_space = tmp_env.observation_space
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    tmp_env.close()
    
    # Create dataset
    dataset = OfflineRLDataset(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=original_obs_space,
        include_rgb=include_rgb,
        device=device,
        num_traj=args.num_demos,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        act_horizon=args.act_horizon,
        control_mode=args.control_mode,
        gamma=args.gamma,  # For SMDP cumulative reward computation
    )
    
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )
    
    # Get action dimension from environment
    action_dim = envs.single_action_space.shape[0]
    
    # Determine state dimension from dataset
    sample_obs = dataset.trajectories["observations"][0]
    state_dim = sample_obs["state"].shape[-1]
    
    # Create visual encoder
    include_rgb_flag = "rgb" in dataset.obs_keys
    
    in_channels = 0
    if include_rgb_flag:
        in_channels += 3
    
    visual_encoder = None
    visual_feature_dim = 0
    if in_channels > 0:
        visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
        visual_feature_dim = args.visual_feature_dim
    
    # Compute global conditioning dimension
    # obs_features = concat(visual_features, state) per timestep, then concat across obs_horizon
    global_cond_dim = args.obs_horizon * (visual_feature_dim + state_dim)
    print(f"action_dim: {action_dim}, state_dim: {state_dim}, visual_feature_dim: {visual_feature_dim}")
    print(f"global_cond_dim: {global_cond_dim} = {args.obs_horizon} * ({visual_feature_dim} + {state_dim})")
    
    # Create agent
    agent = create_agent(args.algorithm, action_dim, global_cond_dim, args).to(device)
    print(f"Agent ({args.algorithm}) parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M")
    
    # Create agent wrapper for evaluation (handles obs encoding)
    # AgentWrapper is imported from diffusion_policy.utils
    agent_wrapper = AgentWrapper(
        agent, visual_encoder, include_rgb_flag, 
        args.obs_horizon, args.act_horizon
    ).to(device)
    
    # Setup optimizers
    # IL algorithms (no critic): BC-only training
    # DPPO and ReinFlow are included because their offline training is pure BC
    il_algorithms = ["diffusion_policy", "flow_matching", "reflected_flow", "consistency_flow", "shortcut_flow", "dppo", "reinflow"]
    
    # Collect all trainable parameters
    all_params = list(agent.parameters())
    if visual_encoder is not None:
        all_params += list(visual_encoder.parameters())
    
    if args.algorithm in il_algorithms:
        optimizer = optim.AdamW(
            params=all_params,
            lr=args.lr,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
        )
        critic_optimizer = None
    else:
        # Separate optimizers for actor and critic (offline RL algorithms)
        # IMPORTANT: Actor params include visual_encoder, critic params are separate
        # This ensures critic gradients don't affect visual encoder through shared optimizer
        actor_params = [p for n, p in agent.named_parameters() if "critic" not in n]
        if visual_encoder is not None:
            actor_params += list(visual_encoder.parameters())
        critic_params = list(agent.critic.parameters())
        
        optimizer = optim.AdamW(
            params=actor_params,
            lr=args.lr,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
        )
        critic_optimizer = optim.AdamW(
            params=critic_params,
            lr=args.lr_critic,
            betas=(0.9, 0.999),
            weight_decay=1e-6,
        )
        
        # Store actor_params and critic_params for separate gradient clipping
        actor_params_for_clip = actor_params
        critic_params_for_clip = critic_params
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )
    
    # EMA setup - different for IL vs Offline RL algorithms
    if args.algorithm in il_algorithms:
        # For IL algorithms: EMA for entire agent
        ema = EMAModel(parameters=agent.parameters(), power=0.75)
        ema_agent = create_agent(args.algorithm, action_dim, global_cond_dim, args).to(device)
        ema_critic = None
        ema_critic_params = None
    else:
        # For Offline RL algorithms: Separate EMA for actor and critic
        # Actor EMA: only actor parameters (excludes critic)
        actor_params_for_ema = [p for n, p in agent.named_parameters() if "critic" not in n]
        ema = EMAModel(parameters=actor_params_for_ema, power=0.75)
        
        # Critic EMA: only critic parameters
        critic_params_for_ema = list(agent.critic.parameters())
        ema_critic = EMAModel(parameters=critic_params_for_ema, power=0.75)
        
        # Store param names for proper EMA copying
        ema_actor_param_names = [n for n, p in agent.named_parameters() if "critic" not in n]
        ema_critic_param_names = [n for n, p in agent.named_parameters() if "critic" in n and "target" not in n]
        
        ema_agent = create_agent(args.algorithm, action_dim, global_cond_dim, args).to(device)
    
    ema_agent_wrapper = AgentWrapper(ema_agent, visual_encoder, include_rgb_flag, args.obs_horizon).to(device)
    
    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)
    
    def copy_ema_to_eval_agent():
        """Copy EMA parameters to evaluation agent."""
        if args.algorithm in il_algorithms:
            # IL algorithms: simple copy
            ema.copy_to(ema_agent.parameters())
        else:
            # Offline RL: copy actor EMA to actor params, critic EMA to critic params
            # Copy actor EMA
            ema_actor_params = [p for n, p in ema_agent.named_parameters() if "critic" not in n]
            ema.copy_to(ema_actor_params)
            # Copy critic EMA
            if ema_critic is not None:
                ema_critic_params = list(ema_agent.critic.parameters())
                ema_critic.copy_to(ema_critic_params)
    
    def evaluate_and_save_best(iteration):
        if iteration % args.eval_freq == 0 and iteration > 0:
            last_tick = time.time()
            copy_ema_to_eval_agent()
            eval_metrics = evaluate(
                args.num_eval_episodes, ema_agent_wrapper, envs, device, args.sim_backend
            )
            timings["eval"] += time.time() - last_tick
            
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")
            
            for k in ["success_once", "success_at_end"]:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    # copy_ema_to_eval_agent is already called above, so ema_agent is up-to-date
                    save_ckpt(run_name, f"best_eval_{k}", agent, ema_agent, visual_encoder)
                    print(f"New best {k}: {eval_metrics[k]:.4f}. Saving checkpoint.")
    
    def log_metrics(iteration, losses):
        if iteration % args.log_freq == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            for k, v in losses.items():
                writer.add_scalar(f"losses/{k}", v, iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)
    
    # Training loop
    agent.train()
    if visual_encoder is not None:
        visual_encoder.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    
    # Define IL algorithms (no critic)
    # DPPO and ReinFlow are included because their offline training is pure BC
    il_algorithms = ["diffusion_policy", "flow_matching", "reflected_flow", "consistency_flow", "shortcut_flow", "dppo", "reinflow"]
    
    # Helper function to encode observations
    def encode_observations(obs_seq):
        """Encode observations to get obs_features for agents."""
        B = obs_seq["state"].shape[0]
        T = obs_seq["state"].shape[1]
        
        features_list = []
        
        # Visual features
        if visual_encoder is not None:
            if "rgb" in obs_seq:
                rgb = obs_seq["rgb"]  # [B, T, C, H, W]
                rgb_flat = rgb.view(B * T, *rgb.shape[2:]).float() / 255.0
                visual_input = rgb_flat
            else:
                visual_input = None
            
            if visual_input is not None:
                visual_feat = visual_encoder(visual_input)  # [B*T, visual_dim]
                visual_feat = visual_feat.view(B, T, -1)  # [B, T, visual_dim]
                features_list.append(visual_feat)
        
        # State features
        state = obs_seq["state"]  # [B, T, state_dim]
        features_list.append(state)
        
        # Concatenate features: [B, T, visual_dim + state_dim]
        obs_features = torch.cat(features_list, dim=-1)
        
        return obs_features
    
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick
        
        last_tick = time.time()
        
        # Common: encode observations
        obs_seq = data_batch["observations"]
        action_seq = data_batch["actions"]
        obs_features = encode_observations(obs_seq)
        
        if args.algorithm in il_algorithms:
            # IL algorithm training (flow_matching, reflected_flow, consistency_flow, shortcut_flow)
            loss_dict = agent.compute_loss(
                obs_features=obs_features,
                actions=action_seq,
            )
            
            if isinstance(loss_dict, dict):
                total_loss = loss_dict["loss"]
                losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            else:
                total_loss = loss_dict
                losses = {"total_loss": total_loss.item()}
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            if visual_encoder is not None:
                torch.nn.utils.clip_grad_norm_(visual_encoder.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            # Update EMA for consistency_flow
            if hasattr(agent, "update_ema"):
                agent.update_ema()
        
        else:
            # Offline RL training (diffusion_double_q or cpql)
            next_obs_seq = data_batch["next_observations"]
            rewards = data_batch["rewards"]
            dones = data_batch["dones"]
            
            # actions_for_q: act_horizon length, matches SMDP reward horizon
            actions_for_q = data_batch["actions_for_q"]
            
            # SMDP chunk-level fields for proper action chunking Bellman equation
            cumulative_reward = data_batch["cumulative_reward"]
            chunk_done = data_batch["chunk_done"]
            discount_factor = data_batch["discount_factor"]
            
            # Encode next observations
            next_obs_features = encode_observations(next_obs_seq)
            
            # Flatten obs_features for compute_loss (agents expect flattened)
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
            next_obs_cond = next_obs_features.reshape(next_obs_features.shape[0], -1)
            
            # Compute combined loss using the unified interface with SMDP fields
            # actions: full pred_horizon for BC training
            # actions_for_q: act_horizon for Q-learning (matches reward horizon)
            loss_dict = agent.compute_loss(
                obs_features=obs_features,
                actions=action_seq,  # pred_horizon for BC
                actions_for_q=actions_for_q,  # act_horizon for Q-learning
                rewards=rewards,
                next_obs_features=next_obs_features,
                dones=dones,
                # SMDP fields for proper action chunking
                cumulative_reward=cumulative_reward,
                chunk_done=chunk_done,
                discount_factor=discount_factor,
            )
            
            # Extract actor and critic losses separately
            actor_loss = loss_dict.get("actor_loss", loss_dict.get("policy_loss", loss_dict["loss"]))
            critic_loss = loss_dict.get("critic_loss", torch.tensor(0.0))
            
            losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            
            # ===== Separate backward passes for actor and critic =====
            # This ensures gradients don't interfere between actor and critic
            
            # Step 1: Backward pass for actor (includes visual_encoder)
            optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)  # retain_graph for critic backward
            
            # Gradient clipping for actor only
            torch.nn.utils.clip_grad_norm_(actor_params_for_clip, 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            
            # Step 2: Backward pass for critic (separate)
            if critic_optimizer is not None and isinstance(critic_loss, torch.Tensor) and critic_loss.requires_grad:
                critic_optimizer.zero_grad()
                critic_loss.backward()
                
                # Gradient clipping for critic only
                torch.nn.utils.clip_grad_norm_(critic_params_for_clip, 1.0)
                
                critic_optimizer.step()
            
            # Update targets and EMA
            agent.update_target()
            if hasattr(agent, "update_ema"):
                agent.update_ema()
        
        timings["forward"] += time.time() - last_tick
        
        # EMA step - separate for actor and critic
        last_tick = time.time()
        if args.algorithm in il_algorithms:
            ema.step(agent.parameters())
        else:
            # Update actor EMA (excludes critic)
            actor_params_for_ema_step = [p for n, p in agent.named_parameters() if "critic" not in n]
            ema.step(actor_params_for_ema_step)
            # Update critic EMA
            if ema_critic is not None:
                critic_params_for_ema_step = list(agent.critic.parameters())
                ema_critic.step(critic_params_for_ema_step)
        timings["ema"] += time.time() - last_tick
        
        # Evaluation
        evaluate_and_save_best(iteration)
        log_metrics(iteration, losses)
        
        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            copy_ema_to_eval_agent()
            save_ckpt(run_name, str(iteration), agent, ema_agent, visual_encoder)
        
        pbar.update(1)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
        last_tick = time.time()
    
    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters, losses)
    
    envs.close()
    writer.close()
