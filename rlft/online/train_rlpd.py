"""
RLPD Online Training Script

Online reinforcement learning with offline data mixing (RLPD style).
Supports SAC and AWSC (Advantage-Weighted ShortCut Flow) with action chunking.

Usage:
    # SAC (default):
    python -m rlft.online.train_rlpd \
        --env_id PickCube-v1 \
        --demo_path ~/demonstrations.h5 \
        --total_timesteps 1000000
    
    # AWSC (with pretrained shortcut flow):
    python -m rlft.online.train_rlpd \
        --algorithm awsc \
        --env_id PickCube-v1 \
        --demo_path ~/demonstrations.h5 \
        --pretrain_path runs/shortcut_bc/best.pt \
        --total_timesteps 1000000
"""

ALGO_NAME = "RLPD"

import os
import random
import time
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter

# Import from rlft package
from rlft.networks import PlainConv, ShortCutVelocityUNet1D
from rlft.algorithms.online_rl import SACAgent, AWSCAgent
from rlft.buffers import OnlineReplayBuffer, OnlineReplayBufferRaw, SMDPChunkCollector, SuccessReplayBuffer
from rlft.envs import make_eval_envs, evaluate
from rlft.datasets import ManiSkillDataset, OfflineRLDataset, ActionNormalizer
from rlft.datasets.data_utils import ObservationStacker, encode_observations as _encode_observations_shared


@dataclass
class Args:
    """RLPD training arguments."""
    # Experiment settings
    exp_name: Optional[str] = None
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "RLPD"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    
    # Algorithm selection
    algorithm: Literal["sac", "awsc"] = "sac"
    """algorithm to use: sac (Soft Actor-Critic) or awsc (Advantage-Weighted ShortCut Flow)"""
    
    # Environment settings
    env_id: str = "LiftPegUpright-v1"
    num_envs: int = 16
    num_eval_envs: int = 16
    max_episode_steps: int = 100
    control_mode: str = "pd_ee_delta_pose"
    obs_mode: str = "rgb"
    sim_backend: str = "physx_cuda"
    
    # Data settings
    demo_path: Optional[str] = "~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"
    num_demos: Optional[int] = None
    online_ratio: float = 0.15
    """Fraction of online data in each batch. Sweep v3: or=0.15 best stability;
    v4: confirmed as optimal for 250K-500K training. Lower ratio = stronger demo anchoring."""
    
    # Training settings
    total_timesteps: int = 500_000
    """Total environment steps. Sweep v4: 500K shows diminishing returns,
    most configs plateau at 250K. 1M is unnecessarily long."""
    num_seed_steps: int = 5000
    utd_ratio: int = 20
    batch_size: int = 256
    eval_freq: int = 10000
    save_freq: int = 50000
    log_freq: int = 100
    num_eval_episodes: int = 50
    
    # Optimizer settings
    lr_actor: float = 1e-4
    """Actor learning rate. Sweep v2: 3e-4 causes catastrophic forgetting;
    v3/v4: 1e-4 optimal for 250K, 7e-5 for 500K. 1e-4 is safe default."""
    lr_critic: float = 1e-4
    """Critic learning rate. Should match lr_actor for AWSC."""
    lr_temp: float = 3e-4
    max_grad_norm: float = 10.0
    
    # SAC hyperparameters
    gamma: float = 0.9
    tau: float = 0.005
    init_temperature: float = 1.0
    target_entropy: Optional[float] = None
    backup_entropy: bool = False
    reward_scale: float = 1.0
    
    # Ensemble Q settings
    num_qs: int = 10
    num_min_qs: int = 2
    q_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    
    # Policy settings
    obs_horizon: int = 2
    act_horizon: int = 8
    """Action horizon (number of action steps fed to critic). Default 8 to match
    offline pretrained checkpoints. Must align with pretrain checkpoint's critic
    input_dim = obs_dim + action_dim * act_horizon."""
    pred_horizon: int = 8
    
    # Visual encoder
    visual_feature_dim: int = 256
    freeze_visual_encoder: bool = False
    
    # Replay buffer
    replay_buffer_capacity: int = 500_000
    
    # AWSC-specific settings (only used when algorithm="awsc")
    pretrain_path: Optional[str] = None
    """Path to pretrained ShortCut Flow or AW-ShortCut checkpoint for AWSC"""
    load_pretrain_critic: bool = False
    """Whether to load critic from pretrain checkpoint (recommended for AW-SC offline checkpoints)"""
    
    # AWSC hyperparameters (unified with old version)
    awsc_beta: float = 50.0
    """Temperature for advantage weighting. Sweep v2-v4: beta=50 most robust.
    Higher beta (80+) needs accurate Q-values and amplifies noise with low K."""
    awsc_bc_weight: float = 2.0
    """Weight for flow matching loss. Sweep v2: bc=2.0 critical for stability
    by anchoring actor near pretrained policy."""
    awsc_shortcut_weight: float = 0.3
    """Weight for shortcut consistency loss"""
    awsc_self_consistency_k: float = 0.25
    """Fraction of batch for consistency loss (match IL/offline_rl)"""
    awsc_ema_decay: float = 0.9995
    """EMA decay rate for velocity network (best from wave3)"""
    awsc_weight_clip: float = 200.0
    """Maximum advantage weight to prevent outliers (best from wave3)"""
    awsc_exploration_noise_std: float = 0.1
    """Standard deviation of exploration noise"""
    awsc_num_inference_steps: int = 8
    """Number of inference steps for flow sampling"""
    awsc_q_target_clip: float = 100.0
    """Q-target clipping range"""
    awsc_filter_policy_data: bool = False
    """Whether to filter policy training data by advantage"""
    awsc_advantage_threshold: float = -0.5
    """Minimum advantage for online samples in policy training"""
    
    # Advantage computation mode (AWSC)
    awsc_advantage_mode: Literal["batch_mean", "per_state_v"] = "per_state_v"
    """How to compute advantage baseline for Q-weighting:
    - 'batch_mean': A(s,a) = Q(s,a) - mean(Q) over batch (fast, but state-independent)
    - 'per_state_v': A(s,a) = Q(s,a) - V(s) where V(s) = E_{a'~π}[Q(s,a')]
      (proper AWAC, distinguishes good/bad actions per state, costs extra forward passes)
    Sweep v2-v4: per_state_v consistently superior with stable training (low LR + low OR)."""
    awsc_num_v_samples: int = 4
    """Number of policy samples to estimate V(s) for per_state_v mode"""
    
    # Actor policy training mode (AWSC)
    actor_policy_mode: Literal["all", "success_only"] = "all"
    """Data source for actor (policy) updates:
    - 'all': Use full mixed batch (demo + online buffer), same as critic
    - 'success_only': Use only demo + online success buffer for actor;
      prevents policy from imitating failed rollout actions"""
    success_criteria: Literal["success_once", "success_at_end"] = "success_once"
    """How to determine if a transition is 'successful' for success_buffer:
    - 'success_once': Mark as success if the agent succeeded at any point during
      the episode (cumulative). Requires tracking per-env success_once state.
    - 'success_at_end': Mark as success only if info['success'] is True at episode
      termination (instantaneous). This is stricter and may result in zero successes
      for tasks where the agent can't maintain success at the final step."""
    
    # ShortCut Flow U-Net settings (for AWSC, matches offline training)
    unet_down_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """U-Net down block dimensions (must match pretrained checkpoint)"""
    unet_kernel_size: int = 5
    """U-Net kernel size"""
    unet_n_groups: int = 8
    """U-Net group norm groups"""
    diffusion_step_embed_dim: int = 64
    """Diffusion step embedding dimension (must match pretrained checkpoint)"""
    
    # Success Replay Buffer settings (for AWSC)
    use_success_buffer: bool = False
    """Whether to store successful episodes in SuccessReplayBuffer"""
    success_buffer_capacity: int = 100_000
    """Capacity of success replay buffer"""
    
    # Action normalization settings
    normalize_actions: bool = True
    """Whether to normalize actions during training (for offline data)"""
    action_norm_mode: Literal["standard", "minmax"] = "standard"
    """Action normalization mode: 'standard' (zero mean, unit var) or 'minmax' (scale to [-1, 1])"""
    action_bounds: Optional[tuple] = (-1.0, 1.0)
    """Action bounds for clamping during inference. Set to None to disable clamping.
    ManiSkill environments have action space [-1, 1], so we clamp by default."""


class AgentWrapper:
    """Wrapper for unified evaluation interface.
    
    Note: eval_envs already applies FrameStack, so obs comes in as
    (num_envs, obs_horizon, state_dim) format - no additional stacking needed.
    """
    def __init__(self, agent, visual_encoder, include_rgb, obs_horizon, act_horizon, device, 
                 action_normalizer=None, include_depth=False):
        self.agent = agent
        self.visual_encoder = visual_encoder
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.device = device
        self.action_normalizer = action_normalizer
    
    def reset(self, obs):
        """Reset is a no-op for eval since FrameStack handles history."""
        pass
    
    def encode_obs(self, obs):
        """Encode observation to conditioning features.
        
        Args:
            obs: Dict with 'state' key, shape (B, T, state_dim) from FrameStack
                 OR tensor shape (B, T, state_dim)
        
        Returns:
            Flattened features (B, T * feature_dim)
        """
        # Handle dict or tensor input
        if isinstance(obs, dict):
            state = obs["state"]
        else:
            state = obs
        
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        state = state.float()
        
        # state shape: (B, T, state_dim) from FrameStack
        B = state.shape[0]
        T = state.shape[1]
        features = []
        
        if self.include_rgb and self.visual_encoder is not None:
            rgb = obs.get("rgb") if isinstance(obs, dict) else None
            if rgb is not None:
                if isinstance(rgb, np.ndarray):
                    rgb = torch.from_numpy(rgb).to(self.device)
                rgb = rgb.contiguous()  # Ensure contiguous memory layout
                rgb_flat = rgb.reshape(B * T, *rgb.shape[2:]).float() / 255.0
                if rgb_flat.ndim == 4 and rgb_flat.shape[-1] == 3:
                    rgb_flat = rgb_flat.permute(0, 3, 1, 2).contiguous()
                
                # Handle depth if available
                if self.include_depth and "depth" in obs:
                    depth = obs["depth"]
                    if isinstance(depth, np.ndarray):
                        depth = torch.from_numpy(depth).to(self.device)
                    depth = depth.contiguous()
                    depth_flat = depth.reshape(B * T, *depth.shape[2:]).float() / 1024.0
                    if depth_flat.ndim == 4 and depth_flat.shape[-1] in [1, 2, 4]:
                        depth_flat = depth_flat.permute(0, 3, 1, 2).contiguous()
                    visual_input = torch.cat([rgb_flat, depth_flat], dim=1)
                else:
                    visual_input = rgb_flat
                
                visual_feat = self.visual_encoder(visual_input)
                visual_feat = visual_feat.view(B, T, -1)
                features.append(visual_feat)
        
        features.append(state)
        
        obs_features = torch.cat(features, dim=-1)
        return obs_features.reshape(B, -1)  # (B, T * feature_dim)
    
    @torch.no_grad()
    def get_action(self, obs, deterministic=True, **kwargs):
        """Get action from agent.
        
        Uses get_action() for full pred_horizon output, then applies temporal
        offset slice [obs_horizon-1 : obs_horizon-1+act_horizon] to align with
        training data (matching the offline evaluation convention).
        
        Args:
            obs: Observation from eval_envs (already FrameStacked)
                 Shape: (B, T, state_dim) or dict with 'state' key
            deterministic: Whether to use deterministic action
        
        Returns:
            actions: (B, act_horizon, act_dim)
        """
        obs_cond = self.encode_obs(obs)
        
        # For flow/diffusion agents (AWSC), use get_action() which returns the
        # full pred_horizon sequence, then apply temporal offset slice
        # [obs_horizon-1 : obs_horizon-1+act_horizon] to match offline eval.
        # For other agents (e.g. SAC), use select_action() which already
        # returns (B, act_horizon, action_dim) without offset.
        if isinstance(self.agent, AWSCAgent):
            actions_full = self.agent.get_action(obs_cond, deterministic=deterministic)
            # Temporal offset slice: [obs_horizon-1 : obs_horizon-1+act_horizon]
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            actions = actions_full[:, start:end]
        else:
            actions = self.agent.select_action(obs_cond, deterministic=deterministic)
        
        # Denormalize actions if normalizer is provided
        if self.action_normalizer is not None:
            actions_np = actions.cpu().numpy()
            actions_denorm = self.action_normalizer.inverse_transform(actions_np)
            return torch.from_numpy(actions_denorm).float().to(actions.device)
        
        return actions
    
    def eval(self):
        """Set agent and visual encoder to evaluation mode."""
        self.agent.eval()
        if self.visual_encoder is not None:
            self.visual_encoder.eval()
        return self
    
    def train(self, mode=True):
        """Set agent and visual encoder to training mode."""
        self.agent.train(mode)
        if self.visual_encoder is not None:
            self.visual_encoder.train(mode)
        return self


def make_train_envs(args):
    """Create parallel training environments."""
    try:
        import gymnasium as gym
        import mani_skill.envs
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    except ImportError:
        raise ImportError("ManiSkill3 is required. Install with: pip install mani-skill")
    
    env_kwargs = dict(
        obs_mode="rgbd" if "rgb" in args.obs_mode else "state",
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        num_envs=args.num_envs,
        reward_mode="dense",
    )
    
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    
    env = gym.make(args.env_id, **env_kwargs)
    
    if "rgb" in args.obs_mode:
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    
    return env


def validate_pretrained_checkpoint(
    checkpoint_path: str,
    args: "Args",
    device: torch.device,
    obs_dim: int = 0,
    action_dim: int = 0,
):
    """Validate pretrained checkpoint architecture against current training config.
    
    Inspects weight tensor shapes to infer the checkpoint's architecture parameters
    and compares them against the current args. Raises RuntimeError on mismatch.
    
    This catches silent failures where a pretrained model loads successfully
    (Conv1d doesn't bind sequence length) but produces garbage outputs due to
    mismatched pred_horizon, obs_dim, etc.
    
    Args:
        checkpoint_path: Path to the pretrained checkpoint.
        args: Current training arguments.
        device: Torch device.
        obs_dim: Computed obs_dim (obs_horizon * feature_dim) from current config.
        action_dim: Environment action dimension.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get agent state dict (prefer ema_agent)
    if "ema_agent" in checkpoint:
        agent_state = checkpoint["ema_agent"]
    elif "agent" in checkpoint:
        agent_state = checkpoint["agent"]
    else:
        agent_state = checkpoint
    
    # Extract velocity_net weights (may have "velocity_net." prefix)
    vnet_state = {}
    for key, value in agent_state.items():
        if key.startswith("velocity_net."):
            vnet_state[key.replace("velocity_net.", "")] = value
    
    if not vnet_state:
        print(f"  [Checkpoint Validation] No velocity_net weights found, skipping validation")
        return
    
    # Determine UNet prefix (ShortCutVelocityUNet1D wraps UNet under "unet.")
    unet_prefix = "unet." if any(k.startswith("unet.") for k in vnet_state) else ""
    
    errors = []
    warnings = []
    
    # --- 1. Infer diffusion_step_embed_dim ---
    dsed_key = f"{unet_prefix}diffusion_step_encoder.3.bias"
    if dsed_key in vnet_state:
        ckpt_dsed = vnet_state[dsed_key].shape[0]
        if ckpt_dsed != args.diffusion_step_embed_dim:
            errors.append(
                f"diffusion_step_embed_dim: checkpoint={ckpt_dsed}, "
                f"current={args.diffusion_step_embed_dim}. "
                f"Fix: --diffusion_step_embed_dim {ckpt_dsed}"
            )
    
    # --- 2. Infer action_dim (input_dim) ---
    action_dim_key = f"{unet_prefix}final_conv.1.weight"
    if action_dim_key in vnet_state:
        ckpt_action_dim = vnet_state[action_dim_key].shape[0]
        if action_dim > 0 and ckpt_action_dim != action_dim:
            errors.append(
                f"action_dim: checkpoint={ckpt_action_dim}, "
                f"environment={action_dim}. "
                f"The checkpoint was trained for a different action space."
            )
        else:
            warnings.append(f"Checkpoint action_dim={ckpt_action_dim}")
    
    # --- 3. Infer global_cond_dim (obs_dim = obs_horizon * per_step_feature_dim) ---
    cond_key = f"{unet_prefix}down_modules.0.0.cond_encoder.1.weight"
    ckpt_obs_dim = None
    if cond_key in vnet_state and dsed_key in vnet_state:
        ckpt_cond_dim = vnet_state[cond_key].shape[1]
        ckpt_dsed = vnet_state[dsed_key].shape[0]
        ckpt_obs_dim = ckpt_cond_dim - ckpt_dsed
        if obs_dim > 0 and ckpt_obs_dim != obs_dim:
            errors.append(
                f"obs_dim (global_cond_dim): checkpoint={ckpt_obs_dim}, "
                f"current={obs_dim} (obs_horizon={args.obs_horizon} × feature_dim). "
                f"This is likely an obs_horizon or visual_feature_dim mismatch. "
                f"Fix: adjust --obs_horizon or --visual_feature_dim"
            )
        else:
            warnings.append(f"Checkpoint obs_dim (global_cond_dim)={ckpt_obs_dim}")
    
    # --- 4. Infer down_dims ---
    ckpt_down_dims = []
    i = 0
    while True:
        dd_key = f"{unet_prefix}down_modules.{i}.0.blocks.0.block.0.weight"
        if dd_key in vnet_state:
            ckpt_down_dims.append(vnet_state[dd_key].shape[0])
            i += 1
        else:
            break
    
    if ckpt_down_dims and tuple(ckpt_down_dims) != tuple(args.unet_down_dims):
        errors.append(
            f"unet_down_dims: checkpoint={tuple(ckpt_down_dims)}, "
            f"current={tuple(args.unet_down_dims)}. "
            f"Fix: --unet_down_dims {' '.join(str(d) for d in ckpt_down_dims)}"
        )
    
    # --- 5. Infer kernel_size ---
    ks_key = f"{unet_prefix}down_modules.0.0.blocks.0.block.0.weight"
    if ks_key in vnet_state:
        ckpt_kernel_size = vnet_state[ks_key].shape[2]
        if ckpt_kernel_size != args.unet_kernel_size:
            errors.append(
                f"unet_kernel_size: checkpoint={ckpt_kernel_size}, "
                f"current={args.unet_kernel_size}. "
                f"Fix: --unet_kernel_size {ckpt_kernel_size}"
            )
    
    # --- 6. Check config dict if saved in checkpoint ---
    ckpt_config = checkpoint.get("config", None)
    if ckpt_config is not None:
        # Compare critical horizon parameters
        for param in ["pred_horizon", "obs_horizon"]:
            ckpt_val = ckpt_config.get(param)
            curr_val = getattr(args, param)
            if ckpt_val is not None and ckpt_val != curr_val:
                errors.append(
                    f"{param}: checkpoint={ckpt_val}, current={curr_val}. "
                    f"Fix: --{param} {ckpt_val}"
                )
    
    # --- 7. Validate critic dimensions (act_horizon must match for load_pretrain_critic) ---
    if args.load_pretrain_critic:
        critic_key = None
        for key in agent_state:
            if key.startswith("critic.q_nets.0.0.weight"):
                critic_key = key
                break
        if critic_key is not None:
            ckpt_critic_input_dim = agent_state[critic_key].shape[1]
            expected_critic_input_dim = obs_dim + action_dim * args.act_horizon
            if ckpt_critic_input_dim != expected_critic_input_dim:
                ckpt_act_horizon = (ckpt_critic_input_dim - obs_dim) // action_dim
                errors.append(
                    f"critic input_dim: checkpoint={ckpt_critic_input_dim} "
                    f"(act_horizon={ckpt_act_horizon}), "
                    f"current={expected_critic_input_dim} "
                    f"(act_horizon={args.act_horizon}). "
                    f"Fix: --act_horizon {ckpt_act_horizon}"
                )
            else:
                warnings.append(
                    f"Critic input_dim={ckpt_critic_input_dim} matches "
                    f"(act_horizon={args.act_horizon}) ✓"
                )
    
    # --- Report ---
    if warnings:
        for w in warnings:
            print(f"  [Checkpoint Info] {w}")
    
    if errors:
        error_msg = (
            f"\n{'=' * 60}\n"
            f"PRETRAINED CHECKPOINT PARAMETER CONFLICT DETECTED\n"
            f"{'=' * 60}\n"
            f"Checkpoint: {checkpoint_path}\n\n"
            f"The following parameters in the current config do NOT match\n"
            f"the pretrained checkpoint's architecture. This will cause\n"
            f"the model to produce incorrect outputs despite loading\n"
            f"without errors.\n\n"
        )
        for i, e in enumerate(errors, 1):
            error_msg += f"  {i}. {e}\n"
        error_msg += (
            f"\nPlease fix the parameters above or remove --pretrain_path.\n"
            f"{'=' * 60}\n"
        )
        raise RuntimeError(error_msg)
    
    print(f"  [Checkpoint Validation] All architecture parameters match ✓")


def main():
    args = tyro.cli(Args)
    
    if args.exp_name is None:
        args.exp_name = f"rlpd-{args.env_id}-seed{args.seed}"
    
    run_name = f"{args.exp_name}__{int(time.time())}"
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
    
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )
    
    writer = SummaryWriter(log_dir)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environments
    print("Creating training environments...")
    train_envs = make_train_envs(args)
    
    print("Creating evaluation environments...")
    eval_env_kwargs = dict(
        control_mode=args.control_mode,
        obs_mode="rgbd" if "rgb" in args.obs_mode else "state",
        render_mode="rgb_array",
    )
    if args.max_episode_steps is not None:
        eval_env_kwargs["max_episode_steps"] = args.max_episode_steps
    
    eval_other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    # Import wrapper for evaluation environment
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    eval_wrappers = [FlattenRGBDObservationWrapper] if "rgb" in args.obs_mode else []
    
    eval_envs = make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        sim_backend=args.sim_backend,
        env_kwargs=eval_env_kwargs,
        other_kwargs=eval_other_kwargs,
        video_dir=f"{log_dir}/videos" if args.capture_video else None,
        wrappers=eval_wrappers,
    )
    
    # Get environment info
    obs_space = train_envs.single_observation_space
    act_space = train_envs.single_action_space
    action_dim = act_space.shape[0]
    
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    
    include_rgb = "rgb" in args.obs_mode
    # Note: obs_space is gymnasium.spaces.Dict, not Python dict
    # Check if it has a "state" key (for flattened RGB observations)
    if hasattr(obs_space, 'spaces') and "state" in obs_space.spaces:
        state_dim = obs_space["state"].shape[0]
    elif hasattr(obs_space, '__getitem__') and "state" in obs_space:
        state_dim = obs_space["state"].shape[0]
    else:
        state_dim = obs_space.shape[0]
    
    # Visual encoder
    visual_encoder = None
    visual_feature_dim = 0
    
    if include_rgb:
        visual_encoder = PlainConv(
            in_channels=3,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
        visual_feature_dim = args.visual_feature_dim
        
        if args.freeze_visual_encoder:
            for param in visual_encoder.parameters():
                param.requires_grad = False
    
    obs_dim = args.obs_horizon * (visual_feature_dim + state_dim)
    print(f"State dim: {state_dim}, Visual dim: {visual_feature_dim}, Total obs dim: {obs_dim}")
    
    # Create agent based on algorithm selection
    if args.algorithm == "sac":
        print("Creating SAC agent...")
        agent = SACAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_horizon=args.act_horizon,
            hidden_dims=args.q_hidden_dims,
            num_qs=args.num_qs,
            num_min_qs=args.num_min_qs,
            gamma=args.gamma,
            tau=args.tau,
            init_temperature=args.init_temperature,
            target_entropy=args.target_entropy,
            backup_entropy=args.backup_entropy,
            reward_scale=args.reward_scale,
            action_bounds=args.action_bounds,
            device=device,
        ).to(device)
        use_temperature = True
    elif args.algorithm == "awsc":
        print("Creating AWSC agent...")
        # Create velocity network for ShortCut Flow
        velocity_net = ShortCutVelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_down_dims),
            kernel_size=args.unet_kernel_size,
            n_groups=args.unet_n_groups,
        ).to(device)
        
        agent = AWSCAgent(
            velocity_net=velocity_net,
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            act_horizon=args.act_horizon,
            num_qs=args.num_qs,
            num_min_qs=args.num_min_qs,
            q_hidden_dims=args.q_hidden_dims,
            num_inference_steps=args.awsc_num_inference_steps,
            beta=args.awsc_beta,
            bc_weight=args.awsc_bc_weight,
            shortcut_weight=args.awsc_shortcut_weight,
            self_consistency_k=args.awsc_self_consistency_k,
            gamma=args.gamma,
            tau=args.tau,
            reward_scale=args.reward_scale,
            q_target_clip=args.awsc_q_target_clip,
            ema_decay=args.awsc_ema_decay,
            weight_clip=args.awsc_weight_clip,
            action_bounds=args.action_bounds,
            exploration_noise_std=args.awsc_exploration_noise_std,
            filter_policy_data=args.awsc_filter_policy_data,
            advantage_threshold=args.awsc_advantage_threshold,
            advantage_mode=args.awsc_advantage_mode,
            num_v_samples=args.awsc_num_v_samples,
            device=device,
        ).to(device)
        use_temperature = False
        
        # Load pretrained checkpoint if provided
        if args.pretrain_path:
            print(f"Loading pretrained checkpoint from {args.pretrain_path}...")
            
            # Validate checkpoint architecture against current config BEFORE loading
            validate_pretrained_checkpoint(args.pretrain_path, args, device,
                                           obs_dim=obs_dim, action_dim=action_dim)
            
            agent.load_pretrained(
                args.pretrain_path,
                load_critic=args.load_pretrain_critic,
                strict=False,
                use_ema=True,
            )
            # Also load visual encoder if available
            if include_rgb and visual_encoder is not None:
                checkpoint = torch.load(args.pretrain_path, map_location=device)
                if "visual_encoder" in checkpoint:
                    visual_encoder.load_state_dict(checkpoint["visual_encoder"])
                    print(f"  Loaded visual encoder from checkpoint")
        
        # Print policy data filtering settings
        if args.awsc_filter_policy_data:
            print(f"Policy-Critic data separation enabled:")
            print(f"  - Advantage threshold: {args.awsc_advantage_threshold}")
            print(f"  - Policy uses: demos + high-advantage online samples")
            print(f"  - Critic uses: all data")
        
        # Print actor policy mode and advantage mode
        print(f"Actor policy mode: {args.actor_policy_mode}")
        if args.actor_policy_mode == "success_only":
            print(f"  - Actor trains on: demo buffer + online success buffer only")
            print(f"  - Critic trains on: all data (demo + full online buffer)")
        print(f"Advantage mode: {args.awsc_advantage_mode}")
        if args.awsc_advantage_mode == "per_state_v":
            print(f"  - V(s) estimated with {args.awsc_num_v_samples} policy samples per state")
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Optimizers - different setup based on algorithm
    if args.algorithm == "sac":
        actor_params = list(agent.actor.parameters())
        if visual_encoder is not None and not args.freeze_visual_encoder:
            actor_params += list(visual_encoder.parameters())
        
        actor_optimizer = optim.Adam(actor_params, lr=args.lr_actor)
        critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.lr_critic)
        temp_optimizer = optim.Adam(agent.temperature.parameters(), lr=args.lr_temp)
    elif args.algorithm == "awsc":
        actor_params = list(agent.velocity_net.parameters())
        if visual_encoder is not None and not args.freeze_visual_encoder:
            actor_params += list(visual_encoder.parameters())
        
        actor_optimizer = optim.Adam(actor_params, lr=args.lr_actor)
        critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.lr_critic)
        temp_optimizer = None  # AWSC doesn't use temperature
    
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent parameters: {total_params / 1e6:.2f}M")
    
    # Replay buffers
    rgb_shape = (128, 128, 3)
    online_buffer = OnlineReplayBufferRaw(
        capacity=args.replay_buffer_capacity,
        num_envs=args.num_envs,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=args.act_horizon,
        obs_horizon=args.obs_horizon,
        include_rgb=include_rgb,
        rgb_shape=rgb_shape,
        gamma=args.gamma,
        device=device,
    )
    
    # Success buffer (for storing successful online episodes)
    # Auto-enable when actor_policy_mode requires it
    success_buffer = None
    need_success_buffer = args.use_success_buffer or (
        args.algorithm == "awsc" and args.actor_policy_mode == "success_only"
    )
    if need_success_buffer:
        success_buffer = SuccessReplayBuffer(
            capacity=args.success_buffer_capacity,
            num_envs=args.num_envs,
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=args.act_horizon,
            obs_horizon=args.obs_horizon,
            include_rgb=include_rgb,
            rgb_shape=rgb_shape,
            gamma=args.gamma,
            device=device,
        )
        print(f"Success buffer enabled (capacity: {args.success_buffer_capacity})")
        print(f"Success criteria: {args.success_criteria}")
        if args.success_criteria == "success_once":
            print("  - Transitions marked successful if agent succeeded at ANY point in episode")
        else:
            print("  - Transitions marked successful only if success=True at episode termination")
    
    # Create action normalizer if needed (for offline data)
    # When loading a pretrained model, check if it was trained with normalization.
    # If the checkpoint has no action_normalizer, the model outputs raw actions;
    # applying inverse_transform would corrupt them.
    action_normalizer = None
    if args.normalize_actions:
        if args.pretrain_path:
            pretrain_ckpt = torch.load(args.pretrain_path, map_location=device)
            if "action_normalizer" in pretrain_ckpt and pretrain_ckpt["action_normalizer"] is not None:
                # Load normalizer from pretrained checkpoint for consistency
                normalizer_info = pretrain_ckpt["action_normalizer"]
                action_normalizer = ActionNormalizer(mode=normalizer_info["mode"])
                import numpy as _np
                action_normalizer.stats = {
                    k: _np.array(v) if isinstance(v, list) else v
                    for k, v in normalizer_info["stats"].items()
                }
                print(f"Loaded action normalizer from pretrained checkpoint (mode={normalizer_info['mode']})")
            else:
                # Pretrained model was trained WITHOUT normalization
                print(f"WARNING: Pretrained checkpoint has no action_normalizer.")
                print(f"  Disabling action normalization to match pretrained model.")
                print(f"  (The model outputs raw actions; denormalizing would corrupt them.)")
                args.normalize_actions = False
            del pretrain_ckpt
        else:
            action_normalizer = ActionNormalizer(mode=args.action_norm_mode)
    
    # Offline dataset for RLPD (use OfflineRLDataset for SMDP formulation)
    offline_dataset = None
    if args.demo_path:
        offline_dataset = OfflineRLDataset(
            data_path=args.demo_path,
            include_rgb=include_rgb,
            num_traj=args.num_demos,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            act_horizon=args.act_horizon,
            control_mode=args.control_mode,
            env_id=args.env_id,
            rgb_format="NCHW",
            gamma=args.gamma,
            device=device,
            action_normalizer=action_normalizer,
        )
        print(f"Offline dataset size: {len(offline_dataset)}")
        
        # Precompute cache for faster sampling (up to 50k samples)
        offline_dataset.precompute_cache(max_cache_size=min(50000, len(offline_dataset)))
    
    # Agent wrapper for evaluation (created after action_normalizer is fitted)
    agent_wrapper = AgentWrapper(
        agent=agent,
        visual_encoder=visual_encoder,
        include_rgb=include_rgb,
        obs_horizon=args.obs_horizon,
        act_horizon=args.act_horizon,
        device=device,
        action_normalizer=action_normalizer,
    )
    
    # ========== Pre-training Evaluation (Baseline) ==========
    if args.pretrain_path:
        print("\n" + "=" * 50)
        print("Evaluating pretrained model (baseline)...")
        print("=" * 50)
        agent.eval() if hasattr(agent, 'eval') else None
        
        pretrain_eval_metrics = evaluate(
            args.num_eval_episodes,
            agent_wrapper,
            eval_envs,
            device,
            sim_backend=args.sim_backend,
        )
        
        print(f"Pretrain evaluation ({len(pretrain_eval_metrics['success_at_end'])} episodes):")
        pretrain_log = {"step": 0}
        for k in pretrain_eval_metrics.keys():
            pretrain_eval_metrics[k] = np.mean(pretrain_eval_metrics[k])
            writer.add_scalar(f"eval/{k}", pretrain_eval_metrics[k], 0)
            pretrain_log[f"pretrain_eval/{k}"] = pretrain_eval_metrics[k]
            print(f"  {k}: {pretrain_eval_metrics[k]:.4f}")
        
        if args.track:
            wandb.log(pretrain_log)
        
        print("=" * 50 + "\n")
    
    # Training loop
    print("\n" + "=" * 50)
    print("Starting RLPD training...")
    print("=" * 50 + "\n")
    
    obs, info = train_envs.reset()
    obs_stacker = ObservationStacker(args.obs_horizon)
    obs_stacker.reset(obs)
    
    def encode_observations(stacked_obs):
        return _encode_observations_shared(
            obs_seq=stacked_obs,
            visual_encoder=visual_encoder,
            include_rgb=include_rgb,
            device=device,
        )
    
    total_steps = 0
    last_eval_step = -args.eval_freq  # Track last eval to handle non-exact intervals
    episode_rewards = defaultdict(float)
    episode_lengths = defaultdict(int)
    episode_successes = []
    best_success_rate = 0.0
    
    # Per-env success_once tracking: accumulates success across steps within an episode
    # Reset on episode done. Used when success_criteria='success_once'.
    episode_has_succeeded = np.zeros(args.num_envs, dtype=np.float32)
    
    chunk_collector = SMDPChunkCollector(
        num_envs=args.num_envs,
        gamma=args.gamma,
        action_horizon=args.act_horizon,
    )
    
    # Training metrics accumulator
    training_metrics = defaultdict(list)
    
    pbar = tqdm(total=args.total_timesteps, desc="Training")
    
    while total_steps < args.total_timesteps:
        # Collect experience
        agent.eval() if hasattr(agent, 'eval') else None
        
        with torch.no_grad():
            stacked_obs = obs_stacker.get_stacked()
            for k in stacked_obs:
                if isinstance(stacked_obs[k], np.ndarray):
                    stacked_obs[k] = torch.from_numpy(stacked_obs[k]).to(device)
            obs_features = encode_observations(stacked_obs)
            
            if total_steps < args.num_seed_steps:
                action_chunk = np.random.uniform(-1, 1, (args.num_envs, args.act_horizon, action_dim))
            else:
                action_chunk = agent.select_action(obs_features, deterministic=False).cpu().numpy()
        
        # Execute action chunk
        chunk_collector.reset()
        chunk_rewards = []
        chunk_dones = []
        chunk_success_per_env = np.zeros(args.num_envs, dtype=np.float32)
        
        first_obs_raw = obs_stacker.get_stacked()
        first_obs_raw_np = {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in first_obs_raw.items()
        }
        
        for step_idx in range(args.act_horizon):
            action = action_chunk[:, step_idx, :]
            next_obs, reward, terminated, truncated, info = train_envs.step(action)
            done = terminated | truncated
            
            reward_np = reward.cpu().numpy() if torch.is_tensor(reward) else reward
            done_np = done.cpu().numpy() if torch.is_tensor(done) else done
            
            obs_stacker.append(next_obs)
            chunk_collector.add(reward=reward_np, done=done_np.astype(np.float32))
            chunk_rewards.append(reward_np)
            chunk_dones.append(done_np)
            
            for env_idx in range(args.num_envs):
                episode_rewards[env_idx] += reward_np[env_idx]
                episode_lengths[env_idx] += 1
                
                # Track per-step success for success_once accumulation
                step_success = info.get("success", [False] * args.num_envs)[env_idx]
                if hasattr(step_success, "item"):
                    step_success = step_success.item()
                episode_has_succeeded[env_idx] = max(episode_has_succeeded[env_idx], float(step_success))
                
                if done_np[env_idx]:
                    episode_successes.append(float(step_success))
                    # Track success per env for this chunk (for success buffer)
                    if args.success_criteria == "success_once":
                        # Use accumulated success: True if agent succeeded at any point
                        chunk_success_per_env[env_idx] = max(
                            chunk_success_per_env[env_idx], episode_has_succeeded[env_idx]
                        )
                    else:
                        # Use instantaneous success at termination
                        chunk_success_per_env[env_idx] = max(
                            chunk_success_per_env[env_idx], float(step_success)
                        )
                    # Reset per-env success_once tracker on episode boundary
                    episode_has_succeeded[env_idx] = 0.0
                    episode_rewards[env_idx] = 0.0
                    episode_lengths[env_idx] = 0
            
            obs = next_obs
            total_steps += args.num_envs
            pbar.update(args.num_envs)
        
        # Compute SMDP rewards
        cumulative_reward, chunk_done, discount_factor, effective_length = chunk_collector.compute_smdp_rewards()
        
        next_obs_raw = obs_stacker.get_stacked()
        next_obs_raw_np = {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in next_obs_raw.items()
        }
        
        online_buffer.store(
            obs=first_obs_raw_np,
            action=action_chunk,
            reward=chunk_rewards[0],
            next_obs=next_obs_raw_np,
            done=np.any(np.stack(chunk_dones, axis=0), axis=0).astype(np.float32),
            cumulative_reward=cumulative_reward,
            chunk_done=chunk_done,
            discount_factor=discount_factor,
            effective_length=effective_length,
        )
        
        # Store successful transitions in success buffer (for actor_policy_mode="success_only")
        if success_buffer is not None:
            success_buffer.store(
                obs=first_obs_raw_np,
                action=action_chunk,
                reward=chunk_rewards[0],
                next_obs=next_obs_raw_np,
                done=np.any(np.stack(chunk_dones, axis=0), axis=0).astype(np.float32),
                cumulative_reward=cumulative_reward,
                chunk_done=chunk_done,
                discount_factor=discount_factor,
                effective_length=effective_length,
                success=chunk_success_per_env,
            )
        
        # Training updates
        if total_steps >= args.num_seed_steps and online_buffer.size >= args.batch_size:
            agent.train() if hasattr(agent, 'train') else None
            
            for _ in range(args.utd_ratio):
                batch = online_buffer.sample_mixed(
                    batch_size=args.batch_size,
                    offline_dataset=offline_dataset,
                    online_ratio=args.online_ratio if offline_dataset else 1.0,
                )
                
                # Log SMDP reward distribution for verification (periodically)
                if total_steps % (args.log_freq * 10) == 0 and offline_dataset is not None:
                    is_demo = batch.get("is_demo", None)
                    if is_demo is not None:
                        online_mask = ~is_demo
                        offline_mask = is_demo
                        
                        if online_mask.any():
                            online_cum_reward = batch["cumulative_reward"][online_mask]
                            training_metrics["smdp/online_cum_reward_mean"].append(online_cum_reward.mean().item())
                            training_metrics["smdp/online_cum_reward_std"].append(online_cum_reward.std().item())
                        
                        if offline_mask.any():
                            offline_cum_reward = batch["cumulative_reward"][offline_mask]
                            training_metrics["smdp/offline_cum_reward_mean"].append(offline_cum_reward.mean().item())
                            training_metrics["smdp/offline_cum_reward_std"].append(offline_cum_reward.std().item())
                        
                        # Log discount factor distribution
                        if online_mask.any():
                            training_metrics["smdp/online_discount_mean"].append(batch["discount_factor"][online_mask].mean().item())
                        if offline_mask.any():
                            training_metrics["smdp/offline_discount_mean"].append(batch["discount_factor"][offline_mask].mean().item())
                
                # Encode observations for critic update
                obs_features = encode_observations(batch["observations"])
                next_obs_features = encode_observations(batch["next_observations"])
                
                # Update critic
                critic_optimizer.zero_grad()
                critic_loss, critic_metrics = agent.compute_critic_loss(
                    obs_features=obs_features.detach(),  # Detach to avoid double backward
                    actions=batch["actions"],
                    next_obs_features=next_obs_features.detach(),
                    rewards=batch["reward"],
                    dones=batch["done"],
                    cumulative_reward=batch["cumulative_reward"],
                    chunk_done=batch["chunk_done"],
                    discount_factor=batch["discount_factor"],
                )
                critic_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                critic_optimizer.step()
                
                # Accumulate critic metrics
                training_metrics["critic_loss"].append(critic_loss.item())
                for k, v in critic_metrics.items():
                    training_metrics[f"critic/{k}"].append(v.item() if torch.is_tensor(v) else v)
                
                # Re-encode for actor update (need fresh computation graph)
                # In "success_only" mode, sample a separate batch for actor from
                # success_buffer + offline_dataset (no failed transitions).
                # In "all" mode, reuse the same batch as critic.
                use_success_batch = (
                    args.algorithm == "awsc"
                    and args.actor_policy_mode == "success_only"
                    and success_buffer is not None
                    and success_buffer.size > 0
                )
                
                if use_success_batch:
                    actor_batch = success_buffer.sample_mixed(
                        batch_size=args.batch_size,
                        offline_dataset=offline_dataset,
                        online_ratio=args.online_ratio if offline_dataset else 1.0,
                        policy_mode=True,
                        success_only=True,
                    )
                    obs_features_actor = encode_observations(actor_batch["observations"])
                else:
                    actor_batch = batch
                    obs_features_actor = encode_observations(batch["observations"])
                
                # Update actor - algorithm-specific
                actor_optimizer.zero_grad()
                if args.algorithm == "sac":
                    actor_loss, actor_metrics = agent.compute_actor_loss(obs_features_actor)
                elif args.algorithm == "awsc":
                    # AWSC needs actions and is_demo for advantage-weighted loss
                    actor_loss, actor_metrics = agent.compute_actor_loss(
                        obs_features_actor,
                        actor_batch["actions"],
                        actor_batch["actions"][:, :args.act_horizon],
                        actor_batch.get("is_demo"),  # May be None if not tracked
                    )
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                actor_optimizer.step()
                
                # Accumulate actor metrics
                training_metrics["actor_loss"].append(actor_loss.item())
                for k, v in actor_metrics.items():
                    training_metrics[f"actor/{k}"].append(v.item() if torch.is_tensor(v) else v)
                
                # Temperature update (SAC only)
                if args.algorithm == "sac":
                    # Re-encode for temperature update (need fresh computation graph)
                    obs_features_temp = encode_observations(batch["observations"])
                    
                    # Update temperature
                    temp_optimizer.zero_grad()
                    temp_loss, temp_metrics = agent.compute_temperature_loss(obs_features_temp.detach())
                    temp_loss.backward()
                    temp_optimizer.step()
                    
                    # Accumulate temperature metrics
                    training_metrics["temperature_loss"].append(temp_loss.item())
                    for k, v in temp_metrics.items():
                        training_metrics[f"temperature/{k}"].append(v.item() if torch.is_tensor(v) else v)
                
                # Soft update target network
                agent.update_target()
                
                # AWSC-specific: update EMA velocity network
                if args.algorithm == "awsc":
                    agent.update_ema()
        
        # Logging
        if total_steps % args.log_freq == 0 and len(episode_successes) > 0:
            # Log episode statistics
            writer.add_scalar("train/success_rate", np.mean(episode_successes[-100:]), total_steps)
            writer.add_scalar("train/episode_count", len(episode_successes), total_steps)
            
            log_dict = {
                "train/success_rate": np.mean(episode_successes[-100:]),
                "train/episode_count": len(episode_successes),
            }
            
            # Log training losses and metrics
            if training_metrics:
                for metric_name, values in training_metrics.items():
                    if values:
                        # Skip string metrics (like advantage_mode)
                        if isinstance(values[0], str):
                            continue
                        avg_value = np.mean(values)
                        writer.add_scalar(f"train/{metric_name}", avg_value, total_steps)
                        log_dict[f"train/{metric_name}"] = avg_value
                
                # Clear metrics after logging
                training_metrics.clear()
            
            # Log success buffer statistics
            if success_buffer is not None:
                sb_stats = success_buffer.get_statistics()
                for k, v in sb_stats.items():
                    log_dict[f"train/success_buffer/{k}"] = v
            
            if args.track:
                wandb.log(log_dict, step=total_steps)
        
        # Evaluation
        if total_steps - last_eval_step >= args.eval_freq:
            last_eval_step = total_steps
            agent.eval() if hasattr(agent, 'eval') else None
            
            eval_metrics = evaluate(
                args.num_eval_episodes,
                agent_wrapper,
                eval_envs,
                device,
                args.sim_backend,
            )
            
            print(f"\n[Step {total_steps}] Evaluation:")
            for k in eval_metrics:
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], total_steps)
                print(f"  {k}: {eval_metrics[k]:.4f}")
            
            # WandB logging for evaluation
            if args.track:
                wandb_eval = {f"eval/{k}": v for k, v in eval_metrics.items()}
                wandb.log(wandb_eval, step=total_steps)
            
            if eval_metrics.get("success_once", 0) > best_success_rate:
                best_success_rate = eval_metrics["success_once"]
                from rlft.utils.checkpoint import save_checkpoint as _save_ckpt
                _save_ckpt(
                    path=f"{log_dir}/checkpoints/best.pt",
                    agent=agent,
                    visual_encoder=visual_encoder,
                    action_normalizer=action_normalizer,
                    args=args,
                    save_args_json=False,
                )
                print(f"  New best! Saved checkpoint.")
        
        # Save checkpoint
        if total_steps % args.save_freq == 0:
            from rlft.utils.checkpoint import save_checkpoint as _save_ckpt
            _save_ckpt(
                path=f"{log_dir}/checkpoints/step_{total_steps}.pt",
                agent=agent,
                visual_encoder=visual_encoder,
                action_normalizer=action_normalizer,
                args=args,
                save_args_json=False,
                extra={
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                },
                total_steps=total_steps,
            )
        
        pbar.set_postfix({
            "success": f"{np.mean(episode_successes[-100:]) if episode_successes else 0:.2%}",
            "steps": total_steps,
        })
    
    pbar.close()
    
    # Final save
    from rlft.utils.checkpoint import save_checkpoint as _save_ckpt
    _save_ckpt(
        path=f"{log_dir}/checkpoints/final.pt",
        agent=agent,
        visual_encoder=visual_encoder,
        action_normalizer=action_normalizer,
        args=args,
        save_args_json=False,
    )
    
    train_envs.close()
    eval_envs.close()
    writer.close()
    
    if args.track:
        # Log final best metrics
        wandb.log({
            "final/best_success_rate": best_success_rate,
        })
        wandb.finish()
    
    print(f"\nTraining complete! Best success rate: {best_success_rate:.2%}")


if __name__ == "__main__":
    main()
