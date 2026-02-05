"""
Unified Offline RL Training Script

Supports training multiple algorithms on ManiSkill environments:
- Imitation Learning: diffusion_policy, flow_matching, consistency_flow, shortcut_flow
- Offline RL: cpql, awcp, aw_shortcut_flow

Usage:
    python -m rlft.offline.train_maniskill --env_id PickCube-v1 --algorithm flow_matching
    python -m rlft.offline.train_maniskill --env_id PickCube-v1 --algorithm aw_shortcut_flow
"""

ALGO_NAME = "OfflineRL_UNet"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

# Import from rlft package
from rlft.envs import make_eval_envs, evaluate
from rlft.networks import (
    PlainConv, ConditionalUnet1D, VelocityUNet1D, ShortCutVelocityUNet1D,
    DoubleQNetwork, EnsembleQNetwork,
)
from rlft.algorithms import (
    DiffusionPolicyAgent, FlowMatchingAgent, ShortCutFlowAgent,
    ConsistencyFlowAgent, ReflectedFlowAgent,
    CPQLAgent, AWCPAgent, AWShortCutFlowAgent,
)
from rlft.datasets import OfflineRLDataset, IterationBasedBatchSampler, worker_init_fn, ActionNormalizer
from rlft.datasets.data_utils import build_state_obs_extractor, create_obs_process_fn


@dataclass
class Args:
    """Training arguments."""
    # Experiment settings
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = True

    # Environment settings
    env_id: str = "LiftPegUpright-v1"
    demo_path: str = "~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"
    num_demos: Optional[int] = None
    max_episode_steps: Optional[int] = None
    control_mode: str = "pd_ee_delta_pose"
    obs_mode: str = "rgb"
    sim_backend: str = "physx_cuda"

    # Training settings
    total_iters: int = 1_000_000
    batch_size: int = 256
    lr: float = 3e-4  # Best from sweep (works well for most algorithms)
    lr_critic: float = 3e-4

    # Policy architecture settings
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 8  # Best from sweep (8 > 16)
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    visual_feature_dim: int = 256

    # Algorithm selection
    algorithm: Literal[
        "diffusion_policy", "flow_matching", "consistency_flow", "reflected_flow", "shortcut_flow",
        "cpql", "awcp", "aw_shortcut_flow",
    ] = "flow_matching"
    
    # Diffusion/Flow settings
    num_diffusion_iters: int = 100  # Diffusion policy iterations
    num_flow_steps: int = 20  # Best from sweep (20 > 10 > 5)
    ema_decay: float = 0.999
    
    # Reflected Flow settings
    reflection_mode: Literal["hard", "soft"] = "soft"  # Best from sweep
    boundary_reg_weight: float = 0.01
    
    # Consistency Flow settings
    cons_delta: float = 0.1
    cons_use_flow_t: bool = False
    """reuse flow t for consistency branch instead of resampling"""
    cons_full_t_range: bool = False
    """sample consistency t in [0,1] instead of clipped range"""
    cons_t_min: float = 0.05
    """minimum t for consistency sampling when not using full range"""
    cons_t_max: float = 0.95
    """maximum t for consistency sampling when not using full range"""
    cons_t_upper: float = 0.95
    """upper clamp for t_plus"""
    cons_delta_mode: Literal["random", "fixed"] = "fixed"
    """delta sampling strategy for consistency (fixed works best from sweep)"""
    cons_delta_min: float = 0.02
    """minimum delta when using random delta"""
    cons_delta_max: float = 0.15
    """maximum delta when using random delta"""
    cons_delta_fixed: float = 0.04
    """fixed delta when cons_delta_mode=fixed (best from sweep)"""
    cons_delta_dynamic_max: bool = False
    """cap random delta by remaining time"""
    cons_delta_cap: float = 0.99
    """ceiling used when cons_delta_dynamic_max is enabled"""
    cons_teacher_steps: int = 2
    """teacher rollout steps to t=1"""
    cons_teacher_from: Literal["t_plus", "t_cons"] = "t_plus"
    """where teacher rollout starts"""
    cons_student_point: Literal["t_plus", "t_cons"] = "t_plus"
    """student evaluation point for consistency loss"""
    cons_loss_space: Literal["velocity", "endpoint"] = "velocity"
    """consistency loss space: velocity or endpoint"""

    # ShortCut Flow settings
    sc_fixed_step_size: float = 0.125
    sc_num_inference_steps: int = 8
    """number of inference steps (best from sweep)"""
    sc_max_denoising_steps: int = 8
    """maximum denoising steps for step size sampling"""
    sc_self_consistency_k: float = 0.25
    """fraction of batch for consistency (best from sweep)"""
    sc_t_min: float = 0.0
    """minimum t for time sampling"""
    sc_t_max: float = 1.0
    """maximum t for time sampling"""
    sc_t_sampling_mode: Literal["uniform", "truncated"] = "uniform"
    """time sampling mode"""
    sc_step_size_mode: Literal["power2", "uniform", "fixed"] = "fixed"
    """step size sampling mode (fixed works best from sweep)"""
    sc_min_step_size: float = 0.0625
    """minimum step size"""
    sc_max_step_size: float = 0.5
    """maximum step size"""
    sc_target_mode: Literal["velocity", "endpoint"] = "velocity"
    """shortcut target mode"""
    sc_teacher_steps: int = 1
    """teacher rollout steps for shortcut target"""
    sc_use_ema_teacher: bool = True
    """whether to use EMA network as teacher"""
    sc_inference_mode: Literal["adaptive", "uniform"] = "uniform"
    """inference mode"""

    # Offline RL settings
    bc_weight: float = 0.5  # Best from sweep for cpql
    consistency_weight: float = 0.3
    """consistency regularization weight (best from sweep)"""
    alpha: float = 0.001
    """CPQL entropy coefficient (best from sweep: 0.001)"""
    beta: float = 10.0
    """AWR temperature (best from sweep: 10.0 for aggressive config)"""
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 0.1
    """reward scaling factor (best from sweep)"""
    q_target_clip: float = 100.0
    weight_clip: float = 100.0
    
    # Ensemble Q settings
    use_ensemble_q: bool = True
    num_qs: int = 10
    num_min_qs: int = 2
    q_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    
    # Action normalization settings
    normalize_actions: bool = False
    """Whether to normalize actions during training"""
    action_norm_mode: Literal["standard", "minmax"] = "standard"
    """Action normalization mode: 'standard' (zero mean, unit var) or 'minmax' (scale to [-1, 1])"""
    action_bounds: Optional[tuple] = (-1.0, 1.0)
    """Action bounds for clamping during inference. Set to None to disable clamping.
    ManiSkill environments have action space [-1, 1], so we clamp by default."""

    # Logging settings
    log_freq: int = 1000
    eval_freq: int = 2500
    save_freq: Optional[int] = None
    num_eval_episodes: int = 100
    num_eval_envs: int = 25
    num_dataload_workers: int = 0


class AgentWrapper(nn.Module):
    """Wrapper for agent with visual encoder for evaluation."""
    
    def __init__(self, agent, visual_encoder, include_rgb, obs_horizon, act_horizon=None, 
                 action_normalizer=None, include_depth=False):
        super().__init__()
        self.agent = agent
        self.visual_encoder = visual_encoder
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon if act_horizon else agent.act_horizon if hasattr(agent, 'act_horizon') else 8
        self.action_normalizer = action_normalizer

    def get_action(self, obs, deterministic=False, **kwargs):
        """Get action from observation.
        
        Handles both RGB/RGBD mode (Dict obs with 'state', 'rgb', 'depth' keys) and
        State mode (direct tensor observation).
        """
        # Handle different observation formats
        # RGB mode: obs is Dict with 'state' and 'rgb' keys
        # State mode: obs is directly a tensor
        if self.include_rgb:
            # RGB mode: Dict observation
            state = obs["state"]
            B = state.shape[0]
            T = self.obs_horizon
            
            features_list = []
            
            if self.visual_encoder is not None:
                rgb = obs["rgb"]
                # Convert NHWC to NCHW if needed
                if rgb.dim() == 5 and rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    rgb = rgb.permute(0, 1, 4, 2, 3)
                rgb_flat = rgb.reshape(B * T, *rgb.shape[2:]).float()
                # Always normalize RGB to [0, 1]
                rgb_flat = rgb_flat / 255.0
                
                # Handle depth if available
                if self.include_depth and "depth" in obs:
                    depth = obs["depth"]
                    if depth.dim() == 5 and depth.shape[-1] in [1, 2, 4]:
                        depth = depth.permute(0, 1, 4, 2, 3)
                    depth_flat = depth.reshape(B * T, *depth.shape[2:]).float()
                    # Normalize depth (ManiSkill uses mm, typical range 0-10000)
                    depth_flat = depth_flat / 1024.0
                    visual_input = torch.cat([rgb_flat, depth_flat], dim=1)
                else:
                    visual_input = rgb_flat
                
                visual_feat = self.visual_encoder(visual_input)
                visual_feat = visual_feat.view(B, T, -1)
                features_list.append(visual_feat)
            
            features_list.append(state.float())
            obs_features = torch.cat(features_list, dim=-1)
        else:
            # State mode: direct tensor observation
            # obs shape: (B, obs_horizon, state_dim) from FrameStack
            state = obs
            B = state.shape[0]
            obs_features = state.float()
        
        obs_cond = obs_features.reshape(B, -1)
        
        # Note: not all agents support 'deterministic' kwarg, so we don't pass it
        # The wrapper handles determinism at the wrapper level if needed
        actions = self.agent.get_action(obs_cond, **kwargs)
        
        # Slice action sequence with temporal alignment
        # Use obs_horizon-1 as start offset to align with training data
        # This matches the original diffusion_policy implementation
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        action_seq = actions[:, start:end]
        
        # Denormalize actions if normalizer is provided
        if self.action_normalizer is not None:
            actions_np = action_seq.cpu().numpy()
            actions_denorm = self.action_normalizer.inverse_transform(actions_np)
            return torch.from_numpy(actions_denorm).float().to(actions.device)
        
        return action_seq
    
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


def create_agent(algorithm: str, action_dim: int, global_cond_dim: int, args):
    """Create agent based on algorithm name."""
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
            action_bounds=args.action_bounds,
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
            action_bounds=args.action_bounds,
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
            max_denoising_steps=args.sc_max_denoising_steps,
            self_consistency_k=args.sc_self_consistency_k,
            flow_weight=args.bc_weight,
            shortcut_weight=args.consistency_weight,
            ema_decay=args.ema_decay,
            # Time sampling
            t_min=args.sc_t_min,
            t_max=args.sc_t_max,
            t_sampling_mode=args.sc_t_sampling_mode,
            # Step size
            step_size_mode=args.sc_step_size_mode,
            min_step_size=args.sc_min_step_size,
            max_step_size=args.sc_max_step_size,
            fixed_step_size=args.sc_fixed_step_size,
            # Target computation
            target_mode=args.sc_target_mode,
            teacher_steps=args.sc_teacher_steps,
            use_ema_teacher=args.sc_use_ema_teacher,
            # Inference
            inference_mode=args.sc_inference_mode,
            num_inference_steps=args.sc_num_inference_steps,
            action_bounds=args.action_bounds,
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
            consistency_delta=args.cons_delta,
            # Consistency design toggles
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
            action_bounds=args.action_bounds,
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
    
    elif algorithm == "cpql":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
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
            act_horizon=args.act_horizon,
            num_flow_steps=args.num_flow_steps,
            alpha=args.alpha,
            bc_weight=args.bc_weight,
            gamma=args.gamma,
            tau=args.tau,
            reward_scale=args.reward_scale,
            q_target_clip=args.q_target_clip,
            ema_decay=args.ema_decay,
            action_bounds=args.action_bounds,
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
            beta=args.beta,
            bc_weight=args.bc_weight,
            gamma=args.gamma,
            tau=args.tau,
            reward_scale=args.reward_scale,
            q_target_clip=args.q_target_clip,
            ema_decay=args.ema_decay,
            weight_clip=args.weight_clip,
            action_bounds=args.action_bounds,
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
            beta=args.beta,
            bc_weight=args.bc_weight,
            shortcut_weight=args.consistency_weight,
            gamma=args.gamma,
            tau=args.tau,
            reward_scale=args.reward_scale,
            q_target_clip=args.q_target_clip,
            ema_decay=args.ema_decay,
            weight_clip=args.weight_clip,
            fixed_step_size=args.sc_fixed_step_size,
            num_inference_steps=args.sc_num_inference_steps,
            action_bounds=args.action_bounds,
            device=device,
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def save_ckpt(run_name, tag, agent, ema_agent, visual_encoder, action_normalizer=None):
    """Save checkpoint."""
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    checkpoint = {
        "agent": agent.state_dict(),
        "ema_agent": ema_agent.state_dict() if ema_agent else None,
    }
    if visual_encoder is not None:
        checkpoint["visual_encoder"] = visual_encoder.state_dict()
    if action_normalizer is not None and action_normalizer.stats is not None:
        checkpoint["action_normalizer"] = {
            "mode": action_normalizer.mode,
            "stats": {k: v.tolist() for k, v in action_normalizer.stats.items()},
        }
    torch.save(checkpoint, f"runs/{run_name}/checkpoints/{tag}.pt")


def main():
    args = tyro.cli(Args)
    
    # Generate experiment name
    if args.exp_name is None:
        args.exp_name = f"{args.algorithm}-{args.env_id}-seed{args.seed}"
    
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # Set up logging
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # =========================================================================
    # Environment Setup and Dimension Calculation
    # =========================================================================
    # ManiSkill observations and dimensions are complex. Here's the pipeline:
    #
    # 1. Raw ManiSkill Environment Output (obs_mode="rgbd"):
    #    obs = {
    #        "agent": {"qpos": (T, N_joints), "qvel": (T, N_joints)},
    #        "extra": {"tcp_pose": (T, 7), ...task-specific...},
    #        "sensor_data": {"camera_name": {"rgb": (T, H, W, 3), "depth": ...}},
    #        "sensor_param": {...camera intrinsics...}
    #    }
    #
    # 2. After FlattenRGBDObservationWrapper:
    #    obs = {
    #        "state": (num_envs, state_dim),  # Flattened agent + extra
    #        "rgb": (num_envs, H, W, C),      # Concatenated camera images
    #        "depth": (num_envs, H, W, 1),    # Optional depth
    #    }
    #
    # 3. After FrameStack(num_stack=obs_horizon):
    #    obs = {
    #        "state": (num_envs, obs_horizon, state_dim),  # Stacked over time
    #        "rgb": (num_envs, obs_horizon, H, W, C),
    #    }
    #    NOTE: observation_space["state"].shape = (obs_horizon, state_dim)
    #          We need state_dim = shape[-1], NOT shape[0]!
    #
    # 4. For Policy Network Input:
    #    - Visual: rgb -> visual_encoder -> (B, obs_horizon, visual_feature_dim)
    #    - State: (B, obs_horizon, state_dim)
    #    - Combined: concat then flatten -> (B, obs_horizon * (visual_dim + state_dim))
    #    This is the global_cond_dim for the U-Net
    # =========================================================================
    
    # Create evaluation environments
    env_kwargs = dict(
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,  # "rgbd" includes rgb and depth
        render_mode="rgb_array",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    # Environment wrapper pipeline:
    # gym.make() -> [FlattenRGBDObservationWrapper if RGB] -> FrameStack -> ManiSkillVectorEnv
    # NOTE: FlattenRGBDObservationWrapper is ONLY for RGB mode. It flattens nested obs to {state, rgb, depth}
    # In state mode, the observation is already flat, so we don't need this wrapper.
    include_rgb = "rgb" in args.obs_mode
    include_depth = "depth" in args.obs_mode
    wrappers = [FlattenRGBDObservationWrapper] if include_rgb else []
    
    envs = make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        sim_backend=args.sim_backend,
        env_kwargs=env_kwargs,
        other_kwargs=other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=wrappers,
    )
    
    # =========================================================================
    # Dimension Extraction
    # =========================================================================
    # After wrappers, single_observation_space["state"] has shape (obs_horizon, state_dim)
    # due to FrameStack. We need the FEATURE dimension (last axis), not time dimension!
    # 
    # Example for LiftPegUpright-v1:
    #   - Before FrameStack: state shape = (1, 25)  [num_envs, state_dim]
    #   - After FrameStack(2): state shape = (2, 25) [obs_horizon, state_dim]
    #   - state_dim = shape[-1] = 25, NOT shape[0] = 2
    # =========================================================================
    
    act_dim = envs.single_action_space.shape[0]
    
    # Build state extractor (used for processing demo data, not env obs)
    # This extracts agent + extra values from raw ManiSkill obs format
    state_obs_extractor = build_state_obs_extractor(args.env_id)
    
    # Get state dimension from env observation space
    # CRITICAL: Use shape[-1] to get feature dim, not shape[0] (which is obs_horizon)
    sample_obs = envs.single_observation_space
    
    # In RGB mode with FlattenRGBDObservationWrapper: observation_space is a Dict with "state" key
    # In state mode without wrapper: observation_space is a simple Box
    if include_rgb:
        # Dict observation space: {"state": Box, "rgb": Box, ...}
        state_dim = sample_obs["state"].shape[-1]
    else:
        # Simple Box observation space
        state_dim = sample_obs.shape[-1]
    
    # =========================================================================
    # Visual Encoder and Observation Dimension
    # =========================================================================
    # obs_dim is the flattened dimension fed to the policy network (U-Net global_cond)
    # 
    # With RGB:
    #   - Each frame: visual_encoder(rgb) -> visual_feature_dim
    #   - Each frame: state -> state_dim  
    #   - Per frame total: visual_feature_dim + state_dim
    #   - After stacking obs_horizon frames: obs_dim = (visual_dim + state_dim) * obs_horizon
    #
    # Without RGB (state only):
    #   - obs_dim = state_dim * obs_horizon
    # =========================================================================
    
    if include_rgb:
        # Compute visual input channels dynamically
        # RGB: 3 channels per camera, Depth: 1 channel per camera
        # FlattenRGBDObservationWrapper stacks cameras, so check observation space
        if include_depth:
            # RGBD mode: rgb channels + depth channels
            total_visual_channels = sample_obs["rgb"].shape[-1] + sample_obs["depth"].shape[-1]
        else:
            # RGB only mode
            total_visual_channels = sample_obs["rgb"].shape[-1]
        
        print(f"Visual encoder input channels: {total_visual_channels}")
        visual_encoder = PlainConv(
            in_channels=total_visual_channels,
            out_dim=args.visual_feature_dim,  # Default: 256
            pool_feature_map=True,
        ).to(device)
        # obs_dim = (256 + 25) * 2 = 562 for LiftPegUpright with obs_horizon=2
        obs_dim = (args.visual_feature_dim + state_dim) * args.obs_horizon
    else:
        visual_encoder = None
        obs_dim = state_dim * args.obs_horizon
    
    print(f"Action dim: {act_dim}, Obs dim: {obs_dim}")
    
    # Create agent
    agent = create_agent(args.algorithm, act_dim, obs_dim, args).to(device)
    ema_agent = create_agent(args.algorithm, act_dim, obs_dim, args).to(device)
    
    # Create action normalizer if needed
    action_normalizer = ActionNormalizer(mode=args.action_norm_mode) if args.normalize_actions else None
    
    # Create dataset and dataloader
    obs_process_fn = create_obs_process_fn(args.env_id, output_format="NCHW")
    
    il_algorithms = ["diffusion_policy", "flow_matching", "shortcut_flow", "consistency_flow", "reflected_flow"]
    
    if args.algorithm in il_algorithms:
        from rlft.datasets import ManiSkillDataset
        dataset = ManiSkillDataset(
            data_path=args.demo_path,
            include_rgb=include_rgb,
            device=device,
            num_traj=args.num_demos,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            control_mode=args.control_mode,
            env_id=args.env_id,
            obs_process_fn=obs_process_fn,
            action_normalizer=action_normalizer,
            include_depth=include_depth,
        )
    else:
        dataset = OfflineRLDataset(
            data_path=args.demo_path,
            include_rgb=include_rgb,
            device=device,
            num_traj=args.num_demos,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            act_horizon=args.act_horizon,
            control_mode=args.control_mode,
            env_id=args.env_id,
            obs_process_fn=obs_process_fn,
            gamma=args.gamma,
            action_normalizer=action_normalizer,
            include_depth=include_depth,
        )
    
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id),
    )
    
    # Set up optimizer
    if visual_encoder is not None:
        params = list(agent.parameters()) + list(visual_encoder.parameters())
    else:
        params = list(agent.parameters())
    
    optimizer = optim.AdamW(params, lr=args.lr)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )
    
    # EMA
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    
    # Agent wrapper for evaluation
    agent_wrapper = AgentWrapper(
        agent, visual_encoder, include_rgb, args.obs_horizon, args.act_horizon, 
        action_normalizer, include_depth=include_depth
    ).to(device)
    ema_agent_wrapper = AgentWrapper(
        ema_agent, visual_encoder, include_rgb, args.obs_horizon, args.act_horizon, 
        action_normalizer, include_depth=include_depth
    ).to(device)
    
    best_eval_metrics = defaultdict(float)
    
    def encode_observations(obs_seq):
        """Encode observations to get obs_features.
        
        Handles RGB-only and RGBD modes:
        - RGB: rgb / 255.0
        - RGBD: concat(rgb / 255.0, depth / 1024.0)
        """
        B = obs_seq["state"].shape[0]
        T = obs_seq["state"].shape[1]
        
        features_list = []
        
        if visual_encoder is not None and "rgb" in obs_seq:
            rgb = obs_seq["rgb"]
            rgb_flat = rgb.view(B * T, *rgb.shape[2:]).float() / 255.0
            
            # If depth is available, concatenate it
            if include_depth and "depth" in obs_seq:
                depth = obs_seq["depth"]
                depth_flat = depth.view(B * T, *depth.shape[2:]).float() / 1024.0
                visual_input = torch.cat([rgb_flat, depth_flat], dim=1)  # Concat along channel dim
            else:
                visual_input = rgb_flat
            
            visual_feat = visual_encoder(visual_input)
            visual_feat = visual_feat.view(B, T, -1)
            features_list.append(visual_feat)
        
        state = obs_seq["state"]
        features_list.append(state)
        
        obs_features = torch.cat(features_list, dim=-1)
        return obs_features
    
    # Training loop
    agent.train()
    if visual_encoder is not None:
        visual_encoder.train()
    
    pbar = tqdm(total=args.total_iters)
    
    for iteration, data_batch in enumerate(train_dataloader):
        obs_seq = data_batch["observations"]
        action_seq = data_batch["actions"]
        obs_features = encode_observations(obs_seq)
        
        if args.algorithm in il_algorithms:
            loss_dict = agent.compute_loss(obs_features=obs_features, actions=action_seq)
            total_loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            if visual_encoder is not None:
                torch.nn.utils.clip_grad_norm_(visual_encoder.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            if hasattr(agent, "update_ema"):
                agent.update_ema()
        else:
            # Offline RL
            next_obs_seq = data_batch["next_observations"]
            actions_for_q = data_batch["actions_for_q"]
            rewards = data_batch["rewards"]
            dones = data_batch["dones"]
            cumulative_reward = data_batch["cumulative_reward"]
            chunk_done = data_batch["chunk_done"]
            discount_factor = data_batch["discount_factor"]
            
            next_obs_features = encode_observations(next_obs_seq)
            
            loss_dict = agent.compute_loss(
                obs_features=obs_features,
                actions=action_seq,
                rewards=rewards,
                next_obs_features=next_obs_features,
                dones=dones,
                actions_for_q=actions_for_q,
                cumulative_reward=cumulative_reward,
                chunk_done=chunk_done,
                discount_factor=discount_factor,
            )
            
            total_loss = loss_dict["loss"]
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            if visual_encoder is not None:
                torch.nn.utils.clip_grad_norm_(visual_encoder.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            agent.update_target()
            if hasattr(agent, "update_ema"):
                agent.update_ema()
        
        # EMA update
        ema.step(agent.parameters())
        
        # Logging
        if iteration % args.log_freq == 0:
            losses = {k: v.item() if isinstance(v, torch.Tensor) else v 
                     for k, v in (loss_dict.items() if isinstance(loss_dict, dict) else {"loss": loss_dict})}
            for k, v in losses.items():
                writer.add_scalar(f"losses/{k}", v, iteration)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            
            # WandB logging
            if args.track:
                wandb_log = {f"losses/{k}": v for k, v in losses.items()}
                wandb_log["charts/learning_rate"] = optimizer.param_groups[0]["lr"]
                wandb_log["charts/iteration"] = iteration
                wandb.log(wandb_log, step=iteration)
        
        # Evaluation
        if iteration % args.eval_freq == 0 and iteration > 0:
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(
                args.num_eval_episodes, ema_agent_wrapper, envs, device, args.sim_backend
            )
            
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
            
            # WandB logging for evaluation
            if args.track:
                wandb_eval = {f"eval/{k}": v for k, v in eval_metrics.items()}
                wandb.log(wandb_eval, step=iteration)
            
            for k in ["success_once", "success_at_end"]:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}", agent, ema_agent, visual_encoder, action_normalizer)
        
        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            ema.copy_to(ema_agent.parameters())
            save_ckpt(run_name, str(iteration), agent, ema_agent, visual_encoder, action_normalizer)
        
        pbar.update(1)
        if isinstance(loss_dict, dict):
            pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
    
    envs.close()
    writer.close()
    
    # Close WandB
    if args.track:
        # Log final best metrics
        wandb.log({
            "final/best_success_once": best_eval_metrics.get("success_once", 0.0),
            "final/best_success_at_end": best_eval_metrics.get("success_at_end", 0.0),
        })
        wandb.finish()
    
    # Print completion message for monitoring scripts
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Run: {run_name}")
    print(f"Total iterations: {args.total_iters}")
    print("=" * 50)


if __name__ == "__main__":
    main()
