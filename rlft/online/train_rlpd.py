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
from rlft.datasets.data_utils import ObservationStacker


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
    capture_video: bool = True
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
    online_ratio: float = 0.5
    
    # Training settings
    total_timesteps: int = 1_000_000
    num_seed_steps: int = 5000
    utd_ratio: int = 20
    batch_size: int = 256
    eval_freq: int = 10000
    save_freq: int = 50000
    log_freq: int = 100
    num_eval_episodes: int = 50
    
    # Optimizer settings
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
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
    act_horizon: int = 2
    pred_horizon: int = 4
    
    # Visual encoder
    visual_feature_dim: int = 256
    freeze_visual_encoder: bool = False
    
    # Replay buffer
    replay_buffer_capacity: int = 500_000
    
    # AWSC-specific settings (only used when algorithm="awsc")
    pretrain_path: Optional[str] = None
    """Path to pretrained ShortCut Flow or AW-ShortCut checkpoint for AWSC"""
    load_pretrain_critic: bool = False
    """Whether to load critic from pretrain checkpoint"""
    
    # AWSC hyperparameters (unified with old version)
    awsc_beta: float = 100.0
    """Temperature for advantage weighting in AWSC"""
    awsc_bc_weight: float = 1.0
    """Weight for flow matching loss"""
    awsc_shortcut_weight: float = 0.3
    """Weight for shortcut consistency loss"""
    awsc_self_consistency_k: float = 0.25
    """Fraction of batch for consistency loss (match IL/offline_rl)"""
    awsc_ema_decay: float = 0.999
    """EMA decay rate for velocity network"""
    awsc_weight_clip: float = 100.0
    """Maximum advantage weight to prevent outliers"""
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
        
        Args:
            obs: Observation from eval_envs (already FrameStacked)
                 Shape: (B, T, state_dim) or dict with 'state' key
            deterministic: Whether to use deterministic action
        
        Returns:
            actions: (B, pred_horizon, act_dim)
        """
        obs_cond = self.encode_obs(obs)
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
            device=device,
        ).to(device)
        use_temperature = False
        
        # Load pretrained checkpoint if provided
        if args.pretrain_path:
            print(f"Loading pretrained checkpoint from {args.pretrain_path}...")
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
    success_buffer = None
    if args.use_success_buffer:
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
    
    # Create action normalizer if needed (for offline data)
    action_normalizer = ActionNormalizer(mode=args.action_norm_mode) if args.normalize_actions else None
    
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
        B = stacked_obs["state"].shape[0]
        T = stacked_obs["state"].shape[1]
        features = []
        
        if include_rgb and visual_encoder is not None:
            rgb = stacked_obs["rgb"]
            if isinstance(rgb, np.ndarray):
                rgb = torch.from_numpy(rgb).to(device)
            rgb = rgb.contiguous()  # Ensure contiguous memory layout
            rgb_flat = rgb.reshape(B * T, *rgb.shape[2:]).float() / 255.0
            if rgb_flat.ndim == 4 and rgb_flat.shape[-1] == 3:
                rgb_flat = rgb_flat.permute(0, 3, 1, 2).contiguous()
            visual_feat = visual_encoder(rgb_flat)
            visual_feat = visual_feat.view(B, T, -1)
            features.append(visual_feat)
        
        state = stacked_obs["state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(device)
        features.append(state)
        
        return torch.cat(features, dim=-1).reshape(B, -1)
    
    total_steps = 0
    last_eval_step = -args.eval_freq  # Track last eval to handle non-exact intervals
    episode_rewards = defaultdict(float)
    episode_lengths = defaultdict(int)
    episode_successes = []
    best_success_rate = 0.0
    
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
                
                if done_np[env_idx]:
                    success = info.get("success", [False] * args.num_envs)[env_idx]
                    if hasattr(success, "item"):
                        success = success.item()
                    episode_successes.append(float(success))
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
                obs_features_actor = encode_observations(batch["observations"])
                
                # Update actor - algorithm-specific
                actor_optimizer.zero_grad()
                if args.algorithm == "sac":
                    actor_loss, actor_metrics = agent.compute_actor_loss(obs_features_actor)
                elif args.algorithm == "awsc":
                    # AWSC needs actions and is_demo for advantage-weighted loss
                    actor_loss, actor_metrics = agent.compute_actor_loss(
                        obs_features_actor,
                        batch["actions"],
                        batch["actions"][:, :args.act_horizon],
                        batch.get("is_demo"),  # May be None if not tracked
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
                        avg_value = np.mean(values)
                        writer.add_scalar(f"train/{metric_name}", avg_value, total_steps)
                        log_dict[f"train/{metric_name}"] = avg_value
                
                # Clear metrics after logging
                training_metrics.clear()
            
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
                checkpoint = {
                    "agent": agent.state_dict(),
                    "visual_encoder": visual_encoder.state_dict() if visual_encoder else None,
                    "config": vars(args),
                }
                if action_normalizer is not None and action_normalizer.stats is not None:
                    checkpoint["action_normalizer"] = {
                        "mode": action_normalizer.mode,
                        "stats": {k: v.tolist() for k, v in action_normalizer.stats.items()},
                    }
                torch.save(checkpoint, f"{log_dir}/checkpoints/best.pt")
                print(f"  New best! Saved checkpoint.")
        
        # Save checkpoint
        if total_steps % args.save_freq == 0:
            checkpoint = {
                "agent": agent.state_dict(),
                "visual_encoder": visual_encoder.state_dict() if visual_encoder else None,
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "total_steps": total_steps,
                "config": vars(args),
            }
            if action_normalizer is not None and action_normalizer.stats is not None:
                checkpoint["action_normalizer"] = {
                    "mode": action_normalizer.mode,
                    "stats": {k: v.tolist() for k, v in action_normalizer.stats.items()},
                }
            torch.save(checkpoint, f"{log_dir}/checkpoints/step_{total_steps}.pt")
        
        pbar.set_postfix({
            "success": f"{np.mean(episode_successes[-100:]) if episode_successes else 0:.2%}",
            "steps": total_steps,
        })
    
    pbar.close()
    
    # Final save
    checkpoint = {
        "agent": agent.state_dict(),
        "visual_encoder": visual_encoder.state_dict() if visual_encoder else None,
        "config": vars(args),
    }
    if action_normalizer is not None and action_normalizer.stats is not None:
        checkpoint["action_normalizer"] = {
            "mode": action_normalizer.mode,
            "stats": {k: v.tolist() for k, v in action_normalizer.stats.items()},
        }
    torch.save(checkpoint, f"{log_dir}/checkpoints/final.pt")
    
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
