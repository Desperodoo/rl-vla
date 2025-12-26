"""
RLPD Online Training Script

Online reinforcement learning with offline data mixing (RLPD style).
Supports two algorithms:
- sac: Soft Actor-Critic with Ensemble Q and action chunking
- awsc: Advantage-Weighted ShortCut Flow with Ensemble Q

Key features:
- Online environment interaction with action chunking (SMDP)
- Offline demonstration buffer mixing (50% online + 50% offline)
- Ensemble Q-networks with subsample + min for sample efficiency
- RGB observation support via PlainConv visual encoder
- Pretrained checkpoint loading for AWSC

Usage:
    # SAC from scratch
    python train_rlpd_online.py --algorithm sac --env_id PickCube-v1
    
    # AWSC with pretrained ShortCut Flow
    python train_rlpd_online.py --algorithm awsc --env_id PickCube-v1 \
        --pretrained_path runs/shortcut_flow-PickCube-v1-seed0/checkpoints/best.pt

Based on ManiSkill RLPD implementation and diffusion_policy framework.
"""

ALGO_NAME = "RLPD"

import os
import random
import time
import warnings
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Any

# Suppress warnings
os.environ["SAPIEN_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore", message=".*PhysX CPU system.*")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter
import h5py

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.evaluate import evaluate
from diffusion_policy.utils import (
    ObservationStacker, encode_observations,
    AgentWrapper, load_traj_hdf5, convert_obs,
)
from train_offline_rl import OfflineRLDataset

from diffusion_policy.rlpd import (
    SACAgent,
    AWSCAgent,
    OnlineReplayBufferRaw,
    EnsembleQNetwork,
    SMDPChunkCollector,
    SuccessReplayBuffer,
)
from diffusion_policy.rlpd.awsc_agent import ShortCutVelocityUNet1D


@dataclass
class Args:
    # Experiment settings
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 42
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
    
    # Algorithm selection
    algorithm: Literal["sac", "awsc"] = "sac"
    """algorithm to use: sac or awsc"""
    
    # Environment settings
    env_id: str = "LiftPegUpright-v1"
    """the id of the environment"""
    num_envs: int = 50
    """number of parallel environments for training"""
    num_eval_envs: int = 25
    """number of parallel eval environments"""
    max_episode_steps: int = 100
    """max episode steps"""
    control_mode: str = "pd_ee_delta_pose"
    """the control mode"""
    obs_mode: str = "rgb"
    """observation mode: state, rgb, or rgbd"""
    sim_backend: str = "physx_cuda"
    """simulation backend"""
    
    # Data settings
    demo_path: Optional[str] = None
    """path to offline demonstrations (h5 file)"""
    num_demos: Optional[int] = None
    """number of demonstrations to use (None for all)"""
    online_ratio: float = 0.5
    """ratio of online data in mixed batch (RLPD default: 0.5)"""
    
    # Pretrained checkpoint (for AWSC)
    pretrained_path: Optional[str] = None
    """path to pretrained checkpoint (ShortCut Flow or AW-ShortCut Flow)"""
    load_critic: bool = False
    """whether to load critic from checkpoint"""
    freeze_visual_encoder: bool = True
    """whether to freeze visual encoder during fine-tuning"""
    
    # Training settings
    total_timesteps: int = 1_000_000
    """total environment steps"""
    num_seed_steps: int = 5000
    """initial exploration steps (random policy)"""
    utd_ratio: int = 4
    """update-to-data ratio (gradient steps per chunk).
    Official RLPD uses grad_updates_per_step=16 with steps_per_env=4, giving effective UTD=4.
    Since we use action chunks (act_horizon steps per chunk), utd_ratio=4 is a reasonable default.
    Increase to 8-16 for more sample-efficient training."""
    batch_size: int = 256
    """batch size for training"""
    eval_freq: int = 10000
    """evaluation frequency (env steps)"""
    save_freq: int = 5000000
    """checkpoint save frequency (env steps)"""
    log_freq: int = 1
    """logging frequency (env steps)"""
    num_eval_episodes: int = 50
    """number of evaluation episodes"""
    
    # Optimizer settings
    lr_actor: float = 3e-4
    """learning rate for actor"""
    lr_critic: float = 3e-4
    """learning rate for critic"""
    lr_temp: float = 3e-4
    """learning rate for temperature (SAC only)"""
    max_grad_norm: float = 10.0
    """maximum gradient norm"""
    
    # SAC hyperparameters
    gamma: float = 0.9
    """discount factor (RLPD ManiSkill default: 0.9 for short episodes)"""
    tau: float = 0.005
    """soft update coefficient"""
    init_temperature: float = 1.0
    """initial entropy temperature (SAC)"""
    target_entropy: Optional[float] = None
    """target entropy (None for -act_dim * act_horizon)"""
    backup_entropy: bool = False
    """whether to use entropy backup in Q-target (RLPD default: False)"""
    reward_scale: float = 1.0
    """reward scaling factor"""
    
    # Ensemble Q settings (RLPD sample-efficient)
    num_qs: int = 10
    """number of Q-networks in ensemble"""
    num_min_qs: int = 2
    """number of Q-networks for subsample + min"""
    q_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    """hidden dimensions for Q-networks"""
    
    # Policy settings
    obs_horizon: int = 2
    """observation horizon"""
    act_horizon: int = 8
    """action execution horizon (SMDP chunk length)"""
    pred_horizon: int = 16
    """action prediction horizon (for AWSC)"""
    
    # AWSC specific settings
    beta: float = 100.0
    """AWAC temperature for advantage weighting"""
    bc_weight: float = 1.0
    """weight for flow matching BC loss"""
    shortcut_weight: float = 0.3
    """weight for shortcut consistency loss"""
    self_consistency_k: float = 0.1
    """fraction of batch for shortcut consistency"""
    exploration_noise_std: float = 0.1
    """exploration noise standard deviation"""
    ema_decay: float = 0.999
    """EMA decay for velocity network"""
    num_inference_steps: int = 8
    """number of flow integration steps"""
    weight_clip: float = 100.0
    """maximum AWAC weight"""
    q_target_clip: float = 100.0
    """Q-target clipping range"""
    
    # Policy-Critic Data Separation (AWSC only)
    filter_policy_data: bool = False
    """whether to filter policy training data by advantage threshold (AWSC only)"""
    advantage_threshold: float = -0.5
    """minimum advantage for online samples to be used in policy training"""
    use_success_buffer: bool = False
    """whether to store successful episodes in SuccessReplayBuffer (for future use)"""
    success_buffer_capacity: int = 100_000
    """capacity of success replay buffer"""
        # Network architecture (for AWSC, must match pretrained)
    diffusion_step_embed_dim: int = 64
    """timestep embedding dimension"""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """U-Net channel dimensions"""
    n_groups: int = 8
    """number of groups for GroupNorm"""
    
    # Visual encoder
    visual_feature_dim: int = 256
    """visual feature dimension"""
    
    # Replay buffer
    replay_buffer_capacity: int = 1_000_000
    """replay buffer capacity per environment"""


def make_train_envs(
    env_id: str,
    num_envs: int,
    sim_backend: str,
    control_mode: str,
    obs_mode: str,
    reward_mode: str = "dense",
    max_episode_steps: Optional[int] = None,
):
    """Create parallel training environments."""
    env_kwargs = dict(
        obs_mode="rgbd" if "rgb" in obs_mode else "state",
        control_mode=control_mode,
        sim_backend=sim_backend,
        num_envs=num_envs,
        reward_mode=reward_mode,
    )
    
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps
    
    env = gym.make(env_id, **env_kwargs)
    
    # Wrap for RGB observations
    if "rgb" in obs_mode:
        env = FlattenRGBDObservationWrapper(
            env,
            rgb=True,
            depth=False,
            state=True,
        )
    
    return env


def main():
    args = tyro.cli(Args)
    
    # Generate experiment name
    if args.exp_name is None:
        args.exp_name = f"rlpd-{args.algorithm}-{args.env_id}-seed{args.seed}"
    
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # Set up logging
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
    
    # Save config
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize tracking
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
            group=f"rlpd-{args.algorithm}",
        )
    
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========== Create Environments ==========
    print("Creating training environments...")
    train_envs = make_train_envs(
        env_id=args.env_id,
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,
        max_episode_steps=args.max_episode_steps,
        reward_mode="dense",
    )
    
    print("Creating evaluation environments...")
    eval_env_kwargs = dict(
        control_mode=args.control_mode,
        max_episode_steps=args.max_episode_steps,
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        reward_mode="dense",
    )
    eval_other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    eval_envs = make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        sim_backend=args.sim_backend,
        env_kwargs=eval_env_kwargs,
        other_kwargs=eval_other_kwargs,
        video_dir=f"{log_dir}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    
    # Get environment info
    obs_space = train_envs.single_observation_space
    act_space = train_envs.single_action_space
    action_dim = act_space.shape[0]
    
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    print(f"Action dimension: {action_dim}")
    
    # ========== Determine Observation Dimension ==========
    include_rgb = "rgb" in args.obs_mode
    
    # State dimension
    state_dim = obs_space["state"].shape[0]
    
    # Visual encoder
    visual_encoder = None
    visual_feature_dim = 0
    
    if include_rgb:
        in_channels = 3  # RGB only
        
        visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
        visual_feature_dim = args.visual_feature_dim
    
    # Total observation dimension (flattened across obs_horizon)
    obs_dim = args.obs_horizon * (visual_feature_dim + state_dim)
    print(f"State dim: {state_dim}, Visual dim: {visual_feature_dim}, Total obs dim: {obs_dim}")
    
    # ========== Create Agent ==========
    if args.algorithm == "sac":
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
            device=device,
        ).to(device)
        
        # Create wrapper using unified AgentWrapper
        agent_wrapper = AgentWrapper(
            agent=agent,
            visual_encoder=visual_encoder,
            include_rgb=include_rgb,
            obs_horizon=args.obs_horizon,
            act_horizon=args.act_horizon,
        ).to(device)
        
        # Optimizers
        actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.lr_actor)
        critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.lr_critic)
        temp_optimizer = optim.Adam(agent.temperature.parameters(), lr=args.lr_temp)
        
    elif args.algorithm == "awsc":
        # Create velocity network
        velocity_net = ShortCutVelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=tuple(args.unet_dims),
            n_groups=args.n_groups,
        )
        
        # Create Q-network
        q_network = EnsembleQNetwork(
            action_dim=action_dim,
            obs_dim=obs_dim,
            action_horizon=args.act_horizon,
            hidden_dims=args.q_hidden_dims,
            num_qs=args.num_qs,
            num_min_qs=args.num_min_qs,
        )
        
        agent = AWSCAgent(
            velocity_net=velocity_net,
            q_network=q_network,
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            act_horizon=args.act_horizon,
            num_qs=args.num_qs,
            num_min_qs=args.num_min_qs,
            beta=args.beta,
            bc_weight=args.bc_weight,
            shortcut_weight=args.shortcut_weight,
            self_consistency_k=args.self_consistency_k,
            gamma=args.gamma,
            tau=args.tau,
            reward_scale=args.reward_scale,
            q_target_clip=args.q_target_clip,
            ema_decay=args.ema_decay,
            weight_clip=args.weight_clip,
            exploration_noise_std=args.exploration_noise_std,
            num_inference_steps=args.num_inference_steps,
            # Policy-Critic data separation
            filter_policy_data=args.filter_policy_data,
            advantage_threshold=args.advantage_threshold,
            device=device,
        ).to(device)
        
        # Load pretrained checkpoint
        if args.pretrained_path:
            agent.load_pretrained(args.pretrained_path, load_critic=args.load_critic)
            # Also load visual encoder if available
            if include_rgb and visual_encoder is not None:
                checkpoint = torch.load(args.pretrained_path, map_location=device)
                if "visual_encoder" in checkpoint:
                    visual_encoder.load_state_dict(checkpoint["visual_encoder"])
                    print(f"Loaded visual encoder from {args.pretrained_path}")
        
        # Print policy data filtering settings
        if args.filter_policy_data:
            print(f"Policy-Critic data separation enabled:")
            print(f"  - Advantage threshold: {args.advantage_threshold}")
            print(f"  - Policy uses: demos + high-advantage online samples")
            print(f"  - Critic uses: all data")
        
        # Create wrapper using unified AgentWrapper
        agent_wrapper = AgentWrapper(
            agent=agent,
            visual_encoder=visual_encoder,
            include_rgb=include_rgb,
            obs_horizon=args.obs_horizon,
            act_horizon=args.act_horizon,
        ).to(device)
        
        # Optimizers
        actor_params = list(agent.velocity_net.parameters())
        if visual_encoder is not None and not args.freeze_visual_encoder:
            actor_params.extend(visual_encoder.parameters())
        
        actor_optimizer = optim.Adam(actor_params, lr=args.lr_actor)
        critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.lr_critic)
        temp_optimizer = None  # AWSC doesn't have temperature
    
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Freeze visual encoder if specified
    if args.freeze_visual_encoder and visual_encoder is not None:
        for param in visual_encoder.parameters():
            param.requires_grad = False
        print("Visual encoder frozen")
    
    # Print model info
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"Agent parameters: {total_params / 1e6:.2f}M (trainable: {trainable_params / 1e6:.2f}M)")
    
    # ========== Create Replay Buffers ==========
    # Online buffer (stores raw observations)
    rgb_shape = (128, 128, 3)  # Default ManiSkill RGB shape
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
    
    # Offline dataset (if demo path provided)
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
            rgb_format="NCHW",  # Unified format for offline and online
            gamma=args.gamma,
            device=device,
        )
        print(f"Offline dataset size: {len(offline_dataset)}")
    
    # Success buffer (for storing successful online episodes)
    success_buffer = None
    if args.use_success_buffer:
        success_buffer = SuccessReplayBuffer(
            capacity=args.success_buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=args.act_horizon,
            obs_horizon=args.obs_horizon,
            include_rgb=include_rgb,
            rgb_shape=rgb_shape,
            success_threshold=1.0,
            device=device,
        )
        print(f"Success buffer enabled (capacity: {args.success_buffer_capacity})")
    
    # ========== Training Loop ==========
    print("\n" + "=" * 50)
    print("Starting RLPD training...")
    print("=" * 50 + "\n")
    
    # Initialize using ObservationStacker
    obs, info = train_envs.reset()
    obs_stacker = ObservationStacker(args.obs_horizon)
    obs_stacker.reset(obs)
    
    # Helper function to encode stacked observations using the unified utility
    def get_obs_features(stacker):
        """Get encoded observation features from stacker using encode_observations."""
        stacked_obs = stacker.get_stacked()
        # Ensure on correct device
        stacked_obs_tensor = {
            k: v.float().to(device) if not v.is_cuda else v.float()
            for k, v in stacked_obs.items()
        }
        return encode_observations(stacked_obs_tensor, visual_encoder, include_rgb, device)
    
    total_steps = 0
    episode_rewards = defaultdict(float)
    episode_lengths = defaultdict(int)
    episode_successes = []
    
    best_success_rate = 0.0
    
    # SMDP chunk collector for action chunk execution
    chunk_collector = SMDPChunkCollector(
        num_envs=args.num_envs,
        gamma=args.gamma,
        action_horizon=args.act_horizon,
    )
    
    pbar = tqdm(total=args.total_timesteps, desc="Training")
    
    # ========== Pre-training Evaluation (Baseline) ==========
    if args.pretrained_path:
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
        
        # Log to wandb
        if args.track:
            import wandb
            wandb.log(pretrain_log, step=0)
        
        # Set initial best success rate from pretrained model
        best_success_rate = pretrain_eval_metrics.get('success_once', 0)
        print(f"Initial best success rate: {best_success_rate:.2%}")
        print("=" * 50 + "\n")
    
    while total_steps < args.total_timesteps:
        # ===== Collect Experience (Action Chunk Execution) =====
        agent.eval() if hasattr(agent, 'eval') else None
        
        # Get current observation features using ObservationStacker
        with torch.no_grad():
            obs_features = get_obs_features(obs_stacker)
            
            # Select action (exploration during seed steps)
            if total_steps < args.num_seed_steps:
                # Random action
                action_chunk = np.random.uniform(-1, 1, (args.num_envs, args.act_horizon, action_dim))
            else:
                # Policy action
                action_chunk = agent.select_action(obs_features, deterministic=False).cpu().numpy()
        
        # Execute action chunk (SMDP)
        chunk_collector.reset()
        chunk_rewards = []
        chunk_dones = []
        
        # Store raw observations for OnlineReplayBufferRaw
        first_obs_raw = obs_stacker.get_stacked()  # Dict with state/rgb
        first_obs_raw_np = {
            k: v.cpu().numpy() if hasattr(v, 'cpu') else v
            for k, v in first_obs_raw.items()
        }
        first_obs_features = obs_features.clone()
        
        for step_idx in range(args.act_horizon):
            action = action_chunk[:, step_idx, :]
            
            next_obs, reward, terminated, truncated, info = train_envs.step(action)
            done = terminated | truncated
            
            # Convert to numpy for buffer storage (ManiSkill returns tensors)
            reward_np = reward.cpu().numpy() if torch.is_tensor(reward) else reward
            done_np = done.cpu().numpy() if torch.is_tensor(done) else done
            
            # Update obs_stacker and encode next_obs for buffer storage
            obs_stacker.append(next_obs)
            with torch.no_grad():
                next_obs_features = get_obs_features(obs_stacker)
            
            # Store reward and done for SMDP computation
            chunk_collector.add(reward=reward_np, done=done_np.astype(np.float32))
            
            chunk_rewards.append(reward_np)
            chunk_dones.append(done_np)
            
            # Track episode stats
            for env_idx in range(args.num_envs):
                episode_rewards[env_idx] += reward_np[env_idx]
                episode_lengths[env_idx] += 1
                
                if done_np[env_idx]:
                    if "success" in info:
                        success = info["success"][env_idx].item() if hasattr(info["success"][env_idx], "item") else info["success"][env_idx]
                    else:
                        success = 0.0
                    episode_successes.append(float(success))
                    
                    # Log episode
                    if len(episode_successes) % 10 == 0:
                        writer.add_scalar("train/episode_reward", episode_rewards[env_idx], total_steps)
                        writer.add_scalar("train/episode_length", episode_lengths[env_idx], total_steps)
                    
                    episode_rewards[env_idx] = 0.0
                    episode_lengths[env_idx] = 0
            
            # Update observation (obs_stacker already updated above)
            obs = next_obs
            
            total_steps += args.num_envs
            pbar.update(args.num_envs)
            
            # Handle episode resets (ManiSkill auto-reset handles this)
        
        # Compute SMDP rewards
        cumulative_reward, chunk_done, discount_factor, effective_length = chunk_collector.compute_smdp_rewards()
        
        # Get final raw observations using ObservationStacker
        next_obs_raw = obs_stacker.get_stacked()
        next_obs_raw_np = {
            k: v.cpu().numpy() if hasattr(v, 'cpu') else v
            for k, v in next_obs_raw.items()
        }
        
        # Store transition in online buffer (raw observations)
        online_buffer.store(
            obs=first_obs_raw_np,
            action=action_chunk,
            reward=chunk_rewards[0],  # Single-step reward for logging
            next_obs=next_obs_raw_np,
            done=np.any(np.stack(chunk_dones, axis=0), axis=0).astype(np.float32),
            cumulative_reward=cumulative_reward,
            chunk_done=chunk_done,
            discount_factor=discount_factor,
            effective_length=effective_length,
        )
        
        # Store successful episodes in success buffer (if enabled)
        if success_buffer is not None and "success" in info:
            # Get success flags for all envs at end of chunk
            success_flags = info.get("success", np.zeros(args.num_envs))
            if hasattr(success_flags, 'cpu'):
                success_flags = success_flags.cpu().numpy()
            
            n_stored = success_buffer.store_if_success(
                obs=first_obs_raw_np,
                action=action_chunk,
                reward=chunk_rewards[0],
                next_obs=next_obs_raw_np,
                done=np.any(np.stack(chunk_dones, axis=0), axis=0).astype(np.float32),
                cumulative_reward=cumulative_reward,
                chunk_done=chunk_done,
                discount_factor=discount_factor,
                effective_length=effective_length,
                success=success_flags,
            )
        
        # ===== Training Updates =====
        if total_steps >= args.num_seed_steps and online_buffer.size >= args.batch_size:
            agent.train() if hasattr(agent, 'train') else None
            
            for _ in range(args.utd_ratio):
                # Sample mixed batch (raw observations)
                batch = online_buffer.sample_mixed(
                    batch_size=args.batch_size,
                    offline_dataset=offline_dataset,
                    online_ratio=args.online_ratio if offline_dataset else 1.0,
                )
                
                # Encode observations (allows gradient flow for visual encoder fine-tuning)
                obs_features = encode_observations(
                    batch["observations"], visual_encoder, include_rgb, device
                )
                next_obs_features = encode_observations(
                    batch["next_observations"], visual_encoder, include_rgb, device
                )
                
                # Update critic
                critic_optimizer.zero_grad()
                
                if args.algorithm == "sac":
                    critic_loss, critic_metrics = agent.compute_critic_loss(
                        obs_features=obs_features,
                        actions=batch["actions"],
                        next_obs_features=next_obs_features,
                        rewards=batch["reward"],
                        dones=batch["done"],
                        cumulative_reward=batch["cumulative_reward"],
                        chunk_done=batch["chunk_done"],
                        discount_factor=batch["discount_factor"],
                    )
                else:  # awsc
                    critic_loss, critic_metrics = agent.compute_critic_loss(
                        obs_features=obs_features,
                        actions=batch["actions"],
                        next_obs_features=next_obs_features,
                        rewards=batch["reward"],
                        dones=batch["done"],
                        cumulative_reward=batch["cumulative_reward"],
                        chunk_done=batch["chunk_done"],
                        discount_factor=batch["discount_factor"],
                    )
                
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                critic_optimizer.step()
                
                # Update actor
                actor_optimizer.zero_grad()
                
                if args.algorithm == "sac":
                    actor_loss, actor_metrics = agent.compute_actor_loss(obs_features)
                else:  # awsc
                    # Need full pred_horizon actions for BC
                    # For online data, we only have act_horizon, so pad
                    actions_full = batch["actions"]
                    if actions_full.shape[1] < args.pred_horizon:
                        pad_len = args.pred_horizon - actions_full.shape[1]
                        pad_action = actions_full[:, -1:, :].repeat(1, pad_len, 1)
                        actions_full = torch.cat([actions_full, pad_action], dim=1)
                    
                    # Pass is_demo for policy-critic data separation
                    actor_loss, actor_metrics = agent.compute_actor_loss(
                        obs_features, actions_full, batch["actions"],
                        is_demo=batch.get("is_demo", None)
                    )
                
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.actor.parameters() if args.algorithm == "sac" else agent.velocity_net.parameters(),
                    args.max_grad_norm
                )
                actor_optimizer.step()
                
                # Update temperature (SAC only)
                if args.algorithm == "sac" and temp_optimizer is not None:
                    temp_optimizer.zero_grad()
                    temp_loss, temp_metrics = agent.compute_temperature_loss(obs_features)
                    temp_loss.backward()
                    temp_optimizer.step()
                
                # Update targets
                agent.update_target()
                
                # Update EMA (AWSC only)
                if args.algorithm == "awsc":
                    agent.update_ema()
        
        # ===== Logging =====
        if total_steps % args.log_freq == 0 and total_steps > args.num_seed_steps:
            log_dict = {}
            
            if len(episode_successes) > 0:
                recent_success = np.mean(episode_successes[-100:])
                writer.add_scalar("train/success_rate", recent_success, total_steps)
                log_dict["train/success_rate"] = recent_success
            
            writer.add_scalar("train/buffer_size", online_buffer.size, total_steps)
            log_dict["train/buffer_size"] = online_buffer.size
            
            # Log success buffer stats
            if success_buffer is not None:
                writer.add_scalar("train/success_buffer_size", success_buffer.size, total_steps)
                writer.add_scalar("train/total_successes_stored", success_buffer.total_successes, total_steps)
                log_dict["train/success_buffer_size"] = success_buffer.size
                log_dict["train/total_successes_stored"] = success_buffer.total_successes
            
            if 'critic_metrics' in dir():
                for k, v in critic_metrics.items():
                    writer.add_scalar(f"train/{k}", v, total_steps)
                    log_dict[f"train/{k}"] = v
            if 'actor_metrics' in dir():
                for k, v in actor_metrics.items():
                    writer.add_scalar(f"train/{k}", v, total_steps)
                    log_dict[f"train/{k}"] = v
            
            # Log episode stats
            if len(episode_rewards) > 0:
                recent_reward = np.mean(list(episode_rewards.values())[-100:]) if episode_rewards else 0
                log_dict["charts/episode_reward"] = recent_reward
            if len(episode_lengths) > 0:
                recent_length = np.mean(list(episode_lengths.values())[-100:]) if episode_lengths else 0
                log_dict["charts/episode_length"] = recent_length
            
            log_dict["training/total_steps"] = total_steps
            
            # Log to wandb
            if args.track:
                import wandb
                wandb.log(log_dict, step=total_steps)
        
        # ===== Evaluation =====
        if total_steps % args.eval_freq == 0:
            print(f"\nEvaluating at step {total_steps}...")
            agent.eval() if hasattr(agent, 'eval') else None
            
            eval_metrics = evaluate(
                args.num_eval_episodes,
                agent_wrapper,
                eval_envs,
                device,
                sim_backend=args.sim_backend,
            )
            
            # Simple handling aligned with offline_rl pipeline
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            eval_log = {}
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], total_steps)
                eval_log[f"eval/{k}"] = eval_metrics[k]
                print(f"{k}: {eval_metrics[k]:.4f}")
            
            # Save best model
            if eval_metrics.get('success_once', 0) > best_success_rate:
                best_success_rate = eval_metrics['success_once']
                torch.save({
                    "agent": agent.state_dict(),
                    "visual_encoder": visual_encoder.state_dict() if visual_encoder else None,
                }, f"{log_dir}/checkpoints/best_eval_success_once.pt")
                print(f"New best success rate: {best_success_rate:.2%}")
            
            # Add best metrics and log to wandb
            eval_log["eval/best_success_once"] = best_success_rate
            eval_log["training/eval_timestep"] = total_steps
            if args.track:
                import wandb
                wandb.log(eval_log, step=total_steps)
        
        # ===== Save Checkpoint =====
        if total_steps % args.save_freq == 0:
            torch.save({
                "agent": agent.state_dict(),
                "visual_encoder": visual_encoder.state_dict() if visual_encoder else None,
                "total_steps": total_steps,
            }, f"{log_dir}/checkpoints/ckpt_{total_steps}.pt")
    
    pbar.close()
    
    # Final save
    torch.save({
        "agent": agent.state_dict(),
        "visual_encoder": visual_encoder.state_dict() if visual_encoder else None,
        "total_steps": total_steps,
    }, f"{log_dir}/checkpoints/final.pt")
    
    print("\nTraining complete!")
    print(f"Best success rate: {best_success_rate:.2%}")
    print(f"Logs saved to: {log_dir}")
    
    train_envs.close()
    eval_envs.close()
    writer.close()
    
    if args.track:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
