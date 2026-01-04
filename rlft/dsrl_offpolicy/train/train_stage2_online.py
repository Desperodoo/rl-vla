"""
Stage 2: Online SAC Training for Latent Policy (Off-Policy)

This script fine-tunes the latent steering policy using online interaction
with ManiSkill environments. Uses SAC with high UTD ratio for improved
sample efficiency.

Key differences from on-policy train_latent_online.py:
1. SAC instead of PPO for policy updates
2. Replay buffer instead of rollout buffer
3. UTD (Update-To-Data) ratio for multiple gradient steps per env step
4. No GAE computation (uses TD learning)

Aligns with on-policy train_latent_online.py for unified environment creation,
observation handling, and evaluation.

Usage:
    python train_stage2_online.py --env_id LiftPegUpright-v1 \
        --stage1_checkpoint ../dsrl/runs/dsrl-stage1-LiftPegUpright-v1/checkpoints/best_eval_success_once.pt \
        --awsc_checkpoint ../dsrl/checkpoints/best_eval_success_once.pt
"""

ALGO_NAME = "DSRL_OffPolicy_Stage2"

import os
import random
import sys
import time
import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Optional, Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import tyro

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

# Add parent dir for imports
_root = Path(__file__).parent.parent.parent
_dsrl_offpolicy = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl"))
sys.path.insert(0, str(_dsrl_offpolicy.parent))  # Add parent so dsrl_offpolicy is importable

from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.evaluate import evaluate
from diffusion_policy.utils import (
    AgentWrapper,
    ObservationStacker,
    encode_observations,
)
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.algorithms.networks import DoubleQNetwork
from diffusion_policy.rlpd.networks import EnsembleQNetwork
from diffusion_policy.rlpd import SMDPChunkCollector

# Import from on-policy DSRL (shared components)
from latent_policy import LatentGaussianPolicy

# Import off-policy specific components
from dsrl_offpolicy.agents.dsrl_sac_agent import DSRLSACAgent
from dsrl_offpolicy.models.latent_q_network import DoubleLatentQNetwork
from dsrl_offpolicy.buffers.macro_replay_buffer import MacroReplayBuffer


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
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "maniskill_dsrl_offpolicy"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""

    # Environment settings
    env_id: str = "LiftPegUpright-v1"
    """the id of the environment"""
    num_envs: int = 50
    """number of parallel training environments"""
    num_eval_envs: int = 50
    """number of parallel eval environments"""
    max_episode_steps: int = 100
    """max episode steps"""
    control_mode: str = "pd_ee_delta_pose"
    """the control mode"""
    obs_mode: str = "rgb"
    """observation mode: state or rgb"""
    sim_backend: str = "physx_cuda"
    """simulation backend"""
    
    # Pretrained checkpoints
    awsc_checkpoint: str = "/home/lizh/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
    """path to pretrained AW-ShortCut Flow checkpoint (velocity_net and q_network)"""
    stage1_checkpoint: str = "/home/lizh/rl-vla/rlft/runs/dsrl_offpolicy_stage1-LiftPegUpright-v1-seed1__1767516252/checkpoints/best_eval_success_once.pt"
    """path to Stage 1 latent policy checkpoint"""
    use_ema: bool = True
    """use EMA weights from AWSC checkpoint (recommended for better performance)"""
    
    # Training settings
    total_timesteps: int = 100_000
    """total environment steps"""
    warmup_steps: int = 5000
    """steps before starting updates (fill replay buffer)"""
    eval_freq: int = 10000
    """evaluation frequency (env steps)"""
    save_freq: int = 10000
    """checkpoint save frequency (env steps)"""
    log_freq: int = 1
    """logging frequency (env steps)"""
    num_eval_episodes: int = 50
    """number of evaluation episodes"""
    
    # Off-policy specific settings
    utd_ratio: int = 20
    """Update-to-data ratio (gradient steps per env step)"""
    batch_size: int = 10240
    """minibatch size for SAC updates"""
    replay_buffer_size: int = 200000
    """replay buffer capacity (in macro-steps)"""
    
    # Optimizer settings
    actor_lr: float = 1e-4
    """learning rate for latent policy"""
    critic_lr: float = 3e-4
    """learning rate for latent Q-networks"""
    temp_lr: float = 1e-4
    """learning rate for temperature"""
    max_grad_norm: float = 1.0
    """maximum gradient norm for clipping"""
    actor_update_delay: int = 10000
    """steps before starting actor updates (critic-only warmup for stability)"""
    
    # Observation/Action horizons
    obs_horizon: int = 2
    """observation horizon"""
    act_horizon: int = 8
    """action execution horizon"""
    pred_horizon: int = 16
    """action prediction horizon"""
    
    # Visual encoder settings
    visual_feature_dim: int = 256
    """visual encoder output dimension"""
    diffusion_step_embed_dim: int = 64
    """timestep embedding dimension"""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """U-Net channel dimensions"""
    n_groups: int = 8
    """GroupNorm groups"""
    
    # Latent policy architecture
    latent_hidden_dims: List[int] = field(default_factory=lambda: [2048, 2048, 2048])
    """hidden dimensions for latent policy MLP"""
    steer_mode: str = "full"
    """steering mode: 'full' or 'act_horizon'"""
    state_dependent_std: bool = True
    """whether to use state-dependent std"""
    
    # Latent Q-network architecture (must match Stage 1's q_hidden_dims for checkpoint loading)
    latent_q_hidden_dims: List[int] = field(default_factory=lambda: [2048, 2048, 2048])
    """hidden dimensions for latent Q-network MLP (should match Stage 1's q_hidden_dims)"""
    
    # Ensemble Q settings (must match pretrained, for Stage 1)
    use_double_q: bool = True
    """use DoubleQNetwork (default for AWSC) instead of EnsembleQNetwork"""
    num_qs: int = 10
    """number of Q-networks in ensemble (only used if use_double_q=False)"""
    num_min_qs: int = 2
    """number of Q-networks for subsample + min (only used if use_double_q=False)"""
    q_hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    """hidden dimensions for Q-networks"""
    
    # SAC hyperparameters
    gamma: float = 0.99
    """discount factor"""
    tau: float = 0.005
    """target network soft update rate"""
    init_temperature: float = 0.1
    """initial SAC temperature"""
    learnable_temp: bool = True
    """whether temperature is learnable"""
    target_entropy: Optional[float] = None
    """target entropy for auto-tuning (None = auto)"""
    backup_entropy: bool = True
    """whether to include entropy in TD target"""
    action_magnitude: float = 1.5
    """TanhNormal output range [-magnitude, magnitude]. Official DSRL: 1.0 (Libero), 2.0 (Aloha), 2.5 (real)"""
    
    # Flow inference
    num_inference_steps: int = 8
    """number of flow integration steps"""


def make_train_envs(
    env_id: str,
    num_envs: int,
    sim_backend: str,
    control_mode: str,
    obs_mode: str,
    reward_mode: str = "dense",
    max_episode_steps: Optional[int] = None,
):
    """Create parallel training environments (aligned with on-policy)."""
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
    
    if "rgb" in obs_mode:
        env = FlattenRGBDObservationWrapper(
            env,
            rgb=True,
            depth=False,
            state=True,
        )
    
    return env


def create_dsrl_sac_agent(args, action_dim: int, global_cond_dim: int) -> DSRLSACAgent:
    """Create DSRL SAC agent with latent Q-networks for Stage 2."""
    device = "cuda" if args.cuda else "cpu"
    
    # Create velocity network (aligned with on-policy)
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=tuple(args.unet_dims),
        n_groups=args.n_groups,
    )
    
    # Create action-space Q-network (for Stage 1, frozen)
    if args.use_double_q:
        q_network = DoubleQNetwork(
            action_dim=action_dim,
            obs_dim=global_cond_dim,
            action_horizon=args.act_horizon,
            hidden_dims=args.q_hidden_dims,
        )
    else:
        q_network = EnsembleQNetwork(
            action_dim=action_dim,
            obs_dim=global_cond_dim,
            action_horizon=args.act_horizon,
            hidden_dims=args.q_hidden_dims,
            num_qs=args.num_qs,
            num_min_qs=args.num_min_qs,
        )
    
    # Create latent policy (trainable)
    latent_policy = LatentGaussianPolicy(
        obs_dim=global_cond_dim,
        pred_horizon=args.pred_horizon,
        action_dim=action_dim,
        hidden_dims=args.latent_hidden_dims,
        steer_mode=args.steer_mode,
        act_horizon=args.act_horizon,
        state_dependent_std=args.state_dependent_std,
        action_magnitude=args.action_magnitude,
    )
    
    # Create latent Q-networks (trainable, for SAC)
    latent_q_network = DoubleLatentQNetwork(
        obs_dim=global_cond_dim,
        pred_horizon=args.pred_horizon,
        action_dim=action_dim,
        hidden_dims=args.latent_q_hidden_dims,
        steer_mode=args.steer_mode,
        act_horizon=args.act_horizon,
        tau=args.tau,
    )
    
    # Create DSRL SAC agent
    agent = DSRLSACAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        latent_policy=latent_policy,
        latent_q_network=latent_q_network,
        action_dim=action_dim,
        obs_dim=global_cond_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        act_horizon=args.act_horizon,
        num_inference_steps=args.num_inference_steps,
        action_magnitude=args.action_magnitude,
        gamma=args.gamma,
        tau_target=args.tau,
        init_temperature=args.init_temperature,
        learnable_temp=args.learnable_temp,
        target_entropy=args.target_entropy,
        backup_entropy=args.backup_entropy,
        device=device,
    )
    
    return agent


def load_checkpoints(
    agent: DSRLSACAgent,
    awsc_checkpoint: str,
    stage1_checkpoint: str,
    visual_encoder: Optional[nn.Module],
    device: str,
    use_ema: bool = True,
):
    """Load pretrained velocity net, Q-network, and Stage 1 latent policy.
    
    Aligned with on-policy load_checkpoints for consistency.
    """
    # Load AWSC checkpoint (velocity_net and q_network)
    if awsc_checkpoint and os.path.exists(awsc_checkpoint):
        print(f"Loading AWSC checkpoint from {awsc_checkpoint}")
        print(f"  Using EMA weights: {use_ema}")
        checkpoint = torch.load(awsc_checkpoint, map_location=device)
        
        if use_ema:
            if "ema_agent" not in checkpoint:
                raise ValueError(
                    f"use_ema=True but checkpoint '{awsc_checkpoint}' does not contain 'ema_agent' key."
                )
            agent_state = checkpoint["ema_agent"]
            print("  Loading from ema_agent")
        else:
            agent_state = checkpoint.get("agent", checkpoint)
            print("  Loading from agent")
        
        velocity_net_state = {}
        q_network_state = {}
        
        for key, value in agent_state.items():
            if key.startswith("velocity_net."):
                velocity_net_state[key.replace("velocity_net.", "")] = value
            elif key.startswith("q_network.") or key.startswith("critic."):
                new_key = key.replace("q_network.", "").replace("critic.", "")
                q_network_state[new_key] = value
        
        if velocity_net_state:
            agent.velocity_net.load_state_dict(velocity_net_state)
            agent.velocity_net_ema.load_state_dict(agent.velocity_net.state_dict())
            print(f"  Loaded velocity_net ({len(velocity_net_state)} keys)")
            print("  Synced velocity_net_ema from velocity_net")
        
        if q_network_state:
            try:
                agent.q_network.load_state_dict(q_network_state)
                print(f"  Loaded q_network ({len(q_network_state)} keys)")
            except Exception as e:
                print(f"  WARNING: Could not load q_network: {e}")
        
        if visual_encoder is not None and "visual_encoder" in checkpoint:
            visual_encoder.load_state_dict(checkpoint["visual_encoder"])
            print("  Loaded visual_encoder")
    else:
        print("WARNING: No AWSC checkpoint provided!")
    
    # Load Stage 1 checkpoint (latent_policy and latent_q_network)
    if stage1_checkpoint and os.path.exists(stage1_checkpoint):
        print(f"Loading Stage 1 checkpoint from {stage1_checkpoint}")
        checkpoint = torch.load(stage1_checkpoint, map_location=device)
        
        # Load latent_policy
        if "latent_policy" in checkpoint:
            agent.latent_policy.load_state_dict(checkpoint["latent_policy"])
            print("  Loaded latent_policy")
        elif "agent" in checkpoint:
            agent_state = checkpoint["agent"]
            latent_policy_state = {}
            for key, value in agent_state.items():
                if key.startswith("latent_policy."):
                    latent_policy_state[key.replace("latent_policy.", "")] = value
            if latent_policy_state:
                agent.latent_policy.load_state_dict(latent_policy_state)
                print(f"  Loaded latent_policy ({len(latent_policy_state)} keys)")
        
        # Load latent_q_network (Q^W trained via distillation in Stage 1)
        if "latent_q_network" in checkpoint and agent.latent_q_network is not None:
            agent.latent_q_network.load_state_dict(checkpoint["latent_q_network"])
            print(f"  Loaded latent_q_network ({len(checkpoint['latent_q_network'])} keys)")
        else:
            print("  WARNING: latent_q_network not found in Stage 1 checkpoint, using random init!")
    else:
        print("WARNING: No Stage 1 checkpoint provided, using random latent policy!")


def save_ckpt(run_name: str, tag: str, agent: DSRLSACAgent, visual_encoder: Optional[nn.Module] = None):
    """Save checkpoint in unified format."""
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ckpt = {
        "agent": agent.state_dict(),
        "latent_policy": agent.latent_policy.state_dict(),
        "latent_q_network": agent.latent_q_network.state_dict() if agent.latent_q_network else None,
        "temperature": agent.temperature.state_dict(),
    }
    if visual_encoder is not None:
        ckpt["visual_encoder"] = visual_encoder.state_dict()
    torch.save(ckpt, f"runs/{run_name}/checkpoints/{tag}.pt")


class DSRLSACAgentWrapper(AgentWrapper):
    """Wrapper for DSRL SAC agent evaluation (aligned with on-policy wrapper)."""
    
    def get_action(self, obs_seq, **kwargs):
        """Get action from observations using latent policy steering."""
        with torch.no_grad():
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            
            action_seq, _, _ = self.agent.get_action(
                obs_cond,
                use_latent_policy=True,
                deterministic=True,
            )
            
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            return action_seq[:, start:end]


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.exp_name is None:
        args.exp_name = f"dsrl_offpolicy_stage2-{args.env_id}-seed{args.seed}"
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
            group="dsrl_offpolicy_stage2",
            tags=["dsrl", "offpolicy", "sac", "stage2"],
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
    state_dim = obs_space["state"].shape[0]
    
    visual_encoder = None
    visual_feature_dim = 0
    
    if include_rgb:
        in_channels = 3
        visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,
        ).to(device)
        visual_feature_dim = args.visual_feature_dim
    
    global_cond_dim = args.obs_horizon * (visual_feature_dim + state_dim)
    print(f"State dim: {state_dim}, Visual dim: {visual_feature_dim}, Total obs dim: {global_cond_dim}")
    
    # ========== Create Agent ==========
    agent = create_dsrl_sac_agent(args, action_dim, global_cond_dim).to(device)
    print(f"DSRL SAC Agent parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M")
    
    # Load checkpoints
    load_checkpoints(
        agent, args.awsc_checkpoint, args.stage1_checkpoint,
        visual_encoder, str(device), use_ema=args.use_ema
    )
    
    # Freeze velocity net and action-space Q-network
    for param in agent.velocity_net.parameters():
        param.requires_grad = False
    for param in agent.q_network.parameters():
        param.requires_grad = False
    if visual_encoder is not None:
        for param in visual_encoder.parameters():
            param.requires_grad = False
    
    # Create wrapper for evaluation
    agent_wrapper = DSRLSACAgentWrapper(
        agent, visual_encoder, include_rgb,
        args.obs_horizon, args.act_horizon
    ).to(device)
    
    # Create optimizers (separate for actor, critic, temperature)
    actor_optimizer = optim.AdamW(
        agent.get_actor_parameters(),
        lr=args.actor_lr,
        weight_decay=1e-5,
    )
    critic_optimizer = optim.AdamW(
        agent.get_critic_parameters(),
        lr=args.critic_lr,
        weight_decay=1e-5,
    )
    if args.learnable_temp:
        temp_optimizer = optim.Adam(
            agent.get_temperature_parameters(),
            lr=args.temp_lr,
        )
    
    # Helper function for encoding observations
    def get_obs_features(stacker):
        """Get encoded observation features from ObservationStacker."""
        stacked_obs = stacker.get_stacked()
        stacked_obs_tensor = {
            k: v.float().to(device) if not v.is_cuda else v.float()
            for k, v in stacked_obs.items()
        }
        return encode_observations(stacked_obs_tensor, visual_encoder, include_rgb, device)
    
    # ========== Training Setup ==========
    obs, info = train_envs.reset()
    obs_stacker = ObservationStacker(args.obs_horizon)
    obs_stacker.reset(obs)
    
    chunk_collector = SMDPChunkCollector(
        num_envs=args.num_envs,
        gamma=args.gamma,
        action_horizon=args.act_horizon,
    )
    
    # Replay buffer for off-policy learning
    replay_buffer = MacroReplayBuffer(
        capacity=args.replay_buffer_size,
        obs_dim=global_cond_dim,
        pred_horizon=args.pred_horizon,
        action_dim=action_dim,
        device=str(device),
    )
    
    total_steps = 0
    episode_rewards = defaultdict(float)
    episode_lengths = defaultdict(int)
    episode_successes = []
    
    best_success_rate = 0.0
    num_updates = 0
    
    # Metrics accumulators
    metrics_accum = defaultdict(list)
    
    print("\n" + "=" * 50)
    print("Starting Stage 2 SAC training (Off-Policy)...")
    print(f"UTD ratio: {args.utd_ratio}")
    print(f"Warmup steps: {args.warmup_steps}")
    print("=" * 50 + "\n")
    
    pbar = tqdm(total=args.total_timesteps, desc="Training")
    
    while total_steps < args.total_timesteps:
        # ===== Collect Data (1 macro-step per iteration) =====
        agent.latent_policy.eval()
        
        with torch.no_grad():
            obs_features = get_obs_features(obs_stacker)
            
            # Sample action from latent policy
            actions, latent_w, log_prob = agent.get_action(
                obs_features,
                use_latent_policy=True,
                deterministic=False,
            )
        
        # Execute action chunk
        action_chunk = actions[:, :args.act_horizon, :].cpu().numpy()
        
        chunk_collector.reset()
        first_obs_features = obs_features.clone()
        
        for step_idx in range(args.act_horizon):
            action = action_chunk[:, step_idx, :]
            
            next_obs, reward, terminated, truncated, info = train_envs.step(action)
            done = terminated | truncated
            
            reward_np = reward.cpu().numpy() if torch.is_tensor(reward) else reward
            done_np = done.cpu().numpy() if torch.is_tensor(done) else done
            
            obs_stacker.append(next_obs)
            chunk_collector.add(reward=reward_np, done=done_np.astype(np.float32))
            
            # Track episode stats
            for env_idx in range(args.num_envs):
                episode_rewards[env_idx] += reward_np[env_idx]
                episode_lengths[env_idx] += 1
                
                if done_np[env_idx]:
                    if "success" in info:
                        success = info["success"][env_idx]
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
        
        # Get next observation features
        with torch.no_grad():
            next_obs_features = get_obs_features(obs_stacker)
        
        # Store transitions in replay buffer
        replay_buffer.add_batch_from_torch(
            obs_cond=first_obs_features,
            latent_w=latent_w,
            reward=torch.from_numpy(cumulative_reward).float(),
            discount_factor=torch.from_numpy(discount_factor).float(),
            done=torch.from_numpy(chunk_done).float(),
            next_obs_cond=next_obs_features,
        )
        
        # ===== SAC Updates (UTD times per env step) =====
        if total_steps >= args.warmup_steps and replay_buffer.is_ready(args.batch_size):
            agent.latent_policy.train()
            agent.latent_q_network.train()
            
            # Determine if we should update actor (critic-only warmup)
            should_update_actor = total_steps >= args.warmup_steps + args.actor_update_delay
            
            for _ in range(args.utd_ratio):
                batch = replay_buffer.sample(args.batch_size)
                
                # ===== Critic Update =====
                critic_metrics = agent.compute_critic_loss(
                    obs_cond=batch["obs_cond"],
                    latent_w=batch["latent_w"],
                    reward=batch["reward"],
                    discount_factor=batch["discount_factor"],
                    done=batch["done"],
                    next_obs_cond=batch["next_obs_cond"],
                )
                
                critic_optimizer.zero_grad()
                critic_metrics["loss"].backward()
                nn.utils.clip_grad_norm_(agent.get_critic_parameters(), args.max_grad_norm)
                critic_optimizer.step()
                
                # ===== Actor Update (delayed for stability) =====
                if should_update_actor:
                    actor_metrics = agent.compute_actor_loss(obs_cond=batch["obs_cond"])
                    
                    actor_optimizer.zero_grad()
                    actor_metrics["loss"].backward()
                    nn.utils.clip_grad_norm_(agent.get_actor_parameters(), args.max_grad_norm)
                    actor_optimizer.step()
                else:
                    # Placeholder metrics during critic-only warmup
                    actor_metrics = {
                        "loss": torch.tensor(0.0),
                        "entropy": torch.tensor(0.0),
                    }
                
                # ===== Temperature Update =====
                if args.learnable_temp and should_update_actor:
                    temp_metrics = agent.compute_temperature_loss(obs_cond=batch["obs_cond"])
                    
                    temp_optimizer.zero_grad()
                    temp_metrics["loss"].backward()
                    temp_optimizer.step()
                else:
                    temp_metrics = {"alpha": agent.temperature.alpha}
                
                # ===== Target Update =====
                agent.update_target()
                
                num_updates += 1
                
                # Accumulate metrics
                metrics_accum["critic_loss"].append(critic_metrics["loss"].item())
                metrics_accum["td_error"].append(critic_metrics["td_error"].item())
                metrics_accum["q_mean"].append(critic_metrics["q_mean"].item())
                metrics_accum["actor_loss"].append(actor_metrics["loss"].item())
                metrics_accum["entropy"].append(actor_metrics["entropy"].item())
                metrics_accum["alpha"].append(temp_metrics["alpha"].item() if hasattr(temp_metrics["alpha"], "item") else temp_metrics["alpha"])
        
        # ===== Logging =====
        if total_steps % args.log_freq < args.num_envs * args.act_horizon and len(metrics_accum["critic_loss"]) > 0:
            train_log = {
                "train/critic_loss": np.mean(metrics_accum["critic_loss"]),
                "train/actor_loss": np.mean(metrics_accum["actor_loss"]),
                "train/td_error": np.mean(metrics_accum["td_error"]),
                "train/q_mean": np.mean(metrics_accum["q_mean"]),
                "train/entropy": np.mean(metrics_accum["entropy"]),
                "train/alpha": np.mean(metrics_accum["alpha"]),
                "train/buffer_size": replay_buffer.size,
                "train/num_updates": num_updates,
            }
            
            for key, value in train_log.items():
                writer.add_scalar(key, value, total_steps)
            
            if episode_successes:
                recent_success = np.mean(episode_successes[-100:])
                writer.add_scalar("train/success_rate", recent_success, total_steps)
                train_log["train/success_rate"] = recent_success
            
            if args.track:
                import wandb
                wandb.log(train_log, step=total_steps)
            
            # Clear accumulators
            metrics_accum = defaultdict(list)
        
        # ===== Evaluation =====
        if total_steps % args.eval_freq < args.num_envs * args.act_horizon:
            agent.latent_policy.eval()
            
            eval_metrics = evaluate(
                args.num_eval_episodes, agent_wrapper, eval_envs, device, args.sim_backend
            )
            
            print(f"\nEvaluation at step {total_steps}:")
            eval_log = {}
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], total_steps)
                eval_log[f"eval/{k}"] = eval_metrics[k]
                print(f"  {k}: {eval_metrics[k]:.4f}")
            
            if args.track:
                import wandb
                wandb.log(eval_log, step=total_steps)
            
            success_rate = eval_metrics.get("success_once", 0)
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                save_ckpt(run_name, "best_eval_success_once", agent, visual_encoder)
                print(f"  New best! Success rate: {success_rate:.2%}")
        
        # ===== Save Checkpoint =====
        if total_steps % args.save_freq < args.num_envs * args.act_horizon:
            save_ckpt(run_name, f"step_{total_steps}", agent, visual_encoder)
    
    # Final evaluation and save
    agent.latent_policy.eval()
    eval_metrics = evaluate(
        args.num_eval_episodes, agent_wrapper, eval_envs, device, args.sim_backend
    )
    
    print(f"\nFinal evaluation:")
    eval_log = {}
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])
        writer.add_scalar(f"eval/{k}", eval_metrics[k], total_steps)
        eval_log[f"eval/{k}"] = eval_metrics[k]
        print(f"  {k}: {eval_metrics[k]:.4f}")
    
    if args.track:
        import wandb
        wandb.log(eval_log, step=total_steps)
    
    save_ckpt(run_name, "final", agent, visual_encoder)
    print(f"\nTraining complete! Final checkpoint saved to {log_dir}/checkpoints/final.pt")
    
    train_envs.close()
    eval_envs.close()
    writer.close()
    
    if args.track:
        import wandb
        wandb.finish()
