"""
Stage 2: Online PPO Training for Latent Policy

This script fine-tunes the latent steering policy using online interaction
with ManiSkill environments. Uses macro-step PPO with GAE advantage estimation.

Key features:
- Macro-step MDP: each step executes act_horizon actions (SMDP)
- GAE computed over macro-steps with gamma_macro = gamma^act_horizon
- Mixed exploration: η probability of using prior vs policy
- KL-to-prior regularization for stable steering

Aligns with train_rlpd_online.py for unified environment creation,
observation handling, and evaluation.

Usage:
    python train_latent_online.py --env_id LiftPegUpright-v1 \
        --stage1_checkpoint runs/dsrl_stage1-LiftPegUpright-v1/checkpoints/best_eval_success_once.pt \
        --awsc_checkpoint runs/aw_shortcut_flow-LiftPegUpright-v1/checkpoints/best_eval_success_once.pt
"""

ALGO_NAME = "DSRL_Stage2"

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
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))

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

# Local imports
from dsrl_agent import DSRLAgent
from latent_policy import LatentGaussianPolicy
from value_network import ValueNetwork
from macro_rollout_buffer import MacroRolloutBuffer


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
    awsc_checkpoint: str = ""
    """path to pretrained AW-ShortCut Flow checkpoint (velocity_net and q_network)"""
    stage1_checkpoint: str = ""
    """path to Stage 1 latent policy checkpoint"""
    use_ema: bool = True
    """use EMA weights from AWSC checkpoint (recommended for better performance)"""
    
    # Training settings
    total_timesteps: int = 500_000
    """total environment steps"""
    rollout_steps: int = 100
    """number of macro-steps per rollout (actual steps = rollout_steps * act_horizon)"""
    ppo_epochs: int = 10
    """number of PPO epochs per update"""
    batch_size: int = 5120
    """minibatch size for PPO updates"""
    eval_freq: int = 102400
    """evaluation frequency (env steps)"""
    save_freq: int = 102400
    """checkpoint save frequency (env steps)"""
    log_freq: int = 1
    """logging frequency (env steps)"""
    num_eval_episodes: int = 50
    """number of evaluation episodes"""
    
    # Optimizer settings
    lr: float = 3e-4
    """learning rate for latent policy"""
    value_lr: float = 1e-3
    """learning rate for value network"""
    max_grad_norm: float = 0.5
    """maximum gradient norm for clipping"""
    
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
    latent_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    """hidden dimensions for latent policy MLP"""
    steer_mode: str = "full"
    """steering mode: 'full' or 'act_horizon'"""
    state_dependent_std: bool = True
    """whether to use state-dependent std"""
    
    # Value network architecture
    value_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    """hidden dimensions for value network"""
    
    # Ensemble Q settings (must match pretrained)
    use_double_q: bool = True
    """use DoubleQNetwork (default for AWSC) instead of EnsembleQNetwork"""
    num_qs: int = 10
    """number of Q-networks in ensemble (only used if use_double_q=False)"""
    num_min_qs: int = 2
    """number of Q-networks for subsample + min (only used if use_double_q=False)"""
    q_hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
    """hidden dimensions for Q-networks"""
    
    # PPO hyperparameters
    ppo_clip: float = 0.2
    """PPO clipping coefficient"""
    entropy_coef: float = 1e-3
    """entropy regularization coefficient"""
    kl_prior_coef: float = 1e-3
    """KL-to-prior regularization coefficient"""
    
    # KL early stopping
    target_kl: Optional[float] = 0.02
    """target KL divergence threshold for early stopping PPO epochs (None to disable)"""
    kl_early_stop: bool = True
    """whether to enable KL-based early stopping of PPO epochs"""
    
    # Adaptive KL settings (for KL-to-prior regularization)
    adaptive_kl: bool = False
    """enable adaptive KL coefficient adjustment"""
    kl_target: float = 0.01
    """target KL divergence value for adaptive adjustment"""
    kl_adapt_factor: float = 1.5
    """multiplicative factor for KL coefficient adjustment"""
    kl_coef_min: float = 1e-5
    """minimum KL coefficient"""
    kl_coef_max: float = 1.0
    """maximum KL coefficient"""
    
    gamma: float = 0.95
    """discount factor (shorter episodes: 0.9, longer: 0.99)"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    normalize_advantage: bool = True
    """whether to normalize advantages"""
    
    # Exploration (prior mixing)
    prior_mix_ratio: float = 0.3
    """initial probability of using prior instead of policy"""
    prior_mix_decay: float = 0.995
    """decay rate for prior mixing per rollout"""
    prior_mix_min: float = 0.05
    """minimum prior mixing probability"""
    
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
    """Create parallel training environments (from train_rlpd_online.py)."""
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


def create_dsrl_agent(args, action_dim: int, global_cond_dim: int) -> DSRLAgent:
    """Create DSRL agent with value network for Stage 2."""
    device = "cuda" if args.cuda else "cpu"
    
    # Create velocity network
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=tuple(args.unet_dims),
        n_groups=args.n_groups,
    )
    
    # Create Q-network
    if args.use_double_q:
        # DoubleQNetwork (default for AWSC checkpoints)
        q_network = DoubleQNetwork(
            action_dim=action_dim,
            obs_dim=global_cond_dim,
            action_horizon=args.act_horizon,
            hidden_dims=args.q_hidden_dims,
        )
    else:
        # EnsembleQNetwork (for RLPD fine-tuned checkpoints)
        q_network = EnsembleQNetwork(
            action_dim=action_dim,
            obs_dim=global_cond_dim,
            action_horizon=args.act_horizon,
            hidden_dims=args.q_hidden_dims,
            num_qs=args.num_qs,
            num_min_qs=args.num_min_qs,
        )
    
    # Create latent policy
    latent_policy = LatentGaussianPolicy(
        obs_dim=global_cond_dim,
        pred_horizon=args.pred_horizon,
        action_dim=action_dim,
        hidden_dims=args.latent_hidden_dims,
        steer_mode=args.steer_mode,
        act_horizon=args.act_horizon,
        state_dependent_std=args.state_dependent_std,
    )
    
    # Create value network (for PPO)
    value_network = ValueNetwork(
        obs_dim=global_cond_dim,
        hidden_dims=args.value_hidden_dims,
    )
    
    # Create DSRL agent
    agent = DSRLAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        latent_policy=latent_policy,
        value_network=value_network,
        action_dim=action_dim,
        obs_dim=global_cond_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        act_horizon=args.act_horizon,
        num_inference_steps=args.num_inference_steps,
        ppo_clip=args.ppo_clip,
        entropy_coef=args.entropy_coef,
        kl_prior_coef=args.kl_prior_coef,
        prior_mix_ratio=args.prior_mix_ratio,
        prior_mix_decay=args.prior_mix_decay,
        prior_mix_min=args.prior_mix_min,
        device=device,
    )
    
    return agent


def load_checkpoints(
    agent: DSRLAgent,
    awsc_checkpoint: str,
    stage1_checkpoint: str,
    visual_encoder: Optional[nn.Module],
    device: str,
    use_ema: bool = True,
):
    """Load pretrained velocity net, Q-network, and Stage 1 latent policy.
    
    Args:
        agent: DSRLAgent to load weights into
        awsc_checkpoint: Path to AWSC checkpoint
        stage1_checkpoint: Path to Stage 1 latent policy checkpoint
        visual_encoder: Optional visual encoder to load
        device: Device for loading
        use_ema: If True, load from ema_agent (recommended for better performance)
    """
    
    # Load AWSC checkpoint (velocity_net and q_network)
    if awsc_checkpoint and os.path.exists(awsc_checkpoint):
        print(f"Loading AWSC checkpoint from {awsc_checkpoint}")
        print(f"  Using EMA weights: {use_ema}")
        checkpoint = torch.load(awsc_checkpoint, map_location=device)
        
        # Prefer EMA weights - strict assertion if requested but missing
        if use_ema:
            if "ema_agent" not in checkpoint:
                raise ValueError(
                    f"use_ema=True but checkpoint '{awsc_checkpoint}' does not contain 'ema_agent' key. "
                    f"Available keys: {list(checkpoint.keys())}. "
                    f"Either set use_ema=False or provide a checkpoint with EMA weights."
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
            # Sync EMA velocity net (critical for correct inference)
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
    
    # Load Stage 1 checkpoint (latent_policy)
    if stage1_checkpoint and os.path.exists(stage1_checkpoint):
        print(f"Loading Stage 1 checkpoint from {stage1_checkpoint}")
        checkpoint = torch.load(stage1_checkpoint, map_location=device)
        
        if "latent_policy" in checkpoint:
            agent.latent_policy.load_state_dict(checkpoint["latent_policy"])
            print("  Loaded latent_policy")
        elif "agent" in checkpoint:
            # Try to extract from full agent state
            agent_state = checkpoint["agent"]
            latent_policy_state = {}
            for key, value in agent_state.items():
                if key.startswith("latent_policy."):
                    latent_policy_state[key.replace("latent_policy.", "")] = value
            if latent_policy_state:
                agent.latent_policy.load_state_dict(latent_policy_state)
                print(f"  Loaded latent_policy ({len(latent_policy_state)} keys)")
    else:
        print("WARNING: No Stage 1 checkpoint provided, using random latent policy!")


def save_ckpt(run_name: str, tag: str, agent: DSRLAgent, visual_encoder: Optional[nn.Module] = None):
    """Save checkpoint in unified format."""
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ckpt = {
        "agent": agent.state_dict(),
        "latent_policy": agent.latent_policy.state_dict(),
        "value_network": agent.value_network.state_dict() if agent.value_network else None,
    }
    if visual_encoder is not None:
        ckpt["visual_encoder"] = visual_encoder.state_dict()
    torch.save(ckpt, f"runs/{run_name}/checkpoints/{tag}.pt")


class DSRLAgentWrapper(AgentWrapper):
    """Wrapper for DSRL agent evaluation."""
    
    def get_action(self, obs_seq, **kwargs):
        """Get action from observations using latent policy steering."""
        with torch.no_grad():
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            
            action_seq, _, _ = self.agent.get_action(
                obs_cond,
                use_latent_policy=True,
                deterministic=True,
                use_prior_mixing=False,
            )
            
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            return action_seq[:, start:end]


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.exp_name is None:
        args.exp_name = f"dsrl_stage2-{args.env_id}-seed{args.seed}"
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
            group="dsrl_stage2",
            tags=["dsrl", "stage2", "ppo"],
        )
    
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Warning for experimental steer_mode
    if args.steer_mode == "act_horizon":
        import warnings
        warnings.warn(
            f"steer_mode='act_horizon' is experimental. Credit assignment may be suboptimal "
            f"compared to steer_mode='full' (default). Consider using 'full' mode for stability.",
            UserWarning
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
    agent = create_dsrl_agent(args, action_dim, global_cond_dim).to(device)
    print(f"DSRL Agent parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M")
    
    # Load checkpoints (use EMA weights for better performance)
    load_checkpoints(
        agent, args.awsc_checkpoint, args.stage1_checkpoint,
        visual_encoder, str(device), use_ema=args.use_ema
    )
    
    # Freeze velocity net and Q-network
    for param in agent.velocity_net.parameters():
        param.requires_grad = False
    for param in agent.q_network.parameters():
        param.requires_grad = False
    if visual_encoder is not None:
        for param in visual_encoder.parameters():
            param.requires_grad = False
    
    # Create wrapper for evaluation
    agent_wrapper = DSRLAgentWrapper(
        agent, visual_encoder, include_rgb,
        args.obs_horizon, args.act_horizon
    ).to(device)
    
    # Create optimizer (latent policy + value network)
    trainable_params = list(agent.latent_policy.parameters())
    if agent.value_network is not None:
        trainable_params.extend(agent.value_network.parameters())
    
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)
    
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
    
    # Rollout buffer for PPO
    buffer = MacroRolloutBuffer(
        capacity=args.rollout_steps,  # Number of macro-steps per env
        obs_dim=global_cond_dim,
        pred_horizon=args.pred_horizon,
        action_dim=action_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        act_horizon=args.act_horizon,
        num_envs=args.num_envs,
        device=device,
    )
    
    total_steps = 0
    episode_rewards = defaultdict(float)
    episode_lengths = defaultdict(int)
    episode_successes = []
    
    best_success_rate = 0.0
    
    print("\n" + "=" * 50)
    print("Starting Stage 2 PPO training...")
    print("=" * 50 + "\n")
    
    pbar = tqdm(total=args.total_timesteps, desc="Training")
    
    while total_steps < args.total_timesteps:
        # ===== Collect Rollout =====
        buffer.reset()
        agent.latent_policy.eval()
        
        # Track current done state for each env (for correct GAE bootstrap)
        current_done = torch.zeros(args.num_envs, device=device)
        
        for _ in range(args.rollout_steps):
            with torch.no_grad():
                obs_features = get_obs_features(obs_stacker)
                
                # Get value estimate
                value = agent.get_value(obs_features)
                
                # Sample action from latent policy
                actions, latent_w, log_prob = agent.get_action(
                    obs_features,
                    use_latent_policy=True,
                    deterministic=False,
                    use_prior_mixing=False,
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
            
            # Update current done state for each env
            current_done = torch.from_numpy(chunk_done).float().to(device)
            
            # Get next observation features
            with torch.no_grad():
                next_obs_features = get_obs_features(obs_stacker)
            
            # Store transition in buffer (batch of num_envs transitions)
            buffer.store_batch(
                obs_cond=first_obs_features,
                latent_w=latent_w,
                log_prob=log_prob,
                reward=torch.from_numpy(cumulative_reward).float().to(device),
                done=current_done,
                value=value,
                next_obs_cond=next_obs_features,
                discount_factor=torch.from_numpy(discount_factor).float().to(device),
            )
        
        # Compute GAE with real done status (not all zeros!)
        with torch.no_grad():
            last_value = agent.get_value(get_obs_features(obs_stacker))
        buffer.compute_gae(last_value, current_done)
        
        # ===== PPO Update =====
        agent.latent_policy.train()
        if agent.value_network is not None:
            agent.value_network.train()
        
        agent.sync_old_policy()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_adv_mean = 0.0
        total_adv_std = 0.0
        num_updates = 0
        kl_early_stopped = False  # Track if we early stopped
        actual_epochs = 0
        
        for epoch in range(args.ppo_epochs):
            # Check for KL early stopping at epoch level
            if kl_early_stopped:
                break
            actual_epochs = epoch + 1
            
            for batch in buffer.iterate_batches(args.batch_size, shuffle=True):
                loss_dict = agent.compute_ppo_loss(
                    obs_cond=batch["obs_cond"],
                    latent_w=batch["latent_w"],
                    old_log_prob=batch["log_prob"],
                    advantage=batch["advantage"],
                    returns=batch["returns"],
                )
                
                loss = loss_dict["loss"]
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                
                total_policy_loss += loss_dict["policy_loss"].item()
                total_value_loss += loss_dict["value_loss"].item()
                total_entropy += loss_dict["entropy"].item()
                total_approx_kl += loss_dict["approx_kl"].item()
                total_clip_frac += loss_dict["clip_frac"].item()
                total_adv_mean += loss_dict["adv_mean"].item()
                total_adv_std += loss_dict["adv_std"].item()
                num_updates += 1
                
                # KL early stopping check (per minibatch, like ppo.py)
                if args.kl_early_stop and args.target_kl is not None:
                    current_kl = loss_dict["approx_kl"].item()
                    if current_kl > args.target_kl * 1.5:  # Allow some margin
                        kl_early_stopped = True
                        break
        
        agent.decay_prior_mix_ratio()
        
        # ===== Logging =====
        if total_steps % args.log_freq < args.num_envs * args.act_horizon * args.rollout_steps:
            # Log PPO epoch utilization info
            early_stop_msg = " (KL early stopped)" if kl_early_stopped else ""
            print(f"  [PPO] {actual_epochs}/{args.ppo_epochs} epochs, {num_updates} minibatch updates{early_stop_msg}")
            
            train_log = {
                "train/policy_loss": total_policy_loss / max(1, num_updates),
                "train/value_loss": total_value_loss / max(1, num_updates),
                "train/entropy": total_entropy / max(1, num_updates),
                "train/prior_mix_ratio": agent.get_prior_mix_ratio(),
                # Critical PPO diagnostics
                "train/approx_kl": total_approx_kl / max(1, num_updates),
                "train/clip_frac": total_clip_frac / max(1, num_updates),
                "train/adv_mean": total_adv_mean / max(1, num_updates),
                "train/adv_std": total_adv_std / max(1, num_updates),
                # PPO epoch utilization
                "train/actual_ppo_epochs": actual_epochs,
                "train/kl_early_stopped": float(kl_early_stopped),
            }
            
            writer.add_scalar("train/policy_loss", train_log["train/policy_loss"], total_steps)
            writer.add_scalar("train/value_loss", train_log["train/value_loss"], total_steps)
            writer.add_scalar("train/entropy", train_log["train/entropy"], total_steps)
            writer.add_scalar("train/prior_mix_ratio", train_log["train/prior_mix_ratio"], total_steps)
            writer.add_scalar("train/approx_kl", train_log["train/approx_kl"], total_steps)
            writer.add_scalar("train/clip_frac", train_log["train/clip_frac"], total_steps)
            writer.add_scalar("train/adv_mean", train_log["train/adv_mean"], total_steps)
            writer.add_scalar("train/adv_std", train_log["train/adv_std"], total_steps)
            writer.add_scalar("train/actual_ppo_epochs", train_log["train/actual_ppo_epochs"], total_steps)
            writer.add_scalar("train/kl_early_stopped", train_log["train/kl_early_stopped"], total_steps)
            
            if episode_successes:
                recent_success = np.mean(episode_successes[-100:])
                writer.add_scalar("train/success_rate", recent_success, total_steps)
                train_log["train/success_rate"] = recent_success
            
            # Log to wandb
            if args.track:
                import wandb
                wandb.log(train_log, step=total_steps)
        
        # ===== Evaluation =====
        if total_steps % args.eval_freq < args.num_envs * args.act_horizon * args.rollout_steps:
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
            
            # Log to wandb
            if args.track:
                import wandb
                wandb.log(eval_log, step=total_steps)
            
            success_rate = eval_metrics.get("success_once", 0)
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                save_ckpt(run_name, "best_eval_success_once", agent, visual_encoder)
                print(f"  New best! Success rate: {success_rate:.2%}")
        
        # ===== Save Checkpoint =====
        if total_steps % args.save_freq < args.num_envs * args.act_horizon * args.rollout_steps:
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
    
    # Log final evaluation to wandb
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
