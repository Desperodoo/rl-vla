"""
Online RL Fine-tuning Script for ReinFlow

Fine-tunes a pretrained flow matching policy (from AW-ShortCut Flow)
using PPO-style policy gradient with environment interaction.

Three-stage pipeline:
- Stage 1: ShortCut Flow BC pretrain (pure BC) → train_offline_rl.py --algorithm shortcut_flow
- Stage 2: AW-ShortCut Flow offline RL (Q-weighted BC) → train_offline_rl.py --algorithm aw_shortcut_flow
- Stage 3: ReinFlow online RL (this script, PPO fine-tuning)

Key features:
- SMDP formulation: executes full act_horizon before policy update
- Critic warmup: trains value network before policy updates
- Noise scheduling: controllable exploration decay
- Compatible with AW-ShortCut Flow checkpoints

Usage:
    python train_online_finetune.py \
        --env_id PickCube-v1 \
        --pretrained_path runs/aw_shortcut_flow-PickCube-v1-seed0/best_model.pt \
        --num_inference_steps 8 \
        --critic_warmup_steps 5000

Based on ManiSkill PPO implementation and ReinFlow paper.
"""

ALGO_NAME = "ReinFlow"

import os
import random
import time
import warnings
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Any
import json

# Suppress SAPIEN PhysX CPU warnings when using GPU training with CPU eval
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

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import (
    AgentWrapper, ObservationStacker, build_state_obs_extractor, 
    convert_obs, encode_observations
)
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.algorithms.reinflow import ReinFlowAgent
from diffusion_policy.rlpd import RolloutBufferPPO
from diffusion_policy.evaluate import evaluate


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
    num_envs: int = 50
    """number of parallel environments for training"""
    num_eval_envs: int = 5
    """number of parallel eval environments"""
    max_episode_steps: Optional[int] = None
    """max episode steps"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode"""
    obs_mode: str = "rgb"
    """observation mode: rgb or state"""
    sim_backend: str = "physx_cuda"
    """simulation backend (physx_cuda for parallel training)"""

    # Pretrained checkpoint
    pretrained_path: str = "runs/awsc-LiftPegUpright-v1-seed0/checkpoints/best_eval_success_once.pt"
    """path to pretrained AW-ShortCut Flow checkpoint"""
    freeze_visual_encoder: bool = True
    """whether to freeze visual encoder during fine-tuning"""

    # Training settings
    total_updates: int = 10_000
    """total training timesteps (env steps, not policy steps)"""
    rollout_steps: int = 64
    """number of SMDP chunks to collect before each update"""
    ppo_epochs: int = 20
    """number of PPO epochs per update"""
    minibatch_size: int = 5120
    """minibatch size for PPO updates"""
    lr: float = 1e-6
    """learning rate for policy"""
    lr_critic: float = 1e-6
    """learning rate for value network"""
    max_grad_norm: float = 10.0
    """maximum gradient norm for clipping"""

    # PPO hyperparameters
    gamma: float = 0.99
    """discount factor"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    clip_ratio: float = 0.1
    """PPO clip ratio"""
    entropy_coef: float = 0.00
    """entropy coefficient"""
    value_coef: float = 0.5
    """value loss coefficient"""

    # Policy settings
    obs_horizon: int = 2
    """observation horizon"""
    act_horizon: int = 8
    """action execution horizon (full chunk for SMDP)"""
    pred_horizon: int = 16
    """action prediction horizon"""
    num_inference_steps: int = 8
    """number of flow integration steps (fixed mode)"""
    ema_decay: float = 0.999
    """EMA decay rate"""

    # Network architecture (defaults match train_offline_rl.py for checkpoint compatibility)
    diffusion_step_embed_dim: int = 64
    """timestep embedding dimension (must match pretrained checkpoint)"""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """U-Net channel dimensions (must match pretrained checkpoint)"""
    n_groups: int = 8
    """GroupNorm groups (must match pretrained checkpoint)"""
    visual_feature_dim: int = 256
    """visual encoder output dimension (must match pretrained checkpoint)"""

    # Noise scheduling
    min_noise_std: float = 0.01
    """minimum exploration noise std"""
    max_noise_std: float = 0.15
    """maximum exploration noise std"""
    noise_decay_type: Literal["constant", "linear", "exponential"] = "linear"
    """noise decay schedule type"""

    # Critic warmup
    critic_warmup_steps: int = 50
    """number of steps to train critic only (policy frozen)"""

    # Reward processing
    normalize_rewards: bool = True
    """whether to normalize rewards using running mean/std"""
    normalize_advantages: bool = False
    """whether to normalize advantages in buffer (set False since we normalize per minibatch)"""
    clip_value_loss: bool = True
    """whether to clip value loss to reduce vloss explosion"""
    value_clip_range: float = 10.0
    """clip value predictions to [-range, range] for stable training"""
    
    # === NEW: Critic stability improvements (borrowed from AWCP) ===
    reward_scale: float = 0.5
    """scale factor for rewards (like AWCP's reward_scale, helps stabilize critic)"""
    value_target_tau: float = 0.005
    """soft update rate for target value network (like AWCP's tau)"""
    use_target_value_net: bool = True
    """whether to use target value network for stable value estimation"""
    value_target_clip: float = 100.0
    """clip value targets to prevent extreme values (like AWCP's q_target_clip)"""
    normalize_returns: bool = True
    """whether to normalize returns using running mean/std"""
    
    # === NEW: KL divergence early stopping ===
    target_kl: Optional[float] = 0.02
    """target KL divergence for early stopping PPO epochs (None to disable)"""
    kl_early_stop: bool = True
    """whether to enable KL-based early stopping of PPO epochs"""

    # Logging settings (in number of updates, not timesteps)
    log_freq: int = 1
    """logging frequency (in number of updates)"""
    eval_freq: int = 10
    """evaluation frequency (in number of updates)"""
    save_freq: int = 10000000
    """checkpoint save frequency (in number of updates)"""
    num_eval_episodes: int = 20
    """number of evaluation episodes"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd:
    """Running mean/std calculator using Welford's algorithm."""
    
    def __init__(self, epsilon: float = 1e-8, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Avoid division by zero
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray):
        """Update running statistics with a batch of values."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.epsilon)


class RewardNormalizer:
    """Running mean/std for reward normalization."""
    
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.rms = RunningMeanStd(epsilon=epsilon)
    
    def update(self, rewards: np.ndarray):
        """Update running statistics."""
        self.rms.update(rewards.flatten())
    
    def normalize(self, reward: float) -> float:
        """Normalize a single reward by dividing by std."""
        return reward / (self.rms.std + self.epsilon)


class ReturnNormalizer:
    """Running mean/std for return normalization (like AWCP's reward scaling).
    
    Tracks the running statistics of returns and normalizes them to have
    approximately unit variance, which helps stabilize value network training.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.rms = RunningMeanStd(epsilon=epsilon)
    
    def update(self, returns: np.ndarray):
        """Update running statistics with batch of returns."""
        self.rms.update(returns.flatten())
    
    def normalize(self, returns: np.ndarray) -> np.ndarray:
        """Normalize returns by dividing by std (not subtracting mean)."""
        return returns / (self.rms.std + self.epsilon)
    
    @property
    def std(self) -> float:
        return self.rms.std


def make_train_envs(env_id: str, num_envs: int, sim_backend: str, 
                    control_mode: str, obs_mode: str, 
                    max_episode_steps: Optional[int] = None,
                    reward_mode: str = "dense",
                    record_video: bool = False,
                    video_path: str = None):
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
    
    if record_video and video_path:
        env_kwargs["enable_shadow"] = True
        env_kwargs["render_mode"] = "rgb_array"
    
    env = gym.make(env_id, **env_kwargs)
    
    # Wrap for RGB observations
    if "rgb" in obs_mode:
        env = FlattenRGBDObservationWrapper(
            env, 
            rgb=True, 
            depth=False, 
            state=True
        )
    
    return env


def main():
    args = tyro.cli(Args)
    
    # Generate experiment name
    if args.exp_name is None:
        args.exp_name = f"reinflow-{args.env_id}-seed{args.seed}"
    
    run_name = f"{args.exp_name}__{int(time.time())}"
    
    # Set up logging
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize logging
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )
    
    writer = SummaryWriter(log_dir)
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environments
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
        obs_mode=args.obs_mode,  # Must match training obs_mode for AgentWrapper
        render_mode="rgb_array",  # For video recording compatibility
        reward_mode="dense",  # Match training reward mode
    )
    eval_other_kwargs = dict(
        obs_horizon=args.obs_horizon,
    )
    eval_envs = make_eval_envs(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        sim_backend=args.sim_backend,
        env_kwargs=eval_env_kwargs,
        other_kwargs=eval_other_kwargs,
        video_dir=f"{log_dir}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],  # Flatten RGBD observations before FrameStack
    )
    
    # Get environment info
    obs_space = train_envs.single_observation_space
    act_space = train_envs.single_action_space
    act_dim = act_space.shape[0]
    
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    print(f"Action dimension: {act_dim}")
    
    # Build state observation extractor and get state dimension
    # Note: state_obs_extractor is NOT used for online training since FlattenRGBDObservationWrapper
    # already flattens the state. We keep it only for compatibility.
    state_obs_extractor = build_state_obs_extractor(args.env_id)
    # State dimension is simply the flattened state size from observation space
    state_dim = train_envs.single_observation_space["state"].shape[0]
    print(f"State dimension: {state_dim}")
    
    # Check observation mode
    include_rgb = "rgb" in args.obs_mode
    
    # Build visual encoder
    if include_rgb:
        in_channels = 3  # RGB only
        
        visual_encoder = PlainConv(
            in_channels=in_channels,
            out_dim=args.visual_feature_dim,
            pool_feature_map=True,  # Must match pretrained checkpoint
        ).to(device)
        
        obs_dim = (args.visual_feature_dim + state_dim) * args.obs_horizon
    else:
        visual_encoder = None
        obs_dim = state_dim * args.obs_horizon
    
    print(f"Observation dimension (flattened): {obs_dim}")
    
    # Build velocity network (ShortCut Flow architecture)
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=act_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=args.unet_dims,
        n_groups=args.n_groups,
    ).to(device)
    
    # Build ReinFlow agent
    agent = ReinFlowAgent(
        velocity_net=velocity_net,
        obs_dim=obs_dim,
        act_dim=act_dim,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        act_horizon=args.act_horizon,
        num_inference_steps=args.num_inference_steps,
        ema_decay=args.ema_decay,
        use_ema=True,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        min_noise_std=args.min_noise_std,
        max_noise_std=args.max_noise_std,
        noise_decay_type=args.noise_decay_type,
        noise_decay_steps=args.total_updates,
        critic_warmup_steps=args.critic_warmup_steps,
        # === NEW: Critic stability parameters ===
        reward_scale=args.reward_scale,
        value_target_tau=args.value_target_tau,
        use_target_value_net=args.use_target_value_net,
        value_target_clip=args.value_target_clip,
    ).to(device)
    
    # Load pretrained checkpoint
    if args.pretrained_path:
        print(f"Loading pretrained model from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        
        # Load velocity network weights
        if "velocity_net" in checkpoint:
            agent.load_from_aw_shortcut_flow(args.pretrained_path, device=str(device))
        else:
            agent.load_pretrained(args.pretrained_path, device=str(device))
        
        # Load visual encoder if available
        if include_rgb:
            if "visual_encoder" in checkpoint:
                visual_encoder.load_state_dict(checkpoint["visual_encoder"])
                print("Loaded visual encoder from checkpoint")
        
        print("Pretrained model loaded successfully")
    
    # Freeze visual encoder if specified
    if args.freeze_visual_encoder and include_rgb:
        for param in visual_encoder.parameters():
            param.requires_grad = False
        print("Visual encoder frozen")
    
    # Calculate and log model statistics
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    visual_params = 0
    if include_rgb:
        visual_params = sum(p.numel() for p in visual_encoder.parameters())
        trainable_visual_params = sum(p.numel() for p in visual_encoder.parameters() if p.requires_grad)
        print(f"Visual encoder parameters: {visual_params / 1e6:.2f}M (trainable: {trainable_visual_params / 1e6:.2f}M)")
    
    print(f"Agent parameters: {total_params / 1e6:.2f}M (trainable: {trainable_params / 1e6:.2f}M)")
    print(f"Total model parameters: {(total_params + visual_params) / 1e6:.2f}M")
    
    # Set up optimizers
    policy_params = list(agent.noisy_velocity_net.parameters())
    critic_params = list(agent.value_net.parameters())
    
    if not args.freeze_visual_encoder and include_rgb:
        policy_params += list(visual_encoder.parameters())
    
    policy_optimizer = optim.AdamW(policy_params, lr=args.lr)
    critic_optimizer = optim.AdamW(critic_params, lr=args.lr_critic)
    
    # Create PPO rollout buffer (following official PPO implementation)
    buffer = RolloutBufferPPO(
        num_steps=args.rollout_steps,
        num_envs=args.num_envs,
        obs_dim=obs_dim,
        pred_horizon=args.pred_horizon,
        act_dim=act_dim,
        num_inference_steps=args.num_inference_steps,  # For x_chain storage
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        normalize_advantages=args.normalize_advantages,
        device=str(device),
    )
    
    # Create agent wrapper for evaluation (aligns with train_offline_rl.py)
    agent_wrapper = AgentWrapper(
        agent, visual_encoder, include_rgb, 
        args.obs_horizon, args.act_horizon
    ).to(device)
    
    # Local encode_observations function using the public utility from utils.py
    def local_encode_observations(obs_seq):
        """Encode observation sequence to get conditioning features.
        
        Args:
            obs_seq: dict with 'state' and optionally 'rgb'
                    shapes: state [B, T, state_dim], rgb [B, T, H, W, C]
        
        Returns:
            obs_cond: [B, obs_horizon * (visual_dim + state_dim)]
        """
        # Use the public encode_observations from utils.py
        # It handles NCHW auto-detection and RGB normalization
        obs_features = encode_observations(obs_seq, visual_encoder, include_rgb, device)
        
        # Flatten to [B, T * (visual_dim + state_dim)]
        B = obs_seq["state"].shape[0]
        obs_cond = obs_features.reshape(B, -1)
        
        return obs_cond
    
    # Training statistics
    reward_normalizer = RewardNormalizer(gamma=args.gamma)
    return_normalizer = ReturnNormalizer()  # NEW: For return normalization
    global_step = 0
    num_updates = 0
    start_time = time.time()
    
    timings = defaultdict(float)
    best_eval_metrics = defaultdict(float)
    
    # Episode tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_successes = deque(maxlen=100)
    
    # Observation stacker using common utility
    obs, _ = train_envs.reset()
    obs_stacker = ObservationStacker(args.obs_horizon)
    obs_stacker.reset(obs)
    
    # Per-env episode tracking
    env_episode_rewards = np.zeros(args.num_envs)
    env_episode_lengths = np.zeros(args.num_envs, dtype=int)
    
    # Calculate expected number of updates
    timesteps_per_update = args.num_envs * args.act_horizon * args.rollout_steps
    total_timesteps = args.total_updates * timesteps_per_update
    
    print("\nStarting training...")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Timesteps per update: {timesteps_per_update} ({args.num_envs} envs × {args.act_horizon} act × {args.rollout_steps} rollout)")
    print(f"Expected updates: {args.total_updates}")
    print(f"Log every {args.log_freq} updates, Eval every {args.eval_freq} updates, Save every {args.save_freq} updates")
    print(f"Critic warmup steps: {args.critic_warmup_steps}")
    print(f"Noise schedule: {args.noise_decay_type}, {args.max_noise_std} -> {args.min_noise_std}")
    
    # Log configuration and model info to tensorboard and wandb
    config_text = f"""
    ## Training Configuration
    - Algorithm: ReinFlow (Online RL Fine-tuning)
    - Environment: {args.env_id}
    - Observation Mode: {args.obs_mode}
    - Simulation Backend: {args.sim_backend}
    
    ## Model Configuration
    - Total Parameters: {(total_params + visual_params) / 1e6:.2f}M
    - Agent Parameters: {total_params / 1e6:.2f}M
    - Visual Encoder Parameters: {visual_params / 1e6:.2f}M
    - Frozen Visual Encoder: {args.freeze_visual_encoder}
    
    ## Training Settings
    - Total Timesteps: {total_timesteps}
    - Total Updates: {args.total_updates}
    - Batch Size (per env): {args.num_envs}
    - Rollout Steps: {args.rollout_steps}
    - PPO Epochs: {args.ppo_epochs}
    - Minibatch Size: {args.minibatch_size}
    - Learning Rate: {args.lr}
    - Critic Learning Rate: {args.lr_critic}
    
    ## PPO Hyperparameters
    - Gamma (discount): {args.gamma}
    - GAE Lambda: {args.gae_lambda}
    - Clip Ratio: {args.clip_ratio}
    - Entropy Coefficient: {args.entropy_coef}
    - Value Coefficient: {args.value_coef}
    - Max Grad Norm: {args.max_grad_norm}
    
    ## Exploration Settings
    - Noise Schedule: {args.noise_decay_type}
    - Max Noise Std: {args.max_noise_std}
    - Min Noise Std: {args.min_noise_std}
    - Critic Warmup Steps: {args.critic_warmup_steps}
    """
    writer.add_text("config/training", config_text)
    if args.track:
        wandb.log({"config/training": wandb.Html(config_text.replace("\n", "<br>"))})
    
    # Track done state for GAE (following ppo_rgb.py)
    next_done = torch.zeros(args.num_envs, device=device)
    
    pbar = tqdm(total=args.total_updates, desc="Training")
    
    while num_updates < args.total_updates:
        # Collect rollouts
        buffer.reset()
        
        for rollout_step in range(args.rollout_steps):
            # Get stacked observations
            obs_seq = obs_stacker.get_stacked()
            
            # Encode observations using unified function
            with torch.no_grad():
                obs_cond = local_encode_observations(obs_seq)
                
                # Get action with exploration
                # x_chain is (B, K+1, H, A) complete denoising trajectory
                actions, x_chain = agent.get_action(
                    obs_cond, 
                    deterministic=False, 
                    use_ema=False,
                    return_chains=True,
                )
                
                # Compute value and log_prob for PPO
                # Use target value network for more stable GAE computation when enabled
                values = agent.compute_value(obs_cond, use_target=args.use_target_value_net)
                # old_log_probs are computed here and stored in buffer
                # They will be compared with new_log_probs in PPO update
                log_probs, _ = agent.compute_action_log_prob(
                    obs_cond, actions, x_chain=x_chain
                )
                
                # # === DEBUG: Log rollout statistics (first step only) ===
                # if rollout_step == 0 and num_updates % args.log_freq == 0:
                #     print(f"\n[DEBUG Rollout] Update {num_updates}")
                #     print(f"  x_chain shape: {x_chain.shape}")
                #     print(f"  x_chain[0] (x_0) stats: mean={x_chain[:, 0].mean():.4f}, std={x_chain[:, 0].std():.4f}")
                #     print(f"  x_chain[-1] (x_K) stats: mean={x_chain[:, -1].mean():.4f}, std={x_chain[:, -1].std():.4f}")
                #     # Check diff between consecutive x_chain states
                #     diffs = []
                #     for k in range(x_chain.shape[1] - 1):
                #         diff = (x_chain[:, k+1] - x_chain[:, k]).abs().mean().item()
                #         diffs.append(diff)
                #     print(f"  x_chain step diffs (mean abs): {[f'{d:.4f}' for d in diffs]}")
                #     print(f"  log_probs: mean={log_probs.mean():.4f}, std={log_probs.std():.4f}, min={log_probs.min():.4f}, max={log_probs.max():.4f}")
                #     print(f"  values: mean={values.mean():.4f}, std={values.std():.4f}")
            
            # Execute action chunk (SMDP)
            chunk_rewards = []
            chunk_dones = []
            
            for step_idx in range(args.act_horizon):
                action = actions[:, step_idx, :]
                next_obs, rewards, terminations, truncations, infos = train_envs.step(action)
                
                dones = terminations | truncations
                
                chunk_rewards.append(rewards)
                chunk_dones.append(dones)
                
                # Update per-env tracking
                for i in range(args.num_envs):
                    env_episode_rewards[i] += rewards[i].item()
                    env_episode_lengths[i] += 1
                    
                    if dones[i]:
                        episode_rewards.append(env_episode_rewards[i])
                        episode_lengths.append(env_episode_lengths[i])
                        
                        if "success" in infos:
                            success = infos["success"][i].item() if hasattr(infos["success"][i], "item") else infos["success"][i]
                        else:
                            success = rewards[i].item() > 0
                        episode_successes.append(float(success))
                        
                        # Reset tracking
                        env_episode_rewards[i] = 0.0
                        env_episode_lengths[i] = 0
                
                obs_stacker.append(next_obs)
                obs = next_obs
                global_step += args.num_envs
                
                # Check if any episode ended - break to handle episode boundary
                if dones.any():
                    break
            
            # Compute SMDP rewards (vectorized)
            chunk_rewards_tensor = torch.stack(chunk_rewards)  # (chunk_len, num_envs)
            chunk_dones_tensor = torch.stack(chunk_dones)      # (chunk_len, num_envs)
            chunk_len = len(chunk_rewards)
            
            # Cumulative discounted reward for each env
            cum_rewards = torch.zeros(args.num_envs, device=device)
            chunk_done_flags = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            discount = 1.0
            
            for i in range(chunk_len):
                # Only add reward if episode hasn't ended yet for this env
                still_running = ~chunk_done_flags
                cum_rewards = cum_rewards + discount * chunk_rewards_tensor[i] * still_running.float()
                
                # Update done flags
                chunk_done_flags = chunk_done_flags | chunk_dones_tensor[i]
                discount *= args.gamma
            
            # === NEW: Apply reward scaling (like AWCP's reward_scale) ===
            # This is critical for stabilizing critic training
            cum_rewards = cum_rewards * args.reward_scale
            
            # Optional: normalize rewards (additional normalization on top of scaling)
            if args.normalize_rewards:
                reward_normalizer.update(cum_rewards.cpu().numpy())
                # Vectorized normalization instead of Python loop
                cum_rewards = cum_rewards / (reward_normalizer.rms.std + 1e-8)
            
            # Add to buffer (vectorized)
            # x_chain is essential for accurate log_prob computation in PPO updates
            # Store log_probs computed during rollout as the "old" log probs for PPO ratio
            buffer.add(
                obs=obs_cond,
                actions=actions,
                rewards=cum_rewards,
                values=values,
                log_probs=log_probs,  # old_log_probs for PPO ratio computation
                dones=chunk_done_flags.float(),
                x_chain=x_chain,  # (num_envs, K+1, pred_horizon, act_dim)
            )
            
            # Handle final values for episodes that ended (following ppo_rgb.py)
            if chunk_done_flags.any():
                done_mask = chunk_done_flags
                # Get value estimate for the new state after reset
                with torch.no_grad():
                    # The obs after episode end is already the new episode's first obs
                    final_obs_seq = obs_stacker.get_stacked()
                    final_obs_cond = local_encode_observations(final_obs_seq)
                    # Use target value network for bootstrap value when enabled
                    final_vals = agent.compute_value(final_obs_cond[done_mask], use_target=args.use_target_value_net)
                buffer.set_final_values(rollout_step, done_mask, final_vals)
            
            # Update next_done for GAE
            next_done = chunk_done_flags.float()
        
        # Compute returns and advantages
        with torch.no_grad():
            # Get last value for bootstrapping
            obs_seq = obs_stacker.get_stacked()
            
            obs_cond = local_encode_observations(obs_seq)
            # Use target value network for bootstrap value when enabled
            next_value = agent.compute_value(obs_cond, use_target=args.use_target_value_net).squeeze(-1)
        
        # Vectorized GAE computation (following ppo_rgb.py)
        # Note: advantage normalization is handled inside buffer if enabled
        buffer.compute_returns_and_advantages(
            next_value=next_value,
            next_done=next_done,
        )
        
        # === NEW: Optional return normalization for stable critic training ===
        if args.normalize_returns:
            # Update return statistics and normalize
            returns_np = buffer.returns.cpu().numpy().flatten()
            return_normalizer.update(returns_np)
            # Normalize returns in buffer (in-place)
            buffer.returns = buffer.returns / (return_normalizer.std + 1e-8)
        
        # PPO update
        last_tick = time.time()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_policy_grad_norm = 0.0
        total_critic_grad_norm = 0.0
        total_kl_div = 0.0
        num_batches = 0
        kl_early_stopped = False  # NEW: Track if we early stopped
        
        for epoch in range(args.ppo_epochs):
            # === NEW: Check for KL early stopping at epoch level ===
            if kl_early_stopped:
                break
                
            batches = buffer.get_batches(args.minibatch_size, shuffle=True)
            
            for batch in batches:
                # Forward pass with value clipping to prevent vloss explosion
                # x_chain is essential for accurate log_prob computation
                loss_dict = agent.compute_ppo_loss(
                    obs_cond=batch["obs"],
                    actions=batch["actions"],
                    old_log_probs=batch["log_probs"],
                    advantages=batch["advantages"],
                    returns=batch["returns"],
                    x_chain=batch["x_chain"],  # (B, K+1, H, A) denoising trajectory
                    old_values=batch.get("values"),  # For value clipping
                    clip_value=args.clip_value_loss,
                    value_clip_range=args.value_clip_range,
                )
                
                # === DEBUG: Log PPO statistics (first batch of first epoch only) ===
                # if epoch == 0 and num_batches == 0 and num_updates % args.log_freq == 0:
                #     print(f"\n[DEBUG PPO] Update {num_updates}, Epoch {epoch}, Batch 0")
                #     print(f"  Batch size: {batch['obs'].shape[0]}")
                #     print(f"  old_log_probs (from buffer): mean={batch['log_probs'].mean():.4f}, std={batch['log_probs'].std():.4f}")
                #     print(f"  new_log_probs (recomputed): mean={loss_dict['new_log_probs_mean']:.4f}, std={loss_dict['new_log_probs_std']:.4f}")
                #     print(f"  log_ratio (new - old): mean={loss_dict['log_ratio_mean']:.6f}, std={loss_dict['log_ratio_std']:.6f}")
                #     print(f"  ratio: mean={loss_dict['ratio_mean']:.6f}, std={loss_dict['ratio_std']:.6f}, min={loss_dict['ratio_min']:.6f}, max={loss_dict['ratio_max']:.6f}")
                #     print(f"  advantages: mean={batch['advantages'].mean():.4f}, std={batch['advantages'].std():.4f}")
                #     print(f"  returns: mean={batch['returns'].mean():.4f}, std={batch['returns'].std():.4f}")
                #     print(f"  policy_loss: {loss_dict['policy_loss']:.8f}")
                #     print(f"  value_loss: {loss_dict['value_loss']:.6f}")
                #     print(f"  entropy: {loss_dict['entropy']:.4f}")
                #     print(f"  approx_kl: {loss_dict['approx_kl']:.8f}")
                                
                # Backward pass
                if num_updates < args.critic_warmup_steps:
                    # During warmup, only update critic
                    critic_optimizer.zero_grad()
                    loss_dict["loss"].backward()
                    critic_grad_norm = nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                    critic_optimizer.step()
                    total_critic_grad_norm += critic_grad_norm
                else:
                    # Full PPO update
                    policy_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    loss_dict["loss"].backward()
                    
                    # === DEBUG: Check gradients before clipping (first batch of first epoch only) ===
                    if epoch == 0 and num_batches == 0 and num_updates % args.log_freq == 0:
                        # Check velocity net gradients
                        velocity_grad_norm = 0.0
                        noise_net_grad_norm = 0.0
                        for name, param in agent.noisy_velocity_net.named_parameters():
                            if param.grad is not None:
                                if 'explore_noise_net' in name:
                                    noise_net_grad_norm += param.grad.norm().item() ** 2
                                else:
                                    velocity_grad_norm += param.grad.norm().item() ** 2
                        velocity_grad_norm = velocity_grad_norm ** 0.5
                        noise_net_grad_norm = noise_net_grad_norm ** 0.5
                        print(f"  [GRAD] velocity_net grad norm: {velocity_grad_norm:.6f}")
                        print(f"  [GRAD] explore_noise_net grad norm: {noise_net_grad_norm:.6f}")
                    
                    policy_grad_norm = nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm)
                    critic_grad_norm = nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                    policy_optimizer.step()
                    critic_optimizer.step()
                    total_policy_grad_norm += policy_grad_norm
                    total_critic_grad_norm += critic_grad_norm
                
                # Accumulate statistics (MUST happen before KL early stop check)
                total_policy_loss += loss_dict["policy_loss"].item()
                total_value_loss += loss_dict["value_loss"].item()
                total_entropy += loss_dict["entropy"].item()
                if "approx_kl" in loss_dict:
                    kl_val = loss_dict["approx_kl"]
                    if isinstance(kl_val, torch.Tensor):
                        kl_val = kl_val.item()
                    total_kl_div += kl_val
                num_batches += 1
                
                # Free memory after each batch to prevent accumulation
                # del loss_dict
                
                # === KL early stopping check (after stats accumulation) ===
                if args.kl_early_stop and args.target_kl is not None:
                    if kl_val > args.target_kl * 1.5:  # Allow some margin
                        kl_early_stopped = True
                        break
            
            # Clear CUDA cache after each epoch to prevent memory buildup
            # del batches
            # torch.cuda.empty_cache()
        
        # === DEBUG: Log rollout utilization stats ===
        actual_epochs = epoch if not kl_early_stopped else epoch + 1  # epoch is 0-indexed
        if num_updates % args.log_freq == 0:
            early_stop_msg = " (KL early stopped)" if kl_early_stopped else ""
            print(f"  [ROLLOUT] Completed {actual_epochs}/{args.ppo_epochs} epochs, {num_batches} minibatch updates{early_stop_msg}")
        
        # Update EMA
        agent.update_ema()
        # Update noise schedule
        agent.update_noise_schedule()
        # === NEW: Update target value network (like AWCP's update_target()) ===
        agent.update_target_value_net()
        num_updates += 1
        
        # Timing for PPO update
        timings["forward"] += time.time() - last_tick
        
        # Compute average losses for this update (always compute for pbar)
        avg_policy_loss = total_policy_loss / max(1, num_batches)
        avg_value_loss = total_value_loss / max(1, num_batches)
        avg_entropy = total_entropy / max(1, num_batches)
        avg_policy_grad_norm = total_policy_grad_norm / max(1, num_batches)
        avg_critic_grad_norm = total_critic_grad_norm / max(1, num_batches)
        avg_kl_div = total_kl_div / max(1, num_batches) if total_kl_div > 0 else 0.0
        min_noise, max_noise = agent.get_current_noise_std()
        
        # Prepare losses dict for pbar display (always available)
        losses_dict = {
            "ploss": avg_policy_loss,
            "vloss": avg_value_loss,
            "ent": avg_entropy,
            "noise": max_noise,
            "episode_reward": np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0,
            "success_rate": np.mean(episode_successes) if len(episode_successes) > 0 else 0.0,
            "global_step": global_step,
        }
        if num_updates < args.critic_warmup_steps:
            losses_dict["warmup"] = 1.0
        
        # Logging (periodic)
        if num_updates % args.log_freq == 0 and num_updates > 0:
            last_tick = time.time()
            
            # Prepare logging dict
            log_dict = {
                "charts/learning_rate": policy_optimizer.param_groups[0]["lr"],
                "charts/critic_learning_rate": critic_optimizer.param_groups[0]["lr"],
                "losses/policy_loss": avg_policy_loss,
                "losses/value_loss": avg_value_loss,
                "losses/entropy": avg_entropy,
                "losses/noise_std_max": max_noise,
                "losses/noise_std_min": min_noise,
                "losses/in_warmup": float(bool(num_updates < args.critic_warmup_steps)),
                "optimizer/policy_grad_norm": avg_policy_grad_norm,
                "optimizer/critic_grad_norm": avg_critic_grad_norm,
                # === NEW: Add critic stability diagnostics ===
                "critic/return_normalizer_std": return_normalizer.std if args.normalize_returns else 1.0,
                "critic/reward_scale": args.reward_scale,
                # === NEW: Add rollout utilization stats ===
                "training/actual_epochs": actual_epochs,
                "training/minibatch_updates": num_batches,
                "training/kl_early_stopped": float(kl_early_stopped),
            }
            
            # Add KL divergence if available
            if avg_kl_div > 0:
                log_dict["losses/kl_div"] = avg_kl_div
            
            # Add timings
            for k, v in timings.items():
                log_dict[f"time/{k}"] = v
            
            # Add episode statistics
            if len(episode_rewards) > 0:
                log_dict["charts/episode_reward"] = np.mean(episode_rewards)
                log_dict["charts/episode_length"] = np.mean(episode_lengths)
                log_dict["charts/success_rate"] = np.mean(episode_successes)
                log_dict["charts/episode_reward_std"] = np.std(episode_rewards) if len(episode_rewards) > 1 else 0
            
            # Add global step and num updates
            log_dict["training/global_step"] = global_step
            log_dict["training/num_updates"] = num_updates
            log_dict["training/num_epochs"] = num_updates * args.ppo_epochs
            
            # Log to tensorboard
            for k, v in log_dict.items():
                writer.add_scalar(k, v, num_updates)
            
            # Log to wandb
            if args.track:
                wandb.log(log_dict, step=num_updates)
            
            timings["logging"] += time.time() - last_tick
        
        # Evaluation (using official evaluate function, aligned with train_offline_rl.py)
        if num_updates % args.eval_freq == 0:
            last_tick = time.time()
            # ReinFlow-specific: use deterministic=True, use_ema=True for evaluation
            eval_metrics = evaluate(
                args.num_eval_episodes, 
                agent_wrapper, 
                eval_envs, 
                device, 
                args.sim_backend,
                agent_kwargs={"deterministic": True, "use_ema": True},
            )
            timings["eval"] += time.time() - last_tick
            
            print(f"\n[Update {num_updates}] Evaluated {len(eval_metrics['success_at_end'])} episodes in {time.time() - last_tick:.2f}s")
            
            # Average metrics and log (aligned with offline format)
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], num_updates)
                print(f"  {k}: {eval_metrics[k]:.4f}")
            
            # Track best metrics for checkpointing
            for k in ["success_once", "success_at_end"]:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    checkpoint_path = f"{log_dir}/checkpoint_best_eval_{k}.pt"
                    checkpoint = {
                        "agent": agent.state_dict(),
                        "visual_encoder": visual_encoder.state_dict() if include_rgb else None,
                        "config": vars(args),
                    }
                    torch.save(checkpoint, checkpoint_path)
                    print(f"  ✓ New best {k}: {eval_metrics[k]:.4f}. Saved checkpoint.")
            
            if args.track:
                eval_log = {f"eval/{k}": v for k, v in eval_metrics.items()}
                # Add best metrics to wandb
                eval_log["eval/best_success_once"] = best_eval_metrics["success_once"]
                eval_log["eval/best_success_at_end"] = best_eval_metrics["success_at_end"]
                eval_log["training/eval_timestep"] = global_step
                wandb.log(eval_log, step=num_updates)
        
        # Save checkpoint (periodic)
        if num_updates % args.save_freq == 0 and num_updates > 0:
            checkpoint_path = f"{log_dir}/checkpoint_{num_updates}.pt"
            checkpoint = {
                "agent": agent.state_dict(),
                "visual_encoder": visual_encoder.state_dict() if include_rgb else None,
                "policy_optimizer": policy_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "global_step": global_step,
                "num_updates": num_updates,
                "config": vars(args),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Update progress bar with losses (aligned with offline format)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in losses_dict.items()})
        pbar.update(1)
    
    pbar.close()
    
    # Final save
    final_path = f"{log_dir}/final_model.pt"
    checkpoint = {
        "agent": agent.state_dict(),
        "visual_encoder": visual_encoder.state_dict() if include_rgb else None,
        "config": vars(args),
    }
    torch.save(checkpoint, final_path)
    print(f"Final model saved to {final_path}")
    
    # Training summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total training time: {total_time / 3600:.2f} hours ({total_time / 60:.1f} minutes)")
    print(f"Total updates completed: {num_updates}")
    print(f"Total timesteps collected: {global_step}")
    print(f"Best eval success_once: {best_eval_metrics['success_once']:.4f}")
    print(f"Best eval success_at_end: {best_eval_metrics['success_at_end']:.4f}")
    print("\nTiming breakdown:")
    for k, v in sorted(timings.items(), key=lambda x: -x[1]):
        pct = 100 * v / total_time if total_time > 0 else 0
        print(f"  {k}: {v / 60:.1f} min ({pct:.1f}%)")
    print("="*60)
    
    # Log final summary to wandb
    if args.track:
        final_summary = {
            "training/total_time_hours": total_time / 3600,
            "training/total_updates": num_updates,
            "training/total_timesteps": global_step,
            "eval/final_success_once": best_eval_metrics["success_once"],
            "eval/final_success_at_end": best_eval_metrics["success_at_end"],
        }
        for k, v in timings.items():
            final_summary[f"timing/{k}_minutes"] = v / 60
        wandb.log(final_summary)
    
    # Cleanup
    train_envs.close()
    eval_envs.close()
    writer.close()
    
    if args.track:
        wandb.finish()
    
    print("Training complete!")


if __name__ == "__main__":
    main()
