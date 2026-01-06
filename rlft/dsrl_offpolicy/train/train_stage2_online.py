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


# ==================== Sanity Check: Prior vs Policy ====================

def sanity_check_latent_policy(
    agent: DSRLSACAgent,
    eval_envs,
    visual_encoder: Optional[nn.Module],
    include_rgb: bool,
    obs_horizon: int,
    act_horizon: int,
    device: torch.device,
    num_candidates: int = 64,
    num_episodes: int = 10,
) -> Dict[str, float]:
    """
    Sanity check: Compare prior vs policy sampling via real environment rollouts.
    
    For each episode initial state:
    1. Sample M latents from prior (TanhNormal with N(0,I) pre-tanh)
    2. Sample M latents from current policy
    3. Execute best-of-M for each, record episode returns
    
    This checks if the policy learned useful latent steering.
    
    Args:
        agent: DSRL SAC agent
        eval_envs: Evaluation environments (num_envs should be >= num_candidates)
        visual_encoder: Optional visual encoder
        include_rgb: Whether to include RGB observations
        obs_horizon: Observation horizon
        act_horizon: Action execution horizon
        device: Torch device
        num_candidates: M candidates to sample (default 64)
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dict with comparison metrics:
        - prior_best_return: Best return from prior sampling
        - prior_mean_return: Mean return from prior sampling
        - policy_best_return: Best return from policy sampling
        - policy_mean_return: Mean return from policy sampling
        - policy_advantage: policy_best - prior_best
    """
    agent.eval()
    if visual_encoder is not None:
        visual_encoder.eval()
    
    num_envs = eval_envs.num_envs
    # Use all available envs, up to num_candidates
    M = min(num_candidates, num_envs)
    
    prior_returns_all = []
    policy_returns_all = []
    
    with torch.no_grad():
        for ep in range(num_episodes):
            # ===== Test Prior Sampling =====
            # Reset environments and get initial observation
            obs, info = eval_envs.reset()
            
            # Build observation features directly (no ObservationStacker needed for single-step)
            # For obs_horizon > 1, we replicate the initial obs
            obs_features = _build_obs_features(
                obs, visual_encoder, include_rgb, obs_horizon, M, device
            )
            
            # Sample from Prior: N(0, I) in pre-tanh space, then tanh squash
            prior_w_pre = torch.randn(M, agent.pred_horizon, agent.action_dim, device=device)
            prior_w = torch.tanh(prior_w_pre) * agent.action_magnitude
            
            # Generate actions from prior latents
            prior_actions = agent.sample_actions_from_latent(obs_features, prior_w, use_ema=True)
            
            # Rollout with Prior Actions
            prior_returns = _rollout_single_chunk(
                eval_envs, prior_actions, act_horizon, M
            )
            prior_returns_all.extend(prior_returns)
            
            # ===== Test Policy Sampling =====
            # Reset environments again (different initial states, but that's fine for comparison)
            obs, info = eval_envs.reset()
            
            # Build observation features
            obs_features = _build_obs_features(
                obs, visual_encoder, include_rgb, obs_horizon, M, device
            )
            
            # Sample from Policy
            policy_w, _ = agent.latent_policy.sample(obs_features, deterministic=False)
            policy_actions = agent.sample_actions_from_latent(obs_features, policy_w, use_ema=True)
            
            # Rollout with Policy Actions
            policy_returns = _rollout_single_chunk(
                eval_envs, policy_actions, act_horizon, M
            )
            policy_returns_all.extend(policy_returns)
    
    prior_returns_all = np.array(prior_returns_all)
    policy_returns_all = np.array(policy_returns_all)
    
    # Compute metrics per episode group
    prior_best = np.max(prior_returns_all)
    prior_mean = np.mean(prior_returns_all)
    prior_std = np.std(prior_returns_all)
    
    policy_best = np.max(policy_returns_all)
    policy_mean = np.mean(policy_returns_all)
    policy_std = np.std(policy_returns_all)
    
    # Per-episode best comparison
    M_actual = min(num_candidates, num_envs)
    prior_returns_grouped = prior_returns_all.reshape(num_episodes, M_actual)
    policy_returns_grouped = policy_returns_all.reshape(num_episodes, M_actual)
    
    prior_best_per_ep = prior_returns_grouped.max(axis=1).mean()
    policy_best_per_ep = policy_returns_grouped.max(axis=1).mean()
    
    agent.train()
    if visual_encoder is not None:
        visual_encoder.train()
    
    return {
        "prior_best_return": prior_best,
        "prior_mean_return": prior_mean,
        "prior_std_return": prior_std,
        "policy_best_return": policy_best,
        "policy_mean_return": policy_mean,
        "policy_std_return": policy_std,
        "prior_best_per_ep": prior_best_per_ep,
        "policy_best_per_ep": policy_best_per_ep,
        "policy_advantage": policy_best_per_ep - prior_best_per_ep,
        "policy_mean_advantage": policy_mean - prior_mean,
    }


def _build_obs_features(
    obs: Dict[str, torch.Tensor],
    visual_encoder: Optional[nn.Module],
    include_rgb: bool,
    obs_horizon: int,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    """Build observation features from raw environment observation.
    
    For single-step observations, replicates along time dimension to fill obs_horizon.
    
    Args:
        obs: Raw observation dict from env.reset() or env.step()
        visual_encoder: Visual encoder for RGB
        include_rgb: Whether to include RGB features
        obs_horizon: Number of observation frames expected
        num_envs: Number of environments to use
        device: Torch device
        
    Returns:
        obs_features: (num_envs, obs_horizon * feature_dim)
    """
    # Slice to num_envs
    obs_sliced = {k: v[:num_envs] for k, v in obs.items()}
    
    # Convert to tensors on device
    obs_tensor = {}
    for k, v in obs_sliced.items():
        if torch.is_tensor(v):
            obs_tensor[k] = v.float().to(device)
        else:
            obs_tensor[k] = torch.from_numpy(v).float().to(device)
    
    # Handle state: add time dimension if needed and replicate
    state = obs_tensor["state"]
    if state.dim() == 2:  # (B, state_dim) - no time dimension
        # Replicate to (B, T, state_dim)
        state = state.unsqueeze(1).expand(-1, obs_horizon, -1).contiguous()
    obs_tensor["state"] = state
    
    # Handle rgb: add time dimension if needed and replicate
    if include_rgb and "rgb" in obs_tensor:
        rgb = obs_tensor["rgb"]
        if rgb.dim() == 4:  # (B, H, W, C) - no time dimension
            # Replicate to (B, T, H, W, C)
            rgb = rgb.unsqueeze(1).expand(-1, obs_horizon, -1, -1, -1).contiguous()
        obs_tensor["rgb"] = rgb
    
    # Use encode_observations
    return encode_observations(obs_tensor, visual_encoder, include_rgb, device)


def _rollout_single_chunk(
    envs,
    actions: torch.Tensor,
    act_horizon: int,
    num_envs: int,
) -> List[float]:
    """Execute a single action chunk and return episode returns.
    
    Runs action chunk to completion or episode termination.
    
    Args:
        envs: Environment
        actions: (B, pred_horizon, action_dim) action sequence
        act_horizon: Number of actions to execute
        num_envs: Number of environments
        
    Returns:
        List of returns for each environment
    """
    returns = np.zeros(num_envs)
    action_chunk = actions[:num_envs, :act_horizon, :].cpu().numpy()
    
    for step_idx in range(act_horizon):
        action = action_chunk[:, step_idx, :]
        obs, reward, terminated, truncated, info = envs.step(action)
        
        reward_np = reward.cpu().numpy() if torch.is_tensor(reward) else reward
        returns[:num_envs] += reward_np[:num_envs]
        
        done = terminated | truncated
        if done.any():
            break
    
    return returns.tolist()


def _rollout_actions(
    envs,
    obs_stacker,
    actions: torch.Tensor,
    act_horizon: int,
    num_envs: int,
    device: torch.device,
) -> List[float]:
    """Execute a single action chunk and return episode returns.
    
    Runs action chunk to completion or episode termination.
    """
    returns = np.zeros(num_envs)
    action_chunk = actions[:num_envs, :act_horizon, :].cpu().numpy()
    
    for step_idx in range(act_horizon):
        action = action_chunk[:, step_idx, :]
        obs, reward, terminated, truncated, info = envs.step(action)
        
        reward_np = reward.cpu().numpy() if torch.is_tensor(reward) else reward
        returns[:num_envs] += reward_np[:num_envs]
        
        done = terminated | truncated
        if done.any():
            break
    
    return returns.tolist()


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
    num_eval_envs: int = 64
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
    awsc_checkpoint: str = "/home/amax/rl-vla/rlft/dsrl_offpolicy/checkpoints/best_eval_success_once.pt"
    """path to pretrained AW-ShortCut Flow checkpoint (velocity_net and q_network)"""
    stage1_checkpoint: str = "/home/amax/rl-vla/rlft/runs/dsrl_offpolicy_stage1-LiftPegUpright-v1-seed1__1767531955/checkpoints/best_eval_success_at_end.pt"
    """path to Stage 1 latent policy checkpoint"""
    use_ema: bool = True
    """use EMA weights from AWSC checkpoint (recommended for better performance)"""
    
    # Training settings
    total_timesteps: int = 1_000_000
    """total environment steps"""
    warmup_steps: int = 20000
    """steps before starting updates (fill replay buffer). 
    NOTE: Must be large enough to fill buffer with batch_size samples.
    With num_envs=50, act_horizon=8, batch_size=10240: need ~82000 steps minimum.
    Set to 100000 for safety margin."""
    eval_freq: int = 20000
    """evaluation frequency (env steps)"""
    save_freq: int = 20000
    """checkpoint save frequency (env steps)"""
    log_freq: int = 1
    """logging frequency (env steps)"""
    num_eval_episodes: int = 64
    """number of evaluation episodes"""
    
    # Off-policy specific settings
    utd_ratio: int = 20
    """Update-to-data ratio (gradient steps per env step)"""
    batch_size: int = 1024
    """minibatch size for SAC updates"""
    replay_buffer_size: int = 500000
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
    actor_update_delay: int = 100000
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
    init_temperature: float = 0.0
    """initial SAC temperature"""
    learnable_temp: bool = False
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
    
    # Sanity check settings
    sanity_check: bool = True
    """whether to run prior vs policy sanity check during evaluation"""
    sanity_check_candidates: int = 64
    """number of latent candidates to sample for sanity check (M)"""
    sanity_check_episodes: int = 5
    """number of episodes for sanity check (per prior and policy)"""
    
    # Q sensitivity diagnostic settings
    q_sensitivity_freq: int = 5000
    """frequency to run Q sensitivity diagnostic (env steps)"""
    q_sensitivity_samples: int = 64
    """number of latent samples per observation for Q sensitivity diagnostic"""


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
    
    # Compute minimum steps needed to fill buffer
    steps_per_macro = args.num_envs * args.act_horizon  # 50 * 8 = 400
    macro_steps_for_batch = (args.batch_size + args.num_envs - 1) // args.num_envs  # ceil division
    min_steps_for_buffer = macro_steps_for_batch * steps_per_macro
    
    print("\n" + "=" * 50)
    print("Starting Stage 2 SAC training (Off-Policy)...")
    print(f"UTD ratio: {args.utd_ratio}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Min steps to fill buffer (batch_size samples): ~{min_steps_for_buffer}")
    print(f"Critic updates start at: max(warmup_steps, buffer_ready) = ~{max(args.warmup_steps, min_steps_for_buffer)}")
    print(f"Actor updates start at: warmup_steps + actor_update_delay = {args.warmup_steps + args.actor_update_delay}")
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
                        "gaussian_entropy": torch.tensor(0.0),
                        "kl_to_prior": torch.tensor(0.0),
                        "q_policy": torch.tensor(0.0),
                        "log_std_mean": torch.tensor(0.0),
                        "std_mean": torch.tensor(0.0),
                        "latent_std": torch.tensor(0.0),
                        "latent_abs_mean": torch.tensor(0.0),
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
                # Entropy metrics
                metrics_accum["entropy"].append(actor_metrics["entropy"].item())
                metrics_accum["gaussian_entropy"].append(actor_metrics["gaussian_entropy"].item())
                metrics_accum["kl_to_prior"].append(actor_metrics["kl_to_prior"].item())
                # Policy stats
                metrics_accum["q_policy"].append(actor_metrics["q_policy"].item())
                metrics_accum["log_std_mean"].append(actor_metrics["log_std_mean"].item())
                metrics_accum["std_mean"].append(actor_metrics["std_mean"].item())
                metrics_accum["latent_std"].append(actor_metrics["latent_std"].item())
                metrics_accum["latent_abs_mean"].append(actor_metrics["latent_abs_mean"].item())
                # Temperature
                metrics_accum["alpha"].append(temp_metrics["alpha"].item() if hasattr(temp_metrics["alpha"], "item") else temp_metrics["alpha"])
        
        # ===== Logging =====
        if total_steps % args.log_freq < args.num_envs * args.act_horizon and len(metrics_accum["critic_loss"]) > 0:
            train_log = {
                # Critic metrics
                "train/critic_loss": np.mean(metrics_accum["critic_loss"]),
                "train/td_error": np.mean(metrics_accum["td_error"]),
                "train/q_mean": np.mean(metrics_accum["q_mean"]),
                # Actor metrics
                "train/actor_loss": np.mean(metrics_accum["actor_loss"]),
                "train/q_policy": np.mean(metrics_accum["q_policy"]),
                # Entropy metrics
                "train/entropy": np.mean(metrics_accum["entropy"]),  # Sample-based
                "train/gaussian_entropy": np.mean(metrics_accum["gaussian_entropy"]),  # Analytic (pre-tanh)
                "train/kl_to_prior": np.mean(metrics_accum["kl_to_prior"]),
                # Policy distribution stats
                "train/log_std_mean": np.mean(metrics_accum["log_std_mean"]),
                "train/std_mean": np.mean(metrics_accum["std_mean"]),
                # Latent stats
                "train/latent_std": np.mean(metrics_accum["latent_std"]),
                "train/latent_abs_mean": np.mean(metrics_accum["latent_abs_mean"]),
                # Temperature
                "train/alpha": np.mean(metrics_accum["alpha"]),
                # Buffer stats
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
        
        # ===== Q Sensitivity Diagnostic =====
        if total_steps % args.q_sensitivity_freq < args.num_envs * args.act_horizon and replay_buffer.is_ready(args.batch_size):
            # Use a batch from replay buffer for diagnostic
            diag_batch = replay_buffer.sample(min(args.batch_size, 256))
            q_sens_metrics = agent.diagnose_q_sensitivity(
                obs_cond=diag_batch["obs_cond"],
                num_samples=args.q_sensitivity_samples,
            )
            
            print(f"\n[Step {total_steps}] Q Sensitivity Diagnostic:")
            print(f"  Prior:  Q_std={q_sens_metrics['prior_q_std']:.4f}, Q_range={q_sens_metrics['prior_q_range']:.4f}, top1-median={q_sens_metrics['prior_q_top1_vs_median']:.4f}")
            print(f"  Policy: Q_std={q_sens_metrics['policy_q_std']:.4f}, Q_range={q_sens_metrics['policy_q_range']:.4f}, top1-median={q_sens_metrics['policy_q_top1_vs_median']:.4f}")
            
            # Log to tensorboard and wandb
            diag_log = {}
            for k, v in q_sens_metrics.items():
                writer.add_scalar(f"diag/{k}", v, total_steps)
                diag_log[f"diag/{k}"] = v
            
            if args.track:
                import wandb
                wandb.log(diag_log, step=total_steps)
        
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
            
            # ===== Sanity Check: Prior vs Policy =====
            if args.sanity_check and args.num_eval_envs >= args.sanity_check_candidates:
                print(f"\n  Running sanity check (M={args.sanity_check_candidates}, episodes={args.sanity_check_episodes})...")
                sanity_metrics = sanity_check_latent_policy(
                    agent=agent,
                    eval_envs=eval_envs,
                    visual_encoder=visual_encoder,
                    include_rgb=include_rgb,
                    obs_horizon=args.obs_horizon,
                    act_horizon=args.act_horizon,
                    device=device,
                    num_candidates=args.sanity_check_candidates,
                    num_episodes=args.sanity_check_episodes,
                )
                
                print(f"  Sanity Check Results:")
                print(f"    Prior:  best={sanity_metrics['prior_best_return']:.2f}, mean={sanity_metrics['prior_mean_return']:.2f} ± {sanity_metrics['prior_std_return']:.2f}")
                print(f"    Policy: best={sanity_metrics['policy_best_return']:.2f}, mean={sanity_metrics['policy_mean_return']:.2f} ± {sanity_metrics['policy_std_return']:.2f}")
                print(f"    Best-of-M per episode: Prior={sanity_metrics['prior_best_per_ep']:.2f}, Policy={sanity_metrics['policy_best_per_ep']:.2f}")
                print(f"    Policy Advantage (best-of-M): {sanity_metrics['policy_advantage']:.2f}")
                print(f"    Policy Advantage (mean): {sanity_metrics['policy_mean_advantage']:.2f}")
                
                # Log sanity check metrics
                for k, v in sanity_metrics.items():
                    writer.add_scalar(f"sanity/{k}", v, total_steps)
                    eval_log[f"sanity/{k}"] = v
            
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
