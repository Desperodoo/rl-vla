#!/usr/bin/env python3
"""Test AWSC checkpoint loading and evaluation.

This script verifies:
1. Checkpoint loading logic matches train_latent_offline.py
2. Evaluation logic works correctly
3. AWSC checkpoint achieves ~75% success rate on LiftPegUpright-v1
4. Compare torch.randn initial latent (AWSC) vs policy-generated latent (DSRL)

Usage:
    cd rlft/dsrl
    python tests/test_awsc_evaluation.py
    
    # With fewer episodes for quick test
    python tests/test_awsc_evaluation.py --num_eval_episodes 20 --num_eval_envs 10
"""
import sys
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import Optional

# Add paths
_dsrl_path = Path(__file__).resolve().parent.parent
_rlft_path = _dsrl_path.parent
sys.path.insert(0, str(_rlft_path))
sys.path.insert(0, str(_dsrl_path))
sys.path.insert(0, str(_rlft_path / "diffusion_policy"))

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import tyro

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

# Import from diffusion_policy
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.evaluate import evaluate
from diffusion_policy.utils import (
    AgentWrapper,
    build_state_obs_extractor,
    convert_obs,
)
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.algorithms.networks import DoubleQNetwork

# Import DSRL components
from latent_policy import LatentGaussianPolicy
from dsrl_agent import DSRLAgent
from value_network import ValueNetwork


@dataclass
class Args:
    # Checkpoint path
    awsc_checkpoint: str = "checkpoints/best_eval_success_once.pt"
    """path to AWSC checkpoint"""
    use_ema: bool = True
    """use EMA weights from checkpoint (recommended)"""
    
    # Environment settings
    env_id: str = "LiftPegUpright-v1"
    """environment ID"""
    obs_mode: str = "rgb"
    """observation mode"""
    control_mode: str = "pd_ee_delta_pose"
    """control mode"""
    sim_backend: str = "physx_cuda"
    """simulation backend"""
    max_episode_steps: int = 100
    """max episode steps"""
    
    # Evaluation settings
    num_eval_episodes: int = 100
    """number of evaluation episodes"""
    num_eval_envs: int = 25
    """number of parallel eval environments"""
    
    # Architecture settings (must match checkpoint)
    obs_horizon: int = 2
    """observation horizon"""
    act_horizon: int = 8
    """action execution horizon"""
    pred_horizon: int = 16
    """action prediction horizon"""
    visual_feature_dim: int = 256
    """visual feature dimension"""
    diffusion_step_embed_dim: int = 64
    """diffusion step embedding dimension"""
    unet_dims: tuple = (64, 128, 256)
    """U-Net channel dimensions"""
    n_groups: int = 8
    """GroupNorm groups"""
    q_hidden_dims: tuple = (512, 512, 512)
    """Q-network hidden dimensions"""
    latent_hidden_dims: tuple = (256, 256, 256)
    """latent policy hidden dimensions"""
    num_inference_steps: int = 8
    """number of flow inference steps"""
    
    # Device
    cuda: bool = True
    """use CUDA"""


class AWShortCutFlowAgentWrapper(AgentWrapper):
    """Wrapper for AWSC agent evaluation with configurable latent source.
    
    Supports two latent initialization modes:
    1. randn: Use torch.randn (original AWSC method)
    2. policy: Use latent policy (DSRL method)
    
    This allows comparing the effect of different initial latent sources.
    """
    
    def __init__(self, agent, visual_encoder, include_rgb, obs_horizon, act_horizon, 
                 latent_source: str = "randn"):
        """
        Args:
            latent_source: "randn" for torch.randn, "policy" for latent policy
        """
        super().__init__(agent, visual_encoder, include_rgb, obs_horizon, act_horizon)
        self.latent_source = latent_source
        
    def get_action(self, obs_seq, **kwargs):
        """Get action from observations using flow policy."""
        with torch.no_grad():
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            
            if self.latent_source == "randn":
                # AWSC original method: use torch.randn as initial latent
                action_seq, _, _ = self.agent.get_action(
                    obs_cond,
                    use_latent_policy=False,  # Use randn
                    deterministic=True,
                    use_prior_mixing=False,
                )
            elif self.latent_source == "policy":
                # DSRL method: use latent policy to generate initial latent
                action_seq, _, _ = self.agent.get_action(
                    obs_cond,
                    use_latent_policy=True,  # Use latent policy
                    deterministic=True,  # Use policy mean
                    use_prior_mixing=False,
                )
            else:
                raise ValueError(f"Unknown latent_source: {self.latent_source}")
            
            # Only return act_horizon actions
            start = self.obs_horizon - 1
            end = start + self.act_horizon
            return action_seq[:, start:end]


def create_agent_and_encoder(args, action_dim: int, state_dim: int, device: str):
    """Create DSRL agent and visual encoder with correct architecture."""
    
    # Visual encoder
    visual_encoder = PlainConv(
        in_channels=3,  # RGB
        out_dim=args.visual_feature_dim,
        pool_feature_map=True,
    ).to(device)
    
    # Compute global_cond_dim
    global_cond_dim = args.obs_horizon * (args.visual_feature_dim + state_dim)
    print(f"Architecture:")
    print(f"  action_dim: {action_dim}")
    print(f"  state_dim: {state_dim}")
    print(f"  visual_feature_dim: {args.visual_feature_dim}")
    print(f"  global_cond_dim: {global_cond_dim} = {args.obs_horizon} * ({args.visual_feature_dim} + {state_dim})")
    
    # Create velocity network
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=args.unet_dims,
        n_groups=args.n_groups,
    ).to(device)
    
    # Create Q-network (DoubleQNetwork for AWSC)
    q_network = DoubleQNetwork(
        action_dim=action_dim,
        obs_dim=global_cond_dim,
        action_horizon=args.act_horizon,
        hidden_dims=list(args.q_hidden_dims),
    ).to(device)
    
    # Create latent policy (not used for AWSC evaluation, but needed for DSRLAgent)
    latent_policy = LatentGaussianPolicy(
        obs_dim=global_cond_dim,
        pred_horizon=args.pred_horizon,
        action_dim=action_dim,
        hidden_dims=list(args.latent_hidden_dims),
        steer_mode="full",
        act_horizon=args.act_horizon,
    ).to(device)
    
    # Create DSRL agent
    agent = DSRLAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        latent_policy=latent_policy,
        value_network=None,
        action_dim=action_dim,
        obs_dim=global_cond_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        act_horizon=args.act_horizon,
        num_inference_steps=args.num_inference_steps,
        device=device,
    ).to(device)
    
    return agent, visual_encoder, global_cond_dim


def load_awsc_checkpoint(agent: DSRLAgent, visual_encoder: nn.Module, checkpoint_path: str, device: str, use_ema: bool = True):
    """Load AWSC checkpoint using train_latent_offline.py logic.
    
    Args:
        use_ema: If True, load from ema_agent instead of agent
    """
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\nLoading checkpoint from {checkpoint_path}")
    print(f"  Using EMA weights: {use_ema}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Choose agent source
    if use_ema and "ema_agent" in checkpoint:
        agent_state = checkpoint["ema_agent"]
        print("  Loading from ema_agent")
    elif "agent" in checkpoint:
        agent_state = checkpoint["agent"]
        print("  Loading from agent")
    elif "velocity_net" in checkpoint:
        agent_state = checkpoint
    else:
        agent_state = checkpoint
    
    # Extract velocity_net weights
    velocity_net_state = {}
    q_network_state = {}
    
    for key, value in agent_state.items():
        if key.startswith("velocity_net."):
            velocity_net_state[key.replace("velocity_net.", "")] = value
        elif key.startswith("q_network.") or key.startswith("critic."):
            new_key = key.replace("q_network.", "").replace("critic.", "")
            q_network_state[new_key] = value
    
    # Load velocity net
    if velocity_net_state:
        missing, unexpected = agent.velocity_net.load_state_dict(velocity_net_state, strict=False)
        print(f"  Loaded velocity_net: {len(velocity_net_state)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"    Missing: {missing[:5]}...")
    else:
        print("  WARNING: No velocity_net keys found!")
    
    # Load Q-network
    if q_network_state:
        try:
            missing, unexpected = agent.q_network.load_state_dict(q_network_state, strict=False)
            print(f"  Loaded q_network: {len(q_network_state)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"  WARNING: Could not load q_network: {e}")
    else:
        print("  WARNING: No q_network keys found!")
    
    # Load visual encoder
    if "visual_encoder" in checkpoint:
        visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        print(f"  Loaded visual_encoder")
    else:
        print("  WARNING: No visual_encoder found in checkpoint!")
    
    # Update EMA velocity net
    agent.velocity_net_ema.load_state_dict(agent.velocity_net.state_dict())
    print("  Synced velocity_net_ema from velocity_net")


def main():
    args = tyro.cli(Args)
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========== Create Evaluation Environment ==========
    print("\n" + "=" * 60)
    print("Creating evaluation environment...")
    print("=" * 60)
    
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="dense",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    
    eval_envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    
    # Get environment info
    action_dim = eval_envs.single_action_space.shape[0]
    print(f"Action dimension: {action_dim}")
    
    # Get state dimension from environment
    # Note: state shape is (num_envs, state_dim) or (1, state_dim) for single env
    obs_space = eval_envs.single_observation_space
    state_shape = obs_space["state"].shape
    # Handle both (state_dim,) and (1, state_dim) shapes
    state_dim = state_shape[-1] if len(state_shape) > 0 else state_shape[0]
    print(f"State dimension: {state_dim}")
    
    # ========== Create Agent ==========
    print("\n" + "=" * 60)
    print("Creating agent...")
    print("=" * 60)
    
    agent, visual_encoder, global_cond_dim = create_agent_and_encoder(
        args, action_dim, state_dim, str(device)
    )
    
    print(f"\nAgent parameters:")
    print(f"  Total: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M")
    print(f"  Velocity net: {sum(p.numel() for p in agent.velocity_net.parameters()) / 1e6:.2f}M")
    print(f"  Q-network: {sum(p.numel() for p in agent.q_network.parameters()) / 1e6:.2f}M")
    
    # ========== Load Checkpoint ==========
    print("\n" + "=" * 60)
    print("Loading checkpoint...")
    print("=" * 60)
    
    checkpoint_path = _dsrl_path / args.awsc_checkpoint
    load_awsc_checkpoint(agent, visual_encoder, str(checkpoint_path), str(device), use_ema=args.use_ema)
    
    # ========== Create Wrapper ==========
    print("\n" + "=" * 60)
    print("Creating agent wrapper...")
    print("=" * 60)
    
    include_rgb = "rgb" in args.obs_mode
    
    # ========== Test 1: Using torch.randn (AWSC original) ==========
    print("\n" + "=" * 60)
    print("Test 1: Evaluating with torch.randn initial latent (AWSC original)")
    print("=" * 60)
    
    agent_wrapper_randn = AWShortCutFlowAgentWrapper(
        agent, visual_encoder, include_rgb,
        args.obs_horizon, args.act_horizon,
        latent_source="randn"
    ).to(device)
    
    # Set to eval mode
    agent.eval()
    visual_encoder.eval()
    
    print(f"Running evaluation: {args.num_eval_episodes} episodes...")
    
    eval_metrics_randn = evaluate(
        args.num_eval_episodes,
        agent_wrapper_randn,
        eval_envs,
        device,
        args.sim_backend,
    )
    
    success_rate_randn = np.mean(eval_metrics_randn.get("success_once", [0]))
    print(f"\n[torch.randn] Results:")
    for k in eval_metrics_randn.keys():
        values = eval_metrics_randn[k]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {k}: {mean_val:.4f} ± {std_val:.4f}")
    
    # ========== Test 2: Using latent policy (DSRL) ==========
    print("\n" + "=" * 60)
    print("Test 2: Evaluating with latent policy initial latent (DSRL)")
    print("  Note: Latent policy is randomly initialized (not trained)")
    print("=" * 60)
    
    agent_wrapper_policy = AWShortCutFlowAgentWrapper(
        agent, visual_encoder, include_rgb,
        args.obs_horizon, args.act_horizon,
        latent_source="policy"
    ).to(device)
    
    print(f"Running evaluation: {args.num_eval_episodes} episodes...")
    
    eval_metrics_policy = evaluate(
        args.num_eval_episodes,
        agent_wrapper_policy,
        eval_envs,
        device,
        args.sim_backend,
    )
    
    success_rate_policy = np.mean(eval_metrics_policy.get("success_once", [0]))
    print(f"\n[Latent Policy] Results:")
    for k in eval_metrics_policy.keys():
        values = eval_metrics_policy[k]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {k}: {mean_val:.4f} ± {std_val:.4f}")
    
    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("SUMMARY: Comparison of Initial Latent Sources")
    print("=" * 60)
    print(f"  torch.randn (AWSC original):  success_once = {success_rate_randn:.2%}")
    print(f"  Latent Policy (DSRL untrained): success_once = {success_rate_policy:.2%}")
    print(f"  Difference: {(success_rate_policy - success_rate_randn):.2%}")
    
    print("\n" + "=" * 60)
    if success_rate_randn >= 0.70:
        print(f"✓ torch.randn SUCCESS! Success rate: {success_rate_randn:.2%} (target: ~75%)")
    else:
        print(f"✗ torch.randn FAILED! Success rate: {success_rate_randn:.2%} (target: ~75%)")
    
    if success_rate_policy >= 0.70:
        print(f"✓ Latent Policy SUCCESS! Success rate: {success_rate_policy:.2%}")
    else:
        print(f"✗ Latent Policy lower than target: {success_rate_policy:.2%}")
        print("  (Expected: untrained latent policy may perform worse than randn)")
    print("=" * 60)
    
    eval_envs.close()
    
    return success_rate_randn >= 0.70


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
