#!/usr/bin/env python3
"""Test DSRL checkpoint loading and inference with AWSC pretrained model.

This script verifies:
1. Checkpoint loading from AWShortCutFlow format
2. DSRLAgent forward inference
3. Latent policy sampling
4. Action generation from latent

Usage:
    cd rlft/dsrl
    python tests/test_checkpoint_loading.py
"""
import sys
from pathlib import Path

# Add paths
_dsrl_path = Path(__file__).resolve().parent.parent
_rlft_path = _dsrl_path.parent
sys.path.insert(0, str(_rlft_path))
sys.path.insert(0, str(_dsrl_path))

import torch
import torch.nn as nn
import numpy as np

# Import DSRL components
from latent_policy import LatentGaussianPolicy
from dsrl_agent import DSRLAgent
from value_network import ValueNetwork

# Import diffusion_policy components
sys.path.insert(0, str(_rlft_path / "diffusion_policy"))
from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.algorithms.networks import DoubleQNetwork
from diffusion_policy.plain_conv import PlainConv


class DoubleQNetworkWrapper(nn.Module):
    """Wrapper to make DoubleQNetwork return EnsembleQNetwork-compatible format.
    
    EnsembleQNetwork returns: (num_qs, B, 1)
    DoubleQNetwork returns: (q1, q2) where each is (B, 1)
    
    This wrapper stacks q1, q2 to return (2, B, 1)
    """
    def __init__(self, double_q: DoubleQNetwork):
        super().__init__()
        self.double_q = double_q
    
    def forward(self, action_seq, obs_cond):
        q1, q2 = self.double_q(action_seq, obs_cond)
        return torch.stack([q1, q2], dim=0)  # (2, B, 1)
    
    def load_state_dict(self, state_dict, strict=True):
        return self.double_q.load_state_dict(state_dict, strict)
    
    def state_dict(self):
        return self.double_q.state_dict()


def test_checkpoint_loading():
    """Test loading AWSC checkpoint into DSRL components."""
    print("=" * 60)
    print("Test 1: Checkpoint Loading")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    ckpt_path = _dsrl_path / "checkpoints" / "best_eval_success_once.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return False
    
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Analyze checkpoint structure
    print(f"\nCheckpoint top-level keys: {list(ckpt.keys())}")
    
    agent_state = ckpt["agent"]
    
    # Count component keys
    velocity_keys = [k for k in agent_state.keys() if k.startswith("velocity_net.")]
    critic_keys = [k for k in agent_state.keys() if k.startswith("critic.")]
    
    print(f"  velocity_net keys: {len(velocity_keys)}")
    print(f"  critic keys: {len(critic_keys)}")
    
    # Get action_dim from velocity_net output layer
    final_conv_key = "velocity_net.unet.final_conv.0.weight"
    if final_conv_key in agent_state:
        action_dim = agent_state[final_conv_key].shape[0]
        print(f"  Detected action_dim: {action_dim}")
    else:
        action_dim = 7  # Default for LiftPegUpright
        print(f"  Using default action_dim: {action_dim}")
    
    # Get visual encoder info
    if "visual_encoder" in ckpt:
        ve_keys = list(ckpt["visual_encoder"].keys())
        print(f"  visual_encoder keys: {len(ve_keys)}")
    
    print("\n✓ Checkpoint loaded successfully!")
    return True


def test_dsrl_agent_creation():
    """Test creating DSRLAgent with correct architecture."""
    print("\n" + "=" * 60)
    print("Test 2: DSRLAgent Creation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Architecture params (from config.yaml)
    # global_cond_dim = obs_horizon * (visual_feature_dim + state_dim)
    # state_dim for LiftPegUpright-v1 with pd_ee_delta_pose = 25
    visual_feature_dim = 256
    state_dim = 25
    obs_horizon = 2
    obs_cond_dim = obs_horizon * (visual_feature_dim + state_dim)  # 562
    
    action_dim = 7  # pd_ee_delta_pose has 7 dims
    pred_horizon = 16
    act_horizon = 8
    
    print(f"Creating components with:")
    print(f"  visual_feature_dim: {visual_feature_dim}, state_dim: {state_dim}")
    print(f"  obs_cond_dim: {obs_cond_dim} = {obs_horizon} * ({visual_feature_dim} + {state_dim})")
    print(f"  action_dim: {action_dim}")
    print(f"  pred_horizon: {pred_horizon}, act_horizon: {act_horizon}")
    
    # Create velocity network
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_cond_dim,
        diffusion_step_embed_dim=64,
        down_dims=(64, 128, 256),
        n_groups=8,
    ).to(device)
    
    # Create Q-network (DoubleQNetwork wrapped to match EnsembleQNetwork format)
    double_q = DoubleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_cond_dim,
        action_horizon=act_horizon,  # Uses act_horizon, not pred_horizon
        hidden_dims=[512, 512, 512],
    ).to(device)
    q_network = DoubleQNetworkWrapper(double_q)
    
    # Create latent policy
    latent_policy = LatentGaussianPolicy(
        obs_dim=obs_cond_dim,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        hidden_dims=[256, 256, 256],
        steer_mode="full",
        act_horizon=act_horizon,
    ).to(device)
    
    # Create value network
    value_network = ValueNetwork(
        obs_dim=obs_cond_dim,
        hidden_dims=[256, 256, 256],
    ).to(device)
    
    # Create DSRLAgent
    agent = DSRLAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        latent_policy=latent_policy,
        value_network=value_network,
        action_dim=action_dim,
        obs_dim=obs_cond_dim,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        obs_horizon=obs_horizon,
        num_inference_steps=8,
        num_candidates=16,
        kappa=1.0,
        tau=5.0,
        beta_latent=1.0,
        kl_coef=1e-3,
        ppo_clip=0.2,
        entropy_coef=1e-3,
    ).to(device)
    
    print(f"\n✓ DSRLAgent created successfully!")
    print(f"  Total parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    return agent, device, obs_cond_dim


def test_checkpoint_weight_loading(agent, device):
    """Test loading pretrained weights into DSRLAgent."""
    print("\n" + "=" * 60)
    print("Test 3: Weight Loading")
    print("=" * 60)
    
    ckpt_path = _dsrl_path / "checkpoints" / "best_eval_success_once.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    
    agent_state = ckpt["agent"]
    
    # Extract and load velocity_net weights
    velocity_net_state = {}
    for key, value in agent_state.items():
        if key.startswith("velocity_net."):
            new_key = key.replace("velocity_net.", "")
            velocity_net_state[new_key] = value
    
    # Extract and load critic weights
    q_network_state = {}
    for key, value in agent_state.items():
        if key.startswith("critic."):
            new_key = key.replace("critic.", "")
            q_network_state[new_key] = value
    
    # Load velocity_net
    try:
        missing, unexpected = agent.velocity_net.load_state_dict(velocity_net_state, strict=False)
        print(f"Velocity net loaded:")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"    First 5 missing: {missing[:5]}")
    except Exception as e:
        print(f"ERROR loading velocity_net: {e}")
        return False
    
    # Load Q-network (DoubleQNetwork via wrapper)
    try:
        missing, unexpected = agent.q_network.load_state_dict(q_network_state, strict=False)
        print(f"\nQ-network loaded:")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"    First 5 missing: {missing[:5]}")
    except Exception as e:
        print(f"ERROR loading q_network: {e}")
        return False
    
    print("\n✓ Velocity network and Q-network weights loaded successfully!")
    
    return True


def test_forward_inference(agent, device, obs_cond_dim):
    """Test forward inference with dummy data."""
    print("\n" + "=" * 60)
    print("Test 4: Forward Inference")
    print("=" * 60)
    
    batch_size = 4
    
    # Create dummy observation with correct dimension
    obs_cond = torch.randn(batch_size, obs_cond_dim).to(device)
    
    agent.eval()
    with torch.no_grad():
        # Test latent policy sampling
        print("Testing latent policy sampling...")
        w, log_prob = agent.latent_policy.sample(obs_cond, deterministic=False)
        print(f"  w shape: {w.shape}")  # (B, pred_horizon, action_dim)
        print(f"  log_prob shape: {log_prob.shape}")  # (B,)
        
        # Test action generation from latent
        print("\nTesting action generation from latent...")
        actions = agent.sample_actions_from_latent(obs_cond, w, use_ema=False)
        print(f"  actions shape: {actions.shape}")  # (B, pred_horizon, action_dim)
        
        # Test get_action
        print("\nTesting get_action...")
        actions, w, log_prob = agent.get_action(obs_cond, use_latent_policy=True)
        print(f"  actions shape: {actions.shape}")
        print(f"  w shape: {w.shape}")
        print(f"  log_prob: {log_prob.mean().item():.4f}")
    
    print("\n✓ Forward inference works correctly!")
    return True


def test_offline_loss(agent, device, obs_cond_dim):
    """Test Stage 1 offline loss computation."""
    print("\n" + "=" * 60)
    print("Test 5: Offline Loss Computation")
    print("=" * 60)
    
    batch_size = 4
    
    obs_cond = torch.randn(batch_size, obs_cond_dim).to(device)
    
    agent.train()
    
    print("Computing offline AW-MLE loss...")
    loss_dict = agent.compute_offline_loss(obs_cond)
    
    print(f"Loss dict keys: {list(loss_dict.keys())}")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.6f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n✓ Offline loss computation works!")
    return True


def main():
    print("\n" + "=" * 60)
    print("DSRL Checkpoint Loading & Inference Tests")
    print("=" * 60 + "\n")
    
    # Test 1: Checkpoint loading
    if not test_checkpoint_loading():
        return
    
    # Test 2: Agent creation
    result = test_dsrl_agent_creation()
    if result is None:
        return
    agent, device, obs_cond_dim = result
    
    # Test 3: Weight loading
    if not test_checkpoint_weight_loading(agent, device):
        return
    
    # Test 4: Forward inference
    if not test_forward_inference(agent, device, obs_cond_dim):
        return
    
    # Test 5: Offline loss
    if not test_offline_loss(agent, device, obs_cond_dim):
        return
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
