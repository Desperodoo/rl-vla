"""
Test DSRL SAC Agent checkpoint loading and rollout.

This test verifies:
1. Agent can be constructed correctly
2. Rollout produces valid actions
3. SAC loss computation works
4. Checkpoint save/load

Usage: python test_agent.py
"""

import sys
from pathlib import Path
import os
import tempfile

import torch
import torch.nn as nn
import numpy as np

# Add paths for imports
_root = Path(__file__).parent.parent.parent
_dsrl_offpolicy = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl"))
sys.path.insert(0, str(_dsrl_offpolicy.parent))

from latent_policy import LatentGaussianPolicy
from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.algorithms.networks import DoubleQNetwork

from dsrl_offpolicy.agents.dsrl_sac_agent import DSRLSACAgent
from dsrl_offpolicy.models.latent_q_network import DoubleLatentQNetwork


# ============================================================
# Helper: Create Agent
# ============================================================

def create_test_agent():
    """Create a minimal agent for testing."""
    action_dim = 7
    obs_dim = 512
    pred_horizon = 16
    act_horizon = 8
    
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=64,
        down_dims=(64, 128),
        n_groups=8,
    )
    
    q_network = DoubleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim,
        action_horizon=act_horizon,
        hidden_dims=[256, 256],
    )
    
    latent_policy = LatentGaussianPolicy(
        obs_dim=obs_dim,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        steer_mode="full",
        act_horizon=act_horizon,
    )
    
    latent_q_network = DoubleLatentQNetwork(
        obs_dim=obs_dim,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        steer_mode="full",
        act_horizon=act_horizon,
        tau=0.005,
    )
    
    agent = DSRLSACAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        latent_policy=latent_policy,
        latent_q_network=latent_q_network,
        action_dim=action_dim,
        obs_dim=obs_dim,
        obs_horizon=2,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        num_inference_steps=4,  # Reduced for faster testing
        gamma=0.99,
        tau_target=0.005,
        init_temperature=0.1,
        learnable_temp=True,
        target_entropy=None,
        backup_entropy=True,
        prior_mix_ratio=0.3,
        device="cpu",
    )
    
    return agent


# ============================================================
# Agent Action Tests
# ============================================================

def test_get_action():
    """Test action sampling."""
    print("  Testing get_action...", end=" ")
    agent = create_test_agent()
    
    B = 8
    obs_features = torch.randn(B, 512)
    
    actions, latent_w, log_prob = agent.get_action(
        obs_features,
        use_latent_policy=True,
        deterministic=False,
    )
    
    assert actions.shape == (B, 16, 7), f"Wrong actions shape: {actions.shape}"
    assert latent_w.shape == (B, 16, 7), f"Wrong latent shape: {latent_w.shape}"
    assert log_prob.shape == (B,), f"Wrong log_prob shape: {log_prob.shape}"
    assert torch.isfinite(actions).all(), "Actions contain inf/nan"
    print("PASSED")


def test_get_action_deterministic():
    """Test deterministic action sampling."""
    print("  Testing get_action deterministic...", end=" ")
    agent = create_test_agent()
    
    B = 8
    obs_features = torch.randn(B, 512)
    
    actions1, _, _ = agent.get_action(
        obs_features,
        use_latent_policy=True,
        deterministic=True,
    )
    
    actions2, _, _ = agent.get_action(
        obs_features,
        use_latent_policy=True,
        deterministic=True,
    )
    
    assert torch.allclose(actions1, actions2, atol=1e-5), "Deterministic actions should match"
    print("PASSED")


# ============================================================
# Loss Computation Tests
# ============================================================

def test_offline_loss():
    """Test Stage 1 offline loss computation."""
    print("  Testing compute_offline_loss...", end=" ")
    agent = create_test_agent()
    
    B = 8
    obs_cond = torch.randn(B, 512)
    
    loss_dict = agent.compute_offline_loss(obs_cond)
    
    assert "loss" in loss_dict, "Missing 'loss' key"
    assert "nll_loss" in loss_dict, "Missing 'nll_loss' key"
    assert "kl_loss" in loss_dict, "Missing 'kl_loss' key"
    assert torch.isfinite(loss_dict["loss"]), "Loss is inf/nan"
    
    # Test backward
    loss_dict["loss"].backward()
    
    grad_count = 0
    for param in agent.latent_policy.parameters():
        if param.requires_grad and param.grad is not None:
            grad_count += 1
    assert grad_count > 0, "No gradients computed for latent_policy"
    print("PASSED")


def test_critic_loss():
    """Test SAC critic loss computation."""
    print("  Testing compute_critic_loss...", end=" ")
    agent = create_test_agent()
    
    B = 8
    obs_cond = torch.randn(B, 512)
    latent_w = torch.randn(B, 16, 7)
    reward = torch.randn(B)
    discount_factor = torch.ones(B) * 0.99
    done = torch.zeros(B)
    next_obs_cond = torch.randn(B, 512)
    
    loss_dict = agent.compute_critic_loss(
        obs_cond=obs_cond,
        latent_w=latent_w,
        reward=reward,
        discount_factor=discount_factor,
        done=done,
        next_obs_cond=next_obs_cond,
    )
    
    assert "loss" in loss_dict, "Missing 'loss' key"
    assert "td_error" in loss_dict, "Missing 'td_error' key"
    assert "q_mean" in loss_dict, "Missing 'q_mean' key"
    assert torch.isfinite(loss_dict["loss"]), "Critic loss is inf/nan"
    print("PASSED")


def test_actor_loss():
    """Test SAC actor loss computation."""
    print("  Testing compute_actor_loss...", end=" ")
    agent = create_test_agent()
    
    B = 8
    obs_cond = torch.randn(B, 512)
    
    loss_dict = agent.compute_actor_loss(obs_cond)
    
    assert "loss" in loss_dict, "Missing 'loss' key"
    assert "entropy" in loss_dict, "Missing 'entropy' key"
    assert torch.isfinite(loss_dict["loss"]), "Actor loss is inf/nan"
    print("PASSED")


def test_temperature_loss():
    """Test SAC temperature loss computation."""
    print("  Testing compute_temperature_loss...", end=" ")
    agent = create_test_agent()
    
    B = 8
    obs_cond = torch.randn(B, 512)
    
    loss_dict = agent.compute_temperature_loss(obs_cond)
    
    assert "loss" in loss_dict, "Missing 'loss' key"
    assert "alpha" in loss_dict, "Missing 'alpha' key"
    print("PASSED")


# ============================================================
# Target Update Tests
# ============================================================

def test_target_update():
    """Test target network update."""
    print("  Testing update_target...", end=" ")
    agent = create_test_agent()
    
    # Should not raise
    agent.update_target()
    print("PASSED")


def test_prior_mix_decay():
    """Test prior mixing ratio decay."""
    print("  Testing prior_mix_decay...", end=" ")
    agent = create_test_agent()
    
    initial_ratio = agent.get_prior_mix_ratio()
    
    for _ in range(10):
        agent.decay_prior_mix_ratio()
    
    final_ratio = agent.get_prior_mix_ratio()
    
    assert final_ratio < initial_ratio, f"Ratio should decay: {initial_ratio} -> {final_ratio}"
    assert final_ratio >= agent.prior_mix_min, "Ratio should not go below min"
    print("PASSED")


# ============================================================
# Checkpoint Tests
# ============================================================

def test_save_load_checkpoint():
    """Test checkpoint save/load."""
    print("  Testing save/load checkpoint...", end=" ")
    agent = create_test_agent()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_ckpt.pt"
        
        # Get original weights
        original_weights = {
            name: param.clone() for name, param in agent.latent_policy.named_parameters()
        }
        
        # Save
        agent.save_checkpoint(str(ckpt_path))
        assert ckpt_path.exists(), "Checkpoint file not created"
        
        # Modify agent
        with torch.no_grad():
            for param in agent.latent_policy.parameters():
                param.add_(torch.randn_like(param) * 10)
        
        # Load
        agent.load_checkpoint(str(ckpt_path))
        
        # Verify weights restored
        for name, param in agent.latent_policy.named_parameters():
            assert torch.allclose(param, original_weights[name], atol=1e-5), \
                f"Weight {name} not restored correctly"
    
    print("PASSED")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DSRL Off-Policy Agent Tests")
    print("=" * 60)
    
    print("\n[Action Sampling Tests]")
    test_get_action()
    test_get_action_deterministic()
    
    print("\n[Loss Computation Tests]")
    test_offline_loss()
    test_critic_loss()
    test_actor_loss()
    test_temperature_loss()
    
    print("\n[Target Update Tests]")
    test_target_update()
    test_prior_mix_decay()
    
    print("\n[Checkpoint Tests]")
    test_save_load_checkpoint()
    
    print("\n" + "=" * 60)
    print("ALL AGENT TESTS PASSED!")
    print("=" * 60)
