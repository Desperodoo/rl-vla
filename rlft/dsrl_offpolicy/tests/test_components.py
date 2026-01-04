"""
Test components for DSRL Off-Policy.

This test verifies:
1. Latent Q-network forward/backward works
2. Double Q-network with target update
3. Temperature module (learnable/fixed)
4. Macro replay buffer operations

Usage: python test_components.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add paths for imports
_root = Path(__file__).parent.parent.parent
_dsrl_offpolicy = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl"))
sys.path.insert(0, str(_dsrl_offpolicy.parent))

from dsrl_offpolicy.models.latent_q_network import (
    LatentQNetwork,
    DoubleLatentQNetwork,
    Temperature,
)
from dsrl_offpolicy.buffers.macro_replay_buffer import MacroReplayBuffer


# ============================================================
# Latent Q-Network Tests
# ============================================================

def test_q_network_forward():
    """Test single Q-network forward pass."""
    print("  Testing LatentQNetwork forward...", end=" ")
    q_network = LatentQNetwork(
        obs_dim=512,
        pred_horizon=16,
        action_dim=7,
        hidden_dims=[256, 256],
        steer_mode="full",
        act_horizon=8,
    )
    
    B = 32
    obs_cond = torch.randn(B, 512)
    latent_w = torch.randn(B, 16, 7)
    
    q_value = q_network(obs_cond, latent_w)
    
    assert q_value.shape == (B, 1), f"Expected shape (32, 1), got {q_value.shape}"
    assert torch.isfinite(q_value).all(), "Q values contain inf/nan"
    print("PASSED")


def test_q_network_backward():
    """Test single Q-network backward pass."""
    print("  Testing LatentQNetwork backward...", end=" ")
    q_network = LatentQNetwork(
        obs_dim=512,
        pred_horizon=16,
        action_dim=7,
        hidden_dims=[256, 256],
        steer_mode="full",
        act_horizon=8,
    )
    
    B = 32
    obs_cond = torch.randn(B, 512)
    latent_w = torch.randn(B, 16, 7)
    
    q_value = q_network(obs_cond, latent_w)
    loss = q_value.mean()
    loss.backward()
    
    # Check gradients exist
    for name, param in q_network.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
    print("PASSED")


def test_double_q_network_forward():
    """Test double Q-network forward pass."""
    print("  Testing DoubleLatentQNetwork forward...", end=" ")
    double_q = DoubleLatentQNetwork(
        obs_dim=512,
        pred_horizon=16,
        action_dim=7,
        hidden_dims=[256, 256],
        steer_mode="full",
        act_horizon=8,
        tau=0.005,
    )
    
    B = 32
    obs_cond = torch.randn(B, 512)
    latent_w = torch.randn(B, 16, 7)
    
    q1, q2 = double_q(obs_cond, latent_w)
    
    assert q1.shape == (B, 1), f"Expected q1 shape (32, 1), got {q1.shape}"
    assert q2.shape == (B, 1), f"Expected q2 shape (32, 1), got {q2.shape}"
    print("PASSED")


def test_double_q_network_target():
    """Test target network forward pass."""
    print("  Testing DoubleLatentQNetwork target forward...", end=" ")
    double_q = DoubleLatentQNetwork(
        obs_dim=512,
        pred_horizon=16,
        action_dim=7,
        hidden_dims=[256, 256],
        steer_mode="full",
        act_horizon=8,
        tau=0.005,
    )
    
    B = 32
    obs_cond = torch.randn(B, 512)
    latent_w = torch.randn(B, 16, 7)
    
    q1_t, q2_t = double_q.forward_target(obs_cond, latent_w)
    
    assert q1_t.shape == (B, 1), f"Expected q1_target shape (32, 1), got {q1_t.shape}"
    assert q2_t.shape == (B, 1), f"Expected q2_target shape (32, 1), got {q2_t.shape}"
    print("PASSED")


def test_double_q_network_min_q():
    """Test min Q computation."""
    print("  Testing DoubleLatentQNetwork get_min_q...", end=" ")
    double_q = DoubleLatentQNetwork(
        obs_dim=512,
        pred_horizon=16,
        action_dim=7,
        hidden_dims=[256, 256],
        steer_mode="full",
        act_horizon=8,
        tau=0.005,
    )
    
    B = 32
    obs_cond = torch.randn(B, 512)
    latent_w = torch.randn(B, 16, 7)
    
    q_min = double_q.get_min_q(obs_cond, latent_w, use_target=False)
    q_min_target = double_q.get_min_q(obs_cond, latent_w, use_target=True)
    
    assert q_min.shape == (B, 1), f"Expected q_min shape (32, 1), got {q_min.shape}"
    assert q_min_target.shape == (B, 1), f"Expected q_min_target shape (32, 1), got {q_min_target.shape}"
    print("PASSED")


def test_soft_update():
    """Test soft target update."""
    print("  Testing soft_update_target...", end=" ")
    double_q = DoubleLatentQNetwork(
        obs_dim=512,
        pred_horizon=16,
        action_dim=7,
        hidden_dims=[256, 256],
        steer_mode="full",
        act_horizon=8,
        tau=0.005,
    )
    
    # Get initial target weights
    initial_target_params = {
        name: param.clone() for name, param in double_q.q1_target.named_parameters()
    }
    
    # Modify online network significantly
    with torch.no_grad():
        for param in double_q.q1.parameters():
            param.add_(torch.randn_like(param) * 10)
    
    # Soft update
    double_q.soft_update_target()
    
    # Check target weights changed slightly
    for name, param in double_q.q1_target.named_parameters():
        if param.numel() > 0:
            assert not torch.allclose(param, initial_target_params[name], atol=1e-6), \
                f"Target {name} should have changed"
    print("PASSED")


# ============================================================
# Temperature Tests
# ============================================================

def test_learnable_temperature():
    """Test learnable temperature."""
    print("  Testing learnable Temperature...", end=" ")
    temp = Temperature(initial_temperature=0.1, learnable=True)
    
    assert temp.learnable, "Temperature should be learnable"
    assert abs(temp.alpha.item() - 0.1) < 1e-4, f"Expected alpha=0.1, got {temp.alpha.item()}"
    assert temp.log_alpha.requires_grad, "log_alpha should require grad"
    print("PASSED")


def test_fixed_temperature():
    """Test fixed temperature."""
    print("  Testing fixed Temperature...", end=" ")
    temp = Temperature(initial_temperature=0.5, learnable=False)
    
    assert not temp.learnable, "Temperature should not be learnable"
    assert abs(temp.alpha.item() - 0.5) < 1e-4, f"Expected alpha=0.5, got {temp.alpha.item()}"
    print("PASSED")


# ============================================================
# Replay Buffer Tests
# ============================================================

def test_buffer_add_single():
    """Test adding single transition."""
    print("  Testing MacroReplayBuffer add single...", end=" ")
    buffer = MacroReplayBuffer(
        capacity=1000,
        obs_dim=512,
        pred_horizon=16,
        action_dim=7,
        device="cpu",
    )
    
    obs = np.random.randn(512).astype(np.float32)
    latent = np.random.randn(16, 7).astype(np.float32)
    
    buffer.add(
        obs_cond=obs,
        latent_w=latent,
        reward=1.0,
        discount_factor=0.99,
        done=0.0,
        next_obs_cond=obs,
    )
    
    assert buffer.size == 1, f"Expected size 1, got {buffer.size}"
    print("PASSED")


def test_buffer_add_batch():
    """Test adding batch of transitions."""
    print("  Testing MacroReplayBuffer add batch...", end=" ")
    buffer = MacroReplayBuffer(
        capacity=1000,
        obs_dim=512,
        pred_horizon=16,
        action_dim=7,
        device="cpu",
    )
    
    B = 50
    obs = np.random.randn(B, 512).astype(np.float32)
    latent = np.random.randn(B, 16, 7).astype(np.float32)
    
    buffer.add_batch(
        obs_cond=obs,
        latent_w=latent,
        reward=np.ones(B, dtype=np.float32),
        discount_factor=np.ones(B, dtype=np.float32) * 0.99,
        done=np.zeros(B, dtype=np.float32),
        next_obs_cond=obs,
    )
    
    assert buffer.size == B, f"Expected size {B}, got {buffer.size}"
    print("PASSED")


def test_buffer_sample():
    """Test sampling from buffer."""
    print("  Testing MacroReplayBuffer sample...", end=" ")
    buffer = MacroReplayBuffer(
        capacity=1000,
        obs_dim=512,
        pred_horizon=16,
        action_dim=7,
        device="cpu",
    )
    
    B = 100
    obs = np.random.randn(B, 512).astype(np.float32)
    latent = np.random.randn(B, 16, 7).astype(np.float32)
    
    buffer.add_batch(
        obs_cond=obs,
        latent_w=latent,
        reward=np.ones(B, dtype=np.float32),
        discount_factor=np.ones(B, dtype=np.float32) * 0.99,
        done=np.zeros(B, dtype=np.float32),
        next_obs_cond=obs,
    )
    
    batch = buffer.sample(32)
    
    assert batch["obs_cond"].shape == (32, 512), f"Wrong obs shape: {batch['obs_cond'].shape}"
    assert batch["latent_w"].shape == (32, 16, 7), f"Wrong latent shape: {batch['latent_w'].shape}"
    assert batch["reward"].shape == (32,), f"Wrong reward shape: {batch['reward'].shape}"
    print("PASSED")


def test_buffer_overflow():
    """Test ring buffer behavior on overflow."""
    print("  Testing MacroReplayBuffer overflow...", end=" ")
    buffer = MacroReplayBuffer(
        capacity=100,
        obs_dim=512,
        pred_horizon=16,
        action_dim=7,
        device="cpu",
    )
    
    # Fill buffer beyond capacity
    for i in range(150):
        buffer.add(
            obs_cond=np.random.randn(512).astype(np.float32),
            latent_w=np.random.randn(16, 7).astype(np.float32),
            reward=float(i),
            discount_factor=0.99,
            done=0.0,
            next_obs_cond=np.random.randn(512).astype(np.float32),
        )
    
    # Size should be capped at capacity
    assert buffer.size == 100, f"Expected size 100, got {buffer.size}"
    print("PASSED")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DSRL Off-Policy Component Tests")
    print("=" * 60)
    
    print("\n[Latent Q-Network Tests]")
    test_q_network_forward()
    test_q_network_backward()
    
    print("\n[Double Q-Network Tests]")
    test_double_q_network_forward()
    test_double_q_network_target()
    test_double_q_network_min_q()
    test_soft_update()
    
    print("\n[Temperature Tests]")
    test_learnable_temperature()
    test_fixed_temperature()
    
    print("\n[Replay Buffer Tests]")
    test_buffer_add_single()
    test_buffer_add_batch()
    test_buffer_sample()
    test_buffer_overflow()
    
    print("\n" + "=" * 60)
    print("ALL COMPONENT TESTS PASSED!")
    print("=" * 60)
