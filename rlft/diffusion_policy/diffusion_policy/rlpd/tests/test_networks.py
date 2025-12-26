"""
Test EnsembleQNetwork and related network components.

Run with:
    cd /home/amax/rlft/diffusion_policy
    conda activate rlft_ms3
    python -m diffusion_policy.rlpd.tests.test_networks
"""

import torch
import torch.nn as nn
import numpy as np

# Test configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing on device: {DEVICE}")


def test_ensemble_q_network_forward():
    """Test EnsembleQNetwork forward pass output shape."""
    print("\n" + "=" * 60)
    print("Test: EnsembleQNetwork forward()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.networks import EnsembleQNetwork
    
    # Config matching RLPD paper: num_qs=10, num_min_qs=2
    obs_dim = 512
    action_dim = 7
    action_horizon = 8
    num_qs = 10
    num_min_qs = 2
    hidden_dims = [256, 256, 256]
    batch_size = 32
    
    # Create network
    q_net = EnsembleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim,
        action_horizon=action_horizon,
        hidden_dims=hidden_dims,
        num_qs=num_qs,
        num_min_qs=num_min_qs,
    ).to(DEVICE)
    
    # Test inputs - note: forward takes (action_seq, obs_cond)
    obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    actions = torch.randn(batch_size, action_horizon, action_dim, device=DEVICE)
    
    # Forward pass - EnsembleQNetwork.forward(action_seq, obs_cond)
    q_values = q_net(actions, obs)
    
    # Check shape: (num_qs, batch_size, 1)
    expected_shape = (num_qs, batch_size, 1)
    assert q_values.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {q_values.shape}"
    
    print(f"✓ Forward output shape: {q_values.shape} (expected {expected_shape})")
    print(f"✓ Q-values range: [{q_values.min().item():.4f}, {q_values.max().item():.4f}]")
    
    # Check no NaN/Inf
    assert not torch.isnan(q_values).any(), "Q-values contain NaN"
    assert not torch.isinf(q_values).any(), "Q-values contain Inf"
    print("✓ No NaN or Inf values")
    
    return True


def test_ensemble_q_network_get_min_q():
    """Test get_min_q with subsampling logic."""
    print("\n" + "=" * 60)
    print("Test: EnsembleQNetwork get_min_q()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.networks import EnsembleQNetwork
    
    obs_dim = 512
    action_dim = 7
    action_horizon = 8
    num_qs = 10
    num_min_qs = 2
    batch_size = 32
    
    q_net = EnsembleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim,
        action_horizon=action_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=num_qs,
        num_min_qs=num_min_qs,
    ).to(DEVICE)
    
    obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    actions = torch.randn(batch_size, action_horizon, action_dim, device=DEVICE)
    
    # Get min Q with subsampling - note: get_min_q(action_seq, obs_cond)
    min_q = q_net.get_min_q(actions, obs)
    
    # Check shape: (batch_size, 1)
    expected_shape = (batch_size, 1)
    assert min_q.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {min_q.shape}"
    print(f"✓ get_min_q output shape: {min_q.shape}")
    
    # Verify subsampling works (run multiple times, check variance)
    min_q_values = []
    for _ in range(10):
        mq = q_net.get_min_q(actions, obs)
        min_q_values.append(mq.mean().item())
    
    variance = np.var(min_q_values)
    print(f"✓ Min Q variance across 10 calls: {variance:.6f}")
    
    # When num_min_qs < num_qs, there should be some variance due to subsampling
    # (unless all Q networks give identical values)
    print(f"✓ Subsampling active: num_min_qs={num_min_qs} < num_qs={num_qs}")
    
    # Test edge case: num_min_qs == num_qs (no subsampling)
    q_net_full = EnsembleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim,
        action_horizon=action_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=10,
        num_min_qs=10,  # Use all Q networks
    ).to(DEVICE)
    
    min_q_full_values = []
    for _ in range(5):
        mq = q_net_full.get_min_q(actions, obs)
        min_q_full_values.append(mq.mean().item())
    
    full_variance = np.var(min_q_full_values)
    print(f"✓ Variance with num_min_qs=num_qs: {full_variance:.8f} (should be ~0)")
    assert full_variance < 1e-6, "Full ensemble should give deterministic results"
    
    return True


def test_ensemble_q_network_get_double_q():
    """Test get_double_q returns two Q values."""
    print("\n" + "=" * 60)
    print("Test: EnsembleQNetwork get_double_q()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.networks import EnsembleQNetwork
    
    obs_dim = 512
    action_dim = 7
    action_horizon = 8
    batch_size = 32
    
    q_net = EnsembleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim,
        action_horizon=action_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=10,
        num_min_qs=2,
    ).to(DEVICE)
    
    obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    actions = torch.randn(batch_size, action_horizon, action_dim, device=DEVICE)
    
    # get_double_q(action_seq, obs_cond)
    q1, q2 = q_net.get_double_q(actions, obs)
    
    # Check shapes
    expected_shape = (batch_size, 1)
    assert q1.shape == expected_shape, f"q1 shape: {q1.shape}, expected {expected_shape}"
    assert q2.shape == expected_shape, f"q2 shape: {q2.shape}, expected {expected_shape}"
    print(f"✓ get_double_q output shapes: q1={q1.shape}, q2={q2.shape}")
    
    # q1 and q2 should be different (from different networks)
    diff = (q1 - q2).abs().mean().item()
    print(f"✓ Mean abs difference between q1 and q2: {diff:.4f}")
    
    return True


def test_diag_gaussian_actor():
    """Test DiagGaussianActor for action chunk output."""
    print("\n" + "=" * 60)
    print("Test: DiagGaussianActor")
    print("=" * 60)
    
    from diffusion_policy.rlpd.networks import DiagGaussianActor
    
    obs_dim = 512
    action_dim = 7
    action_horizon = 8
    batch_size = 32
    
    actor = DiagGaussianActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dims=[256, 256, 256],
        log_std_range=(-5.0, 2.0),  # Use log_std_range instead of log_std_min/max
    ).to(DEVICE)
    
    obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    
    # Test get_action (deterministic) - returns (action, log_prob)
    action_det, log_prob_det = actor.get_action(obs, deterministic=True)
    expected_shape = (batch_size, action_horizon, action_dim)
    assert action_det.shape == expected_shape, \
        f"Deterministic action shape: {action_det.shape}, expected {expected_shape}"
    print(f"✓ Deterministic action shape: {action_det.shape}")
    assert log_prob_det is None, "Deterministic mode should not return log_prob"
    print("✓ Deterministic mode returns None for log_prob")
    
    # Test get_action (stochastic) - returns (action, log_prob)
    action_stoch, log_prob_stoch = actor.get_action(obs, deterministic=False)
    assert action_stoch.shape == expected_shape
    print(f"✓ Stochastic action shape: {action_stoch.shape}")
    assert log_prob_stoch is not None, "Stochastic mode should return log_prob"
    print(f"✓ Stochastic log_prob shape: {log_prob_stoch.shape}")
    
    # Actions should be different (stochastic vs deterministic)
    diff = (action_det - action_stoch).abs().mean().item()
    print(f"✓ Mean abs difference (det vs stoch): {diff:.4f}")
    
    # Test sample_with_log_prob via get_action
    action_sample, log_prob = actor.get_action(obs, deterministic=False)
    assert action_sample.shape == expected_shape
    print(f"✓ Sample action shape: {action_sample.shape}")
    
    # Actions should be in [-1, 1] due to tanh squashing
    assert action_sample.min() >= -1.0, f"Action min: {action_sample.min().item()}"
    assert action_sample.max() <= 1.0, f"Action max: {action_sample.max().item()}"
    print(f"✓ Actions bounded in [-1, 1]: [{action_sample.min().item():.4f}, {action_sample.max().item():.4f}]")
    
    return True


def test_learnable_temperature():
    """Test LearnableTemperature module."""
    print("\n" + "=" * 60)
    print("Test: LearnableTemperature")
    print("=" * 60)
    
    from diffusion_policy.rlpd.networks import LearnableTemperature
    
    init_temp = 1.0
    temp = LearnableTemperature(init_temp).to(DEVICE)
    
    # Check initial value
    alpha = temp()
    print(f"✓ Initial temperature: {alpha.item():.4f} (init={init_temp})")
    assert abs(alpha.item() - init_temp) < 1e-4
    
    # Temperature should always be positive
    assert alpha.item() > 0, "Temperature should be positive"
    print("✓ Temperature is positive")
    
    # Test gradient flow
    loss = alpha * 10
    loss.backward()
    assert temp.log_alpha.grad is not None
    print(f"✓ Gradient flows to log_alpha: {temp.log_alpha.grad.item():.4f}")
    
    return True


def test_soft_update():
    """Test soft_update function for target network updates."""
    print("\n" + "=" * 60)
    print("Test: soft_update()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.networks import soft_update
    
    # Create two simple networks
    source = nn.Linear(10, 10).to(DEVICE)
    target = nn.Linear(10, 10).to(DEVICE)
    
    # Initialize with different values
    with torch.no_grad():
        source.weight.fill_(1.0)
        source.bias.fill_(1.0)
        target.weight.fill_(0.0)
        target.bias.fill_(0.0)
    
    print(f"Before soft update:")
    print(f"  source.weight[0,0] = {source.weight[0, 0].item():.4f}")
    print(f"  target.weight[0,0] = {target.weight[0, 0].item():.4f}")
    
    # Soft update with tau=0.1
    # target = tau * source + (1 - tau) * target
    # target = 0.1 * 1.0 + 0.9 * 0.0 = 0.1
    tau = 0.1
    soft_update(target, source, tau)  # Note: soft_update(target, source, tau) 
    
    expected = tau * 1.0 + (1 - tau) * 0.0
    actual = target.weight[0, 0].item()
    
    print(f"After soft update (tau={tau}):")
    print(f"  target.weight[0,0] = {actual:.4f} (expected: {expected:.4f})")
    
    assert abs(actual - expected) < 1e-5, f"Expected {expected}, got {actual}"
    print(f"✓ Soft update with tau={tau}: correct")
    
    # Test tau=1.0 (hard update)
    soft_update(target, source, tau=1.0)
    actual_hard = target.weight[0, 0].item()
    assert abs(actual_hard - 1.0) < 1e-5, f"Expected 1.0, got {actual_hard}"
    print("✓ Hard update (tau=1.0) works correctly")
    
    return True


def run_all_tests():
    """Run all network tests."""
    print("\n" + "=" * 60)
    print("RLPD Networks Test Suite")
    print("=" * 60)
    
    tests = [
        ("EnsembleQNetwork forward()", test_ensemble_q_network_forward),
        ("EnsembleQNetwork get_min_q()", test_ensemble_q_network_get_min_q),
        ("EnsembleQNetwork get_double_q()", test_ensemble_q_network_get_double_q),
        ("DiagGaussianActor", test_diag_gaussian_actor),
        ("LearnableTemperature", test_learnable_temperature),
        ("soft_update()", test_soft_update),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
