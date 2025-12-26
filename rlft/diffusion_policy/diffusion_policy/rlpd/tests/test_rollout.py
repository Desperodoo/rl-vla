"""
Test RolloutBuffer SMDP reward computation.

Run with:
    cd /home/amax/rlft/diffusion_policy
    conda activate rlft_ms3
    python -m diffusion_policy.rlpd.tests.test_rollout
"""

import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing on device: {DEVICE}")


def test_rollout_buffer_basic():
    """Test basic RolloutBuffer functionality."""
    print("\n" + "=" * 60)
    print("Test: RolloutBuffer basic add_step()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.replay_buffer import RolloutBuffer
    
    num_envs = 4
    obs_dim = 32
    action_dim = 7
    action_horizon = 8
    gamma = 0.99
    
    buffer = RolloutBuffer(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        gamma=gamma,
    )
    
    # Add steps (simulate 8-step chunk)
    for step in range(action_horizon):
        obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        action = np.random.randn(num_envs, action_dim).astype(np.float32)
        reward = np.ones(num_envs, dtype=np.float32) * (step + 1)  # Increasing rewards
        next_obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        done = np.zeros(num_envs, dtype=np.float32)
        
        buffer.add_step(obs, action, reward, next_obs, done)
    
    assert len(buffer.obs_list) == action_horizon, f"Expected {action_horizon} steps, got {len(buffer.obs_list)}"
    assert len(buffer.reward_list) == action_horizon
    print(f"✓ Added {action_horizon} steps to RolloutBuffer")
    
    return True


def test_rollout_smdp_no_done():
    """Test SMDP computation when no episode terminates."""
    print("\n" + "=" * 60)
    print("Test: RolloutBuffer SMDP (no early done)")
    print("=" * 60)
    
    from diffusion_policy.rlpd.replay_buffer import RolloutBuffer
    
    num_envs = 4
    obs_dim = 32
    action_dim = 7
    action_horizon = 8
    gamma = 0.99
    
    buffer = RolloutBuffer(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        gamma=gamma,
    )
    
    # Add constant reward=1.0 for all steps, no done
    for step in range(action_horizon):
        obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
        action = np.zeros((num_envs, action_dim), dtype=np.float32)
        reward = np.ones(num_envs, dtype=np.float32)  # r=1.0 each step
        next_obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
        done = np.zeros(num_envs, dtype=np.float32)
        
        buffer.add_step(obs, action, reward, next_obs, done)
    
    # Compute SMDP rewards
    cumulative_reward, chunk_done, discount_factor, effective_length = buffer.compute_smdp_rewards()
    
    # Manual calculation:
    # cumulative_reward = 1 + γ + γ² + ... + γ^7 = (1 - γ^8) / (1 - γ)
    expected_cumulative = sum(gamma ** i for i in range(action_horizon))
    
    print(f"Expected cumulative reward: {expected_cumulative:.6f}")
    print(f"Actual cumulative reward (env 0): {cumulative_reward[0]:.6f}")
    
    assert np.allclose(cumulative_reward, expected_cumulative, atol=1e-5), \
        f"Expected {expected_cumulative}, got {cumulative_reward[0]}"
    print("✓ Cumulative reward matches manual calculation")
    
    # chunk_done should be 0 (no early termination)
    assert np.all(chunk_done == 0.0), f"Expected no chunk_done, got {chunk_done}"
    print("✓ chunk_done = 0 (no early termination)")
    
    # effective_length should be action_horizon
    assert np.all(effective_length == action_horizon), \
        f"Expected effective_length={action_horizon}, got {effective_length}"
    print(f"✓ effective_length = {action_horizon}")
    
    # discount_factor should be γ^8
    expected_discount = gamma ** action_horizon
    assert np.allclose(discount_factor, expected_discount, atol=1e-6), \
        f"Expected discount_factor={expected_discount}, got {discount_factor[0]}"
    print(f"✓ discount_factor = γ^{action_horizon} = {expected_discount:.6f}")
    
    return True


def test_rollout_smdp_early_done():
    """Test SMDP computation with early episode termination."""
    print("\n" + "=" * 60)
    print("Test: RolloutBuffer SMDP (early done)")
    print("=" * 60)
    
    from diffusion_policy.rlpd.replay_buffer import RolloutBuffer
    
    num_envs = 4
    obs_dim = 32
    action_dim = 7
    action_horizon = 8
    gamma = 0.99
    
    buffer = RolloutBuffer(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        gamma=gamma,
    )
    
    # Env 0: done at step 3 (0-indexed, so steps 0,1,2 contribute, total 3 steps)
    # Env 1: done at step 5 
    # Env 2: done at step 7 (last step)
    # Env 3: no done (full chunk)
    
    done_at = [3, 5, 7, -1]  # -1 means no done
    
    for step in range(action_horizon):
        obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
        action = np.zeros((num_envs, action_dim), dtype=np.float32)
        reward = np.ones(num_envs, dtype=np.float32)  # r=1.0 each step
        next_obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
        
        # Set done flags
        done = np.zeros(num_envs, dtype=np.float32)
        for env_idx, d in enumerate(done_at):
            if step == d:
                done[env_idx] = 1.0
        
        buffer.add_step(obs, action, reward, next_obs, done)
    
    # Compute SMDP rewards
    cumulative_reward, chunk_done, discount_factor, effective_length = buffer.compute_smdp_rewards()
    
    print("Results per environment:")
    for env_idx in range(num_envs):
        expected_steps = done_at[env_idx] + 1 if done_at[env_idx] >= 0 else action_horizon
        expected_cumul = sum(gamma ** i for i in range(expected_steps))
        expected_discount = gamma ** expected_steps
        expected_chunk_done = 1.0 if done_at[env_idx] >= 0 else 0.0
        
        print(f"  Env {env_idx}: effective_length={effective_length[env_idx]:.0f} (expected {expected_steps}), "
              f"cumulative={cumulative_reward[env_idx]:.4f} (expected {expected_cumul:.4f}), "
              f"chunk_done={chunk_done[env_idx]}")
        
        assert np.isclose(effective_length[env_idx], expected_steps), \
            f"Env {env_idx}: expected effective_length={expected_steps}, got {effective_length[env_idx]}"
        assert np.isclose(cumulative_reward[env_idx], expected_cumul, atol=1e-5), \
            f"Env {env_idx}: expected cumulative={expected_cumul}, got {cumulative_reward[env_idx]}"
        assert np.isclose(chunk_done[env_idx], expected_chunk_done), \
            f"Env {env_idx}: expected chunk_done={expected_chunk_done}, got {chunk_done[env_idx]}"
        assert np.isclose(discount_factor[env_idx], expected_discount, atol=1e-6), \
            f"Env {env_idx}: expected discount={expected_discount}, got {discount_factor[env_idx]}"
    
    print("✓ All environments computed correctly with early termination")
    
    return True


def test_rollout_smdp_varying_rewards():
    """Test SMDP computation with varying rewards."""
    print("\n" + "=" * 60)
    print("Test: RolloutBuffer SMDP (varying rewards)")
    print("=" * 60)
    
    from diffusion_policy.rlpd.replay_buffer import RolloutBuffer
    
    num_envs = 2
    obs_dim = 32
    action_dim = 7
    action_horizon = 4
    gamma = 0.9  # Use 0.9 for easier manual calculation
    
    buffer = RolloutBuffer(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        gamma=gamma,
    )
    
    # Rewards: [1, 2, 3, 4] for each step
    rewards_per_step = [1.0, 2.0, 3.0, 4.0]
    
    for step, r in enumerate(rewards_per_step):
        obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
        action = np.zeros((num_envs, action_dim), dtype=np.float32)
        reward = np.full(num_envs, r, dtype=np.float32)
        next_obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
        done = np.zeros(num_envs, dtype=np.float32)
        
        buffer.add_step(obs, action, reward, next_obs, done)
    
    cumulative_reward, chunk_done, discount_factor, effective_length = buffer.compute_smdp_rewards()
    
    # Manual calculation:
    # cumulative = 1*γ^0 + 2*γ^1 + 3*γ^2 + 4*γ^3
    #            = 1 + 2*0.9 + 3*0.81 + 4*0.729
    #            = 1 + 1.8 + 2.43 + 2.916 = 8.146
    expected_cumulative = sum(r * (gamma ** i) for i, r in enumerate(rewards_per_step))
    
    print(f"Rewards per step: {rewards_per_step}")
    print(f"Expected cumulative: {expected_cumulative:.6f}")
    print(f"Actual cumulative: {cumulative_reward[0]:.6f}")
    
    assert np.allclose(cumulative_reward, expected_cumulative, atol=1e-5), \
        f"Expected {expected_cumulative}, got {cumulative_reward[0]}"
    print("✓ Varying rewards computed correctly")
    
    return True


def test_rollout_get_transitions():
    """Test get_transitions() returns correct shapes."""
    print("\n" + "=" * 60)
    print("Test: RolloutBuffer get_transitions()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.replay_buffer import RolloutBuffer
    
    num_envs = 4
    obs_dim = 32
    action_dim = 7
    action_horizon = 8
    gamma = 0.99
    
    buffer = RolloutBuffer(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        gamma=gamma,
    )
    
    # Add steps
    for step in range(action_horizon):
        obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        action = np.random.randn(num_envs, action_dim).astype(np.float32)
        reward = np.random.randn(num_envs).astype(np.float32)
        next_obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        done = np.zeros(num_envs, dtype=np.float32)
        
        buffer.add_step(obs, action, reward, next_obs, done)
    
    # Get transitions
    (obs, action, reward, next_obs, done,
     cumulative_reward, chunk_done, discount_factor, effective_length) = buffer.get_transitions()
    
    # Check shapes
    assert obs.shape == (num_envs, obs_dim), f"obs shape: {obs.shape}"
    assert action.shape == (num_envs, action_horizon, action_dim), f"action shape: {action.shape}"
    assert reward.shape == (num_envs,), f"reward shape: {reward.shape}"
    assert next_obs.shape == (num_envs, obs_dim), f"next_obs shape: {next_obs.shape}"
    assert done.shape == (num_envs,), f"done shape: {done.shape}"
    assert cumulative_reward.shape == (num_envs,), f"cumulative_reward shape: {cumulative_reward.shape}"
    assert chunk_done.shape == (num_envs,), f"chunk_done shape: {chunk_done.shape}"
    assert discount_factor.shape == (num_envs,), f"discount_factor shape: {discount_factor.shape}"
    assert effective_length.shape == (num_envs,), f"effective_length shape: {effective_length.shape}"
    
    print(f"✓ obs shape: {obs.shape}")
    print(f"✓ action shape: {action.shape}")
    print(f"✓ reward shape: {reward.shape}")
    print(f"✓ next_obs shape: {next_obs.shape}")
    print(f"✓ SMDP fields shapes correct")
    
    return True


def test_rollout_reset():
    """Test RolloutBuffer reset() clears all data."""
    print("\n" + "=" * 60)
    print("Test: RolloutBuffer reset()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.replay_buffer import RolloutBuffer
    
    num_envs = 4
    obs_dim = 32
    action_dim = 7
    action_horizon = 8
    gamma = 0.99
    
    buffer = RolloutBuffer(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        gamma=gamma,
    )
    
    # Add some steps
    for step in range(4):
        obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
        action = np.zeros((num_envs, action_dim), dtype=np.float32)
        reward = np.ones(num_envs, dtype=np.float32)
        next_obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
        done = np.zeros(num_envs, dtype=np.float32)
        buffer.add_step(obs, action, reward, next_obs, done)
    
    assert len(buffer.obs_list) == 4
    print(f"✓ Before reset: {len(buffer.obs_list)} steps")
    
    # Reset
    buffer.reset()
    
    assert len(buffer.obs_list) == 0
    assert len(buffer.reward_list) == 0
    assert len(buffer.action_list) == 0
    print(f"✓ After reset: {len(buffer.obs_list)} steps")
    
    return True


def run_all_tests():
    """Run all rollout tests."""
    print("\n" + "=" * 60)
    print("RLPD RolloutBuffer Test Suite")
    print("=" * 60)
    
    tests = [
        ("RolloutBuffer basic", test_rollout_buffer_basic),
        ("SMDP no done", test_rollout_smdp_no_done),
        ("SMDP early done", test_rollout_smdp_early_done),
        ("SMDP varying rewards", test_rollout_smdp_varying_rewards),
        ("get_transitions()", test_rollout_get_transitions),
        ("reset()", test_rollout_reset),
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
