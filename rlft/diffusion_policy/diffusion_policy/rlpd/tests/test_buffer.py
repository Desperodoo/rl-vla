"""
Test OnlineReplayBuffer and OfflineDataBuffer.

Run with:
    cd /home/amax/rlft/diffusion_policy
    conda activate rlft_ms3
    python -m diffusion_policy.rlpd.tests.test_buffer
"""

import torch
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing on device: {DEVICE}")

# Demo file path
DEMO_PATH = os.path.expanduser("~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5")


def test_online_buffer_store_sample():
    """Test OnlineReplayBuffer store and sample."""
    print("\n" + "=" * 60)
    print("Test: OnlineReplayBuffer store() and sample()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.replay_buffer import OnlineReplayBuffer
    
    capacity = 100
    num_envs = 4
    obs_dim = 32
    action_dim = 7
    action_horizon = 8
    gamma = 0.99
    
    buffer = OnlineReplayBuffer(
        capacity=capacity,
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        gamma=gamma,
        device=DEVICE,
    )
    
    assert buffer.size == 0, f"Initial size should be 0, got {buffer.size}"
    print(f"✓ Initial buffer size: {buffer.size}")
    
    # Store some transitions
    n_transitions = 10
    for _ in range(n_transitions):
        obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        action = np.random.randn(num_envs, action_horizon, action_dim).astype(np.float32)
        reward = np.random.randn(num_envs).astype(np.float32)
        next_obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        done = np.zeros(num_envs, dtype=np.float32)
        cumulative_reward = np.random.randn(num_envs).astype(np.float32)
        chunk_done = np.zeros(num_envs, dtype=np.float32)
        discount_factor = np.full(num_envs, gamma ** action_horizon, dtype=np.float32)
        effective_length = np.full(num_envs, action_horizon, dtype=np.float32)
        
        buffer.store(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            cumulative_reward=cumulative_reward,
            chunk_done=chunk_done,
            discount_factor=discount_factor,
            effective_length=effective_length,
        )
    
    expected_size = n_transitions * num_envs
    assert buffer.size == expected_size, f"Expected size {expected_size}, got {buffer.size}"
    print(f"✓ After {n_transitions} store() calls: size = {buffer.size}")
    
    # Sample batch
    batch_size = 16
    batch = buffer.sample(batch_size)
    
    # Check batch shapes
    assert batch["obs"].shape == (batch_size, obs_dim), f"obs: {batch['obs'].shape}"
    assert batch["action"].shape == (batch_size, action_horizon, action_dim), f"action: {batch['action'].shape}"
    assert batch["reward"].shape == (batch_size,), f"reward: {batch['reward'].shape}"
    assert batch["next_obs"].shape == (batch_size, obs_dim), f"next_obs: {batch['next_obs'].shape}"
    assert batch["done"].shape == (batch_size,), f"done: {batch['done'].shape}"
    assert batch["cumulative_reward"].shape == (batch_size,), f"cumulative_reward: {batch['cumulative_reward'].shape}"
    assert batch["chunk_done"].shape == (batch_size,), f"chunk_done: {batch['chunk_done'].shape}"
    assert batch["discount_factor"].shape == (batch_size,), f"discount_factor: {batch['discount_factor'].shape}"
    
    print(f"✓ Sampled batch with correct shapes")
    
    # Check tensor device
    assert batch["obs"].device.type == DEVICE.split(":")[0], f"Expected device {DEVICE}, got {batch['obs'].device}"
    print(f"✓ Batch tensors on correct device: {DEVICE}")
    
    return True


def test_online_buffer_ring_overwrite():
    """Test OnlineReplayBuffer ring buffer overwrite behavior."""
    print("\n" + "=" * 60)
    print("Test: OnlineReplayBuffer ring buffer overwrite")
    print("=" * 60)
    
    from diffusion_policy.rlpd.replay_buffer import OnlineReplayBuffer
    
    capacity = 10  # Small capacity to test overwrite
    num_envs = 2
    obs_dim = 8
    action_dim = 4
    action_horizon = 4
    
    buffer = OnlineReplayBuffer(
        capacity=capacity,
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        gamma=0.99,
        device=DEVICE,
    )
    
    # Store more than capacity to trigger overwrite
    n_transitions = capacity + 5  # 15 > 10
    
    for i in range(n_transitions):
        obs = np.full((num_envs, obs_dim), i, dtype=np.float32)  # Use index as value
        action = np.full((num_envs, action_horizon, action_dim), i, dtype=np.float32)
        reward = np.full(num_envs, i, dtype=np.float32)
        next_obs = np.full((num_envs, obs_dim), i + 1, dtype=np.float32)
        done = np.zeros(num_envs, dtype=np.float32)
        
        buffer.store(obs, action, reward, next_obs, done)
    
    # Size should be capped at capacity * num_envs
    max_size = capacity * num_envs
    assert buffer.size == max_size, f"Expected size {max_size}, got {buffer.size}"
    print(f"✓ Buffer size capped at capacity: {buffer.size}")
    
    # The oldest data (indices 0-4) should be overwritten
    # Check that data from early indices is gone
    # Sample multiple times to verify data range
    batch = buffer.sample(max_size)  # Sample all
    
    # The minimum reward should be >= 5 (since 0-4 are overwritten)
    min_reward = batch["reward"].min().item()
    max_reward = batch["reward"].max().item()
    print(f"✓ Reward range in buffer: [{min_reward}, {max_reward}]")
    print(f"  (Old data [0-4] should be overwritten)")
    
    return True


def test_online_buffer_store_without_smdp():
    """Test OnlineReplayBuffer stores correctly without SMDP fields."""
    print("\n" + "=" * 60)
    print("Test: OnlineReplayBuffer without SMDP fields")
    print("=" * 60)
    
    from diffusion_policy.rlpd.replay_buffer import OnlineReplayBuffer
    
    capacity = 100
    num_envs = 4
    obs_dim = 32
    action_dim = 7
    action_horizon = 8
    gamma = 0.99
    
    buffer = OnlineReplayBuffer(
        capacity=capacity,
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        gamma=gamma,
        device=DEVICE,
    )
    
    # Store without SMDP fields (should use defaults)
    obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
    action = np.random.randn(num_envs, action_horizon, action_dim).astype(np.float32)
    reward = np.ones(num_envs, dtype=np.float32) * 5.0
    next_obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
    done = np.zeros(num_envs, dtype=np.float32)
    
    buffer.store(obs, action, reward, next_obs, done)  # No SMDP fields
    
    # Sample and check defaults
    batch = buffer.sample(num_envs)
    
    # cumulative_reward should default to reward
    assert torch.allclose(batch["cumulative_reward"], batch["reward"]), \
        "cumulative_reward should default to reward"
    print(f"✓ cumulative_reward defaults to reward: {batch['cumulative_reward'][0].item()}")
    
    # chunk_done should default to done
    assert torch.allclose(batch["chunk_done"], batch["done"]), \
        "chunk_done should default to done"
    print(f"✓ chunk_done defaults to done")
    
    # discount_factor should default to gamma
    assert torch.allclose(batch["discount_factor"], torch.full_like(batch["discount_factor"], gamma)), \
        f"discount_factor should default to {gamma}"
    print(f"✓ discount_factor defaults to gamma={gamma}")
    
    return True


def test_offline_buffer_from_dict():
    """Test OfflineDataBuffer loading from dict."""
    print("\n" + "=" * 60)
    print("Test: OfflineDataBuffer from dict")
    print("=" * 60)
    
    from diffusion_policy.rlpd.replay_buffer import OfflineDataBuffer
    
    n_samples = 100
    obs_dim = 32
    action_dim = 7
    action_horizon = 8
    
    # Create mock offline data
    data = {
        "obs": np.random.randn(n_samples, obs_dim).astype(np.float32),
        "action": np.random.randn(n_samples, action_horizon, action_dim).astype(np.float32),
        "reward": np.random.randn(n_samples).astype(np.float32),
        "next_obs": np.random.randn(n_samples, obs_dim).astype(np.float32),
        "done": np.zeros(n_samples, dtype=np.float32),
        "cumulative_reward": np.random.randn(n_samples).astype(np.float32),
        "chunk_done": np.zeros(n_samples, dtype=np.float32),
        "discount_factor": np.full(n_samples, 0.99 ** action_horizon, dtype=np.float32),
    }
    
    buffer = OfflineDataBuffer(data, device=DEVICE)
    
    assert buffer.size == n_samples, f"Expected size {n_samples}, got {buffer.size}"
    print(f"✓ OfflineDataBuffer size: {buffer.size}")
    
    # Sample batch
    batch_size = 16
    batch = buffer.sample(batch_size)
    
    # Check shapes
    assert batch["obs"].shape == (batch_size, obs_dim)
    assert batch["action"].shape == (batch_size, action_horizon, action_dim)
    print(f"✓ Sampled batch with correct shapes")
    
    # Check device
    assert batch["obs"].device.type == DEVICE.split(":")[0]
    print(f"✓ Batch on correct device: {DEVICE}")
    
    return True


def test_mixed_sampling():
    """Test sample_mixed() with different online_ratio values."""
    print("\n" + "=" * 60)
    print("Test: sample_mixed() with varying online_ratio")
    print("=" * 60)
    
    from diffusion_policy.rlpd.replay_buffer import OnlineReplayBuffer, OfflineDataBuffer
    
    capacity = 1000
    num_envs = 4
    obs_dim = 32
    action_dim = 7
    action_horizon = 8
    n_offline = 500
    batch_size = 256
    
    # Create online buffer with distinguishable data
    online_buffer = OnlineReplayBuffer(
        capacity=capacity,
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        gamma=0.99,
        device=DEVICE,
    )
    
    # Fill online buffer (reward = 100 to distinguish)
    for _ in range(100):
        obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        action = np.random.randn(num_envs, action_horizon, action_dim).astype(np.float32)
        reward = np.full(num_envs, 100.0, dtype=np.float32)  # Online: reward=100
        next_obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        done = np.zeros(num_envs, dtype=np.float32)
        online_buffer.store(obs, action, reward, next_obs, done)
    
    # Create offline buffer (reward = -100 to distinguish)
    offline_data = {
        "obs": np.random.randn(n_offline, obs_dim).astype(np.float32),
        "action": np.random.randn(n_offline, action_horizon, action_dim).astype(np.float32),
        "reward": np.full(n_offline, -100.0, dtype=np.float32),  # Offline: reward=-100
        "next_obs": np.random.randn(n_offline, obs_dim).astype(np.float32),
        "done": np.zeros(n_offline, dtype=np.float32),
        "cumulative_reward": np.full(n_offline, -100.0, dtype=np.float32),
        "chunk_done": np.zeros(n_offline, dtype=np.float32),
        "discount_factor": np.full(n_offline, 0.99 ** action_horizon, dtype=np.float32),
    }
    offline_buffer = OfflineDataBuffer(offline_data, device=DEVICE)
    
    # Test online_ratio = 0.5 (RLPD default)
    print("\nTesting online_ratio = 0.5:")
    batch = online_buffer.sample_mixed(batch_size, offline_buffer, online_ratio=0.5)
    
    n_online = (batch["reward"] > 0).sum().item()
    n_offline = (batch["reward"] < 0).sum().item()
    actual_ratio = n_online / batch_size
    
    print(f"  Online samples: {n_online}, Offline samples: {n_offline}")
    print(f"  Actual online ratio: {actual_ratio:.2f} (expected ~0.5)")
    assert 0.3 < actual_ratio < 0.7, f"Expected ratio ~0.5, got {actual_ratio}"
    print(f"  ✓ Mixed sampling with ratio=0.5 works")
    
    # Test online_ratio = 1.0 (pure online)
    print("\nTesting online_ratio = 1.0:")
    batch = online_buffer.sample_mixed(batch_size, offline_buffer, online_ratio=1.0)
    
    n_online = (batch["reward"] > 0).sum().item()
    assert n_online == batch_size, f"Expected all online, got {n_online}/{batch_size}"
    print(f"  ✓ Pure online sampling works")
    
    # Test online_ratio = 0.0 (pure offline)
    print("\nTesting online_ratio = 0.0:")
    batch = online_buffer.sample_mixed(batch_size, offline_buffer, online_ratio=0.0)
    
    n_offline = (batch["reward"] < 0).sum().item()
    assert n_offline == batch_size, f"Expected all offline, got {n_offline}/{batch_size}"
    print(f"  ✓ Pure offline sampling works")
    
    return True


def test_load_real_offline_data():
    """Test loading real offline demo data."""
    print("\n" + "=" * 60)
    print("Test: Load real offline demo data")
    print("=" * 60)
    
    import h5py
    
    if not os.path.exists(DEMO_PATH):
        print(f"⚠ Demo file not found: {DEMO_PATH}")
        print("  Skipping real data test")
        return True
    
    print(f"Loading: {DEMO_PATH}")
    
    # Open and inspect
    with h5py.File(DEMO_PATH, "r") as f:
        print(f"✓ File opened successfully")
        print(f"  Number of trajectories: {len(f.keys())}")
        
        # Check first trajectory
        first_key = list(f.keys())[0]
        traj = f[first_key]
        
        print(f"\n  First trajectory '{first_key}':")
        print(f"    Keys: {list(traj.keys())}")
        
        if "actions" in traj:
            actions = np.array(traj["actions"])
            print(f"    actions shape: {actions.shape}")
        
        if "obs" in traj:
            obs_keys = list(traj["obs"].keys())
            print(f"    obs keys: {obs_keys}")
            
            if "state" in traj["obs"]:
                state = np.array(traj["obs"]["state"])
                print(f"    state shape: {state.shape}")
    
    print(f"\n✓ Real demo data inspection complete")
    
    return True


def run_all_tests():
    """Run all buffer tests."""
    print("\n" + "=" * 60)
    print("RLPD Buffer Test Suite")
    print("=" * 60)
    
    tests = [
        ("OnlineReplayBuffer store/sample", test_online_buffer_store_sample),
        ("OnlineReplayBuffer ring overwrite", test_online_buffer_ring_overwrite),
        ("OnlineReplayBuffer without SMDP", test_online_buffer_store_without_smdp),
        ("OfflineDataBuffer from dict", test_offline_buffer_from_dict),
        ("sample_mixed() ratios", test_mixed_sampling),
        ("Load real offline data", test_load_real_offline_data),
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
