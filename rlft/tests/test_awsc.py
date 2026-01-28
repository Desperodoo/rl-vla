"""
Tests for AWSC Agent and SuccessReplayBuffer.

Tests:
1. AWSC Agent instantiation (state and RGB modes)
2. Pretrained checkpoint loading (ShortCut Flow BC and AW-ShortCut Flow)
3. Forward pass and action sampling
4. Loss computation (actor and critic)
5. SuccessReplayBuffer filtering functionality
"""

import pytest
import torch
import numpy as np
import tempfile
import os

# Skip tests if dependencies not available
try:
    from rlft.networks import ShortCutVelocityUNet1D, EnsembleQNetwork
    from rlft.algorithms.online_rl import AWSCAgent
    from rlft.buffers import SuccessReplayBuffer
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    import_error = str(e)


@pytest.mark.skipif(not HAS_DEPS, reason=f"Dependencies not available: {import_error if not HAS_DEPS else ''}")
class TestAWSCAgent:
    """Test suite for AWSC Agent."""
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @pytest.fixture
    def state_agent(self, device):
        """Create AWSC agent for state-only observations."""
        obs_dim = 32
        action_dim = 7
        
        velocity_net = ShortCutVelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=128,
            down_dims=(64, 128, 256),
            kernel_size=5,
            n_groups=8,
        ).to(device)
        
        agent = AWSCAgent(
            velocity_net=velocity_net,
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=2,
            pred_horizon=16,
            act_horizon=8,
            num_qs=4,
            num_min_qs=2,
            q_hidden_dims=[64, 64],
            num_inference_steps=4,
            beta=10.0,
            bc_weight=1.0,
            shortcut_weight=0.3,
            gamma=0.99,
            tau=0.005,
            device=device,
        ).to(device)
        
        return agent
    
    @pytest.fixture
    def rgb_agent(self, device):
        """Create AWSC agent for RGB observations (larger obs_dim)."""
        obs_dim = 2 * (256 + 32)  # obs_horizon * (visual_dim + state_dim)
        action_dim = 7
        
        velocity_net = ShortCutVelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=128,
            down_dims=(64, 128, 256),
            kernel_size=5,
            n_groups=8,
        ).to(device)
        
        agent = AWSCAgent(
            velocity_net=velocity_net,
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=2,
            pred_horizon=16,
            act_horizon=8,
            num_qs=4,
            num_min_qs=2,
            q_hidden_dims=[64, 64],
            num_inference_steps=4,
            device=device,
        ).to(device)
        
        return agent
    
    def test_agent_initialization_state(self, state_agent, device):
        """Test AWSC agent initialization with state observations."""
        assert state_agent is not None
        assert hasattr(state_agent, 'velocity_net')
        assert hasattr(state_agent, 'velocity_net_ema')
        assert hasattr(state_agent, 'critic')
        assert hasattr(state_agent, 'critic_target')
        
        # Check EMA is a copy
        for p_main, p_ema in zip(state_agent.velocity_net.parameters(), 
                                  state_agent.velocity_net_ema.parameters()):
            assert torch.allclose(p_main, p_ema)
            assert not p_ema.requires_grad
    
    def test_agent_initialization_rgb(self, rgb_agent, device):
        """Test AWSC agent initialization with RGB observations."""
        assert rgb_agent is not None
        assert rgb_agent.obs_dim == 2 * (256 + 32)
    
    def test_select_action(self, state_agent, device):
        """Test action selection."""
        batch_size = 4
        obs_features = torch.randn(batch_size, state_agent.obs_dim, device=device)
        
        # Deterministic action
        action_det = state_agent.select_action(obs_features, deterministic=True)
        assert action_det.shape == (batch_size, state_agent.act_horizon, state_agent.action_dim)
        assert (action_det >= -1.0).all() and (action_det <= 1.0).all()
        
        # Stochastic action (with exploration noise)
        action_stoch = state_agent.select_action(obs_features, deterministic=False)
        assert action_stoch.shape == (batch_size, state_agent.act_horizon, state_agent.action_dim)
    
    def test_get_action(self, state_agent, device):
        """Test get_action (full pred_horizon sequence)."""
        batch_size = 4
        obs_features = torch.randn(batch_size, state_agent.obs_dim, device=device)
        
        actions = state_agent.get_action(obs_features)
        assert actions.shape == (batch_size, state_agent.pred_horizon, state_agent.action_dim)
    
    def test_compute_actor_loss(self, state_agent, device):
        """Test actor loss computation."""
        batch_size = 8
        obs_features = torch.randn(batch_size, state_agent.obs_dim, device=device)
        actions = torch.randn(batch_size, state_agent.pred_horizon, state_agent.action_dim, device=device)
        
        loss, metrics = state_agent.compute_actor_loss(obs_features, actions)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert "actor_loss" in metrics
        assert "flow_loss" in metrics
        assert "shortcut_loss" in metrics
        assert "q_mean" in metrics
        assert "weight_mean" in metrics
    
    def test_compute_actor_loss_with_filter(self, state_agent, device):
        """Test actor loss computation with policy-critic data separation."""
        state_agent.filter_policy_data = True
        state_agent.advantage_threshold = 0.0
        
        batch_size = 8
        obs_features = torch.randn(batch_size, state_agent.obs_dim, device=device)
        actions = torch.randn(batch_size, state_agent.pred_horizon, state_agent.action_dim, device=device)
        is_demo = torch.tensor([True, True, False, False, False, False, True, False], device=device)
        
        loss, metrics = state_agent.compute_actor_loss(obs_features, actions, is_demo=is_demo)
        
        assert isinstance(loss, torch.Tensor)
        assert metrics["n_demo_samples"] == 3
    
    def test_compute_critic_loss(self, state_agent, device):
        """Test critic loss computation."""
        batch_size = 8
        obs_features = torch.randn(batch_size, state_agent.obs_dim, device=device)
        next_obs_features = torch.randn(batch_size, state_agent.obs_dim, device=device)
        actions = torch.randn(batch_size, state_agent.act_horizon, state_agent.action_dim, device=device)
        rewards = torch.randn(batch_size, device=device)
        dones = torch.zeros(batch_size, device=device)
        
        loss, metrics = state_agent.compute_critic_loss(
            obs_features, actions, next_obs_features, rewards, dones
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert "critic_loss" in metrics
        assert "q_mean" in metrics
        assert "td_target_mean" in metrics
    
    def test_update_ema(self, state_agent, device):
        """Test EMA network update."""
        # Modify main network
        for p in state_agent.velocity_net.parameters():
            p.data += 0.1
        
        # Update EMA
        state_agent.update_ema()
        
        # Check EMA moved toward main (not exact match due to decay)
        for p_main, p_ema in zip(state_agent.velocity_net.parameters(),
                                  state_agent.velocity_net_ema.parameters()):
            # EMA should have moved
            pass  # Just check it runs without error
    
    def test_update_target(self, state_agent, device):
        """Test target critic update."""
        state_agent.update_target()
        # Should not raise
    
    def test_load_pretrained_bc_checkpoint(self, state_agent, device):
        """Test loading from ShortCut Flow BC checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock BC checkpoint
            checkpoint = {
                "agent": state_agent.velocity_net.state_dict(),
            }
            # Add velocity_net. prefix
            checkpoint["agent"] = {
                f"velocity_net.{k}": v for k, v in state_agent.velocity_net.state_dict().items()
            }
            
            checkpoint_path = os.path.join(tmpdir, "bc_checkpoint.pt")
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint
            state_agent.load_pretrained(checkpoint_path, load_critic=False)
            # Should not raise
    
    def test_load_pretrained_awsc_checkpoint(self, state_agent, device):
        """Test loading from AW-ShortCut Flow checkpoint with critic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock AW-ShortCut checkpoint with EMA
            checkpoint = {
                "ema_agent": {},
            }
            
            # Add velocity network
            for k, v in state_agent.velocity_net.state_dict().items():
                checkpoint["ema_agent"][f"velocity_net.{k}"] = v
            
            # Add critic (ensemble)
            for k, v in state_agent.critic.state_dict().items():
                checkpoint["ema_agent"][f"critic.{k}"] = v
            
            checkpoint_path = os.path.join(tmpdir, "awsc_checkpoint.pt")
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint with critic
            state_agent.load_pretrained(checkpoint_path, load_critic=True, use_ema=True)
            # Should not raise


@pytest.mark.skipif(not HAS_DEPS, reason=f"Dependencies not available")
class TestSuccessReplayBuffer:
    """Test suite for SuccessReplayBuffer."""
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @pytest.fixture
    def buffer(self, device):
        """Create success replay buffer."""
        return SuccessReplayBuffer(
            capacity=100,
            num_envs=2,
            state_dim=32,
            action_dim=7,
            action_horizon=8,
            obs_horizon=2,
            include_rgb=False,
            gamma=0.99,
            device=device,
            min_success_ratio=0.3,
        )
    
    def test_buffer_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.size == 0
        assert buffer.capacity == 100
        assert buffer.num_success == 0
        assert buffer.num_fail == 0
    
    def test_store_transitions(self, buffer):
        """Test storing transitions."""
        batch_size = 2
        obs = {"state": np.random.randn(batch_size, 2, 32).astype(np.float32)}
        next_obs = {"state": np.random.randn(batch_size, 2, 32).astype(np.float32)}
        action = np.random.randn(batch_size, 8, 7).astype(np.float32)
        reward = np.random.randn(batch_size).astype(np.float32)
        done = np.array([0.0, 1.0], dtype=np.float32)
        cumulative_reward = np.random.randn(batch_size).astype(np.float32)
        chunk_done = done.copy()
        discount_factor = np.ones(batch_size, dtype=np.float32) * 0.99
        effective_length = np.ones(batch_size, dtype=np.float32) * 8
        success = np.array([0.0, 1.0], dtype=np.float32)
        
        buffer.store(
            obs, action, reward, next_obs, done,
            cumulative_reward, chunk_done, discount_factor, effective_length,
            success=success
        )
        
        assert buffer.size == 2
        assert buffer.num_success == 1
        assert buffer.num_fail == 1
    
    def test_sample(self, buffer, device):
        """Test standard sampling."""
        # Fill buffer
        for _ in range(10):
            batch_size = 2
            obs = {"state": np.random.randn(batch_size, 2, 32).astype(np.float32)}
            next_obs = {"state": np.random.randn(batch_size, 2, 32).astype(np.float32)}
            action = np.random.randn(batch_size, 8, 7).astype(np.float32)
            reward = np.random.randn(batch_size).astype(np.float32)
            done = np.random.choice([0.0, 1.0], size=batch_size).astype(np.float32)
            success = done.copy()
            
            buffer.store(
                obs, action, reward, next_obs, done,
                cumulative_reward=reward,
                chunk_done=done,
                discount_factor=np.ones(batch_size) * 0.99,
                effective_length=np.ones(batch_size) * 8,
                success=success
            )
        
        batch = buffer.sample(4)
        
        assert batch["observations"]["state"].shape == (4, 2, 32)
        assert batch["actions"].shape == (4, 8, 7)
        assert batch["reward"].shape == (4,)
        assert "success" in batch
    
    def test_sample_policy_success_only(self, buffer):
        """Test policy sampling with success-only filter."""
        # Fill buffer with mixed success/fail
        for i in range(20):
            batch_size = 2
            obs = {"state": np.random.randn(batch_size, 2, 32).astype(np.float32)}
            next_obs = {"state": np.random.randn(batch_size, 2, 32).astype(np.float32)}
            action = np.random.randn(batch_size, 8, 7).astype(np.float32)
            reward = np.random.randn(batch_size).astype(np.float32)
            done = np.array([0.0, 1.0], dtype=np.float32)
            success = np.array([float(i % 2), float((i + 1) % 2)], dtype=np.float32)
            
            buffer.store(
                obs, action, reward, next_obs, done,
                cumulative_reward=reward,
                chunk_done=done,
                discount_factor=np.ones(batch_size) * 0.99,
                effective_length=np.ones(batch_size) * 8,
                success=success
            )
        
        # Sample success only
        batch = buffer.sample_policy(8, success_only=True)
        
        # All samples should be from successful episodes
        assert batch["success"].sum() == 8
    
    def test_sample_policy_min_success_ratio(self, buffer):
        """Test policy sampling with minimum success ratio."""
        # Fill buffer
        for i in range(20):
            batch_size = 2
            obs = {"state": np.random.randn(batch_size, 2, 32).astype(np.float32)}
            next_obs = {"state": np.random.randn(batch_size, 2, 32).astype(np.float32)}
            action = np.random.randn(batch_size, 8, 7).astype(np.float32)
            reward = np.random.randn(batch_size).astype(np.float32)
            done = np.array([0.0, 1.0], dtype=np.float32)
            success = np.array([0.0, 1.0], dtype=np.float32)  # 50% success rate
            
            buffer.store(
                obs, action, reward, next_obs, done,
                cumulative_reward=reward,
                chunk_done=done,
                discount_factor=np.ones(batch_size) * 0.99,
                effective_length=np.ones(batch_size) * 8,
                success=success
            )
        
        # Sample with 50% minimum success ratio
        batch = buffer.sample_policy(10, min_success_ratio=0.5)
        
        # At least 50% should be successful
        assert batch["success"].sum() >= 5
    
    def test_get_statistics(self, buffer):
        """Test buffer statistics."""
        # Fill buffer
        for _ in range(5):
            batch_size = 2
            obs = {"state": np.random.randn(batch_size, 2, 32).astype(np.float32)}
            next_obs = {"state": np.random.randn(batch_size, 2, 32).astype(np.float32)}
            action = np.random.randn(batch_size, 8, 7).astype(np.float32)
            reward = np.random.randn(batch_size).astype(np.float32)
            done = np.array([0.0, 1.0], dtype=np.float32)
            success = np.array([0.0, 1.0], dtype=np.float32)
            
            buffer.store(
                obs, action, reward, next_obs, done,
                cumulative_reward=reward,
                chunk_done=done,
                discount_factor=np.ones(batch_size) * 0.99,
                effective_length=np.ones(batch_size) * 8,
                success=success
            )
        
        stats = buffer.get_statistics()
        
        assert stats["size"] == 10
        assert stats["num_success"] == 5
        assert stats["num_fail"] == 5
        assert stats["success_rate"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
