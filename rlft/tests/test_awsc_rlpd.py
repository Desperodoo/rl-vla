"""
Comprehensive tests for AWSC algorithm in train_rlpd.py

Tests cover:
1. State observation mode (no visual encoder)
2. RGB/Image observation mode (with visual encoder)
3. With offline demo data (RLPD style)
4. Without offline demo data (pure online)

Usage:
    pytest rlft/tests/test_awsc_rlpd.py -v
    
    # Run specific test
    pytest rlft/tests/test_awsc_rlpd.py::TestAWSCTraining::test_awsc_state_mode_no_demo -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
from collections import defaultdict

# Import components to test
from rlft.algorithms.online_rl import AWSCAgent
from rlft.networks import ShortCutVelocityUNet1D, EnsembleQNetwork, PlainConv
from rlft.buffers import OnlineReplayBufferRaw
from rlft.datasets.data_utils import ObservationStacker


@pytest.fixture
def device():
    """Test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def action_dim():
    return 7


@pytest.fixture
def obs_horizon():
    return 2


@pytest.fixture
def pred_horizon():
    return 16


@pytest.fixture
def act_horizon():
    return 8


@pytest.fixture
def state_dim():
    return 25


@pytest.fixture
def visual_feature_dim():
    return 256


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def num_envs():
    return 4


# ============== Agent Fixtures ==============

@pytest.fixture
def state_obs_dim(state_dim, obs_horizon):
    """Observation dimension for state-only mode."""
    return obs_horizon * state_dim


@pytest.fixture
def rgb_obs_dim(state_dim, visual_feature_dim, obs_horizon):
    """Observation dimension for RGB mode."""
    return obs_horizon * (state_dim + visual_feature_dim)


@pytest.fixture
def velocity_net_state(action_dim, state_obs_dim, device):
    """Velocity network for state mode."""
    return ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=state_obs_dim,
        diffusion_step_embed_dim=128,
        down_dims=(128, 256),
        kernel_size=5,
        n_groups=8,
    ).to(device)


@pytest.fixture
def velocity_net_rgb(action_dim, rgb_obs_dim, device):
    """Velocity network for RGB mode."""
    return ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=rgb_obs_dim,
        diffusion_step_embed_dim=128,
        down_dims=(128, 256),
        kernel_size=5,
        n_groups=8,
    ).to(device)


@pytest.fixture
def awsc_agent_state(velocity_net_state, action_dim, state_obs_dim, obs_horizon, pred_horizon, act_horizon, device):
    """AWSC agent for state mode."""
    return AWSCAgent(
        velocity_net=velocity_net_state,
        obs_dim=state_obs_dim,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        num_qs=5,
        num_min_qs=2,
        q_hidden_dims=[128, 128],
        num_inference_steps=4,
        beta=10.0,
        bc_weight=1.0,
        shortcut_weight=0.3,
        gamma=0.9,
        device=str(device),
    ).to(device)


@pytest.fixture
def awsc_agent_rgb(velocity_net_rgb, action_dim, rgb_obs_dim, obs_horizon, pred_horizon, act_horizon, device):
    """AWSC agent for RGB mode."""
    return AWSCAgent(
        velocity_net=velocity_net_rgb,
        obs_dim=rgb_obs_dim,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        num_qs=5,
        num_min_qs=2,
        q_hidden_dims=[128, 128],
        num_inference_steps=4,
        beta=10.0,
        bc_weight=1.0,
        shortcut_weight=0.3,
        gamma=0.9,
        filter_policy_data=True,  # Enable for testing data separation
        advantage_threshold=0.0,
        device=str(device),
    ).to(device)


@pytest.fixture
def visual_encoder(visual_feature_dim, device):
    """Visual encoder for RGB mode."""
    return PlainConv(
        in_channels=3,
        out_dim=visual_feature_dim,
        pool_feature_map=True,
    ).to(device)


# ============== Buffer Fixtures ==============

@pytest.fixture
def online_buffer_state(state_dim, action_dim, act_horizon, obs_horizon, num_envs, device):
    """Replay buffer for state mode."""
    return OnlineReplayBufferRaw(
        capacity=1000,
        num_envs=num_envs,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=act_horizon,
        obs_horizon=obs_horizon,
        include_rgb=False,
        rgb_shape=None,
        gamma=0.9,
        device=device,
    )


@pytest.fixture
def online_buffer_rgb(state_dim, action_dim, act_horizon, obs_horizon, num_envs, device):
    """Replay buffer for RGB mode."""
    return OnlineReplayBufferRaw(
        capacity=1000,
        num_envs=num_envs,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=act_horizon,
        obs_horizon=obs_horizon,
        include_rgb=True,
        rgb_shape=(128, 128, 3),
        gamma=0.9,
        device=device,
    )


# ============== Helper Functions ==============

def create_mock_state_obs(batch_size, obs_horizon, state_dim, device):
    """Create mock state observation."""
    return {
        "state": torch.randn(batch_size, obs_horizon, state_dim, device=device),
    }


def create_mock_rgb_obs(batch_size, obs_horizon, state_dim, device, rgb_shape=(128, 128, 3)):
    """Create mock RGB observation."""
    return {
        "state": torch.randn(batch_size, obs_horizon, state_dim, device=device),
        "rgb": torch.randint(0, 255, (batch_size, obs_horizon, *rgb_shape), device=device, dtype=torch.uint8),
    }


def encode_state_obs(obs, obs_horizon):
    """Encode state-only observation."""
    state = obs["state"]
    B = state.shape[0]
    return state.reshape(B, -1)


def encode_rgb_obs(obs, visual_encoder, device, obs_horizon):
    """Encode RGB observation."""
    state = obs["state"]
    rgb = obs["rgb"]
    B, T = state.shape[0], state.shape[1]
    
    # Encode RGB
    rgb_flat = rgb.view(B * T, *rgb.shape[2:]).float() / 255.0
    if rgb_flat.ndim == 4 and rgb_flat.shape[-1] == 3:
        rgb_flat = rgb_flat.permute(0, 3, 1, 2)
    visual_feat = visual_encoder(rgb_flat)
    visual_feat = visual_feat.view(B, T, -1)
    
    # Concatenate
    features = torch.cat([visual_feat, state], dim=-1)
    return features.reshape(B, -1)


def create_mock_batch(batch_size, obs_horizon, state_dim, action_dim, act_horizon, device, include_rgb=False):
    """Create a mock training batch."""
    if include_rgb:
        obs = create_mock_rgb_obs(batch_size, obs_horizon, state_dim, device)
        next_obs = create_mock_rgb_obs(batch_size, obs_horizon, state_dim, device)
    else:
        obs = create_mock_state_obs(batch_size, obs_horizon, state_dim, device)
        next_obs = create_mock_state_obs(batch_size, obs_horizon, state_dim, device)
    
    return {
        "observations": obs,
        "next_observations": next_obs,
        "actions": torch.randn(batch_size, act_horizon, action_dim, device=device),
        "reward": torch.randn(batch_size, device=device),
        "done": torch.zeros(batch_size, device=device),
        "cumulative_reward": torch.randn(batch_size, device=device),
        "chunk_done": torch.zeros(batch_size, device=device),
        "discount_factor": torch.full((batch_size,), 0.9, device=device),
        "is_demo": torch.randint(0, 2, (batch_size,), device=device).bool(),
    }


# ============== Test Classes ==============

class TestAWSCAgentBasic:
    """Basic functionality tests for AWSC agent."""
    
    def test_agent_creation_state(self, awsc_agent_state):
        """Test AWSC agent creation in state mode."""
        assert awsc_agent_state is not None
        assert hasattr(awsc_agent_state, "velocity_net")
        assert hasattr(awsc_agent_state, "critic")
        assert hasattr(awsc_agent_state, "critic_target")
        assert hasattr(awsc_agent_state, "velocity_net_ema")
    
    def test_agent_creation_rgb(self, awsc_agent_rgb):
        """Test AWSC agent creation in RGB mode."""
        assert awsc_agent_rgb is not None
        assert awsc_agent_rgb.filter_policy_data is True
    
    def test_select_action_state(self, awsc_agent_state, state_obs_dim, batch_size, act_horizon, device):
        """Test action selection in state mode."""
        obs_features = torch.randn(batch_size, state_obs_dim, device=device)
        
        # Deterministic
        actions = awsc_agent_state.select_action(obs_features, deterministic=True)
        assert actions.shape == (batch_size, act_horizon, awsc_agent_state.action_dim)
        assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)
        
        # Non-deterministic (with exploration noise)
        actions_noisy = awsc_agent_state.select_action(obs_features, deterministic=False)
        assert actions_noisy.shape == (batch_size, act_horizon, awsc_agent_state.action_dim)
    
    def test_select_action_rgb(self, awsc_agent_rgb, rgb_obs_dim, batch_size, act_horizon, device):
        """Test action selection in RGB mode."""
        obs_features = torch.randn(batch_size, rgb_obs_dim, device=device)
        
        actions = awsc_agent_rgb.select_action(obs_features, deterministic=True)
        assert actions.shape == (batch_size, act_horizon, awsc_agent_rgb.action_dim)
    
    def test_get_action_full_sequence(self, awsc_agent_state, state_obs_dim, batch_size, pred_horizon, device):
        """Test get_action returns full pred_horizon sequence."""
        obs_features = torch.randn(batch_size, state_obs_dim, device=device)
        
        actions = awsc_agent_state.get_action(obs_features)
        assert actions.shape == (batch_size, pred_horizon, awsc_agent_state.action_dim)


class TestAWSCTraining:
    """Training loop tests for AWSC."""
    
    def test_awsc_state_mode_no_demo(
        self, awsc_agent_state, state_obs_dim, state_dim, obs_horizon,
        action_dim, act_horizon, batch_size, device
    ):
        """Test AWSC training in state mode without demo data."""
        print("\n=== Test: State mode, no demo ===")
        
        # Create optimizers
        actor_optimizer = torch.optim.Adam(awsc_agent_state.velocity_net.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(awsc_agent_state.critic.parameters(), lr=3e-4)
        
        # Create batch (no is_demo, simulating pure online)
        batch = create_mock_batch(batch_size, obs_horizon, state_dim, action_dim, act_horizon, device, include_rgb=False)
        batch["is_demo"] = None  # No demo indicator
        
        # Encode observations
        obs_features = encode_state_obs(batch["observations"], obs_horizon)
        next_obs_features = encode_state_obs(batch["next_observations"], obs_horizon)
        
        print(f"  obs_features shape: {obs_features.shape}")
        
        # Critic update
        critic_optimizer.zero_grad()
        critic_loss, critic_metrics = awsc_agent_state.compute_critic_loss(
            obs_features=obs_features,
            actions=batch["actions"],
            next_obs_features=next_obs_features,
            rewards=batch["reward"],
            dones=batch["done"],
            cumulative_reward=batch["cumulative_reward"],
            chunk_done=batch["chunk_done"],
            discount_factor=batch["discount_factor"],
        )
        critic_loss.backward()
        critic_optimizer.step()
        print(f"  critic_loss: {critic_loss.item():.4f}")
        
        # Actor update
        actor_optimizer.zero_grad()
        actor_loss, actor_metrics = awsc_agent_state.compute_actor_loss(
            obs_features=obs_features,
            actions=batch["actions"],
            actions_for_q=batch["actions"],
            is_demo=batch["is_demo"],
        )
        actor_loss.backward()
        actor_optimizer.step()
        print(f"  actor_loss: {actor_loss.item():.4f}")
        print(f"  flow_loss: {actor_metrics['flow_loss']:.4f}")
        print(f"  shortcut_loss: {actor_metrics['shortcut_loss']:.4f}")
        
        # Update targets
        awsc_agent_state.update_target()
        awsc_agent_state.update_ema()
        
        assert not torch.isnan(torch.tensor(critic_loss.item()))
        assert not torch.isnan(torch.tensor(actor_loss.item()))
    
    def test_awsc_state_mode_with_demo(
        self, awsc_agent_state, state_obs_dim, state_dim, obs_horizon,
        action_dim, act_horizon, batch_size, device
    ):
        """Test AWSC training in state mode with demo data (RLPD style)."""
        print("\n=== Test: State mode, with demo ===")
        
        actor_optimizer = torch.optim.Adam(awsc_agent_state.velocity_net.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(awsc_agent_state.critic.parameters(), lr=3e-4)
        
        # Create batch with is_demo indicator (50% online, 50% demo)
        batch = create_mock_batch(batch_size, obs_horizon, state_dim, action_dim, act_horizon, device, include_rgb=False)
        batch["is_demo"] = torch.zeros(batch_size, device=device, dtype=torch.bool)
        batch["is_demo"][batch_size // 2:] = True
        
        print(f"  Demo samples: {batch['is_demo'].sum().item()}/{batch_size}")
        
        obs_features = encode_state_obs(batch["observations"], obs_horizon)
        next_obs_features = encode_state_obs(batch["next_observations"], obs_horizon)
        
        # Critic update
        critic_optimizer.zero_grad()
        critic_loss, critic_metrics = awsc_agent_state.compute_critic_loss(
            obs_features=obs_features,
            actions=batch["actions"],
            next_obs_features=next_obs_features,
            rewards=batch["reward"],
            dones=batch["done"],
            cumulative_reward=batch["cumulative_reward"],
            chunk_done=batch["chunk_done"],
            discount_factor=batch["discount_factor"],
        )
        critic_loss.backward()
        critic_optimizer.step()
        
        # Actor update (with is_demo)
        actor_optimizer.zero_grad()
        actor_loss, actor_metrics = awsc_agent_state.compute_actor_loss(
            obs_features=obs_features,
            actions=batch["actions"],
            actions_for_q=batch["actions"],
            is_demo=batch["is_demo"],
        )
        actor_loss.backward()
        actor_optimizer.step()
        
        print(f"  critic_loss: {critic_loss.item():.4f}")
        print(f"  actor_loss: {actor_loss.item():.4f}")
        
        awsc_agent_state.update_target()
        awsc_agent_state.update_ema()
        
        assert not torch.isnan(torch.tensor(critic_loss.item()))
        assert not torch.isnan(torch.tensor(actor_loss.item()))
    
    def test_awsc_rgb_mode_no_demo(
        self, awsc_agent_rgb, visual_encoder, rgb_obs_dim, state_dim, obs_horizon,
        action_dim, act_horizon, batch_size, device
    ):
        """Test AWSC training in RGB mode without demo data."""
        print("\n=== Test: RGB mode, no demo ===")
        
        actor_params = list(awsc_agent_rgb.velocity_net.parameters()) + list(visual_encoder.parameters())
        actor_optimizer = torch.optim.Adam(actor_params, lr=3e-4)
        critic_optimizer = torch.optim.Adam(awsc_agent_rgb.critic.parameters(), lr=3e-4)
        
        batch = create_mock_batch(batch_size, obs_horizon, state_dim, action_dim, act_horizon, device, include_rgb=True)
        batch["is_demo"] = None
        
        # Encode with visual encoder
        obs_features = encode_rgb_obs(batch["observations"], visual_encoder, device, obs_horizon)
        next_obs_features = encode_rgb_obs(batch["next_observations"], visual_encoder, device, obs_horizon)
        
        print(f"  obs_features shape: {obs_features.shape}")
        
        # Critic update (detached obs)
        critic_optimizer.zero_grad()
        critic_loss, critic_metrics = awsc_agent_rgb.compute_critic_loss(
            obs_features=obs_features.detach(),
            actions=batch["actions"],
            next_obs_features=next_obs_features.detach(),
            rewards=batch["reward"],
            dones=batch["done"],
            cumulative_reward=batch["cumulative_reward"],
            chunk_done=batch["chunk_done"],
            discount_factor=batch["discount_factor"],
        )
        critic_loss.backward()
        critic_optimizer.step()
        
        # Re-encode for actor (need fresh computation graph)
        obs_features_actor = encode_rgb_obs(batch["observations"], visual_encoder, device, obs_horizon)
        
        # Actor update
        actor_optimizer.zero_grad()
        actor_loss, actor_metrics = awsc_agent_rgb.compute_actor_loss(
            obs_features=obs_features_actor,
            actions=batch["actions"],
            actions_for_q=batch["actions"],
            is_demo=batch["is_demo"],
        )
        actor_loss.backward()
        actor_optimizer.step()
        
        print(f"  critic_loss: {critic_loss.item():.4f}")
        print(f"  actor_loss: {actor_loss.item():.4f}")
        
        awsc_agent_rgb.update_target()
        awsc_agent_rgb.update_ema()
        
        assert not torch.isnan(torch.tensor(critic_loss.item()))
        assert not torch.isnan(torch.tensor(actor_loss.item()))
    
    def test_awsc_rgb_mode_with_demo_and_filtering(
        self, awsc_agent_rgb, visual_encoder, rgb_obs_dim, state_dim, obs_horizon,
        action_dim, act_horizon, batch_size, device
    ):
        """Test AWSC training in RGB mode with demo data and policy-critic data separation."""
        print("\n=== Test: RGB mode, with demo + policy filtering ===")
        
        # awsc_agent_rgb has filter_policy_data=True
        assert awsc_agent_rgb.filter_policy_data is True
        
        actor_params = list(awsc_agent_rgb.velocity_net.parameters()) + list(visual_encoder.parameters())
        actor_optimizer = torch.optim.Adam(actor_params, lr=3e-4)
        critic_optimizer = torch.optim.Adam(awsc_agent_rgb.critic.parameters(), lr=3e-4)
        
        batch = create_mock_batch(batch_size, obs_horizon, state_dim, action_dim, act_horizon, device, include_rgb=True)
        batch["is_demo"] = torch.zeros(batch_size, device=device, dtype=torch.bool)
        batch["is_demo"][batch_size // 2:] = True
        
        print(f"  Demo samples: {batch['is_demo'].sum().item()}/{batch_size}")
        print(f"  filter_policy_data: {awsc_agent_rgb.filter_policy_data}")
        print(f"  advantage_threshold: {awsc_agent_rgb.advantage_threshold}")
        
        obs_features = encode_rgb_obs(batch["observations"], visual_encoder, device, obs_horizon)
        next_obs_features = encode_rgb_obs(batch["next_observations"], visual_encoder, device, obs_horizon)
        
        # Critic update
        critic_optimizer.zero_grad()
        critic_loss, _ = awsc_agent_rgb.compute_critic_loss(
            obs_features=obs_features.detach(),
            actions=batch["actions"],
            next_obs_features=next_obs_features.detach(),
            rewards=batch["reward"],
            dones=batch["done"],
            cumulative_reward=batch["cumulative_reward"],
            chunk_done=batch["chunk_done"],
            discount_factor=batch["discount_factor"],
        )
        critic_loss.backward()
        critic_optimizer.step()
        
        # Re-encode for actor
        obs_features_actor = encode_rgb_obs(batch["observations"], visual_encoder, device, obs_horizon)
        
        # Actor update (with filtering)
        actor_optimizer.zero_grad()
        actor_loss, actor_metrics = awsc_agent_rgb.compute_actor_loss(
            obs_features=obs_features_actor,
            actions=batch["actions"],
            actions_for_q=batch["actions"],
            is_demo=batch["is_demo"],
        )
        actor_loss.backward()
        actor_optimizer.step()
        
        print(f"  critic_loss: {critic_loss.item():.4f}")
        print(f"  actor_loss: {actor_loss.item():.4f}")
        print(f"  n_demo_samples: {actor_metrics['n_demo_samples']}")
        print(f"  n_online_kept: {actor_metrics['n_online_kept']}")
        print(f"  n_online_filtered: {actor_metrics['n_online_filtered']}")
        print(f"  policy_batch_size: {actor_metrics['policy_batch_size']}")
        
        awsc_agent_rgb.update_target()
        awsc_agent_rgb.update_ema()
        
        assert not torch.isnan(torch.tensor(critic_loss.item()))
        assert not torch.isnan(torch.tensor(actor_loss.item()))


class TestAWSCBuffer:
    """Buffer interaction tests."""
    
    def test_buffer_state_mode(
        self, online_buffer_state, state_dim, action_dim, act_horizon, obs_horizon, num_envs, device
    ):
        """Test replay buffer in state mode."""
        print("\n=== Test: Buffer state mode ===")
        
        # Simulate storing experience
        obs = {
            "state": np.random.randn(num_envs, obs_horizon, state_dim).astype(np.float32),
        }
        next_obs = {
            "state": np.random.randn(num_envs, obs_horizon, state_dim).astype(np.float32),
        }
        actions = np.random.randn(num_envs, act_horizon, action_dim).astype(np.float32)
        rewards = np.random.randn(num_envs).astype(np.float32)
        dones = np.zeros(num_envs, dtype=np.float32)
        cumulative_rewards = rewards.copy()
        chunk_dones = dones.copy()
        discount_factors = np.full(num_envs, 0.9, dtype=np.float32)
        effective_lengths = np.full(num_envs, act_horizon, dtype=np.float32)
        
        # Store multiple times
        for _ in range(10):
            online_buffer_state.store(
                obs=obs,
                action=actions,
                reward=rewards,
                next_obs=next_obs,
                done=dones,
                cumulative_reward=cumulative_rewards,
                chunk_done=chunk_dones,
                discount_factor=discount_factors,
                effective_length=effective_lengths,
            )
        
        print(f"  Buffer size: {online_buffer_state.size}")
        
        # Sample batch
        batch = online_buffer_state.sample(batch_size=8)
        
        assert "observations" in batch
        assert "next_observations" in batch
        assert "actions" in batch
        assert batch["actions"].shape == (8, act_horizon, action_dim)
        print(f"  Sampled batch actions shape: {batch['actions'].shape}")
    
    def test_buffer_rgb_mode(
        self, online_buffer_rgb, state_dim, action_dim, act_horizon, obs_horizon, num_envs, device
    ):
        """Test replay buffer in RGB mode."""
        print("\n=== Test: Buffer RGB mode ===")
        
        rgb_shape = (128, 128, 3)
        obs = {
            "state": np.random.randn(num_envs, obs_horizon, state_dim).astype(np.float32),
            "rgb": np.random.randint(0, 255, (num_envs, obs_horizon, *rgb_shape), dtype=np.uint8),
        }
        next_obs = {
            "state": np.random.randn(num_envs, obs_horizon, state_dim).astype(np.float32),
            "rgb": np.random.randint(0, 255, (num_envs, obs_horizon, *rgb_shape), dtype=np.uint8),
        }
        actions = np.random.randn(num_envs, act_horizon, action_dim).astype(np.float32)
        rewards = np.random.randn(num_envs).astype(np.float32)
        dones = np.zeros(num_envs, dtype=np.float32)
        cumulative_rewards = rewards.copy()
        chunk_dones = dones.copy()
        discount_factors = np.full(num_envs, 0.9, dtype=np.float32)
        effective_lengths = np.full(num_envs, act_horizon, dtype=np.float32)
        
        for _ in range(10):
            online_buffer_rgb.store(
                obs=obs,
                action=actions,
                reward=rewards,
                next_obs=next_obs,
                done=dones,
                cumulative_reward=cumulative_rewards,
                chunk_done=chunk_dones,
                discount_factor=discount_factors,
                effective_length=effective_lengths,
            )
        
        print(f"  Buffer size: {online_buffer_rgb.size}")
        
        batch = online_buffer_rgb.sample(batch_size=8)
        
        assert "observations" in batch
        assert "rgb" in batch["observations"]
        assert batch["observations"]["rgb"].shape == (8, obs_horizon, *rgb_shape)
        print(f"  Sampled batch rgb shape: {batch['observations']['rgb'].shape}")


class TestAWSCFullTrainingLoop:
    """Full training loop simulation tests."""
    
    def test_full_loop_state_no_demo(
        self, awsc_agent_state, online_buffer_state, state_dim, action_dim,
        obs_horizon, act_horizon, num_envs, batch_size, device
    ):
        """Simulate full training loop: state mode, no demo."""
        print("\n=== Test: Full loop - state, no demo ===")
        
        actor_optimizer = torch.optim.Adam(awsc_agent_state.velocity_net.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(awsc_agent_state.critic.parameters(), lr=3e-4)
        
        # Fill buffer
        for _ in range(20):
            obs = {"state": np.random.randn(num_envs, obs_horizon, state_dim).astype(np.float32)}
            next_obs = {"state": np.random.randn(num_envs, obs_horizon, state_dim).astype(np.float32)}
            actions = np.random.randn(num_envs, act_horizon, action_dim).astype(np.float32)
            rewards = np.random.randn(num_envs).astype(np.float32)
            dones = np.zeros(num_envs, dtype=np.float32)
            
            online_buffer_state.store(
                obs=obs,
                action=actions,
                reward=rewards,
                next_obs=next_obs,
                done=dones,
                cumulative_reward=rewards,
                chunk_done=dones,
                discount_factor=np.full(num_envs, 0.9, dtype=np.float32),
                effective_length=np.full(num_envs, act_horizon, dtype=np.float32),
            )
        
        # Simulate training steps
        num_steps = 5
        for step in range(num_steps):
            batch = online_buffer_state.sample(batch_size=batch_size)
            
            # Convert to tensors
            obs_t = torch.from_numpy(batch["observations"]["state"]).to(device)
            next_obs_t = torch.from_numpy(batch["next_observations"]["state"]).to(device)
            obs_features = obs_t.reshape(batch_size, -1)
            next_obs_features = next_obs_t.reshape(batch_size, -1)
            
            actions_t = torch.from_numpy(batch["actions"]).to(device)
            rewards_t = torch.from_numpy(batch["reward"]).to(device)
            dones_t = torch.from_numpy(batch["done"]).to(device)
            cumulative_t = torch.from_numpy(batch["cumulative_reward"]).to(device)
            chunk_done_t = torch.from_numpy(batch["chunk_done"]).to(device)
            discount_t = torch.from_numpy(batch["discount_factor"]).to(device)
            
            # Critic update
            critic_optimizer.zero_grad()
            critic_loss, _ = awsc_agent_state.compute_critic_loss(
                obs_features, actions_t, next_obs_features,
                rewards_t, dones_t, cumulative_t, chunk_done_t, discount_t
            )
            critic_loss.backward()
            critic_optimizer.step()
            
            # Actor update
            actor_optimizer.zero_grad()
            actor_loss, metrics = awsc_agent_state.compute_actor_loss(
                obs_features, actions_t, actions_t, None
            )
            actor_loss.backward()
            actor_optimizer.step()
            
            awsc_agent_state.update_target()
            awsc_agent_state.update_ema()
            
            print(f"  Step {step+1}: critic={critic_loss.item():.4f}, actor={actor_loss.item():.4f}")
        
        print("  ✓ Full loop completed successfully")
    
    def test_full_loop_rgb_with_demo(
        self, awsc_agent_rgb, visual_encoder, online_buffer_rgb, state_dim, action_dim,
        obs_horizon, act_horizon, num_envs, batch_size, device
    ):
        """Simulate full training loop: RGB mode, with demo mixing."""
        print("\n=== Test: Full loop - RGB, with demo ===")
        
        actor_params = list(awsc_agent_rgb.velocity_net.parameters()) + list(visual_encoder.parameters())
        actor_optimizer = torch.optim.Adam(actor_params, lr=3e-4)
        critic_optimizer = torch.optim.Adam(awsc_agent_rgb.critic.parameters(), lr=3e-4)
        
        rgb_shape = (128, 128, 3)
        
        # Fill buffer
        for _ in range(20):
            obs = {
                "state": np.random.randn(num_envs, obs_horizon, state_dim).astype(np.float32),
                "rgb": np.random.randint(0, 255, (num_envs, obs_horizon, *rgb_shape), dtype=np.uint8),
            }
            next_obs = {
                "state": np.random.randn(num_envs, obs_horizon, state_dim).astype(np.float32),
                "rgb": np.random.randint(0, 255, (num_envs, obs_horizon, *rgb_shape), dtype=np.uint8),
            }
            actions = np.random.randn(num_envs, act_horizon, action_dim).astype(np.float32)
            rewards = np.random.randn(num_envs).astype(np.float32)
            dones = np.zeros(num_envs, dtype=np.float32)
            
            online_buffer_rgb.store(
                obs=obs,
                action=actions,
                reward=rewards,
                next_obs=next_obs,
                done=dones,
                cumulative_reward=rewards,
                chunk_done=dones,
                discount_factor=np.full(num_envs, 0.9, dtype=np.float32),
                effective_length=np.full(num_envs, act_horizon, dtype=np.float32),
            )
        
        # Simulate training steps with demo mixing
        num_steps = 5
        for step in range(num_steps):
            batch = online_buffer_rgb.sample(batch_size=batch_size)
            
            # Convert observations
            state_t = torch.from_numpy(batch["observations"]["state"]).to(device)
            rgb_t = torch.from_numpy(batch["observations"]["rgb"]).to(device)
            next_state_t = torch.from_numpy(batch["next_observations"]["state"]).to(device)
            next_rgb_t = torch.from_numpy(batch["next_observations"]["rgb"]).to(device)
            
            # Encode
            obs_dict = {"state": state_t, "rgb": rgb_t}
            next_obs_dict = {"state": next_state_t, "rgb": next_rgb_t}
            obs_features = encode_rgb_obs(obs_dict, visual_encoder, device, obs_horizon)
            next_obs_features = encode_rgb_obs(next_obs_dict, visual_encoder, device, obs_horizon)
            
            actions_t = torch.from_numpy(batch["actions"]).to(device)
            rewards_t = torch.from_numpy(batch["reward"]).to(device)
            dones_t = torch.from_numpy(batch["done"]).to(device)
            cumulative_t = torch.from_numpy(batch["cumulative_reward"]).to(device)
            chunk_done_t = torch.from_numpy(batch["chunk_done"]).to(device)
            discount_t = torch.from_numpy(batch["discount_factor"]).to(device)
            
            # Simulate is_demo (50% demo)
            is_demo = torch.zeros(batch_size, device=device, dtype=torch.bool)
            is_demo[batch_size // 2:] = True
            
            # Critic update
            critic_optimizer.zero_grad()
            critic_loss, _ = awsc_agent_rgb.compute_critic_loss(
                obs_features.detach(), actions_t, next_obs_features.detach(),
                rewards_t, dones_t, cumulative_t, chunk_done_t, discount_t
            )
            critic_loss.backward()
            critic_optimizer.step()
            
            # Re-encode for actor
            obs_features_actor = encode_rgb_obs(obs_dict, visual_encoder, device, obs_horizon)
            
            # Actor update
            actor_optimizer.zero_grad()
            actor_loss, metrics = awsc_agent_rgb.compute_actor_loss(
                obs_features_actor, actions_t, actions_t, is_demo
            )
            actor_loss.backward()
            actor_optimizer.step()
            
            awsc_agent_rgb.update_target()
            awsc_agent_rgb.update_ema()
            
            print(f"  Step {step+1}: critic={critic_loss.item():.4f}, actor={actor_loss.item():.4f}, "
                  f"demo={metrics['n_demo_samples']}, online_kept={metrics['n_online_kept']}")
        
        print("  ✓ Full loop completed successfully")


class TestAWSCEdgeCases:
    """Edge case and error handling tests."""
    
    def test_single_sample_batch(self, awsc_agent_state, state_obs_dim, action_dim, act_horizon, device):
        """Test with batch size of 1."""
        print("\n=== Test: Single sample batch ===")
        
        obs_features = torch.randn(1, state_obs_dim, device=device)
        actions = torch.randn(1, act_horizon, action_dim, device=device)
        next_obs_features = torch.randn(1, state_obs_dim, device=device)
        rewards = torch.randn(1, device=device)
        dones = torch.zeros(1, device=device)
        
        critic_loss, _ = awsc_agent_state.compute_critic_loss(
            obs_features, actions, next_obs_features, rewards, dones,
            rewards, dones, torch.full((1,), 0.9, device=device)
        )
        
        actor_loss, _ = awsc_agent_state.compute_actor_loss(
            obs_features, actions, actions, None
        )
        
        print(f"  critic_loss: {critic_loss.item():.4f}")
        print(f"  actor_loss: {actor_loss.item():.4f}")
        
        assert not torch.isnan(critic_loss)
        assert not torch.isnan(actor_loss)
    
    def test_all_demo_batch(self, awsc_agent_rgb, rgb_obs_dim, action_dim, act_horizon, batch_size, device):
        """Test when all samples are from demo (is_demo all True)."""
        print("\n=== Test: All demo batch ===")
        
        obs_features = torch.randn(batch_size, rgb_obs_dim, device=device)
        actions = torch.randn(batch_size, act_horizon, action_dim, device=device)
        is_demo = torch.ones(batch_size, device=device, dtype=torch.bool)
        
        actor_loss, metrics = awsc_agent_rgb.compute_actor_loss(
            obs_features, actions, actions, is_demo
        )
        
        print(f"  actor_loss: {actor_loss.item():.4f}")
        print(f"  n_demo_samples: {metrics['n_demo_samples']}")
        print(f"  n_online_kept: {metrics['n_online_kept']}")
        
        assert metrics["n_demo_samples"] == batch_size
        assert metrics["n_online_kept"] == 0
    
    def test_no_demo_batch(self, awsc_agent_rgb, rgb_obs_dim, action_dim, act_horizon, batch_size, device):
        """Test when no samples are from demo (is_demo all False)."""
        print("\n=== Test: No demo batch ===")
        
        obs_features = torch.randn(batch_size, rgb_obs_dim, device=device)
        actions = torch.randn(batch_size, act_horizon, action_dim, device=device)
        is_demo = torch.zeros(batch_size, device=device, dtype=torch.bool)
        
        actor_loss, metrics = awsc_agent_rgb.compute_actor_loss(
            obs_features, actions, actions, is_demo
        )
        
        print(f"  actor_loss: {actor_loss.item():.4f}")
        print(f"  n_demo_samples: {metrics['n_demo_samples']}")
        print(f"  n_online_kept: {metrics['n_online_kept']}")
        
        assert metrics["n_demo_samples"] == 0
        # All online samples with advantage > threshold should be kept
        assert metrics["n_online_kept"] + metrics["n_online_filtered"] == batch_size
    
    def test_high_advantage_threshold(self, awsc_agent_rgb, rgb_obs_dim, action_dim, act_horizon, batch_size, device):
        """Test with high advantage threshold (most online samples filtered)."""
        print("\n=== Test: High advantage threshold ===")
        
        # Temporarily set high threshold
        original_threshold = awsc_agent_rgb.advantage_threshold
        awsc_agent_rgb.advantage_threshold = 100.0  # Very high threshold
        
        obs_features = torch.randn(batch_size, rgb_obs_dim, device=device)
        actions = torch.randn(batch_size, act_horizon, action_dim, device=device)
        is_demo = torch.zeros(batch_size, device=device, dtype=torch.bool)
        is_demo[0] = True  # Only 1 demo sample
        
        actor_loss, metrics = awsc_agent_rgb.compute_actor_loss(
            obs_features, actions, actions, is_demo
        )
        
        print(f"  actor_loss: {actor_loss.item():.4f}")
        print(f"  n_demo_samples: {metrics['n_demo_samples']}")
        print(f"  n_online_kept: {metrics['n_online_kept']}")
        print(f"  n_online_filtered: {metrics['n_online_filtered']}")
        print(f"  policy_batch_size: {metrics['policy_batch_size']}")
        
        # With high threshold, most online samples should be filtered
        assert metrics["n_online_filtered"] >= metrics["n_online_kept"]
        
        # Restore threshold
        awsc_agent_rgb.advantage_threshold = original_threshold
    
    def test_gradient_flow(self, awsc_agent_state, state_obs_dim, action_dim, act_horizon, batch_size, device):
        """Test gradient flow through network."""
        print("\n=== Test: Gradient flow ===")
        
        obs_features = torch.randn(batch_size, state_obs_dim, device=device, requires_grad=True)
        actions = torch.randn(batch_size, act_horizon, action_dim, device=device)
        
        actor_loss, _ = awsc_agent_state.compute_actor_loss(
            obs_features, actions, actions, None
        )
        
        actor_loss.backward()
        
        # Check gradients exist
        has_grad = False
        for param in awsc_agent_state.velocity_net.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        print(f"  Gradients exist: {has_grad}")
        assert has_grad, "No gradients found in velocity network"


# ============== Main ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
