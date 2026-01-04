"""
Minimal training test for DSRL Off-Policy.

This script runs a very short training loop to verify:
1. Environment interaction works
2. Data collection and buffer storage works  
3. SAC updates don't crash
4. Checkpointing works

Usage:
    python test_minimal_training.py
"""

import sys
from pathlib import Path
import tempfile
import os

# Add paths
_root = Path(__file__).parent.parent.parent
_dsrl_offpolicy = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "diffusion_policy"))
sys.path.insert(0, str(_root / "dsrl"))
sys.path.insert(0, str(_dsrl_offpolicy.parent))  # Add parent so dsrl_offpolicy is importable

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from latent_policy import LatentGaussianPolicy
from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.algorithms.networks import DoubleQNetwork

from dsrl_offpolicy.agents.dsrl_sac_agent import DSRLSACAgent
from dsrl_offpolicy.models.latent_q_network import DoubleLatentQNetwork
from dsrl_offpolicy.buffers.macro_replay_buffer import MacroReplayBuffer


def create_minimal_agent(obs_dim=64, action_dim=7):
    """Create a minimal agent for testing."""
    pred_horizon = 16
    act_horizon = 8
    
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        n_groups=8,
    )
    
    q_network = DoubleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim,
        action_horizon=act_horizon,
        hidden_dims=[64, 64],
    )
    
    latent_policy = LatentGaussianPolicy(
        obs_dim=obs_dim,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        steer_mode="full",
        act_horizon=act_horizon,
    )
    
    latent_q_network = DoubleLatentQNetwork(
        obs_dim=obs_dim,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        hidden_dims=[64, 64],
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
        obs_horizon=1,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        num_inference_steps=2,  # Minimal for speed
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


def run_minimal_training():
    """Run minimal training loop."""
    print("=" * 50)
    print("DSRL Off-Policy Minimal Training Test")
    print("=" * 50)
    
    # Parameters
    obs_dim = 64
    action_dim = 7
    pred_horizon = 16
    batch_size = 16
    num_steps = 50
    warmup_steps = 20
    utd_ratio = 2
    
    device = torch.device("cpu")
    
    # Create agent
    print("\n1. Creating agent...")
    agent = create_minimal_agent(obs_dim, action_dim).to(device)
    print(f"   Agent parameters: {sum(p.numel() for p in agent.parameters()) / 1e3:.1f}K")
    
    # Create replay buffer
    print("\n2. Creating replay buffer...")
    buffer = MacroReplayBuffer(
        capacity=500,
        obs_dim=obs_dim,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        device="cpu",
    )
    
    # Create optimizers
    print("\n3. Creating optimizers...")
    actor_optimizer = optim.Adam(agent.get_actor_parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(agent.get_critic_parameters(), lr=1e-3)
    temp_optimizer = optim.Adam(agent.get_temperature_parameters(), lr=1e-3)
    
    # Freeze frozen components
    for param in agent.velocity_net.parameters():
        param.requires_grad = False
    for param in agent.q_network.parameters():
        param.requires_grad = False
    
    # Simulated environment loop
    print("\n4. Running training loop...")
    
    metrics = {
        "critic_loss": [],
        "actor_loss": [],
        "td_error": [],
        "q_mean": [],
        "alpha": [],
    }
    
    for step in range(num_steps):
        # Simulate observation
        obs = torch.randn(1, obs_dim)
        
        # Get action
        with torch.no_grad():
            actions, latent_w, log_prob = agent.get_action(
                obs,
                use_latent_policy=True,
                deterministic=False,
                use_prior_mixing=True,
            )
        
        # Simulate transition
        next_obs = torch.randn(1, obs_dim)
        reward = np.random.randn()
        done = float(np.random.random() < 0.05)
        discount = 0.99 ** 8  # gamma^act_horizon
        
        # Store in buffer
        buffer.add(
            obs_cond=obs.squeeze(0).numpy(),
            latent_w=latent_w.squeeze(0).numpy(),
            reward=reward,
            discount_factor=discount,
            done=done,
            next_obs_cond=next_obs.squeeze(0).numpy(),
        )
        
        # SAC updates
        if step >= warmup_steps and buffer.is_ready(batch_size):
            for _ in range(utd_ratio):
                batch = buffer.sample(batch_size)
                
                # Critic update
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
                critic_optimizer.step()
                
                # Actor update
                actor_metrics = agent.compute_actor_loss(obs_cond=batch["obs_cond"])
                
                actor_optimizer.zero_grad()
                actor_metrics["loss"].backward()
                actor_optimizer.step()
                
                # Temperature update
                temp_metrics = agent.compute_temperature_loss(obs_cond=batch["obs_cond"])
                
                temp_optimizer.zero_grad()
                temp_metrics["loss"].backward()
                temp_optimizer.step()
                
                # Target update
                agent.update_target()
                
                # Collect metrics
                metrics["critic_loss"].append(critic_metrics["loss"].item())
                metrics["actor_loss"].append(actor_metrics["loss"].item())
                metrics["td_error"].append(critic_metrics["td_error"].item())
                metrics["q_mean"].append(critic_metrics["q_mean"].item())
                metrics["alpha"].append(temp_metrics["alpha"].item())
            
            agent.decay_prior_mix_ratio()
        
        if (step + 1) % 10 == 0:
            print(f"   Step {step + 1}/{num_steps}, Buffer size: {buffer.size}")
    
    # Print final metrics
    print("\n5. Training metrics:")
    for key, values in metrics.items():
        if values:
            print(f"   {key}: {np.mean(values[-10:]):.4f} (last 10)")
    
    # Test checkpoint save/load
    print("\n6. Testing checkpoint save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pt")
        agent.save_checkpoint(ckpt_path)
        print(f"   Saved checkpoint to {ckpt_path}")
        
        agent.load_checkpoint(ckpt_path)
        print("   Loaded checkpoint successfully")
    
    print("\n" + "=" * 50)
    print("MINIMAL TRAINING TEST PASSED!")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    success = run_minimal_training()
    exit(0 if success else 1)
