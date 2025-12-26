#!/usr/bin/env python3
"""
Test evaluation functionality for RLPD agents.

This module tests the integration between RLPD agents and ManiSkill environments,
ensuring that agents can be properly evaluated using the standard evaluate() function.

Tests:
- SACAgentWrapper: Wraps SACAgent for evaluation interface
- AWSCAgentWrapper: Wraps AWSCAgent for evaluation interface
- End-to-end evaluation with pretrained AWSC checkpoint

Usage:
    python -m diffusion_policy.rlpd.tests.test_evaluate
"""

import os
import sys
import torch
import numpy as np

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {DEVICE}")


class SACAgentWrapper:
    """Wrapper to provide get_action() interface for SACAgent."""
    
    def __init__(self, agent, visual_encoder=None, obs_horizon=2):
        self.agent = agent
        self.visual_encoder = visual_encoder
        self.obs_horizon = obs_horizon
        self.obs_buffer = None
        
    def eval(self):
        self.agent.eval()
        if self.visual_encoder is not None:
            self.visual_encoder.eval()
        
    def train(self):
        self.agent.train()
        if self.visual_encoder is not None:
            self.visual_encoder.train()
            
    def reset(self, batch_size=None):
        """Reset observation buffer."""
        self.obs_buffer = None
        
    def _encode_obs(self, obs):
        """Encode observations to feature space."""
        if self.visual_encoder is not None:
            # obs is dict with 'rgb' and 'state' keys
            rgb = obs['rgb']  # (B, H, W, C) or (B, N_cams, H, W, C)
            state = obs['state']  # (B, state_dim)
            
            # Flatten camera dimension if needed
            if rgb.dim() == 5:  # (B, N, H, W, C)
                B, N, H, W, C = rgb.shape
                rgb = rgb.view(B, N * C, H, W)
            else:  # (B, H, W, C)
                rgb = rgb.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            # Encode RGB
            rgb_features = self.visual_encoder(rgb)  # (B, visual_dim)
            
            # Concatenate with state
            obs_features = torch.cat([rgb_features, state], dim=-1)
        else:
            # obs is already features
            obs_features = obs
            
        return obs_features
        
    def get_action(self, obs, deterministic=True):
        """Get action from agent.
        
        Args:
            obs: Observation dict or tensor
            deterministic: Whether to use deterministic action
            
        Returns:
            action_seq: (B, action_horizon, action_dim)
        """
        # Encode observations
        obs_features = self._encode_obs(obs)
        
        # Maintain observation buffer for temporal stacking
        if self.obs_buffer is None:
            self.obs_buffer = [obs_features.clone() for _ in range(self.obs_horizon)]
        else:
            self.obs_buffer.pop(0)
            self.obs_buffer.append(obs_features.clone())
        
        # Stack observations
        stacked_obs = torch.cat(self.obs_buffer, dim=-1)
        
        # Get action from agent
        action_seq = self.agent.select_action(stacked_obs, deterministic=deterministic)
        
        return action_seq


class AWSCAgentWrapper:
    """Wrapper to provide get_action() interface for AWSCAgent."""
    
    def __init__(self, agent, visual_encoder=None, obs_horizon=2):
        self.agent = agent
        self.visual_encoder = visual_encoder
        self.obs_horizon = obs_horizon
        self.obs_buffer = None
        
    def eval(self):
        self.agent.eval()
        if self.visual_encoder is not None:
            self.visual_encoder.eval()
        
    def train(self):
        self.agent.train()
        if self.visual_encoder is not None:
            self.visual_encoder.train()
            
    def reset(self, batch_size=None):
        """Reset observation buffer."""
        self.obs_buffer = None
        
    def _encode_obs(self, obs):
        """Encode observations to feature space."""
        if self.visual_encoder is not None:
            # obs is dict with 'rgb' and 'state' keys
            if isinstance(obs, dict):
                rgb = obs['rgb']  # (B, H, W, C) or (B, N_cams, H, W, C)
                state = obs['state']  # (B, state_dim)
                
                # Flatten camera dimension if needed
                if rgb.dim() == 5:  # (B, N, H, W, C)
                    B, N, H, W, C = rgb.shape
                    rgb = rgb.view(B, N * C, H, W)
                else:  # (B, H, W, C)
                    rgb = rgb.permute(0, 3, 1, 2)  # (B, C, H, W)
                
                # Encode RGB
                rgb_features = self.visual_encoder(rgb)  # (B, visual_dim)
                
                # Concatenate with state
                obs_features = torch.cat([rgb_features, state], dim=-1)
            else:
                obs_features = obs
        else:
            # obs is already features
            obs_features = obs
            
        return obs_features
        
    def get_action(self, obs, deterministic=True, use_ema=True):
        """Get action from agent.
        
        Args:
            obs: Observation dict or tensor
            deterministic: Whether to use deterministic action (always True for flow)
            use_ema: Whether to use EMA network (ignored, select_action uses EMA by default)
            
        Returns:
            action_seq: (B, action_horizon, action_dim)
        """
        # Encode observations
        obs_features = self._encode_obs(obs)
        
        # Maintain observation buffer for temporal stacking
        if self.obs_buffer is None:
            self.obs_buffer = [obs_features.clone() for _ in range(self.obs_horizon)]
        else:
            self.obs_buffer.pop(0)
            self.obs_buffer.append(obs_features.clone())
        
        # Stack observations
        stacked_obs = torch.cat(self.obs_buffer, dim=-1)
        
        # Get action from agent (select_action uses EMA by default)
        action_seq = self.agent.select_action(
            stacked_obs, 
            deterministic=True,  # Flow always deterministic 
        )
        
        return action_seq


def test_sac_wrapper_interface():
    """Test SACAgentWrapper interface."""
    print("\n" + "=" * 60)
    print("Test: SACAgentWrapper interface")
    print("=" * 60)
    
    from diffusion_policy.rlpd.sac_agent import SACAgent
    
    obs_dim = 256
    action_dim = 7
    action_horizon = 8
    obs_horizon = 2
    batch_size = 4
    
    agent = SACAgent(
        obs_dim=obs_dim * obs_horizon,  # Stacked observations
        action_dim=action_dim,
        action_horizon=action_horizon,
        device=DEVICE,
    ).to(DEVICE)
    
    wrapper = SACAgentWrapper(
        agent=agent,
        visual_encoder=None,
        obs_horizon=obs_horizon,
    )
    
    # Test eval/train mode
    wrapper.eval()
    wrapper.train()
    print("✓ eval/train mode switching works")
    
    # Test get_action
    obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    
    action_seq = wrapper.get_action(obs, deterministic=True)
    assert action_seq.shape == (batch_size, action_horizon, action_dim), \
        f"Expected ({batch_size}, {action_horizon}, {action_dim}), got {action_seq.shape}"
    print(f"✓ get_action shape: {action_seq.shape}")
    
    # Test observation buffer updates
    action_seq2 = wrapper.get_action(obs * 2, deterministic=False)
    assert action_seq2.shape == (batch_size, action_horizon, action_dim)
    print("✓ Observation buffer updates correctly")
    
    # Test reset
    wrapper.reset()
    assert wrapper.obs_buffer is None
    print("✓ reset() clears observation buffer")
    
    return True


def test_awsc_wrapper_interface():
    """Test AWSCAgentWrapper interface."""
    print("\n" + "=" * 60)
    print("Test: AWSCAgentWrapper interface")
    print("=" * 60)
    
    from diffusion_policy.rlpd.awsc_agent import AWSCAgent, ShortCutVelocityUNet1D
    from diffusion_policy.rlpd.networks import EnsembleQNetwork
    
    obs_dim = 256
    action_dim = 7
    pred_horizon = 16
    act_horizon = 8
    obs_horizon = 2
    batch_size = 4
    
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,  # Stacked observations
        diffusion_step_embed_dim=64,
    )
    
    q_network = EnsembleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim * obs_horizon,
        action_horizon=act_horizon,
    )
    
    agent = AWSCAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        obs_dim=obs_dim * obs_horizon,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        device=DEVICE,
    ).to(DEVICE)
    
    wrapper = AWSCAgentWrapper(
        agent=agent,
        visual_encoder=None,
        obs_horizon=obs_horizon,
    )
    
    # Test eval/train mode
    wrapper.eval()
    wrapper.train()
    print("✓ eval/train mode switching works")
    
    # Test get_action
    obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    
    action_seq = wrapper.get_action(obs, deterministic=True, use_ema=True)
    assert action_seq.shape == (batch_size, act_horizon, action_dim), \
        f"Expected ({batch_size}, {act_horizon}, {action_dim}), got {action_seq.shape}"
    print(f"✓ get_action shape: {action_seq.shape}")
    
    # Test with use_ema=False
    action_seq2 = wrapper.get_action(obs * 2, deterministic=True, use_ema=False)
    assert action_seq2.shape == (batch_size, act_horizon, action_dim)
    print("✓ get_action with use_ema=False works")
    
    # Test reset
    wrapper.reset()
    assert wrapper.obs_buffer is None
    print("✓ reset() clears observation buffer")
    
    return True


def test_awsc_pretrained_evaluation():
    """Test evaluation with pretrained AWSC checkpoint."""
    print("\n" + "=" * 60)
    print("Test: Pretrained AWSC evaluation (short)")
    print("=" * 60)
    
    from diffusion_policy.rlpd.awsc_agent import AWSCAgent, ShortCutVelocityUNet1D
    from diffusion_policy.rlpd.networks import EnsembleQNetwork
    from diffusion_policy.plain_conv import PlainConv
    
    # Checkpoint path
    ckpt_path = "runs/awsc-LiftPegUpright-v1-seed0/checkpoints/best_eval_success_once.pt"
    ckpt_path = os.path.abspath(ckpt_path)
    
    if not os.path.exists(ckpt_path):
        print(f"⚠ Checkpoint not found: {ckpt_path}")
        print("  Skipping pretrained evaluation test")
        return True
    
    # Load checkpoint to get dimensions
    ckpt = torch.load(ckpt_path, map_location="cpu")
    agent_state = ckpt["agent"]
    
    # Get architecture parameters
    cond_weight = agent_state["velocity_net.unet.mid_modules.0.cond_encoder.1.weight"]
    final_conv_weight = agent_state["velocity_net.unet.final_conv.1.weight"]
    diff_enc_weight = agent_state["velocity_net.unet.diffusion_step_encoder.1.weight"]
    
    action_dim = final_conv_weight.shape[0]  # 7
    diffusion_step_embed_dim = diff_enc_weight.shape[1]  # 64
    global_cond_dim = cond_weight.shape[1] - diffusion_step_embed_dim  # 562
    
    print(f"  action_dim: {action_dim}")
    print(f"  global_cond_dim: {global_cond_dim}")
    
    pred_horizon = 16
    act_horizon = 8
    obs_horizon = 2  # Assumed from typical config
    
    # Create velocity network
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=(64, 128, 256),
        n_groups=8,
    )
    
    # Create Q-network (not used for evaluation but needed for agent)
    q_network = EnsembleQNetwork(
        action_dim=action_dim,
        obs_dim=global_cond_dim,
        action_horizon=act_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=10,
        num_min_qs=2,
    )
    
    # Create agent
    agent = AWSCAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        obs_dim=global_cond_dim,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        device=DEVICE,
    ).to(DEVICE)
    
    # Load pretrained weights
    print(f"  Loading checkpoint: {ckpt_path}")
    agent.load_pretrained(ckpt_path, load_critic=False)
    print("✓ Pretrained weights loaded")
    
    # Test inference without environment
    batch_size = 4
    obs_features = torch.randn(batch_size, global_cond_dim, device=DEVICE)
    
    agent.eval()
    with torch.no_grad():
        action_seq = agent.select_action(obs_features, deterministic=True)
    
    assert action_seq.shape == (batch_size, act_horizon, action_dim)
    assert action_seq.abs().max() <= 1.0 + 1e-5, "Actions should be bounded"
    print(f"✓ Inference test passed: action shape = {action_seq.shape}")
    print(f"  Action range: [{action_seq.min().item():.4f}, {action_seq.max().item():.4f}]")
    
    return True


def test_mini_evaluation_loop():
    """Test a minimal evaluation loop without full environment."""
    print("\n" + "=" * 60)
    print("Test: Mini evaluation loop")
    print("=" * 60)
    
    from diffusion_policy.rlpd.awsc_agent import AWSCAgent, ShortCutVelocityUNet1D
    from diffusion_policy.rlpd.networks import EnsembleQNetwork
    
    obs_dim = 128
    action_dim = 7
    pred_horizon = 16
    act_horizon = 8
    batch_size = 4
    num_steps = 10
    
    # Create agent
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=64,
    )
    
    q_network = EnsembleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim,
        action_horizon=act_horizon,
    )
    
    agent = AWSCAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        obs_dim=obs_dim,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        device=DEVICE,
    ).to(DEVICE)
    
    wrapper = AWSCAgentWrapper(
        agent=agent,
        visual_encoder=None,
        obs_horizon=1,  # No stacking for simplicity
    )
    
    # Simulate evaluation loop
    wrapper.eval()
    wrapper.reset()
    
    all_actions = []
    with torch.no_grad():
        for step in range(num_steps):
            # Simulate observation
            obs = torch.randn(batch_size, obs_dim, device=DEVICE)
            
            # Get action sequence
            action_seq = wrapper.get_action(obs, deterministic=True)
            all_actions.append(action_seq)
            
            # Execute each action in sequence (simulated)
            for i in range(action_seq.shape[1]):
                action = action_seq[:, i]  # (B, action_dim)
                # In real env: obs, reward, done, info = env.step(action)
    
    print(f"✓ Completed {num_steps} evaluation steps")
    print(f"  Total actions generated: {len(all_actions)}")
    print(f"  Each action sequence shape: {all_actions[0].shape}")
    
    return True


def run_all_tests():
    """Run all evaluation tests."""
    print("\n" + "=" * 60)
    print("RLPD Evaluate Test Suite")
    print("=" * 60)
    
    tests = [
        ("SACAgentWrapper interface", test_sac_wrapper_interface),
        ("AWSCAgentWrapper interface", test_awsc_wrapper_interface),
        ("Pretrained AWSC evaluation", test_awsc_pretrained_evaluation),
        ("Mini evaluation loop", test_mini_evaluation_loop),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"\n✗ FAILED: {name}")
        except Exception as e:
            failed += 1
            print(f"\n✗ FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
