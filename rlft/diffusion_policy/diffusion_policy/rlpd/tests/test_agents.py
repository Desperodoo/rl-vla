"""
Test SACAgent and AWSCAgent.

Run with:
    cd /home/amax/rlft/diffusion_policy
    conda activate rlft_ms3
    python -m diffusion_policy.rlpd.tests.test_agents
"""

import torch
import torch.nn as nn
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing on device: {DEVICE}")

# Checkpoint path for AWSC
CHECKPOINT_PATH = "runs/awsc-LiftPegUpright-v1-seed0/checkpoints/best_eval_success_once.pt"


def test_sac_agent_init():
    """Test SACAgent initialization."""
    print("\n" + "=" * 60)
    print("Test: SACAgent initialization")
    print("=" * 60)
    
    from diffusion_policy.rlpd.sac_agent import SACAgent
    
    obs_dim = 512
    action_dim = 7
    action_horizon = 8
    num_qs = 10
    num_min_qs = 2
    
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=num_qs,
        num_min_qs=num_min_qs,
        gamma=0.99,
        tau=0.005,
        init_temperature=1.0,
        device=DEVICE,
    ).to(DEVICE)
    
    # Check components
    assert agent.actor is not None
    assert agent.critic is not None
    assert agent.critic_target is not None
    assert agent.temperature is not None
    
    print(f"✓ SACAgent initialized with:")
    print(f"    obs_dim={obs_dim}, action_dim={action_dim}, action_horizon={action_horizon}")
    print(f"    num_qs={num_qs}, num_min_qs={num_min_qs}")
    
    # Check parameter counts
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    print(f"    Actor params: {actor_params:,}")
    print(f"    Critic params: {critic_params:,}")
    
    return True


def test_sac_agent_select_action():
    """Test SACAgent.select_action() output shape."""
    print("\n" + "=" * 60)
    print("Test: SACAgent select_action()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.sac_agent import SACAgent
    
    obs_dim = 512
    action_dim = 7
    action_horizon = 8
    batch_size = 32
    
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=10,
        num_min_qs=2,
        device=DEVICE,
    ).to(DEVICE)
    
    # Test batched input
    obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    action = agent.select_action(obs, deterministic=False)
    
    expected_shape = (batch_size, action_horizon, action_dim)
    assert action.shape == expected_shape, f"Expected {expected_shape}, got {action.shape}"
    print(f"✓ Batched action shape: {action.shape}")
    
    # Test single input
    obs_single = torch.randn(obs_dim, device=DEVICE)
    action_single = agent.select_action(obs_single, deterministic=True)
    
    expected_single_shape = (action_horizon, action_dim)
    assert action_single.shape == expected_single_shape, \
        f"Expected {expected_single_shape}, got {action_single.shape}"
    print(f"✓ Single action shape: {action_single.shape}")
    
    # Actions should be in [-1, 1]
    assert action.min() >= -1.0 and action.max() <= 1.0, \
        f"Actions out of [-1, 1]: [{action.min()}, {action.max()}]"
    print(f"✓ Actions bounded in [-1, 1]")
    
    return True


def test_sac_agent_losses():
    """Test SACAgent loss computation."""
    print("\n" + "=" * 60)
    print("Test: SACAgent loss computation")
    print("=" * 60)
    
    from diffusion_policy.rlpd.sac_agent import SACAgent
    
    obs_dim = 512
    action_dim = 7
    action_horizon = 8
    batch_size = 32
    
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=10,
        num_min_qs=2,
        device=DEVICE,
    ).to(DEVICE)
    
    # Create mock batch
    obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    actions = torch.randn(batch_size, action_horizon, action_dim, device=DEVICE).clamp(-1, 1)
    next_obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    rewards = torch.randn(batch_size, device=DEVICE)
    dones = torch.zeros(batch_size, device=DEVICE)
    cumulative_reward = torch.randn(batch_size, device=DEVICE)
    chunk_done = torch.zeros(batch_size, device=DEVICE)
    discount_factor = torch.full((batch_size,), 0.99 ** action_horizon, device=DEVICE)
    
    # Test actor loss
    actor_loss, actor_metrics = agent.compute_actor_loss(obs)
    
    assert actor_loss.ndim == 0, "Actor loss should be scalar"
    assert not torch.isnan(actor_loss), "Actor loss is NaN"
    print(f"✓ Actor loss: {actor_loss.item():.4f}")
    print(f"    Entropy: {actor_metrics['actor_entropy']:.4f}")
    print(f"    Q value: {actor_metrics['actor_q']:.4f}")
    
    # Test critic loss
    critic_loss, critic_metrics = agent.compute_critic_loss(
        obs_features=obs,
        actions=actions,
        next_obs_features=next_obs,
        rewards=rewards,
        dones=dones,
        cumulative_reward=cumulative_reward,
        chunk_done=chunk_done,
        discount_factor=discount_factor,
    )
    
    assert critic_loss.ndim == 0, "Critic loss should be scalar"
    assert not torch.isnan(critic_loss), "Critic loss is NaN"
    print(f"✓ Critic loss: {critic_loss.item():.4f}")
    print(f"    Q mean: {critic_metrics['q_mean']:.4f}")
    print(f"    TD target mean: {critic_metrics['td_target_mean']:.4f}")
    
    # Test temperature loss
    temp_loss, temp_metrics = agent.compute_temperature_loss(obs)
    
    assert temp_loss.ndim == 0, "Temperature loss should be scalar"
    assert not torch.isnan(temp_loss), "Temperature loss is NaN"
    print(f"✓ Temperature loss: {temp_loss.item():.4f}")
    
    return True


def test_sac_agent_update_target():
    """Test SACAgent target network update."""
    print("\n" + "=" * 60)
    print("Test: SACAgent update_target()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.sac_agent import SACAgent
    
    obs_dim = 512
    action_dim = 7
    action_horizon = 8
    
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=10,
        num_min_qs=2,
        tau=0.1,  # Large tau for visible change
        device=DEVICE,
    ).to(DEVICE)
    
    # Manually modify critic parameters to create a difference from target
    # (Initially target = copy of critic, so diff is 0)
    with torch.no_grad():
        for param in agent.critic.parameters():
            param.add_(torch.randn_like(param) * 0.5)
    
    # Get params after modification
    target_param_before = list(agent.critic_target.parameters())[0].clone()
    critic_param = list(agent.critic.parameters())[0].clone()
    
    # Calculate diff before update
    diff_before = (target_param_before - critic_param).abs().mean().item()
    
    # Update target (soft update: target = tau * critic + (1 - tau) * target)
    agent.update_target()
    
    target_param_after = list(agent.critic_target.parameters())[0]
    
    # Check that target moved toward critic
    diff_after = (target_param_after - critic_param).abs().mean().item()
    
    print(f"✓ Target param diff before soft update: {diff_before:.6f}")
    print(f"✓ Target param diff after soft update: {diff_after:.6f}")
    assert diff_before > 0, "Critic and target should differ after modification"
    assert diff_after < diff_before, "Target should move toward critic"
    print("✓ Target network updated correctly")
    
    return True


def test_awsc_agent_init():
    """Test AWSCAgent initialization."""
    print("\n" + "=" * 60)
    print("Test: AWSCAgent initialization")
    print("=" * 60)
    
    from diffusion_policy.rlpd.awsc_agent import AWSCAgent, ShortCutVelocityUNet1D
    from diffusion_policy.rlpd.networks import EnsembleQNetwork
    
    obs_dim = 512
    action_dim = 7
    obs_horizon = 2
    pred_horizon = 16
    act_horizon = 8
    num_qs = 10
    num_min_qs = 2
    
    # Create velocity network
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=64,
        down_dims=(64, 128, 256),
        n_groups=8,
    )
    
    # Create Q-network
    q_network = EnsembleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim,
        action_horizon=act_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=num_qs,
        num_min_qs=num_min_qs,
    )
    
    agent = AWSCAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        num_qs=num_qs,
        num_min_qs=num_min_qs,
        beta=10.0,
        bc_weight=1.0,
        shortcut_weight=0.3,
        gamma=0.99,
        tau=0.005,
        device=DEVICE,
    ).to(DEVICE)
    
    # Check components
    assert agent.velocity_net is not None
    assert agent.velocity_net_ema is not None
    assert agent.critic is not None
    assert agent.critic_target is not None
    
    print(f"✓ AWSCAgent initialized with:")
    print(f"    obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"    pred_horizon={pred_horizon}, act_horizon={act_horizon}")
    print(f"    num_qs={num_qs}, num_min_qs={num_min_qs}")
    
    # Check parameter counts
    velocity_params = sum(p.numel() for p in agent.velocity_net.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    print(f"    Velocity net params: {velocity_params:,}")
    print(f"    Critic params: {critic_params:,}")
    
    return True


def test_awsc_agent_select_action():
    """Test AWSCAgent.select_action() output shape."""
    print("\n" + "=" * 60)
    print("Test: AWSCAgent select_action()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.awsc_agent import AWSCAgent, ShortCutVelocityUNet1D
    from diffusion_policy.rlpd.networks import EnsembleQNetwork
    
    obs_dim = 512
    action_dim = 7
    pred_horizon = 16
    act_horizon = 8
    batch_size = 32
    
    # Create agent
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=64,
        down_dims=(64, 128, 256),
        n_groups=8,
    )
    
    q_network = EnsembleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim,
        action_horizon=act_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=10,
        num_min_qs=2,
    )
    
    agent = AWSCAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        obs_dim=obs_dim,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
        num_inference_steps=4,  # Fewer steps for faster test
        device=DEVICE,
    ).to(DEVICE)
    
    # Test batched input
    obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    action = agent.select_action(obs, deterministic=False)
    
    expected_shape = (batch_size, act_horizon, action_dim)
    assert action.shape == expected_shape, f"Expected {expected_shape}, got {action.shape}"
    print(f"✓ Batched action shape: {action.shape}")
    
    # Test deterministic mode
    action_det = agent.select_action(obs, deterministic=True)
    assert action_det.shape == expected_shape
    print(f"✓ Deterministic action shape: {action_det.shape}")
    
    # Actions should be in [-1, 1]
    assert action.min() >= -1.0 and action.max() <= 1.0, \
        f"Actions out of [-1, 1]: [{action.min()}, {action.max()}]"
    print(f"✓ Actions bounded in [-1, 1]: [{action.min().item():.4f}, {action.max().item():.4f}]")
    
    return True


def test_awsc_agent_load_pretrained():
    """Test AWSCAgent.load_pretrained() with real checkpoint."""
    print("\n" + "=" * 60)
    print("Test: AWSCAgent load_pretrained()")
    print("=" * 60)
    
    from diffusion_policy.rlpd.awsc_agent import AWSCAgent, ShortCutVelocityUNet1D
    from diffusion_policy.rlpd.networks import EnsembleQNetwork
    
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", CHECKPOINT_PATH)
    ckpt_path = os.path.abspath(ckpt_path)
    
    if not os.path.exists(ckpt_path):
        print(f"⚠ Checkpoint not found: {ckpt_path}")
        print("  Skipping load_pretrained test")
        return True
    
    # First, inspect checkpoint to get dimensions
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Get dimensions from checkpoint
    # cond_encoder weight shape: [out, in] -> in = obs_dim + diffusion_step_embed_dim
    cond_weight = ckpt["agent"]["velocity_net.unet.mid_modules.0.cond_encoder.1.weight"]
    print(f"  cond_encoder weight shape: {cond_weight.shape}")
    
    # final_conv output shape tells us action_dim
    final_conv_weight = ckpt["agent"]["velocity_net.unet.final_conv.1.weight"]
    action_dim = final_conv_weight.shape[0]
    print(f"  action_dim from checkpoint: {action_dim}")
    
    # down_modules.0.0 tells us input_dim
    down0_weight = ckpt["agent"]["velocity_net.unet.down_modules.0.0.blocks.0.block.0.weight"]
    input_dim = down0_weight.shape[1]
    print(f"  input_dim (action_dim) from down_modules: {input_dim}")
    
    # Calculate obs_dim from cond_encoder
    # In ConditionalUnet1D: cond_dim = dsed + global_cond_dim
    # cond_encoder.1 is Linear(cond_dim, out_channels*2), so shape = (out*2, cond_dim)
    # Thus cond_weight.shape[1] = dsed + global_cond_dim
    
    # Get diffusion_step_embed_dim (dsed) from diffusion_step_encoder
    # diffusion_step_encoder = [SinusoidalPosEmb(dsed), Linear(dsed, dsed*4), Mish, Linear(dsed*4, dsed)]
    # diffusion_step_encoder.1 is first Linear: weight shape (dsed*4, dsed) -> dsed = shape[1]
    diff_enc_weight = ckpt["agent"]["velocity_net.unet.diffusion_step_encoder.1.weight"]
    print(f"  diffusion_step_encoder.1 weight shape: {diff_enc_weight.shape}")
    diffusion_step_embed_dim = diff_enc_weight.shape[1]  # dsed
    print(f"  diffusion_step_embed_dim: {diffusion_step_embed_dim}")
    
    # Now calculate global_cond_dim
    # cond_weight.shape[1] = dsed + global_cond_dim
    total_cond_in = cond_weight.shape[1]
    global_cond_dim = total_cond_in - diffusion_step_embed_dim
    print(f"  global_cond_dim (obs_dim): {global_cond_dim}")
    
    obs_dim = global_cond_dim
    pred_horizon = 16
    act_horizon = 8
    num_qs = 10
    num_min_qs = 2
    
    # Create agent with matching architecture
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=64,  # From checkpoint inspection
        down_dims=(64, 128, 256),
        n_groups=8,
    )
    
    q_network = EnsembleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim,
        action_horizon=act_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=num_qs,
        num_min_qs=num_min_qs,
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
    
    # Load pretrained weights - this should NOT use try-except
    print(f"\n  Loading checkpoint: {ckpt_path}")
    agent.load_pretrained(ckpt_path, load_critic=False)
    
    print("✓ Pretrained weights loaded successfully")
    
    # Verify by running inference
    obs = torch.randn(4, obs_dim, device=DEVICE)
    action = agent.select_action(obs, deterministic=True)
    print(f"✓ Inference after loading: action shape = {action.shape}")
    
    return True


def test_awsc_agent_losses():
    """Test AWSCAgent loss computation."""
    print("\n" + "=" * 60)
    print("Test: AWSCAgent loss computation")
    print("=" * 60)
    
    from diffusion_policy.rlpd.awsc_agent import AWSCAgent, ShortCutVelocityUNet1D
    from diffusion_policy.rlpd.networks import EnsembleQNetwork
    
    obs_dim = 512
    action_dim = 7
    pred_horizon = 16
    act_horizon = 8
    batch_size = 32
    
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=64,
        down_dims=(64, 128, 256),
        n_groups=8,
    )
    
    q_network = EnsembleQNetwork(
        action_dim=action_dim,
        obs_dim=obs_dim,
        action_horizon=act_horizon,
        hidden_dims=[256, 256, 256],
        num_qs=10,
        num_min_qs=2,
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
    
    # Create mock batch
    obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    actions_bc = torch.randn(batch_size, pred_horizon, action_dim, device=DEVICE).clamp(-1, 1)
    actions_q = torch.randn(batch_size, act_horizon, action_dim, device=DEVICE).clamp(-1, 1)
    next_obs = torch.randn(batch_size, obs_dim, device=DEVICE)
    rewards = torch.randn(batch_size, device=DEVICE)
    dones = torch.zeros(batch_size, device=DEVICE)
    cumulative_reward = torch.randn(batch_size, device=DEVICE)
    chunk_done = torch.zeros(batch_size, device=DEVICE)
    discount_factor = torch.full((batch_size,), 0.99 ** act_horizon, device=DEVICE)
    
    # Test actor loss
    actor_loss, actor_metrics = agent.compute_actor_loss(obs, actions_bc, actions_q)
    
    assert actor_loss.ndim == 0, "Actor loss should be scalar"
    assert not torch.isnan(actor_loss), "Actor loss is NaN"
    print(f"✓ Actor loss: {actor_loss.item():.4f}")
    print(f"    BC loss: {actor_metrics.get('bc_loss', 'N/A')}")
    print(f"    Shortcut loss: {actor_metrics.get('shortcut_loss', 'N/A')}")
    
    # Test critic loss
    critic_loss, critic_metrics = agent.compute_critic_loss(
        obs_features=obs,
        actions=actions_q,
        next_obs_features=next_obs,
        rewards=rewards,
        dones=dones,
        cumulative_reward=cumulative_reward,
        chunk_done=chunk_done,
        discount_factor=discount_factor,
    )
    
    assert critic_loss.ndim == 0, "Critic loss should be scalar"
    assert not torch.isnan(critic_loss), "Critic loss is NaN"
    print(f"✓ Critic loss: {critic_loss.item():.4f}")
    print(f"    Q mean: {critic_metrics['q_mean']:.4f}")
    
    return True


def run_all_tests():
    """Run all agent tests."""
    print("\n" + "=" * 60)
    print("RLPD Agent Test Suite")
    print("=" * 60)
    
    tests = [
        ("SACAgent init", test_sac_agent_init),
        ("SACAgent select_action", test_sac_agent_select_action),
        ("SACAgent losses", test_sac_agent_losses),
        ("SACAgent update_target", test_sac_agent_update_target),
        ("AWSCAgent init", test_awsc_agent_init),
        ("AWSCAgent select_action", test_awsc_agent_select_action),
        ("AWSCAgent load_pretrained", test_awsc_agent_load_pretrained),
        ("AWSCAgent losses", test_awsc_agent_losses),
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
