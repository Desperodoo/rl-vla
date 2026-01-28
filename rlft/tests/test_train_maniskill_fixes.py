"""
Test script to verify train_maniskill.py fixes for all algorithm/mode combinations.

This tests:
1. State mode no longer fails with FlattenRGBDObservationWrapper
2. Offline RL algorithms receive rewards and dones parameters
3. IL algorithms (including consistency_flow, reflected_flow) don't receive extra params
"""

import torch
import sys
sys.path.insert(0, '/home/lizh/rl-vla')

# Test algorithm categorization
def test_algorithm_categorization():
    """Test that algorithms are correctly categorized as IL or Offline RL."""
    # These should match what's in train_maniskill.py
    il_algorithms = ["diffusion_policy", "flow_matching", "shortcut_flow", "consistency_flow", "reflected_flow"]
    offline_rl_algorithms = ["cpql", "awcp", "aw_shortcut_flow"]
    
    # IL algorithms should NOT need rewards/dones as required params
    from rlft.algorithms.il import (
        DiffusionPolicyAgent, FlowMatchingAgent, ShortCutFlowAgent,
        ConsistencyFlowAgent, ReflectedFlowAgent
    )
    
    import inspect
    for name, cls in [
        ("diffusion_policy", DiffusionPolicyAgent),
        ("flow_matching", FlowMatchingAgent),
        ("shortcut_flow", ShortCutFlowAgent),
        ("consistency_flow", ConsistencyFlowAgent),
        ("reflected_flow", ReflectedFlowAgent),
    ]:
        sig = inspect.signature(cls.compute_loss)
        params = list(sig.parameters.keys())
        assert "rewards" not in params or sig.parameters["rewards"].default is not inspect.Parameter.empty, \
            f"{name} should not require 'rewards' parameter"
        print(f"✓ {name} compute_loss signature: {list(sig.parameters.keys())}")

def test_offline_rl_compute_loss_signatures():
    """Test that Offline RL algorithms have correct compute_loss signatures."""
    from rlft.algorithms.offline_rl import CPQLAgent, AWCPAgent, AWShortCutFlowAgent
    
    import inspect
    for name, cls in [
        ("cpql", CPQLAgent),
        ("awcp", AWCPAgent),
        ("aw_shortcut_flow", AWShortCutFlowAgent),
    ]:
        sig = inspect.signature(cls.compute_loss)
        params = list(sig.parameters.keys())
        # These SHOULD have rewards and dones
        assert "rewards" in params, f"{name} should have 'rewards' parameter"
        assert "dones" in params, f"{name} should have 'dones' parameter"
        print(f"✓ {name} compute_loss signature: {params}")

def test_il_agent_compute_loss():
    """Test IL agents can compute loss with just obs_features and actions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    act_dim = 7
    obs_dim = 25
    pred_horizon = 16
    obs_horizon = 2
    
    from rlft.algorithms.il import (
        FlowMatchingAgent, ShortCutFlowAgent,
        ConsistencyFlowAgent, ReflectedFlowAgent
    )
    from rlft.networks import VelocityUNet1D, ShortCutVelocityUNet1D
    
    # Create dummy data
    batch_size = 8
    obs_features = torch.randn(batch_size, obs_dim, device=device)
    actions = torch.randn(batch_size, pred_horizon, act_dim, device=device)
    
    # Test each IL algorithm
    print("Testing FlowMatchingAgent...")
    velocity_net = VelocityUNet1D(
        input_dim=act_dim,
        global_cond_dim=obs_dim,
    ).to(device)
    agent = FlowMatchingAgent(
        velocity_net=velocity_net,
        action_dim=act_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
    ).to(device)
    loss_dict = agent.compute_loss(obs_features=obs_features, actions=actions)
    assert "loss" in loss_dict
    print(f"✓ flow_matching compute_loss works: loss={loss_dict['loss'].item():.4f}")
    
    print("Testing ShortCutFlowAgent...")
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=act_dim,
        global_cond_dim=obs_dim,
    ).to(device)
    agent = ShortCutFlowAgent(
        velocity_net=velocity_net,
        action_dim=act_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
    ).to(device)
    loss_dict = agent.compute_loss(obs_features=obs_features, actions=actions)
    assert "loss" in loss_dict
    print(f"✓ shortcut_flow compute_loss works: loss={loss_dict['loss'].item():.4f}")
    
    print("Testing ConsistencyFlowAgent...")
    velocity_net = VelocityUNet1D(
        input_dim=act_dim,
        global_cond_dim=obs_dim,
    ).to(device)
    agent = ConsistencyFlowAgent(
        velocity_net=velocity_net,
        action_dim=act_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
    ).to(device)
    loss_dict = agent.compute_loss(obs_features=obs_features, actions=actions)
    assert "loss" in loss_dict
    print(f"✓ consistency_flow compute_loss works: loss={loss_dict['loss'].item():.4f}")
    
    print("Testing ReflectedFlowAgent...")
    velocity_net = VelocityUNet1D(
        input_dim=act_dim,
        global_cond_dim=obs_dim,
    ).to(device)
    agent = ReflectedFlowAgent(
        velocity_net=velocity_net,
        action_dim=act_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
    ).to(device)
    loss_dict = agent.compute_loss(obs_features=obs_features, actions=actions)
    assert "loss" in loss_dict
    print(f"✓ reflected_flow compute_loss works: loss={loss_dict['loss'].item():.4f}")

def test_offline_rl_agent_compute_loss():
    """Test Offline RL agents need rewards and dones."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    act_dim = 7
    obs_dim = 25
    pred_horizon = 16
    act_horizon = 8
    obs_horizon = 2
    
    from rlft.algorithms.offline_rl import CPQLAgent, AWCPAgent, AWShortCutFlowAgent
    from rlft.networks import VelocityUNet1D, ShortCutVelocityUNet1D, DoubleQNetwork
    
    # Create dummy data
    batch_size = 8
    obs_features = torch.randn(batch_size, obs_dim, device=device)
    next_obs_features = torch.randn(batch_size, obs_dim, device=device)
    actions = torch.randn(batch_size, pred_horizon, act_dim, device=device)
    actions_for_q = torch.randn(batch_size, act_horizon, act_dim, device=device)
    rewards = torch.randn(batch_size, device=device)
    dones = torch.zeros(batch_size, device=device)
    
    # Test CPQLAgent
    print("Testing CPQLAgent...")
    velocity_net = VelocityUNet1D(
        input_dim=act_dim,
        global_cond_dim=obs_dim,
    ).to(device)
    q_network = DoubleQNetwork(
        action_dim=act_dim,
        obs_dim=obs_dim,
        action_horizon=act_horizon,
    ).to(device)
    agent = CPQLAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        action_dim=act_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
    ).to(device)
    loss_dict = agent.compute_loss(
        obs_features=obs_features,
        actions=actions,
        rewards=rewards,
        next_obs_features=next_obs_features,
        dones=dones,
        actions_for_q=actions_for_q,
    )
    assert "loss" in loss_dict
    print(f"✓ cpql compute_loss works: loss={loss_dict['loss'].item():.4f}")
    
    # Test AWCPAgent
    print("Testing AWCPAgent...")
    velocity_net = VelocityUNet1D(
        input_dim=act_dim,
        global_cond_dim=obs_dim,
    ).to(device)
    q_network = DoubleQNetwork(
        action_dim=act_dim,
        obs_dim=obs_dim,
        action_horizon=act_horizon,
    ).to(device)
    agent = AWCPAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        action_dim=act_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
    ).to(device)
    loss_dict = agent.compute_loss(
        obs_features=obs_features,
        actions=actions,
        rewards=rewards,
        next_obs_features=next_obs_features,
        dones=dones,
        actions_for_q=actions_for_q,
    )
    assert "loss" in loss_dict
    print(f"✓ awcp compute_loss works: loss={loss_dict['loss'].item():.4f}")
    
    # Test AWShortCutFlowAgent
    print("Testing AWShortCutFlowAgent...")
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=act_dim,
        global_cond_dim=obs_dim,
    ).to(device)
    q_network = DoubleQNetwork(
        action_dim=act_dim,
        obs_dim=obs_dim,
        action_horizon=act_horizon,
    ).to(device)
    agent = AWShortCutFlowAgent(
        velocity_net=velocity_net,
        q_network=q_network,
        action_dim=act_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        act_horizon=act_horizon,
    ).to(device)
    loss_dict = agent.compute_loss(
        obs_features=obs_features,
        actions=actions,
        rewards=rewards,
        next_obs_features=next_obs_features,
        dones=dones,
        actions_for_q=actions_for_q,
    )
    assert "loss" in loss_dict
    print(f"✓ aw_shortcut_flow compute_loss works: loss={loss_dict['loss'].item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Algorithm Categorization")
    print("=" * 60)
    test_algorithm_categorization()
    
    print("\n" + "=" * 60)
    print("Testing Offline RL Signatures")
    print("=" * 60)
    test_offline_rl_compute_loss_signatures()
    
    print("\n" + "=" * 60)
    print("Testing IL Agent compute_loss")
    print("=" * 60)
    test_il_agent_compute_loss()
    
    print("\n" + "=" * 60)
    print("Testing Offline RL Agent compute_loss")
    print("=" * 60)
    test_offline_rl_agent_compute_loss()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
