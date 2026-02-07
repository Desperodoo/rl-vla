"""Tests for rlft.utils.model_factory — agent creation factory."""

import pytest
import torch

from rlft.utils.model_factory import create_agent_for_inference, SUPPORTED_ALGORITHMS


class TestSupportedAlgorithms:
    def test_has_five_algorithms(self):
        assert len(SUPPORTED_ALGORITHMS) == 5

    def test_expected_names(self):
        expected = {
            "diffusion_policy", "flow_matching",
            "consistency_flow", "reflected_flow", "shortcut_flow",
        }
        assert set(SUPPORTED_ALGORITHMS) == expected


class TestCreateAgentForInference:
    """Verify create_agent_for_inference builds an agent for every supported algorithm."""

    ACTION_DIM = 7
    GLOBAL_COND_DIM = 512

    @pytest.mark.parametrize("algorithm", SUPPORTED_ALGORITHMS)
    def test_creates_agent(self, algorithm, device):
        agent = create_agent_for_inference(
            algorithm=algorithm,
            action_dim=self.ACTION_DIM,
            global_cond_dim=self.GLOBAL_COND_DIM,
            obs_horizon=2,
            pred_horizon=16,
            num_inference_steps=4,
            device=str(device),
        )
        assert agent is not None
        agent = agent.to(device)
        # Check it's a proper nn.Module
        assert isinstance(agent, torch.nn.Module)

    @pytest.mark.parametrize("algorithm", SUPPORTED_ALGORITHMS)
    def test_agent_forward_deterministic(self, algorithm, device):
        """get_action_deterministic should return (B, pred_horizon, action_dim)."""
        agent = create_agent_for_inference(
            algorithm=algorithm,
            action_dim=self.ACTION_DIM,
            global_cond_dim=self.GLOBAL_COND_DIM,
            pred_horizon=8,
            num_inference_steps=2,
            device=str(device),
        ).to(device)
        agent.eval()
        obs = torch.randn(1, self.GLOBAL_COND_DIM, device=device)
        with torch.no_grad():
            action = agent.get_action_deterministic(obs)
        assert action.shape == (1, 8, self.ACTION_DIM)

    def test_unsupported_algorithm_raises(self, device):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            create_agent_for_inference(
                algorithm="not_a_real_algo",
                action_dim=7,
                global_cond_dim=256,
                device=str(device),
            )
