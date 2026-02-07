"""Shared fixtures for carm_deploy tests."""

import sys
import os
import json
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup – make rlft and carm_deploy importable
# ---------------------------------------------------------------------------
_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RL_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CARM_DEPLOY_ROOT)))

for p in (_CARM_DEPLOY_ROOT, _RL_VLA_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Default policy config (mirrors POLICY_DEFAULTS)
# ---------------------------------------------------------------------------
@pytest.fixture()
def default_policy_config():
    return {
        "obs_horizon": 2,
        "pred_horizon": 16,
        "action_dim": 13,
        "action_dim_full": 15,
        "state_mode": "joint_only",
        "target_image_size": (128, 128),
        "visual_feature_dim": 256,
        "state_encoder_hidden_dim": 128,
        "state_encoder_out_dim": 256,
        "use_state_encoder": True,
        "algorithm": "consistency_flow",
        "num_inference_steps": 10,
        "use_ema": False,
        "gripper_threshold": 0.05,
        "gripper_open_val": 0.078,
        "gripper_close_val": 0.04,
        "gripper_head_hidden_dim": 256,
        "gripper_hysteresis_window": 1,
    }


# ---------------------------------------------------------------------------
# Fake checkpoint directory (with args.json + .pt)
# ---------------------------------------------------------------------------
@pytest.fixture()
def fake_checkpoint_dir(tmp_path, device):
    """Build a fake checkpoint directory with model weights + args.json."""
    from rlft.networks import PlainConv, StateEncoder, GripperHead
    from rlft.utils.model_factory import create_agent_for_inference
    from rlft.utils.checkpoint import build_checkpoint

    obs_horizon = 2
    visual_dim = 256
    state_encoder_out_dim = 256
    action_dim = 13
    pred_horizon = 16

    visual_encoder = PlainConv(in_channels=3, out_dim=visual_dim, pool_feature_map=True)
    state_encoder = StateEncoder(state_dim=7, hidden_dim=128, out_dim=state_encoder_out_dim)
    global_cond_dim = obs_horizon * (visual_dim + state_encoder_out_dim)
    agent = create_agent_for_inference(
        "consistency_flow", action_dim=action_dim,
        global_cond_dim=global_cond_dim, obs_horizon=obs_horizon,
        pred_horizon=pred_horizon, device=str(device),
    )
    gripper_head = GripperHead(obs_dim=global_cond_dim, pred_horizon=pred_horizon, hidden_dim=256)

    ckpt = build_checkpoint(
        agent=agent, visual_encoder=visual_encoder,
        state_encoder=state_encoder, gripper_head=gripper_head,
    )
    ckpt_path = str(tmp_path / "model.pt")
    torch.save(ckpt, ckpt_path)

    args = {
        "obs_horizon": obs_horizon,
        "pred_horizon": pred_horizon,
        "action_dim": action_dim,
        "visual_feature_dim": visual_dim,
        "state_encoder_hidden_dim": 128,
        "state_encoder_out_dim": state_encoder_out_dim,
        "use_state_encoder": True,
        "algorithm": "consistency_flow",
        "state_mode": "joint_only",
        "visual_encoder_type": "plain_conv",
        "auto_image_size": True,
        "action_mode": "full",
        "gripper_threshold": 0.05,
        "gripper_open_val": 0.078,
        "gripper_close_val": 0.04,
        "gripper_head_hidden_dim": 256,
    }
    with open(str(tmp_path / "args.json"), "w") as f:
        json.dump(args, f)

    return tmp_path


# ---------------------------------------------------------------------------
# Random observation helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def random_image():
    """Return a factory for random HWC uint8 images."""
    def _make(h=128, w=128, c=3):
        return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
    return _make


@pytest.fixture()
def random_state():
    """Return a factory for random state vectors."""
    def _make(dim=7):
        return np.random.randn(dim).astype(np.float32)
    return _make
