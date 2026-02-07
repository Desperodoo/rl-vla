"""Tests for rlft.datasets.data_utils.encode_observations."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from rlft.datasets.data_utils import encode_observations
from rlft.networks import PlainConv, StateEncoder


@pytest.fixture()
def visual_encoder(device):
    return PlainConv(in_channels=3, out_dim=256, pool_feature_map=True).to(device).eval()


@pytest.fixture()
def state_enc(device):
    return StateEncoder(state_dim=7, hidden_dim=64, out_dim=128).to(device).eval()


def _obs_seq(B=2, T=2, H=128, W=128, state_dim=7, device="cpu"):
    return {
        "rgb": torch.randint(0, 256, (B, T, 3, H, W), dtype=torch.uint8).to(device),
        "state": torch.randn(B, T, state_dim, device=device),
    }


class TestEncodeObservations:
    def test_flatten_shape(self, device, visual_encoder):
        obs = _obs_seq(B=2, T=2, device=device)
        out = encode_observations(obs, visual_encoder, include_rgb=True, device=device, flatten=True)
        # 2 * (256 + 7) = 526
        assert out.shape == (2, 2 * (256 + 7))

    def test_unflatten_shape(self, device, visual_encoder):
        obs = _obs_seq(B=2, T=2, device=device)
        out = encode_observations(obs, visual_encoder, include_rgb=True, device=device, flatten=False)
        assert out.ndim == 3
        assert out.shape == (2, 2, 256 + 7)

    def test_with_state_encoder(self, device, visual_encoder, state_enc):
        obs = _obs_seq(B=1, T=2, device=device)
        out = encode_observations(
            obs, visual_encoder, include_rgb=True, device=device,
            state_encoder=state_enc, flatten=True,
        )
        # 2 * (256 + 128)
        assert out.shape == (1, 2 * (256 + 128))

    def test_state_only_no_rgb(self, device):
        obs = {"state": torch.randn(3, 2, 7, device=device)}
        out = encode_observations(obs, visual_encoder=None, include_rgb=False, device=device, flatten=True)
        assert out.shape == (3, 2 * 7)

    def test_nhwc_auto_detect(self, device, visual_encoder):
        """NHWC images should be auto-converted to NCHW."""
        obs = {
            "rgb": torch.randint(0, 256, (1, 2, 128, 128, 3), dtype=torch.uint8).to(device),
            "state": torch.randn(1, 2, 7, device=device),
        }
        out = encode_observations(obs, visual_encoder, include_rgb=True, device=device, flatten=True)
        assert out.shape == (1, 2 * (256 + 7))

    def test_output_is_float(self, device, visual_encoder):
        obs = _obs_seq(B=1, T=1, device=device)
        out = encode_observations(obs, visual_encoder, include_rgb=True, device=device)
        assert out.dtype == torch.float32
