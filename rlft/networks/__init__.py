"""
Neural Network Architectures for RLFT.

This module provides all neural network components:
- unet: Conditional 1D U-Net for diffusion/flow models
- velocity: Velocity networks for flow matching
- q_networks: Q-networks for RL (DoubleQ, Ensemble)
- actors: Actor networks (Gaussian, Latent)
- encoders: Visual and state encoders
"""

from .unet import (
    ConditionalUnet1D,
    SinusoidalPosEmb,
    Conv1dBlock,
    ConditionalResidualBlock1D,
)
from .velocity import VelocityUNet1D, ShortCutVelocityUNet1D, GripperHead
from .q_networks import DoubleQNetwork, EnsembleQNetwork, SigmoidQNetwork, soft_update
from .actors import DiagGaussianActor, LearnableTemperature
from .encoders import (
    PlainConv,
    ResNetEncoder,
    StateEncoder,
    create_visual_encoder,
    get_encoder_input_size,
)

__all__ = [
    # U-Net
    "ConditionalUnet1D",
    "SinusoidalPosEmb",
    "Conv1dBlock",
    "ConditionalResidualBlock1D",
    # Velocity
    "VelocityUNet1D",
    "ShortCutVelocityUNet1D",
    "GripperHead",
    # Q-Networks
    "DoubleQNetwork",
    "EnsembleQNetwork",
    "SigmoidQNetwork",
    "soft_update",
    # Actors
    "DiagGaussianActor",
    "LearnableTemperature",
    # Encoders
    "PlainConv",
    "ResNetEncoder",
    "StateEncoder",
    "create_visual_encoder",
    "get_encoder_input_size",
]
