"""
Algorithms module for Diffusion Policy

This module contains offline RL algorithms migrated from toy_diffusion_rl,
aligned with the official ManiSkill diffusion policy implementation.

Available algorithms:
- DiffusionPolicyAgent: DDPM-based imitation learning (original diffusion policy)
- FlowMatchingAgent: Flow Matching based imitation learning
- ReflectedFlowAgent: Reflected Flow for bounded action spaces
- ConsistencyFlowAgent: Consistency Flow with self-consistency loss
- ShortCutFlowAgent: ShortCut Flow with adaptive step sizes
- DiffusionDoubleQAgent: Diffusion Policy + Double Q-Learning for offline RL  
- CPQLAgent: Consistency Policy Q-Learning for offline RL
- AWCPAgent: Advantage-Weighted Consistency Policy (AWAC-style Q-weighted BC)
- AWShortCutFlowAgent: Advantage-Weighted ShortCut Flow (bridge between BC and online RL)
- DPPOAgent: DDPM policy with partial chain fine-tuning (for online RL)
- ReinFlowAgent: Flow matching with learnable exploration noise (for online RL)
"""

from .diffusion_policy import DiffusionPolicyAgent
from .flow_matching import FlowMatchingAgent
from .reflected_flow import ReflectedFlowAgent
from .consistency_flow import ConsistencyFlowAgent
from .shortcut_flow import ShortCutFlowAgent, ShortCutVelocityUNet1D
from .diffusion_double_q import DiffusionDoubleQAgent
from .cpql import CPQLAgent
from .awcp import AWCPAgent
from .aw_shortcut_flow import AWShortCutFlowAgent
from .dppo import DPPOAgent, ValueNetwork
from .reinflow import ReinFlowAgent, NoisyVelocityUNet1D, ExploreNoiseNet
from .networks import DoubleQNetwork, VelocityUNet1D, EnsembleQNetwork

__all__ = [
    "DiffusionPolicyAgent",
    "FlowMatchingAgent",
    "ReflectedFlowAgent",
    "ConsistencyFlowAgent",
    "ShortCutFlowAgent",
    "ShortCutVelocityUNet1D",
    "DiffusionDoubleQAgent", 
    "CPQLAgent",
    "AWCPAgent",
    "AWShortCutFlowAgent",
    "DPPOAgent",
    "ReinFlowAgent",
    "ValueNetwork",
    "NoisyVelocityUNet1D",
    "ExploreNoiseNet",
    "DoubleQNetwork",
    "EnsembleQNetwork",
    "VelocityUNet1D",
]
