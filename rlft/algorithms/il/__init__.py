"""
Imitation Learning algorithms.

- DiffusionPolicyAgent: DDPM-based policy (Chi et al., 2023)
- FlowMatchingAgent: Conditional Flow Matching policy
- ShortCutFlowAgent: Adaptive step-size flow (from ReinFlow)
- ConsistencyFlowAgent: Flow matching with consistency loss
- ReflectedFlowAgent: Flow matching with boundary reflection
"""

from .diffusion_policy import DiffusionPolicyAgent
from .flow_matching import FlowMatchingAgent
from .shortcut_flow import ShortCutFlowAgent
from .consistency_flow import ConsistencyFlowAgent
from .reflected_flow import ReflectedFlowAgent

__all__ = [
    "DiffusionPolicyAgent",
    "FlowMatchingAgent",
    "ShortCutFlowAgent",
    "ConsistencyFlowAgent",
    "ReflectedFlowAgent",
]
