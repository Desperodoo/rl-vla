"""
rlft.algorithms - Policy learning algorithms

Organized by training paradigm:
- il/: Imitation Learning (BC, Flow Matching, ShortCut Flow, etc.)
- offline_rl/: Offline RL (CPQL, AWCP, AW-ShortCut)
- online_rl/: Online RL (SAC, RLPD, ReinFlow, AWSC)
"""

# IL algorithms
from .il.diffusion_policy import DiffusionPolicyAgent
from .il.flow_matching import FlowMatchingAgent
from .il.shortcut_flow import ShortCutFlowAgent
from .il.consistency_flow import ConsistencyFlowAgent
from .il.reflected_flow import ReflectedFlowAgent

# Offline RL algorithms
from .offline_rl.cpql import CPQLAgent
from .offline_rl.awcp import AWCPAgent
from .offline_rl.aw_shortcut_flow import AWShortCutFlowAgent

# Online RL algorithms
from .online_rl.sac import SACAgent
from .online_rl.reinflow import ReinFlowAgent
from .online_rl.awsc import AWSCAgent

__all__ = [
    # IL
    "DiffusionPolicyAgent",
    "FlowMatchingAgent", 
    "ShortCutFlowAgent",
    "ConsistencyFlowAgent",
    "ReflectedFlowAgent",
    # Offline RL
    "CPQLAgent",
    "AWCPAgent",
    "AWShortCutFlowAgent",
    # Online RL
    "SACAgent",
    "ReinFlowAgent",
    "AWSCAgent",
]
