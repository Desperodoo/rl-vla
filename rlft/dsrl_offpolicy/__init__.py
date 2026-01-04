"""
DSRL Off-Policy: SAC-based Dual-Stage Reinforcement Learning with Latent Steering

This module implements the off-policy version of DSRL, replacing PPO with SAC
for improved sample efficiency through higher UTD (Update-To-Data) ratio.

Key differences from on-policy DSRL:
1. Uses SAC instead of PPO for Stage 2 online training
2. Replay buffer instead of rollout buffer for off-policy learning
3. Q-network in latent space Q^W(s, w) instead of V(s) for advantage estimation
4. Automatic temperature tuning for entropy regularization
5. Higher UTD ratio (10-100x) for improved sample efficiency

Structure:
- models/latent_q_network.py: Double Q-network in latent space + target networks
- buffers/macro_replay_buffer.py: SMDP replay buffer for off-policy learning
- agents/dsrl_sac_agent.py: SAC agent with latent steering
- train/train_stage1_offline.py: Offline AW-MLE warm start (same as on-policy)
- train/train_stage2_online.py: Online SAC training with UTD
"""

from .agents.dsrl_sac_agent import DSRLSACAgent
from .models.latent_q_network import LatentQNetwork
from .buffers.macro_replay_buffer import MacroReplayBuffer

__all__ = [
    "DSRLSACAgent",
    "LatentQNetwork",
    "MacroReplayBuffer",
]
