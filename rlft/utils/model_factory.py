"""
Model Factory for CARM Robot Policy.

Shared model creation functions used by both training (train_carm.py) and
inference (carm_deploy/inference_ros.py) to ensure architecture consistency.

Usage (inference):
    from rlft.utils.model_factory import create_agent_for_inference
    agent = create_agent_for_inference(
        algorithm="consistency_flow",
        action_dim=13,
        global_cond_dim=1024,
        obs_horizon=2,
        pred_horizon=16,
        num_inference_steps=10,
    )

Usage (training):
    # Training still uses train_carm.py's create_agent() which accepts the full Args object
    # for algorithm-specific hyperparameters (consistency weights, shortcut params, etc.)
"""

from typing import Optional, List, Tuple

import torch.nn as nn

from rlft.networks import (
    VelocityUNet1D,
    ShortCutVelocityUNet1D,
    ConditionalUnet1D,
)
from rlft.algorithms import (
    DiffusionPolicyAgent,
    FlowMatchingAgent,
    ConsistencyFlowAgent,
    ReflectedFlowAgent,
    ShortCutFlowAgent,
)


# Supported algorithms for inference
SUPPORTED_ALGORITHMS = [
    "diffusion_policy",
    "flow_matching",
    "consistency_flow",
    "reflected_flow",
    "shortcut_flow",
]


def create_agent_for_inference(
    algorithm: str,
    action_dim: int,
    global_cond_dim: int,
    obs_horizon: int = 2,
    pred_horizon: int = 16,
    num_inference_steps: int = 10,
    # UNet architecture params (should match training)
    diffusion_step_embed_dim: int = 64,
    unet_dims: List[int] = None,
    n_groups: int = 8,
    # Optional overrides
    action_bounds: Optional[Tuple[float, float]] = None,
    device: str = "cuda",
    # ShortCut-specific inference params
    sc_inference_mode: str = "uniform",
    sc_num_inference_steps: int = 8,
) -> nn.Module:
    """Create agent for inference with minimal required parameters.
    
    This is a simplified version of train_carm.py's create_agent() that only
    requires parameters needed for inference (architecture + inference steps).
    Training-specific hyperparameters (loss weights, consistency params, etc.)
    are set to defaults since they don't affect inference behavior.
    
    Args:
        algorithm: Algorithm name (see SUPPORTED_ALGORITHMS)
        action_dim: Continuous action dimension (no gripper)
        global_cond_dim: Dimension of global conditioning (obs_horizon * feature_dim)
        obs_horizon: Observation horizon
        pred_horizon: Prediction horizon
        num_inference_steps: Number of flow/diffusion steps for inference
        diffusion_step_embed_dim: Timestep embedding dimension (must match training)
        unet_dims: UNet channel dimensions (must match training)
        n_groups: GroupNorm groups (must match training)
        action_bounds: Action bounds for clamping (optional)
        device: Device string
        sc_inference_mode: ShortCut Flow inference mode ('adaptive' or 'uniform')
        sc_num_inference_steps: ShortCut Flow inference steps
    
    Returns:
        Agent module (not yet moved to device, caller should do .to(device))
    """
    if unet_dims is None:
        unet_dims = [64, 128, 256]
    
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Supported: {SUPPORTED_ALGORITHMS}"
        )
    
    if algorithm == "diffusion_policy":
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
        )
        return DiffusionPolicyAgent(
            noise_pred_net=noise_pred_net,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            num_diffusion_iters=num_inference_steps,
            action_bounds=action_bounds,
            device=device,
        )
    
    elif algorithm == "flow_matching":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=tuple(unet_dims),
            n_groups=n_groups,
        )
        return FlowMatchingAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            num_flow_steps=num_inference_steps,
            action_bounds=action_bounds,
            device=device,
        )
    
    elif algorithm == "reflected_flow":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=tuple(unet_dims),
            n_groups=n_groups,
        )
        kwargs = dict(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            num_flow_steps=num_inference_steps,
            device=device,
        )
        if action_bounds is not None:
            kwargs["action_low"] = action_bounds[0]
            kwargs["action_high"] = action_bounds[1]
        return ReflectedFlowAgent(**kwargs)
    
    elif algorithm == "consistency_flow":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=tuple(unet_dims),
            n_groups=n_groups,
        )
        return ConsistencyFlowAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            num_flow_steps=num_inference_steps,
            action_bounds=action_bounds,
            device=device,
        )
    
    elif algorithm == "shortcut_flow":
        shortcut_velocity_net = ShortCutVelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=tuple(unet_dims),
            n_groups=n_groups,
        )
        return ShortCutFlowAgent(
            velocity_net=shortcut_velocity_net,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            inference_mode=sc_inference_mode,
            num_inference_steps=sc_num_inference_steps,
            action_bounds=action_bounds,
            device=device,
        )
