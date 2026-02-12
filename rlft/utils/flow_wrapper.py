"""
ShortCut Flow Policy Wrapper for DSRL.

Wraps a pretrained ShortCut Flow velocity network to provide a unified
interface for decoding noise to actions via Euler ODE integration.

Corresponds to the official DSRL ``DPPOBasePolicyWrapper``.

Reference: https://github.com/ajwagen/dsrl
"""

import torch
import numpy as np
from typing import Optional, Tuple, Any


class ShortCutFlowWrapper:
    """ShortCut Flow policy wrapper.

    Wraps a pretrained ``ShortCutVelocityUNet1D`` and performs Euler ODE
    integration from *noise* (t=0) to *clean action* (t=1).

    The wrapper is **frozen** — no gradients flow through it.

    Args:
        velocity_net: Pretrained ``ShortCutVelocityUNet1D``.
        obs_horizon: Observation history length.
        pred_horizon: Prediction action-sequence length.
        action_dim: Per-step action dimension.
        num_inference_steps: Number of Euler steps for ODE integration.
        device: Torch device.
    """

    def __init__(
        self,
        velocity_net: torch.nn.Module,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_dim: int = 7,
        num_inference_steps: int = 8,
        device: str = "cuda",
    ):
        self.velocity_net = velocity_net.to(device)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        self.device = device

        self.velocity_net.eval()
        for p in self.velocity_net.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        obs: torch.Tensor,
        initial_noise: torch.Tensor,
        return_numpy: bool = True,
        act_steps: Optional[int] = None,
    ) -> Any:
        """Generate actions from observation + noise via Euler ODE.

        Noise → pad to *pred_horizon* → Euler integration → clamp → (optionally
        slice *act_steps* from ``obs_horizon-1``).

        Args:
            obs: (B, obs_dim) or (B, obs_horizon, obs_dim).
            initial_noise: (B, T, action_dim) or (B, T * action_dim).
            return_numpy: Whether to return ``np.ndarray``.
            act_steps: If given, return only ``act_steps`` actions starting from
                index ``obs_horizon - 1``.

        Returns:
            actions: (B, T, action_dim) — T depends on *act_steps*.
        """
        obs = self._to_tensor(obs)
        initial_noise = self._to_tensor(initial_noise)
        B = initial_noise.shape[0]

        # ---- reshape noise ----
        if initial_noise.dim() == 2:
            T = initial_noise.shape[1] // self.action_dim
            initial_noise = initial_noise.view(B, T, self.action_dim)
        noise_T = initial_noise.shape[1]

        # ---- pad / truncate to pred_horizon ----
        if noise_T < self.pred_horizon:
            pad = torch.zeros(
                B, self.pred_horizon - noise_T, self.action_dim, device=self.device
            )
            x = torch.cat([initial_noise, pad], dim=1)
        elif noise_T > self.pred_horizon:
            x = initial_noise[:, : self.pred_horizon, :]
        else:
            x = initial_noise

        # ---- flatten obs for global conditioning ----
        if obs.dim() == 3:
            obs = obs.reshape(B, -1)

        # ---- Euler integration: t=0 (noise) → t=1 (action) ----
        dt = 1.0 / self.num_inference_steps
        step_size = torch.full((B,), dt, device=self.device)

        for i in range(self.num_inference_steps):
            t = torch.full((B,), i * dt, device=self.device)
            v = self.velocity_net(x, t, step_size, obs)
            x = x + v * dt

        actions = torch.clamp(x, -1.0, 1.0)

        # ---- optional slice ----
        if act_steps is not None:
            start = self.obs_horizon - 1
            actions = actions[:, start : start + act_steps, :]

        return actions.cpu().numpy() if return_numpy else actions

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _to_tensor(self, x: Any) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return x.to(self.device)


# =====================================================================
# Checkpoint loading
# =====================================================================

def load_shortcut_flow_policy(
    checkpoint_path: str,
    *,
    visual_encoder_class=None,
    obs_horizon: int = 2,
    pred_horizon: int = 16,
    action_dim: int = 7,
    visual_feature_dim: int = 256,
    diffusion_step_embed_dim: int = 64,
    unet_dims: Tuple[int, ...] = (64, 128, 256),
    n_groups: int = 8,
    state_dim: Optional[int] = None,
    include_rgb: bool = True,
    use_ema: bool = True,
    device: str = "cuda",
) -> Tuple["ShortCutFlowWrapper", Optional[Any], int]:
    """Load a pretrained ShortCut Flow policy from checkpoint.

    If *state_dim* is ``None``, it is inferred from the checkpoint weights.

    Returns:
        (flow_wrapper, visual_encoder, state_dim)
    """
    from rlft.networks import ShortCutVelocityUNet1D, PlainConv

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Prefer EMA weights
    if use_ema and "ema_agent" in checkpoint:
        agent_state = checkpoint["ema_agent"]
    else:
        agent_state = checkpoint.get("agent", checkpoint)

    # ----- infer state_dim from cond_encoder -----
    if state_dim is None:
        for key, value in agent_state.items():
            if "velocity_net" in key and "cond_encoder.1.weight" in key:
                cond_input = value.shape[1]
                global_cond = cond_input - diffusion_step_embed_dim
                visual_dim = visual_feature_dim if include_rgb else 0
                state_dim = (global_cond // obs_horizon) - visual_dim
                break
        if state_dim is None:
            raise ValueError("Could not infer state_dim from checkpoint")

    visual_dim = visual_feature_dim if include_rgb else 0
    global_cond_dim = obs_horizon * (visual_dim + state_dim)

    # ----- create velocity_net -----
    velocity_net = ShortCutVelocityUNet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=unet_dims,
        n_groups=n_groups,
    ).to(device)

    vnet_state = {
        k.replace("velocity_net.", ""): v
        for k, v in agent_state.items()
        if k.startswith("velocity_net.")
    }
    if not vnet_state:
        raise ValueError(f"No velocity_net weights found in {checkpoint_path}")
    velocity_net.load_state_dict(vnet_state)
    velocity_net.eval()

    # ----- create visual_encoder (optional) -----
    visual_encoder = None
    if include_rgb and visual_encoder_class is not None:
        visual_encoder = visual_encoder_class(
            in_channels=3, out_dim=visual_feature_dim, pool_feature_map=True,
        ).to(device)
        if "visual_encoder" in checkpoint:
            visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        visual_encoder.eval()

    # ----- wrap -----
    wrapper = ShortCutFlowWrapper(
        velocity_net=velocity_net,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        num_inference_steps=8,
        device=device,
    )

    return wrapper, visual_encoder, state_dim
