"""
Checkpoint Utilities for CARM / ManiSkill Training.

Provides ``save_checkpoint`` and ``build_checkpoint`` to standardize the
checkpoint schema across offline IL, offline RL, and online RL pipelines.

Canonical checkpoint schema::

    {
        # Required – model weights
        "agent":            state_dict,
        "ema_agent":        state_dict | None,
        "visual_encoder":   state_dict | None,

        # Optional – CARM-specific components
        "state_encoder":    state_dict | None,
        "gripper_head":     state_dict | None,

        # Optional – action normalization
        "action_normalizer": {"mode": str, "stats": {...}} | None,

        # Optional – optimizer / scheduler state (for resume)
        "optimizer":        state_dict | None,
        "lr_scheduler":     state_dict | None,
        "ema":              state_dict | None,

        # Optional – progress tracking
        "iteration":        int | None,
        "total_steps":      int | None,

        # Optional – full training config
        "config":           dict | None,
    }
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def build_checkpoint(
    agent: nn.Module,
    visual_encoder: Optional[nn.Module] = None,
    *,
    ema_agent: Optional[nn.Module] = None,
    state_encoder: Optional[nn.Module] = None,
    gripper_head: Optional[nn.Module] = None,
    action_normalizer: Any = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Any = None,
    ema: Any = None,
    iteration: Optional[int] = None,
    total_steps: Optional[int] = None,
    config: Optional[dict] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> dict:
    """Build a checkpoint dictionary following the canonical schema.

    Only non-None components are included (except ``agent`` which is always
    present).

    Args:
        agent: The policy agent (required).
        visual_encoder: Visual feature encoder.
        ema_agent: Exponential moving average copy of agent.
        state_encoder: State encoder MLP (CARM).
        gripper_head: Discrete gripper classification head (CARM).
        action_normalizer: ``ActionNormalizer`` instance with ``.mode`` and
            ``.stats`` attributes.
        optimizer: Optimizer for training resume.
        lr_scheduler: LR scheduler for training resume.
        ema: ``EMAModel`` instance for training resume.
        iteration: Current training iteration number.
        total_steps: Current total environment steps (online RL).
        config: Full training configuration dict (e.g. ``vars(args)``).
        extra: Any additional key-value pairs to include.

    Returns:
        Checkpoint dictionary ready for ``torch.save``.
    """
    ckpt: Dict[str, Any] = {
        "agent": agent.state_dict(),
    }

    # Optional model components
    if ema_agent is not None:
        ckpt["ema_agent"] = ema_agent.state_dict()
    if visual_encoder is not None:
        ckpt["visual_encoder"] = visual_encoder.state_dict()
    if state_encoder is not None:
        ckpt["state_encoder"] = state_encoder.state_dict()
    if gripper_head is not None:
        ckpt["gripper_head"] = gripper_head.state_dict()

    # Action normalizer
    if action_normalizer is not None and getattr(action_normalizer, "stats", None) is not None:
        import numpy as np
        ckpt["action_normalizer"] = {
            "mode": action_normalizer.mode,
            "stats": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in action_normalizer.stats.items()
            },
        }

    # Optimizer / scheduler / EMA state
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        ckpt["lr_scheduler"] = lr_scheduler.state_dict()
    if ema is not None:
        ckpt["ema"] = ema.state_dict()

    # Progress
    if iteration is not None:
        ckpt["iteration"] = iteration
    if total_steps is not None:
        ckpt["total_steps"] = total_steps

    # Config
    if config is not None:
        ckpt["config"] = config

    # Extra
    if extra:
        ckpt.update(extra)

    return ckpt


def save_checkpoint(
    path: str,
    agent: nn.Module,
    visual_encoder: Optional[nn.Module] = None,
    *,
    args: Any = None,
    action_normalizer: Any = None,
    save_args_json: bool = True,
    save_normalizer_json: bool = True,
    **kwargs: Any,
) -> str:
    """Build a checkpoint, save it, and optionally write sidecar JSON files.

    This is a convenience wrapper around :func:`build_checkpoint` that also:

    - Saves ``args.json`` next to the checkpoint (if *args* is provided).
    - Saves ``action_normalizer.json`` next to the checkpoint (if normalizer
      has stats).

    Args:
        path: Full file path for the ``.pt`` checkpoint.
        agent: The policy agent.
        visual_encoder: Visual feature encoder.
        args: Training args (namespace or dataclass).  Converted to dict with
            ``vars()`` for the ``config`` field.
        action_normalizer: ``ActionNormalizer`` instance.
        save_args_json: Write ``args.json`` beside the checkpoint.
        save_normalizer_json: Write ``action_normalizer.json`` beside the
            checkpoint.
        **kwargs: Forwarded to :func:`build_checkpoint`.

    Returns:
        The *path* that was saved to.
    """
    checkpoint_dir = os.path.dirname(path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = None
    if args is not None:
        config = {
            k: v if not isinstance(v, (list, tuple)) or not any(isinstance(x, type) for x in v) else str(v)
            for k, v in vars(args).items()
        }

    ckpt = build_checkpoint(
        agent=agent,
        visual_encoder=visual_encoder,
        action_normalizer=action_normalizer,
        config=config,
        **kwargs,
    )
    torch.save(ckpt, path)

    # Sidecar: args.json
    if save_args_json and config is not None:
        args_path = os.path.join(checkpoint_dir, "args.json")
        with open(args_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    # Sidecar: action_normalizer.json
    if (
        save_normalizer_json
        and action_normalizer is not None
        and getattr(action_normalizer, "stats", None) is not None
    ):
        action_normalizer.save(os.path.join(checkpoint_dir, "action_normalizer.json"))

    return path
