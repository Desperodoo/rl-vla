"""
Checkpoint Utilities.
"""

import os
import torch
from typing import Dict, Any, Optional


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ema_model: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """Save training checkpoint.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        ema_model: Optional EMA model
        step: Current training step
        epoch: Current training epoch
        metrics: Optional metrics to save
        config: Optional config to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    if config is not None:
        checkpoint["config"] = config
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ema_model: Optional[Any] = None,
    device: str = "cuda",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        ema_model: Optional EMA model to load state into
        device: Device to load to
        strict: Whether to enforce strict state dict loading
        
    Returns:
        Checkpoint dict with additional metadata (step, epoch, metrics, config)
    """
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if ema_model is not None and "ema_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
    
    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
    }
