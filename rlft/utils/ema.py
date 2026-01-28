"""
Exponential Moving Average for Model Parameters.
"""

import copy
import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of model parameters updated with:
        ema_param = decay * ema_param + (1 - decay) * model_param
    
    Args:
        model: Model to track
        decay: EMA decay rate (default: 0.999)
        device: Device to store EMA parameters
        update_after_step: Start EMA updates after this many steps
        inv_gamma: Inverse multiplicative factor for decay (default: 1.0)
        power: Exponential factor for decay (default: 0.75)
    
    Note:
        decay is computed as: min(decay, (1 + step) / (inv_gamma + step) ** power)
        This allows for a warmup period where decay starts low and increases.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: str = "cuda",
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
    ):
        self.decay = decay
        self.device = device
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.step = 0
        
        # Create shadow copy of model parameters
        self.shadow = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone().to(device)
    
    def get_decay(self) -> float:
        """Get current decay rate with warmup."""
        step = max(0, self.step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** (-self.power)
        return min(self.decay, max(0.0, value))
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters from model."""
        self.step += 1
        
        if self.step <= self.update_after_step:
            # Before warmup: just copy
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].copy_(param.data)
            return
        
        decay = self.get_decay()
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay)
                self.shadow[name].add_(param.data, alpha=1 - decay)
    
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """Copy EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    @torch.no_grad()
    def store(self, model: nn.Module):
        """Store current model parameters for later restore."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
    
    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Restore model parameters from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        """Get state dict for checkpointing."""
        return {
            "shadow": self.shadow,
            "step": self.step,
            "decay": self.decay,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.shadow = state_dict["shadow"]
        self.step = state_dict["step"]
        if "decay" in state_dict:
            self.decay = state_dict["decay"]
