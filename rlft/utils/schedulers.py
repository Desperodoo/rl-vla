"""
Learning Rate Schedulers.
"""

import math
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
):
    """Create a cosine learning rate schedule with linear warmup.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (default: 0.5 for standard cosine)
        min_lr_ratio: Minimum LR as ratio of initial LR (default: 0.0)
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_value
    
    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
):
    """Create a constant learning rate schedule with linear warmup.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)
