import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

from models import transformer


def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
    """PyTorch implementation of JAX-style learning rate scheduler."""
    factors = [f.strip() for f in factors.split('*')]
    
    def lr_lambda(current_step):
        """Compute learning rate multiplier at given step."""
        ret = 1.0
        for name in factors:
            if name == 'constant':
                ret *= base_learning_rate
            elif name == 'linear_warmup':
                ret *= min(1.0, current_step / warmup_steps) if warmup_steps > 0 else 1.0
            elif name == 'rsqrt_decay':
                divisor = math.sqrt(max(current_step, warmup_steps))
                ret /= divisor if divisor > 0 else 1.0
            elif name == 'rsqrt_normalized_decay':
                divisor = math.sqrt(max(current_step, warmup_steps))
                ret *= math.sqrt(warmup_steps) / divisor if divisor > 0 else 1.0
            elif name == 'decay_every':
                exponent = current_step // steps_per_decay
                ret *= (decay_factor ** exponent)
            elif name == 'cosine_decay':
                if current_step < warmup_steps:
                    continue
                progress = (current_step - warmup_steps) / steps_per_cycle
                ret *= max(0.0, 0.5 * (1.0 + math.cos(math.pi * (progress % 1.0))))
            else:
                raise ValueError(f'Unknown factor: {name}')
        return ret
    
    return lr_lambda


def compute_weighted_cross_entropy(logits, targets, num_classes, weights=None):
    loss = F.cross_entropy(logits.view(-1, num_classes), targets.view(-1), reduction='none')
    
    if weights is not None:
        weights = weights.view(-1)
        loss = loss * weights
        normalizing_factor = weights.sum()
    else:
        normalizing_factor = torch.tensor(loss.numel(), dtype=loss.dtype, device=loss.device)
    
    # return loss.sum(), normalizing_factor
    return loss.sum()/normalizing_factor

def compute_weighted_accuracy(logits, targets, weights=None):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float()
    
    if weights is not None:
        correct = correct * weights
        normalizing_factor = weights.sum()
    else:
        normalizing_factor = torch.tensor(correct.numel(), dtype=correct.dtype, device=correct.device)
    
    return correct.sum()/normalizing_factor