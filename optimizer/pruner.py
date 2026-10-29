"""Permanent L1 unstructured magnitude pruner for PyTorch models."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def apply_l1_pruning(model: nn.Module, amount: float = 0.20) -> nn.Module:
    """Deep-copy model and permanently prune Conv2d and Linear weights by L1 magnitude."""
    if not (0.0 <= amount < 1.0):
        raise ValueError(f"Pruning amount must be in [0.0, 1.0), got {amount}.")

    pruned_model = copy.deepcopy(model)
    pruned_model.eval()

    if amount == 0.0:
        return pruned_model

    for module in pruned_model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=float(amount))
            prune.remove(module, "weight")

    return pruned_model


def compute_prunable_sparsity(model: nn.Module) -> float:
    """Compute exact zero-weight percentage specifically across Conv2d and Linear layers."""
    total_weights = 0
    zero_weights = 0

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
                w = module.weight.data
                numel = int(w.numel())
                nonzero = int(torch.count_nonzero(w).item())
                total_weights += numel
                zero_weights += numel - nonzero

    if total_weights == 0:
        return 0.0
    return round((zero_weights / total_weights) * 100.0, 2)
