"""Box bounds and optional residual energy penalty for denoising."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def box_clamp(x: torch.Tensor, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
    """Enforce 0 <= u_hat_k <= L (default L=1 when inputs are normalized)."""
    return torch.clamp(x, low, high)


def residual_energy_penalty(
    v: torch.Tensor,
    u_hat: torch.Tensor,
    epsilon: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Soft penalty max(0, ||v - u_hat||_2^2 - epsilon)^2 averaged over batch.

    For AWGN with per-pixel variance σ², a typical scale for ε is n·σ² with n the number of pixels.
    """
    # Flatten spatial dims per batch element
    diff = v - u_hat
    r = diff.reshape(diff.shape[0], -1).pow(2).sum(dim=1)
    excess = F.relu(r - epsilon)
    pen = excess.pow(2)
    if reduction == "mean":
        return pen.mean()
    if reduction == "sum":
        return pen.sum()
    raise ValueError(reduction)
