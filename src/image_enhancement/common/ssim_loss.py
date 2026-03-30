"""
Pooled SSIM loss between estimate x and observation y (e.g. û vs v).

Uses the simplified two-factor SSIM and averages over valid sliding windows
(Wang et al. style local statistics, where SSIM using local-window statistics: 
    means/variances/covariance/constants then pooling over all windows).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _window_2d(window_size: int, channel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Normalized box filter (1/W^2) for depthwise conv."""
    w = torch.ones(channel, 1, window_size, window_size, dtype=dtype, device=device)
    return w / (window_size * window_size)


def pooled_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    window_size: int = 11,
    K1: float = 0.01,
    K2: float = 0.03,
    L: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Mean SSIM over valid sliding windows; patch count is (M-W+1)(N-W+1).

    x, y: (B, C, H, W), same shape; values in [0, L] with L=1 for normalized floats.
    Returns scalar mean SSIM in [0, 1].
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    _, c, h, w = x.shape
    if h < window_size or w < window_size:
        raise ValueError(f"Image ({h}x{w}) smaller than window_size={window_size}")

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    win = _window_2d(window_size, c, x.dtype, x.device)

    mu_x = F.conv2d(x, win, padding=0, groups=c)
    mu_y = F.conv2d(y, win, padding=0, groups=c)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, win, padding=0, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, win, padding=0, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(x * y, win, padding=0, groups=c) - mu_xy

    sigma_x_sq = sigma_x_sq.clamp_min(0.0)
    sigma_y_sq = sigma_y_sq.clamp_min(0.0)

    # Simplified two-factor SSIM
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2) + eps
    ssim_map = num / den
    return ssim_map.mean()


class PooledSSIMLoss(nn.Module):
    """Loss = 1 - mean local SSIM between x (estimate) and y (observation)."""

    def __init__(
        self,
        window_size: int = 11,
        K1: float = 0.01,
        K2: float = 0.03,
        L: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.K1 = K1
        self.K2 = K2
        self.L = L

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ssim = pooled_ssim(
            x,
            y,
            window_size=self.window_size,
            K1=self.K1,
            K2=self.K2,
            L=self.L,
        )
        return 1.0 - ssim
