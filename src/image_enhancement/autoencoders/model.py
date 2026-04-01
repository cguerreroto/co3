"""Convolutional autoencoder: maps noisy observation v to estimate û (same spatial size)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    """Grayscale (1 channel). Output in [0, 1] via sigmoid (box constraint with L=1)."""

    def __init__(
        self,
        base_channels: int = 32,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        c1, c2 = base_channels, hidden_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input noisy tensor ``x`` to denoised estimate ``u_hat`` in [0, 1]."""
        return self.net(x)
