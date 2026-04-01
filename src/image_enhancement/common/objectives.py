"""Shared objective helpers for AE/GA denoising experiments."""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
from image_enhancement.common.ssim_loss import pooled_ssim


def mse_tensor(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute mean squared error between tensors ``x`` and ``y``."""
    return float(torch.mean((x - y) ** 2).detach().cpu())


def psnr_tensor(x: torch.Tensor, y: torch.Tensor, l: float = 1.0) -> float:
    """Compute PSNR in dB for tensors ``x`` and ``y`` using dynamic range ``l``."""
    mse = torch.mean((x - y) ** 2)
    if mse.item() <= 0:
        return float("inf")
    return float(10.0 * torch.log10((l * l) / mse))


def array_to_tensor_11(image: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    """Convert a 2D float array in [0,1] to a tensor shaped (1,1,H,W)."""
    t = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


def blind_ssim_objective(
    u_hat: np.ndarray,
    v: np.ndarray,
    *,
    window_size: int = 11,
    device: torch.device | None = None,
) -> tuple[float, dict[str, float]]:
    """Return blind objective value and diagnostic metrics for arrays in [0,1]."""
    u_hat_t = array_to_tensor_11(u_hat, device)
    v_t = array_to_tensor_11(v, device)
    ssim_hat_v = float(pooled_ssim(u_hat_t, v_t, window_size=window_size, L=1.0).detach().cpu())
    loss = 1.0 - ssim_hat_v
    return loss, {"ssim_hat_vs_noisy": ssim_hat_v}


def hybrid_ssim_mse_objective(
    u_hat: np.ndarray,
    u: np.ndarray,
    *,
    v: np.ndarray | None = None,
    alpha: float = 0.8,
    beta: float = 0.2,
    window_size: int = 11,
    device: torch.device | None = None,
) -> tuple[float, dict[str, float]]:
    """Return hybrid objective value and diagnostic metrics for arrays in [0,1]."""
    u_hat_t = array_to_tensor_11(u_hat, device)
    u_t = array_to_tensor_11(u, device)
    ssim_hat_u = float(pooled_ssim(u_hat_t, u_t, window_size=window_size, L=1.0).detach().cpu())
    mse_u = mse_tensor(u_hat_t, u_t)
    loss = alpha * (1.0 - ssim_hat_u) + beta * mse_u
    rec: dict[str, float] = {
        "ssim_hat_vs_clean": ssim_hat_u,
        "ssim_vs_clean": ssim_hat_u,
        "mse_vs_clean": mse_u,
        "psnr_vs_clean": psnr_tensor(u_hat_t, u_t),
    }
    if v is not None:
        v_t = array_to_tensor_11(v, device)
        rec["ssim_hat_vs_noisy"] = float(
            pooled_ssim(u_hat_t, v_t, window_size=window_size, L=1.0).detach().cpu()
        )
        rec["ssim_clean_vs_noisy"] = float(
            pooled_ssim(u_t, v_t, window_size=window_size, L=1.0).detach().cpu()
        )
    return loss, rec


def evaluate_objective(
    u_hat: np.ndarray,
    *,
    v: np.ndarray,
    u: np.ndarray | None = None,
    loss_mode: str = "blind_ssim",
    alpha: float = 0.8,
    beta: float = 0.2,
    window_size: int = 11,
    device: torch.device | None = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate blind or hybrid objective on arrays in [0,1]."""
    if loss_mode == "blind_ssim":
        loss, rec = blind_ssim_objective(u_hat, v, window_size=window_size, device=device)
        if u is not None:
            _, extra = hybrid_ssim_mse_objective(
                u_hat,
                u,
                v=v,
                alpha=alpha,
                beta=beta,
                window_size=window_size,
                device=device,
            )
            rec.update(extra)
        return loss, rec
    if loss_mode == "hybrid_ssim_mse":
        if u is None:
            raise ValueError("hybrid_ssim_mse requires clean reference u")
        return hybrid_ssim_mse_objective(
            u_hat,
            u,
            v=v,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
            device=device,
        )
    raise ValueError(f"Unknown loss_mode: {loss_mode}")
