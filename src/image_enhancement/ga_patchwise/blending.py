"""Overlap blending weights and fusion for patch-wise denoising (see LaTeX stitch-blend notation)."""

from __future__ import annotations

from typing import Literal

import numpy as np

WindowMode = Literal["hann", "triangular", "tukey", "flat"]


def tile_starts_1d(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    if stride <= 0:
        raise ValueError("stride must be positive")
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def tukey_1d(n: int, alpha: float) -> np.ndarray:
    """Tukey (tapered cosine) window, alpha in (0,1] fraction of taper at each end."""
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    if alpha <= 0:
        return np.ones(n, dtype=np.float64)
    if alpha >= 1.0:
        return np.hanning(n).astype(np.float64)
    w = np.ones(n, dtype=np.float64)
    x = np.linspace(0.0, 1.0, n, endpoint=True)
    a = alpha / 2.0
    # Rise [0, a)
    m1 = x < a
    w[m1] = 0.5 * (1.0 + np.cos(np.pi * (x[m1] / a - 1.0)))
    # Fall (1-a, 1]
    m2 = x > (1.0 - a)
    w[m2] = 0.5 * (1.0 + np.cos(np.pi * ((x[m2] - 1.0 + a) / a)))
    return w


def triangular_1d(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    return np.bartlett(n).astype(np.float64)


def weight_patch_2d(
    patch: int,
    mode: WindowMode,
    *,
    tukey_alpha: float = 0.5,
) -> np.ndarray:
    """2D separable weights for one BxB tile (outer product of 1D windows)."""
    if mode == "flat":
        w1 = np.ones(patch, dtype=np.float64)
    elif mode == "hann":
        w1 = np.hanning(patch).astype(np.float64)
    elif mode == "triangular":
        w1 = triangular_1d(patch)
    elif mode == "tukey":
        w1 = tukey_1d(patch, tukey_alpha)
    else:
        raise ValueError(f"Unknown window mode: {mode}")
    w2d = np.outer(w1, w1)
    # Normalize so max is 1 (optional; keeps scale stable)
    m = float(w2d.max()) if w2d.size else 1.0
    if m > 0:
        w2d = w2d / m
    return w2d.astype(np.float64)


def padded_canvas_shape(
    height: int,
    width: int,
    patch: int,
    stride: int,
) -> tuple[int, int, list[int], list[int]]:
    """Return (Hc, Wc, ys, xs) tile top-left index lists on padded canvas."""
    ys = tile_starts_1d(height, patch, stride)
    xs = tile_starts_1d(width, patch, stride)
    hc = ys[-1] + patch
    wc = xs[-1] + patch
    return hc, wc, ys, xs


def pad_image_edge(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape
    ph = max(0, target_h - h)
    pw = max(0, target_w - w)
    if ph == 0 and pw == 0:
        return img
    pad = ((0, ph), (0, pw))
    return np.pad(img, pad, mode="edge")


def blend_overlapping_patches(
    height: int,
    width: int,
    patch: int,
    stride: int,
    patches: list[tuple[int, int, np.ndarray]],
    mode: WindowMode,
    *,
    tukey_alpha: float = 0.5,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuse tile results into HcxWc, then crop to height x width.

    patches: list of (y, x, u_hat_patch) with u_hat_patch shape (patch, patch) aligned
    to padded canvas coordinates.
    Returns (u_hat_final_crop, numerator_canvas, denominator_canvas) for diagnostics.
    """
    hc, wc, ys_ref, xs_ref = padded_canvas_shape(height, width, patch, stride)
    w2d = weight_patch_2d(patch, mode, tukey_alpha=tukey_alpha)
    num = np.zeros((hc, wc), dtype=np.float64)
    den = np.zeros((hc, wc), dtype=np.float64)
    for (y, x, phat) in patches:
        if phat.shape != (patch, patch):
            raise ValueError(f"Expected patch ({patch},{patch}), got {phat.shape}")
        num[y : y + patch, x : x + patch] += w2d * phat.astype(np.float64)
        den[y : y + patch, x : x + patch] += w2d
    fused = num / np.maximum(den, eps)
    out = fused[:height, :width]
    return out.astype(np.float32), num.astype(np.float64), den.astype(np.float64)
