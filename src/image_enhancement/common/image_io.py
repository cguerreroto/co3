"""
Generic grayscale image IO for preprocessing and evaluation.

Supports TIFF (.tif/.tiff) as well as raster formats like PNG and JPEG.
All outputs are saved as 8-bit grayscale (0..255) for simplicity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

try:
    import imageio.v3 as iio
except Exception as e:  # pragma: no cover
    raise ImportError(
        "imageio is required for PNG/JPEG support. Install dependencies for this project."
    ) from e


_TIFF_EXTS = {".tif", ".tiff"}
_RASTER_EXTS = {".png", ".jpg", ".jpeg"}
_SUPPORTED_EXTS = _TIFF_EXTS | _RASTER_EXTS


def _normalize_ext(path: str | Path) -> str:
    """Return lowercase file extension for ``path`` (including leading dot)."""
    return Path(path).suffix.lower()


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA to grayscale if needed; otherwise return as-is."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # imageio may return HxWx3 or HxWx4 for RGB(A)
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float64)
            gray = np.dot(rgb, [0.2989, 0.5870, 0.1140])
            return gray
        # Fallback: take first channel
        return arr[..., 0]
    raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}")


def read_grayscale_float_L(path: str | Path) -> tuple[np.ndarray, float, dict[str, str]]:
    """
    Read image and return:
      - array in [0, L] as float64
      - L as 255.0 or 1.0 (auto-detected)
      - a small metadata dict
    """
    path = Path(path)
    ext = _normalize_ext(path)
    if ext not in _SUPPORTED_EXTS:
        raise ValueError(f"Unsupported image extension: {ext}. Supported: {sorted(_SUPPORTED_EXTS)}")

    if ext in _TIFF_EXTS:
        arr = tifffile.imread(str(path))
    else:
        arr = iio.imread(str(path))

    meta = {"original_dtype": str(np.asarray(arr).dtype)}
    arr_g = _to_grayscale(np.asarray(arr))
    arr_f = np.asarray(arr_g, dtype=np.float64)

    amax = float(arr_f.max()) if arr_f.size else 0.0
    if amax > 1.5:
        L = 255.0
        arr_f = arr_f.clip(0.0, L)
    else:
        L = 1.0
        arr_f = arr_f.clip(0.0, 1.0)
    return arr_f, L, meta


def read_grayscale_01(path: str | Path) -> np.ndarray:
    """Read image from ``path`` and return grayscale float32 array in [0, 1]."""
    arr, L, _ = read_grayscale_float_L(path)
    if L == 1.0:
        return np.asarray(arr, dtype=np.float32)
    return (arr / L).astype(np.float32)


def save_uint8_grayscale(path: str | Path, image_u8: np.ndarray) -> Path:
    """Save 8-bit grayscale array ``image_u8`` to ``path`` and return the written path."""
    path = Path(path)
    ext = _normalize_ext(path)
    if ext not in _SUPPORTED_EXTS:
        raise ValueError(f"Unsupported output extension: {ext}. Supported: {sorted(_SUPPORTED_EXTS)}")
    out = np.asarray(image_u8)
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    if ext in _TIFF_EXTS:
        tifffile.imwrite(str(path), out, photometric="minisblack")
    else:
        iio.imwrite(str(path), out)
    return path


def save_grayscale_from_float_L(
    path: str | Path,
    arr: np.ndarray,
    L: float,
) -> Path:
    """Convert float array in [0, L] to uint8 grayscale (0..255) and save to ``path``; return written path."""
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    u8 = np.round(np.asarray(arr, dtype=np.float64) / float(L) * 255.0).clip(0, 255).astype(np.uint8)
    return save_uint8_grayscale(path, u8)

