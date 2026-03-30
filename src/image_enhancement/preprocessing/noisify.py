"""Additive white Gaussian noise (AWGN) for clean images.

Supports input/output images with extensions:
.tif, .tiff, .png, .jpg, .jpeg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from image_enhancement.common.image_io import (
    read_grayscale_float_L,
    save_grayscale_from_float_L,
)


def load_grayscale(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load image ``path`` as grayscale float array in [0, L]; return array and metadata."""
    arr, L, meta0 = read_grayscale_float_L(path)
    meta: dict[str, Any] = {
        "original_shape": None,
        "original_dtype": meta0.get("original_dtype", "unknown"),
        "L": L,
    }
    meta["original_shape"] = list(np.asarray(arr).shape)
    return arr, meta


def add_awgn(
    u: np.ndarray,
    sigma: float,
    *,
    seed: int | None = None,
    clip: bool = True,
    L: float = 255.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Equation: v = u + eta, where eta ~ N(0, sigma^2)
    Add AWGN to clean image ``u`` with std ``sigma``.
    Input: clean array ``u``.
    Output: tuple ``(v, eta)`` where ``v = u + eta`` (optionally clipped to [0, L]).
    """
    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, sigma, size=u.shape).astype(np.float64)
    v = u + eta
    if clip:
        v = np.clip(v, 0.0, L)
    return v, eta


def save_noisy_pair(
    clean_path: str | Path,
    noisy_path: str | Path,
    meta_path: str | Path | None,
    sigma: float,
    *,
    seed: int | None = None,
    clip: bool = True,
) -> dict[str, Any]:
    """Create noisy image from ``clean_path`` and save image + JSON metadata."""
    u, load_meta = load_grayscale(clean_path)
    L = float(load_meta.get("L", 255.0))
    v, eta = add_awgn(u, sigma, seed=seed, clip=clip, L=L)
    noisy_path = Path(noisy_path)
    noisy_path.parent.mkdir(parents=True, exist_ok=True)
    save_grayscale_from_float_L(noisy_path, v, L)

    n = int(np.prod(u.shape))
    record: dict[str, Any] = {
        "clean_path": str(Path(clean_path).resolve()),
        "noisy_path": str(noisy_path.resolve()),
        "sigma": sigma,
        "seed": seed,
        "shape": list(u.shape),
        "n_pixels": n,
        "L": L,
        "clip_after_noise": clip,
        "epsilon_hint": float(n * sigma**2),
        "load": load_meta,
    }
    if meta_path is not None:
        mp = Path(meta_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    return record


def build_parser(sub: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Build CLI parser for ``preprocess noisify`` and return it."""
    p = (
        sub.add_parser("noisify", help="Add AWGN to a clean image; write noisy image + JSON")
        if sub is not None
        else argparse.ArgumentParser("noisify")
    )
    p.add_argument("--input", "-i", required=True, type=Path, help="Clean image (u)")
    p.add_argument("--output", "-o", required=True, type=Path, help="Noisy image (v)")
    p.add_argument("--meta", "-m", type=Path, help="JSON sidecar (default: output with .json)")
    p.add_argument("--sigma", type=float, required=True, help="Noise std (AWGN per pixel)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument(
        "--no-clip",
        action="store_true",
        help="Do not clip v to [0, L] after noise",
    )
    return p


def main_ns(args: argparse.Namespace) -> None:
    """Run noisification from parsed args and print saved metadata JSON."""
    meta = args.meta
    if meta is None:
        meta = Path(str(args.output) + ".json")
    rec = save_noisy_pair(
        args.input,
        args.output,
        meta,
        args.sigma,
        seed=args.seed,
        clip=not args.no_clip,
    )
    print(json.dumps(rec, indent=2))


def main() -> None:
    """Standalone entrypoint for noisify module."""
    parser = build_parser()
    main_ns(parser.parse_args())


if __name__ == "__main__":
    main()
