"""Resize images for the GA/PSO branch (e.g. max side 128, proportional)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from image_enhancement.common.image_io import read_grayscale_float_L, save_grayscale_from_float_L

def load_grayscale_float(path: str | Path) -> np.ndarray:
    """Load image from ``path`` as grayscale float array."""
    arr, _L, _meta = read_grayscale_float_L(path)
    return np.asarray(arr, dtype=np.float32)


def resize_for_small_branch(
    image: np.ndarray,
    max_side: int = 128,
    *,
    mode: str = "bilinear",
) -> np.ndarray:
    """
    Proportionally scale so max(H, W) == max_side (unless already smaller).

    ``image`` is 2D float; returns 2D float same range approximately.
    """
    if image.ndim != 2:
        raise ValueError("Expected 2D array")
    h, w = image.shape
    m = max(h, w)
    if m <= max_side:
        return image.copy()
    scale = max_side / float(m)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    t = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    align = False if mode == "nearest" else True
    out = F.interpolate(
        t,
        size=(new_h, new_w),
        mode=mode,
        align_corners=align if mode in ("bilinear", "bicubic") else None,
    )
    return out.squeeze(0).squeeze(0).numpy().astype(np.float32)


def resize_file(
    input_path: str | Path,
    output_path: str | Path,
    max_side: int = 128,
    *,
    mode: str = "bilinear",
) -> Path:
    """Resize image from ``input_path`` and save to ``output_path``; return output path."""
    img, L, _ = read_grayscale_float_L(input_path)
    out = resize_for_small_branch(img, max_side=max_side, mode=mode)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_grayscale_from_float_L(output_path, out, L)
    return output_path


def build_parser(sub: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Build CLI parser for ``preprocess resize`` and return it."""
    p = (
        sub.add_parser("resize", help="Proportional resize (max side) for GA/PSO inputs")
        if sub is not None
        else argparse.ArgumentParser("resize")
    )
    p.add_argument("--input", "-i", required=True, type=Path)
    p.add_argument("--output", "-o", required=True, type=Path)
    p.add_argument("--max-side", type=int, default=128)
    p.add_argument(
        "--mode",
        choices=("bilinear", "bicubic", "nearest"),
        default="bilinear",
    )
    return p


def main_ns(args: argparse.Namespace) -> None:
    """Run resize operation from parsed args and print written file path."""
    p = resize_file(args.input, args.output, max_side=args.max_side, mode=args.mode)
    print(f"Wrote {p}")


def main() -> None:
    """Standalone entrypoint for resize module."""
    parser = build_parser()
    main_ns(parser.parse_args())


if __name__ == "__main__":
    main()
