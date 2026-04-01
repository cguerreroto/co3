"""Batch AWGN: mirror a directory tree of clean images to noisy images + JSON sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from image_enhancement.preprocessing.noisify import save_noisy_pair

_IMAGE_EXTS = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})


def iter_clean_images(clean_root: Path) -> list[Path]:
    """Return sorted list of image files under ``clean_root`` (recursive)."""
    clean_root = Path(clean_root)
    out: list[Path] = []
    for p in sorted(clean_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            out.append(p)
    return out


def _noisy_filename(clean_name: str) -> str:
    """Map input filename to a noisy counterpart name."""
    p = Path(clean_name)
    stem = p.stem
    suffix = p.suffix
    if stem.startswith("slice_"):
        # Common case from nifti export: slice_257.tif -> slice_noisy_257.tif
        stem = "slice_noisy_" + stem[len("slice_") :]
    elif not stem.startswith("slice_noisy_"):
        stem = f"{stem}_noisy"
    return f"{stem}{suffix}"


def noisify_directory(
    clean_root: Path,
    noisy_root: Path,
    sigma: float,
    *,
    base_seed: int | None = 42,
    clip: bool = True,
) -> list[dict[str, Any]]:
    """
    For each clean image under ``clean_root``, write noisy image under ``noisy_root`` with same relative path.

    Each output gets a deterministic per-file ``seed`` derived from ``base_seed`` and the relative path.
    """
    clean_root = Path(clean_root).resolve()
    noisy_root = Path(noisy_root).resolve()
    records: list[dict[str, Any]] = []
    for clean_path in iter_clean_images(clean_root):
        rel = clean_path.relative_to(clean_root)
        noisy_rel = rel.with_name(_noisy_filename(rel.name))
        noisy_path = noisy_root / noisy_rel
        meta_path = Path(str(noisy_path) + ".json")
        h = hashlib.md5(str(rel).encode()).hexdigest()
        file_seed = (int(h[:8], 16) + (base_seed or 0)) % (2**31)
        rec = save_noisy_pair(
            clean_path,
            noisy_path,
            meta_path,
            sigma,
            seed=file_seed,
            clip=clip,
        )
        records.append(rec)
    if not records:
        raise FileNotFoundError(f"No images found under {clean_root}")
    return records


def build_parser(sub: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Build CLI parser for ``preprocess noisify-dir``."""
    p = (
        sub.add_parser(
            "noisify-dir",
            help="Add AWGN to every image under a clean directory tree; mirror to noisy root",
        )
        if sub is not None
        else argparse.ArgumentParser("noisify-dir")
    )
    p.add_argument("--clean-dir", "-c", required=True, type=Path, help="Root of clean images (u)")
    p.add_argument("--noisy-dir", "-o", required=True, type=Path, help="Root for noisy outputs (v)")
    p.add_argument("--sigma", type=float, required=True, help="Noise std (AWGN per pixel)")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; per-file seed is derived for reproducibility",
    )
    p.add_argument(
        "--no-clip",
        action="store_true",
        help="Do not clip v to [0, L] after noise",
    )
    return p


def main_ns(args: argparse.Namespace) -> None:
    """Run batch noisification and print summary JSON."""
    records = noisify_directory(
        args.clean_dir,
        args.noisy_dir,
        args.sigma,
        base_seed=args.seed,
        clip=not args.no_clip,
    )
    print(json.dumps({"n_files": len(records), "first": records[0] if records else None}, indent=2))


def main() -> None:
    """Standalone entrypoint."""
    parser = build_parser()
    main_ns(parser.parse_args())


if __name__ == "__main__":
    main()
