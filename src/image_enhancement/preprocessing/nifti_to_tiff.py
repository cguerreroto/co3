"""Export 2D slices from NIfTI volumes to image files.

Single-slice output uses the output filename extension to decide format.
Supports .tif/.tiff/.png/.jpg/.jpeg.
"""

from __future__ import annotations

import argparse
import glob as glob_mod
from pathlib import Path

import nibabel as nib
import numpy as np

from image_enhancement.common.image_io import save_uint8_grayscale


def load_volume(path: str | Path) -> np.ndarray:
    """Load NIfTI; return a numpy array with singleton dimensions squeezed."""
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float64)
    return np.squeeze(data)


def extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Extract one 2D slice from ``volume`` using selected ``axis`` and ``index``."""
    if volume.ndim < 3:
        raise ValueError(f"Expected at least 3D volume after squeeze, got shape {volume.shape}")
    if axis < 0 or axis >= volume.ndim:
        raise ValueError(f"axis must be in [0, {volume.ndim}), got {axis}")
    idx: tuple[int | slice, ...] = tuple(
        index if a == axis else slice(None) for a in range(volume.ndim)
    )
    plane = np.asarray(volume[idx])
    if plane.ndim != 2:
        raise ValueError(f"Slice is not 2D (shape {plane.shape}); try a different axis/index.")
    return plane


def normalize_to_uint8(plane: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """Normalize 2D array ``plane`` to uint8 [0, 255] using ``mode``."""
    p = plane.astype(np.float64, copy=False)
    if mode == "minmax":
        pmin, pmax = float(np.min(p)), float(np.max(p))
        if pmax <= pmin:
            out = np.zeros_like(p, dtype=np.uint8)
        else:
            out = ((p - pmin) / (pmax - pmin) * 255.0).clip(0, 255).astype(np.uint8)
        return out
    if mode == "none":
        return p.clip(0, 255).astype(np.uint8)
    raise ValueError(f"Unknown normalize mode: {mode}")


def export_slice(
    input_path: str | Path,
    output_path: str | Path,
    axis: int,
    index: int,
    *,
    normalize: str = "minmax",
) -> Path:
    """Extract one slice from NIfTI and save it as an image (format from extension); return written output path."""
    vol = load_volume(input_path)
    plane = extract_slice(vol, axis, index)
    out_u8 = normalize_to_uint8(plane, mode=normalize)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_uint8_grayscale(output_path, out_u8)
    return output_path


def export_all_slices(
    input_path: str | Path,
    output_dir: str | Path,
    axis: int,
    *,
    prefix: str = "slice",
    normalize: str = "minmax",
    output_ext: str = "tif",
) -> list[Path]:
    """Export all slices along ``axis`` (as ``{prefix}_{i:03}.{output_ext}``) and return list of written file paths."""
    vol = load_volume(input_path)
    if axis < 0 or axis >= vol.ndim:
        raise ValueError(f"axis must be in [0, {vol.ndim}), got {axis}")
    n = vol.shape[axis]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n):
        plane = extract_slice(vol, axis, i)
        out_u8 = normalize_to_uint8(plane, mode=normalize)
        p = output_dir / f"{prefix}_{i:03}.{output_ext}"
        save_uint8_grayscale(p, out_u8)
        paths.append(p)
    return paths


def export_slice_range(
    input_path: str | Path,
    output_dir: str | Path,
    axis: int,
    index_start: int,
    index_end: int,
    *,
    prefix: str = "slice",
    normalize: str = "minmax",
    output_ext: str = "tif",
) -> list[Path]:
    """
    Export slices with indices in ``[index_start, index_end]`` inclusive along ``axis``.

    Filenames are ``{prefix}_{i:03}.{output_ext}`` as in ``export_all_slices``.
    """
    if index_end < index_start:
        raise ValueError(f"index_end ({index_end}) must be >= index_start ({index_start})")
    vol = load_volume(input_path)
    if axis < 0 or axis >= vol.ndim:
        raise ValueError(f"axis must be in [0, {vol.ndim}), got {axis}")
    n = vol.shape[axis]
    if index_start < 0 or index_end >= n:
        raise ValueError(f"Range [{index_start}, {index_end}] out of bounds for axis length {n}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(index_start, index_end + 1):
        plane = extract_slice(vol, axis, i)
        out_u8 = normalize_to_uint8(plane, mode=normalize)
        p = output_dir / f"{prefix}_{i:03}.{output_ext}"
        save_uint8_grayscale(p, out_u8)
        paths.append(p)
    return paths


def export_glob_volumes(
    pattern: str,
    output_root: str | Path,
    axis: int,
    index_start: int,
    index_end: int,
    *,
    prefix: str = "slice",
    normalize: str = "minmax",
    output_ext: str = "tif",
) -> list[tuple[Path, list[Path]]]:
    """
    For each NIfTI path matching ``pattern``, export slice range into ``output_root/<stem>/``.

    Returns list of ``(volume_path, written_slice_paths)``.
    """
    paths = sorted(glob_mod.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern!r}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Path, list[Path]]] = []
    for p in paths:
        p = Path(p)
        name = p.name
        if name.endswith(".nii.gz"):
            stem = name[:-7]
        elif name.endswith(".nii"):
            stem = name[:-4]
        else:
            stem = p.stem
        out_dir = output_root / stem
        written = export_slice_range(
            p,
            out_dir,
            axis,
            index_start,
            index_end,
            prefix=prefix,
            normalize=normalize,
            output_ext=output_ext,
        )
        results.append((p, written))
    return results


def build_parser(sub: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Build CLI parser for ``preprocess nifti-to-tiff`` and return it."""
    p = (
        sub.add_parser(
            "nifti-to-tiff",
            help="Export coronal/axial/sagittal slice(s) to an image file",
        )
        if sub is not None
        else argparse.ArgumentParser("nifti-to-tiff")
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Input .nii or .nii.gz (omit if using --input-glob)",
    )
    p.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help="Glob of NIfTI paths (e.g. assets/*.nii.gz); requires --output-root and index range",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for --input-glob: each volume -> output_root/<stem>/",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output image path (extension decides format: .tif/.tiff/.png/.jpg/.jpeg)",
    )
    g.add_argument("--output-dir", type=Path, help="Directory for all slices along axis")
    p.add_argument(
        "--axis",
        type=int,
        default=1,
        help="Volume axis along which to take slices (0,1,2 for 3D). Default 1 (often coronal in RAS).",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Slice index when using --output (ignored for --output-dir).",
    )
    p.add_argument("--prefix", type=str, default="slice", help="Filename prefix for batch export")
    p.add_argument(
        "--output-ext",
        type=str,
        default="tif",
        help="Output extension used when exporting to --output-dir (e.g. tif, png, jpg).",
    )
    p.add_argument(
        "--normalize",
        choices=("minmax", "none"),
        default="minmax",
        help="minmax: scale slice to 0-255; none: clip to 0-255",
    )
    p.add_argument(
        "--index-start",
        type=int,
        default=None,
        help="With --output-dir: first slice index (inclusive); omit for all slices",
    )
    p.add_argument(
        "--index-end",
        type=int,
        default=None,
        help="With --output-dir: last slice index (inclusive); required if --index-start is set",
    )
    return p


def main_ns(args: argparse.Namespace) -> None:
    """Run export operation from parsed args and print output location(s)."""
    if args.input_glob is not None:
        if args.output_root is None:
            raise SystemExit("--input-glob requires --output-root")
        if args.index_start is None or args.index_end is None:
            raise SystemExit("--input-glob requires --index-start and --index-end")
        results = export_glob_volumes(
            args.input_glob,
            args.output_root,
            args.axis,
            args.index_start,
            args.index_end,
            prefix=args.prefix,
            normalize=args.normalize,
            output_ext=args.output_ext,
        )
        for vol, written in results:
            print(f"{vol}: {len(written)} slices -> {written[0].parent}")
        return

    if args.input is None:
        raise SystemExit("Provide --input or --input-glob")

    if args.output_dir is not None:
        if args.index_start is not None:
            if args.index_end is None:
                raise SystemExit("--index-end is required when --index-start is set")
            export_slice_range(
                args.input,
                args.output_dir,
                args.axis,
                args.index_start,
                args.index_end,
                prefix=args.prefix,
                normalize=args.normalize,
                output_ext=args.output_ext,
            )
            print(
                f"Wrote slices {args.index_start}..{args.index_end} to {args.output_dir}/{args.prefix}_*.{args.output_ext}"
            )
        else:
            export_all_slices(
                args.input,
                args.output_dir,
                args.axis,
                prefix=args.prefix,
                normalize=args.normalize,
                output_ext=args.output_ext,
            )
            print(f"Wrote {args.output_dir}/{args.prefix}_*.{args.output_ext}")
    else:
        path = export_slice(
            args.input,
            args.output,
            args.axis,
            args.index,
            normalize=args.normalize,
        )
        print(f"Wrote {path}")


def main() -> None:
    """Standalone entrypoint for nifti export module."""
    parser = build_parser()
    main_ns(parser.parse_args())


if __name__ == "__main__":
    main()
