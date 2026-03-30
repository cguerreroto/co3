"""CLI entry: preprocess (NIfTI, noisify, resize) and train-ae."""

from __future__ import annotations

import argparse

from image_enhancement.preprocessing import nifti_to_tiff, noisify, resize


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser and register preprocess subcommands."""
    parser = argparse.ArgumentParser(prog="image-enhancement")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("preprocess", help="NIfTI-to-image, AWGN, resize")
    prep_sub = prep.add_subparsers(dest="preprocess_cmd", required=True)
    nifti_to_tiff.build_parser(prep_sub)
    noisify.build_parser(prep_sub)
    resize.build_parser(prep_sub)
    return parser


def main() -> None:
    """Parse CLI args and dispatch to the selected preprocess routine."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "preprocess":
        if args.preprocess_cmd == "nifti-to-tiff":
            nifti_to_tiff.main_ns(args)
        elif args.preprocess_cmd == "noisify":
            noisify.main_ns(args)
        elif args.preprocess_cmd == "resize":
            resize.main_ns(args)
        else:
            raise SystemExit(f"Unknown preprocess subcommand: {args.preprocess_cmd}")
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
