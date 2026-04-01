"""CLI entry: preprocess, AE training, GA, and PSO optimization."""

from __future__ import annotations
import argparse
import image_enhancement.autoencoders.training as train_mod
import image_enhancement.genetic_algorithm.ga_runner as ga_mod
import image_enhancement.particle_swarm_opt.pso_runner as pso_mod
from image_enhancement.preprocessing import nifti_to_tiff, noisify, noisify_dir, resize


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser and register preprocess subcommands."""
    parser = argparse.ArgumentParser(prog="image-enhancement")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("preprocess", help="NIfTI-to-image, AWGN, resize")
    prep_sub = prep.add_subparsers(dest="preprocess_cmd", required=True)
    nifti_to_tiff.build_parser(prep_sub)
    noisify.build_parser(prep_sub)
    noisify_dir.build_parser(prep_sub)
    resize.build_parser(prep_sub)

    train_mod.add_train_parser(sub)
    train_mod.add_infer_parser(sub)
    ga_mod.add_optimize_parser(sub)
    ga_mod.add_infer_parser(sub)
    pso_mod.add_optimize_parser(sub)
    pso_mod.add_infer_parser(sub)
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
        elif args.preprocess_cmd == "noisify-dir":
            noisify_dir.main_ns(args)
        elif args.preprocess_cmd == "resize":
            resize.main_ns(args)
        else:
            raise SystemExit(f"Unknown preprocess subcommand: {args.preprocess_cmd}")
    elif args.command == "train-ae":
        train_mod.train_cli(args)
    elif args.command == "infer-ae":
        train_mod.infer_cli(args)
    elif args.command == "optimize-ga":
        ga_mod.optimize_cli(args)
    elif args.command == "infer-ga":
        ga_mod.infer_cli(args)
    elif args.command == "optimize-pso":
        pso_mod.optimize_cli(args)
    elif args.command == "infer-pso":
        pso_mod.infer_cli(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
