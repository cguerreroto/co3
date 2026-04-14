"""Patch-wise PSO orchestration: per-tile search, overlap blending, and full-image metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from image_enhancement.common.image_io import read_grayscale_01, save_grayscale_from_float_L
from image_enhancement.common.objectives import evaluate_objective
from image_enhancement.common.performance import PerformanceTracker
from image_enhancement.ga_patchwise import blending
from image_enhancement.pso_patchwise.tile_pso import run_tile_pso

WindowMode = Literal["hann", "triangular", "tukey", "flat"]


@dataclass
class PSOPatchwiseConfig:
    patch: int
    stride: int
    loss_mode: str
    alpha: float
    beta: float
    window_size: int
    blend_mode: WindowMode
    tukey_alpha: float
    swarm_size: int
    iterations: int
    inertia: float
    cognitive: float
    social: float
    velocity_max: float
    init_noise: float
    seed: int | None
    device: str | None
    max_tiles: int | None


def _device_obj(cfg: PSOPatchwiseConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_pso_patchwise(
    v_path: str | Path,
    u_path: str | Path | None,
    out_dir: str | Path,
    cfg: PSOPatchwiseConfig,
) -> dict[str, Any]:
    """
    Run patch-wise PSO on noisy image v; optional clean u for hybrid and reporting.
    Writes images/, stats/, checkpoint npz under out_dir.
    """
    v_path = Path(v_path)
    out_dir = Path(out_dir)
    stats_dir = out_dir / "stats"
    img_dir = out_dir / "images"
    stats_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    v = read_grayscale_01(v_path)
    u: np.ndarray | None = None
    if u_path is not None:
        u = read_grayscale_01(u_path)
        if u.shape != v.shape:
            raise ValueError(f"u shape {u.shape} != v shape {v.shape}")
    if cfg.loss_mode == "hybrid_ssim_mse" and u is None:
        raise ValueError("hybrid_ssim_mse requires clean image -u")
    if cfg.swarm_size <= 0:
        raise ValueError("swarm_size must be positive")
    if cfg.iterations <= 0:
        raise ValueError("iterations must be positive")
    if cfg.patch <= 0:
        raise ValueError("patch must be positive")
    if cfg.stride <= 0:
        raise ValueError("stride must be positive")
    if cfg.patch < cfg.window_size:
        raise ValueError(f"patch ({cfg.patch}) must be >= SSIM window_size ({cfg.window_size})")

    dev = _device_obj(cfg)
    perf = PerformanceTracker()

    h, w = v.shape
    hc, wc, ys, xs = blending.padded_canvas_shape(h, w, cfg.patch, cfg.stride)
    vp = blending.pad_image_edge(v, hc, wc)
    up: np.ndarray | None = None
    if u is not None:
        up = blending.pad_image_edge(u, hc, wc)

    tiles: list[tuple[int, int]] = [(y, x) for y in ys for x in xs]
    if cfg.max_tiles is not None:
        tiles = tiles[: int(cfg.max_tiles)]

    patches: list[tuple[int, int, np.ndarray]] = []
    history_lines: list[dict[str, Any]] = []
    tile_seed = cfg.seed

    for ti, (y, x) in enumerate(tiles):
        vc = vp[y : y + cfg.patch, x : x + cfg.patch].copy()
        uc = None if up is None else up[y : y + cfg.patch, x : x + cfg.patch].copy()
        ts = None if tile_seed is None else int(tile_seed + ti * 100_003)
        best_hat, best_loss, hist = run_tile_pso(
            vc,
            uc,
            patch=cfg.patch,
            loss_mode=cfg.loss_mode,
            alpha=cfg.alpha,
            beta=cfg.beta,
            window_size=cfg.window_size,
            device=dev,
            swarm_size=cfg.swarm_size,
            iterations=cfg.iterations,
            inertia=cfg.inertia,
            cognitive=cfg.cognitive,
            social=cfg.social,
            velocity_max=cfg.velocity_max,
            init_noise=cfg.init_noise,
            seed=ts,
        )
        patches.append((y, x, best_hat))
        history_lines.append(
            {
                "tile_index": ti,
                "y": y,
                "x": x,
                "best_tile_loss": best_loss,
                "tile_iterations": len(hist),
            }
        )

    u_hat, _num, _den = blending.blend_overlapping_patches(
        h,
        w,
        cfg.patch,
        cfg.stride,
        patches,
        cfg.blend_mode,
        tukey_alpha=cfg.tukey_alpha,
    )

    loss_final, metrics = evaluate_objective(
        u_hat,
        v=v,
        u=u,
        loss_mode=cfg.loss_mode,
        alpha=cfg.alpha,
        beta=cfg.beta,
        window_size=cfg.window_size,
        device=dev,
    )
    perf_m = perf.metrics()
    metrics_out = {
        **metrics,
        "final_objective": float(loss_final),
        "loss_mode": cfg.loss_mode,
        "patch": cfg.patch,
        "stride": cfg.stride,
        "blend_mode": cfg.blend_mode,
        "num_tiles": len(tiles),
        "tile_positions": [[int(y), int(x)] for y, x in tiles],
        "grid": {"ys": ys, "xs": xs, "canvas": [hc, wc]},
        "swarm_size": cfg.swarm_size,
        "iterations": cfg.iterations,
        "inertia": cfg.inertia,
        "cognitive": cfg.cognitive,
        "social": cfg.social,
        "velocity_max": cfg.velocity_max,
        "init_noise": cfg.init_noise,
        **perf_m,
    }

    with (stats_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    with (stats_dir / "history.jsonl").open("w", encoding="utf-8") as f:
        for row in history_lines:
            f.write(json.dumps(row) + "\n")

    npz_path = out_dir / "pso_patchwise_solution.npz"
    np.savez_compressed(
        str(npz_path),
        u_hat=u_hat,
        v=v,
        u=u if u is not None else np.array([]),
        patch=cfg.patch,
        stride=cfg.stride,
        blend_mode=np.array(cfg.blend_mode),
        ys=np.array(ys),
        xs=np.array(xs),
        canvas_hc=hc,
        canvas_wc=wc,
    )

    save_grayscale_from_float_L(img_dir / "denoised.tif", u_hat, 1.0)

    return {"metrics": metrics_out, "out_dir": str(out_dir), "npz": str(npz_path)}


def infer_pso_patchwise(
    npz_path: str | Path,
    v_path: str | Path | None,
    u_path: str | Path | None,
    out_dir: str | Path,
    *,
    window_size: int = 11,
    loss_mode: str = "blind_ssim",
    alpha: float = 0.5,
    beta: float = 0.5,
    device: str | None = None,
) -> dict[str, Any]:
    """Load fused u_hat from checkpoint and recompute full-image metrics."""
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(npz_path), allow_pickle=True)
    u_hat = np.asarray(data["u_hat"], dtype=np.float32)

    v = read_grayscale_01(v_path) if v_path is not None else None
    u: np.ndarray | None = None
    if u_path is not None:
        u = read_grayscale_01(u_path)
    if v is None and "v" in data.files:
        v = np.asarray(data["v"], dtype=np.float32)
    if u is None and "u" in data.files:
        uu = np.asarray(data["u"])
        if uu.size:
            u = uu.astype(np.float32)

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if v is None:
        raise ValueError("infer requires -v or v stored in npz")

    loss, metrics = evaluate_objective(
        u_hat,
        v=v,
        u=u,
        loss_mode=loss_mode,
        alpha=alpha,
        beta=beta,
        window_size=window_size,
        device=dev,
    )
    metrics_out = {**metrics, "final_objective": float(loss), "loss_mode": loss_mode}

    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    with (stats_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    save_grayscale_from_float_L(img_dir / "denoised.tif", u_hat, 1.0)

    return {"metrics": metrics_out}


def add_optimize_pso_patchwise_parser(
    sub: argparse._SubParsersAction | None = None,
) -> argparse.ArgumentParser:
    p = (
        sub.add_parser(
            "optimize-pso-patchwise",
            help="Patch-wise PSO: independent per-tile search and overlap blending",
        )
        if sub is not None
        else argparse.ArgumentParser("optimize-pso-patchwise")
    )
    p.add_argument("--noisy", "-v", required=True, type=Path, help="Noisy image v")
    p.add_argument("--clean", "-u", type=Path, default=None, help="Clean u (required for hybrid_ssim_mse)")
    p.add_argument("--out-dir", "-o", type=Path, required=True, help="Output directory")
    p.add_argument("--loss-mode", choices=("blind_ssim", "hybrid_ssim_mse"), default="blind_ssim")
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--patch-size", type=int, default=64, help="Tile side B (>= --window-size)")
    p.add_argument("--stride", type=int, default=32, help="Tile stride (overlap = patch-size - stride)")
    p.add_argument("--window-size", type=int, default=11, help="Pooled SSIM window")
    p.add_argument("--blend", choices=("hann", "triangular", "tukey", "flat"), default="hann")
    p.add_argument("--tukey-alpha", type=float, default=0.5)
    p.add_argument("--swarm-size", type=int, default=20)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--inertia", type=float, default=0.7)
    p.add_argument("--cognitive", type=float, default=1.5)
    p.add_argument("--social", type=float, default=1.5)
    p.add_argument("--velocity-max", type=float, default=0.1)
    p.add_argument("--init-noise", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cuda or cpu")
    p.add_argument("--max-tiles", type=int, default=None, help="Limit number of tiles (debug)")
    return p


def add_infer_pso_patchwise_parser(
    sub: argparse._SubParsersAction | None = None,
) -> argparse.ArgumentParser:
    p = (
        sub.add_parser(
            "infer-pso-patchwise",
            help="Load pso_patchwise_solution.npz and write metrics and denoised.tif",
        )
        if sub is not None
        else argparse.ArgumentParser("infer-pso-patchwise")
    )
    p.add_argument("--checkpoint", "-c", required=True, type=Path, help="pso_patchwise_solution.npz")
    p.add_argument("--noisy", "-v", type=Path, default=None, help="Noisy v (else use v from npz)")
    p.add_argument("--clean", "-u", type=Path, default=None, help="Optional clean u")
    p.add_argument("--out-dir", "-o", type=Path, required=True)
    p.add_argument("--loss-mode", choices=("blind_ssim", "hybrid_ssim_mse"), default="blind_ssim")
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--window-size", type=int, default=11)
    p.add_argument("--device", type=str, default=None)
    return p


def optimize_pso_patchwise_cli(args: argparse.Namespace) -> None:
    cfg = PSOPatchwiseConfig(
        patch=args.patch_size,
        stride=args.stride,
        loss_mode=args.loss_mode,
        alpha=args.alpha,
        beta=args.beta,
        window_size=args.window_size,
        blend_mode=args.blend,
        tukey_alpha=args.tukey_alpha,
        swarm_size=args.swarm_size,
        iterations=args.iterations,
        inertia=args.inertia,
        cognitive=args.cognitive,
        social=args.social,
        velocity_max=args.velocity_max,
        init_noise=args.init_noise,
        seed=args.seed,
        device=args.device,
        max_tiles=args.max_tiles,
    )
    out = optimize_pso_patchwise(args.noisy, args.clean, args.out_dir, cfg)
    print(json.dumps({"ok": True, **out}, indent=2))


def infer_pso_patchwise_cli(args: argparse.Namespace) -> None:
    out = infer_pso_patchwise(
        args.checkpoint,
        args.noisy,
        args.clean,
        args.out_dir,
        window_size=args.window_size,
        loss_mode=args.loss_mode,
        alpha=args.alpha,
        beta=args.beta,
        device=args.device,
    )
    print(json.dumps({"ok": True, **out}, indent=2))
