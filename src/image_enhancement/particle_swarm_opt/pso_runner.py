"""Patchwise particle-swarm denoising integrated with the package CLI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from image_enhancement.common.image_io import read_grayscale_01, save_uint8_grayscale
from image_enhancement.common.objectives import evaluate_objective


def _load_gray_01(path: Path) -> np.ndarray:
    return read_grayscale_01(path)


def _pad_to_patch_grid(arr: np.ndarray, patch_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = arr.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    if pad_h == 0 and pad_w == 0:
        return arr, (0, 0)
    out = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
    return out, (pad_h, pad_w)


def _image_to_patch_position(arr: np.ndarray, patch_size: int) -> np.ndarray:
    padded, _pads = _pad_to_patch_grid(arr, patch_size)
    h, w = padded.shape
    patches = (
        padded.reshape(h // patch_size, patch_size, w // patch_size, patch_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, patch_size * patch_size)
    )
    return patches.reshape(-1).astype(np.float32)


def _patch_position_to_image(
    position: np.ndarray,
    image_shape: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    h, w = image_shape
    padded_h = ((h + patch_size - 1) // patch_size) * patch_size
    padded_w = ((w + patch_size - 1) // patch_size) * patch_size
    n_patches = (padded_h // patch_size) * (padded_w // patch_size)
    expected = n_patches * patch_size * patch_size
    if position.size != expected:
        raise ValueError(f"Particle length {position.size} != expected {expected}")
    patches = position.reshape(n_patches, patch_size, patch_size)
    grid = patches.reshape(padded_h // patch_size, padded_w // patch_size, patch_size, patch_size)
    image = grid.transpose(0, 2, 1, 3).reshape(padded_h, padded_w)
    return np.clip(image[:h, :w], 0.0, 1.0).astype(np.float32)


def _evaluate_particles(
    positions: np.ndarray,
    *,
    image_shape: tuple[int, int],
    patch_size: int,
    v: np.ndarray,
    u: np.ndarray | None,
    loss_mode: str,
    alpha: float,
    beta: float,
    window_size: int,
) -> np.ndarray:
    losses = np.empty(positions.shape[0], dtype=np.float32)
    for idx, pos in enumerate(positions):
        u_hat = _patch_position_to_image(pos, image_shape, patch_size)
        loss, _metrics = evaluate_objective(
            u_hat,
            v=v,
            u=u,
            loss_mode=loss_mode,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
        )
        losses[idx] = float(loss)
    return losses


def optimize_pso(
    noisy_path: Path,
    out_dir: Path,
    *,
    clean_path: Path | None = None,
    iterations: int = 60,
    swarm_size: int = 24,
    patch_size: int = 8,
    inertia: float = 0.7,
    cognitive: float = 1.5,
    social: float = 1.5,
    init_noise: float = 0.05,
    velocity_max: float = 0.1,
    loss_mode: str = "blind_ssim",
    alpha: float = 0.8,
    beta: float = 0.2,
    window_size: int = 11,
    seed: int = 42,
) -> dict[str, Any]:
    """Run PSO search over a patchwise image genome and save outputs."""
    rng = np.random.default_rng(seed)

    v = _load_gray_01(noisy_path)
    u = _load_gray_01(clean_path) if clean_path is not None else None
    if u is not None and u.shape != v.shape:
        raise ValueError("clean and noisy images must have same shape")
    if loss_mode == "hybrid_ssim_mse" and u is None:
        raise ValueError("hybrid_ssim_mse requires --clean/-u")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if swarm_size <= 0:
        raise ValueError("swarm_size must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    base_position = _image_to_patch_position(v, patch_size)
    dim = base_position.size
    image_shape = tuple(v.shape)

    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    stats_dir = out_dir / "stats"
    img_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    positions = np.repeat(base_position[None, :], swarm_size, axis=0)
    positions += rng.uniform(-init_noise, init_noise, size=positions.shape).astype(np.float32)
    positions = np.clip(positions, 0.0, 1.0).astype(np.float32)

    velocities = rng.uniform(-velocity_max, velocity_max, size=(swarm_size, dim)).astype(np.float32)
    personal_best_pos = positions.copy()
    personal_best_loss = _evaluate_particles(
        positions,
        image_shape=image_shape,
        patch_size=patch_size,
        v=v,
        u=u,
        loss_mode=loss_mode,
        alpha=alpha,
        beta=beta,
        window_size=window_size,
    )
    best_idx = int(np.argmin(personal_best_loss))
    global_best_pos = personal_best_pos[best_idx].copy()
    global_best_loss = float(personal_best_loss[best_idx])

    history: list[dict[str, float | int]] = []
    t0 = time.time()

    for it in range(1, iterations + 1):
        r1 = rng.random(size=(swarm_size, dim), dtype=np.float32)
        r2 = rng.random(size=(swarm_size, dim), dtype=np.float32)
        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best_pos - positions)
            + social * r2 * (global_best_pos[None, :] - positions)
        )
        velocities = np.clip(velocities, -velocity_max, velocity_max)

        positions = np.clip(positions + velocities, 0.0, 1.0).astype(np.float32)
        losses = _evaluate_particles(
            positions,
            image_shape=image_shape,
            patch_size=patch_size,
            v=v,
            u=u,
            loss_mode=loss_mode,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
        )

        improved = losses < personal_best_loss
        personal_best_pos[improved] = positions[improved]
        personal_best_loss[improved] = losses[improved]

        best_idx = int(np.argmin(personal_best_loss))
        if float(personal_best_loss[best_idx]) < global_best_loss:
            global_best_loss = float(personal_best_loss[best_idx])
            global_best_pos = personal_best_pos[best_idx].copy()

        best_img = _patch_position_to_image(global_best_pos, image_shape, patch_size)
        best_loss, best_metrics = evaluate_objective(
            best_img,
            v=v,
            u=u,
            loss_mode=loss_mode,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
        )
        rec: dict[str, float | int] = {
            "iteration": it,
            "best_loss": float(best_loss),
            "swarm_best_loss": float(np.min(losses)),
            "mean_loss": float(np.mean(losses)),
        }
        rec.update({k: float(vv) for k, vv in best_metrics.items()})
        history.append(rec)

        if it == 1 or it == iterations or it % max(1, iterations // 10) == 0:
            print(f"iter {it}/{iterations} loss={rec['best_loss']:.6f}", end="")
            if "ssim_hat_vs_clean" in rec:
                print(
                    f" SSIM(û,u)={rec['ssim_hat_vs_clean']:.4f} SSIM(û,v)={rec['ssim_hat_vs_noisy']:.4f}",
                    end="",
                )
                if "mse_vs_clean" in rec:
                    print(f" MSE={rec['mse_vs_clean']:.6f}", end="")
                if "psnr_vs_clean" in rec:
                    print(f" PSNR={rec['psnr_vs_clean']:.2f} dB", end="")
            print()

    elapsed = time.time() - t0
    best_img = _patch_position_to_image(global_best_pos, image_shape, patch_size)
    out_tif = img_dir / "denoised.tif"
    save_uint8_grayscale(out_tif, (best_img * 255.0).clip(0, 255).astype(np.uint8))
    np.savez_compressed(
        out_dir / "pso_solution.npz",
        particle=global_best_pos.astype(np.float32),
        image_shape=np.asarray(image_shape, dtype=np.int64),
        patch_size=np.asarray([patch_size], dtype=np.int64),
        noisy_path=np.asarray([str(noisy_path.resolve())]),
        clean_path=np.asarray([str(clean_path.resolve()) if clean_path is not None else ""]),
        loss_mode=np.asarray([loss_mode]),
        alpha=np.asarray([alpha], dtype=np.float32),
        beta=np.asarray([beta], dtype=np.float32),
        window_size=np.asarray([window_size], dtype=np.int64),
        iterations=np.asarray([iterations], dtype=np.int64),
        swarm_size=np.asarray([swarm_size], dtype=np.int64),
        inertia=np.asarray([inertia], dtype=np.float32),
        cognitive=np.asarray([cognitive], dtype=np.float32),
        social=np.asarray([social], dtype=np.float32),
        init_noise=np.asarray([init_noise], dtype=np.float32),
        velocity_max=np.asarray([velocity_max], dtype=np.float32),
        seed=np.asarray([seed], dtype=np.int64),
    )

    final: dict[str, Any] = {
        "mode": "single_image_pso",
        "noisy_path": str(noisy_path.resolve()),
        "clean_path": str(clean_path.resolve()) if clean_path is not None else None,
        "out_denoised": str(out_tif.resolve()),
        "loss_mode": loss_mode,
        "alpha": alpha,
        "beta": beta,
        "window_size": window_size,
        "iterations": iterations,
        "swarm_size": swarm_size,
        "patch_size": patch_size,
        "inertia": inertia,
        "cognitive": cognitive,
        "social": social,
        "init_noise": init_noise,
        "velocity_max": velocity_max,
        "seed": seed,
        "runtime_sec": elapsed,
        "history_tail": history[-5:] if len(history) > 5 else history,
    }
    if history:
        last = history[-1]
        final["final_loss"] = last["best_loss"]
        for key in (
            "ssim_hat_vs_clean",
            "ssim_vs_clean",
            "ssim_hat_vs_noisy",
            "ssim_clean_vs_noisy",
            "mse_vs_clean",
            "psnr_vs_clean",
        ):
            if key in last:
                final[key] = last[key]
    with open(stats_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    with open(stats_dir / "history.jsonl", "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")
    return final


def infer_pso(
    checkpoint: Path,
    noisy_path: Path,
    out_dir: Path,
    *,
    clean_path: Path | None = None,
) -> dict[str, Any]:
    """Reconstruct the saved PSO best particle and write metrics/image."""
    data = np.load(checkpoint, allow_pickle=True)
    particle = np.asarray(data["particle"], dtype=np.float32)
    image_shape = tuple(int(x) for x in data["image_shape"])
    patch_size = int(data["patch_size"][0])
    loss_mode = str(data["loss_mode"][0])
    alpha = float(data["alpha"][0])
    beta = float(data["beta"][0])
    window_size = int(data["window_size"][0])

    v = _load_gray_01(noisy_path)
    if tuple(v.shape) != image_shape:
        raise ValueError(f"Input noisy image shape {v.shape} != checkpoint shape {image_shape}")
    u = _load_gray_01(clean_path) if clean_path is not None else None
    u_hat = _patch_position_to_image(particle, image_shape, patch_size)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / "denoised.tif"
    save_uint8_grayscale(out_tif, (u_hat * 255.0).clip(0, 255).astype(np.uint8))
    _loss, metrics = evaluate_objective(
        u_hat,
        v=v,
        u=u,
        loss_mode=loss_mode,
        alpha=alpha,
        beta=beta,
        window_size=window_size,
    )
    rec: dict[str, Any] = {
        "noisy_path": str(noisy_path.resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "out_denoised": str(out_tif.resolve()),
        "loss_mode": loss_mode,
        "alpha": alpha,
        "beta": beta,
        "patch_size": patch_size,
    }
    rec.update(metrics)
    if clean_path is not None:
        rec["clean_path"] = str(clean_path.resolve())
    with open(out_dir / "infer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return rec


def add_optimize_parser(sub: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    p = (
        sub.add_parser("optimize-pso", help="Run patchwise particle swarm optimization (single-image)")
        if sub is not None
        else argparse.ArgumentParser("optimize-pso")
    )
    p.add_argument("--noisy", "-v", required=True, type=Path, help="Noisy image v")
    p.add_argument("--clean", "-u", type=Path, default=None, help="Optional clean image u")
    p.add_argument("--out-dir", "-o", type=Path, default=Path("output/pso"))
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--swarm-size", type=int, default=24)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--inertia", type=float, default=0.7)
    p.add_argument("--cognitive", type=float, default=1.5)
    p.add_argument("--social", type=float, default=1.5)
    p.add_argument("--init-noise", type=float, default=0.05)
    p.add_argument("--velocity-max", type=float, default=0.1)
    p.add_argument("--loss-mode", choices=("blind_ssim", "hybrid_ssim_mse"), default="blind_ssim")
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--window-size", type=int, default=11)
    p.add_argument("--seed", type=int, default=42)
    return p


def add_infer_parser(sub: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    p = (
        sub.add_parser("infer-pso", help="Write PSO denoised image from saved pso_solution.npz")
        if sub is not None
        else argparse.ArgumentParser("infer-pso")
    )
    p.add_argument("--checkpoint", "-c", required=True, type=Path, help="pso_solution.npz")
    p.add_argument("--noisy", "-v", required=True, type=Path, help="Noisy image v")
    p.add_argument("--clean", "-u", type=Path, default=None, help="Optional clean image u")
    p.add_argument("--out-dir", "-o", type=Path, default=Path("output/pso/infer"))
    return p


def optimize_cli(args: argparse.Namespace) -> None:
    optimize_pso(
        args.noisy,
        args.out_dir,
        clean_path=args.clean,
        iterations=args.iterations,
        swarm_size=args.swarm_size,
        patch_size=args.patch_size,
        inertia=args.inertia,
        cognitive=args.cognitive,
        social=args.social,
        init_noise=args.init_noise,
        velocity_max=args.velocity_max,
        loss_mode=args.loss_mode,
        alpha=args.alpha,
        beta=args.beta,
        window_size=args.window_size,
        seed=args.seed,
    )
    print(f"Done. Outputs under {args.out_dir}")


def infer_cli(args: argparse.Namespace) -> None:
    m = infer_pso(args.checkpoint, args.noisy, args.out_dir, clean_path=args.clean)
    print(json.dumps(m, indent=2))
