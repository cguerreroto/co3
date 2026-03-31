"""Patchwise genetic-algorithm denoising via evolutionary search."""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Any
import numpy as np
from deap import base, creator, tools
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


def _image_to_patch_genome(arr: np.ndarray, patch_size: int) -> np.ndarray:
    padded, _pads = _pad_to_patch_grid(arr, patch_size)
    h, w = padded.shape
    patches = (
        padded.reshape(h // patch_size, patch_size, w // patch_size, patch_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, patch_size * patch_size)
    )
    return patches.reshape(-1).astype(np.float32)


def _patch_genome_to_image(
    genome: np.ndarray,
    image_shape: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    h, w = image_shape
    padded_h = ((h + patch_size - 1) // patch_size) * patch_size
    padded_w = ((w + patch_size - 1) // patch_size) * patch_size
    n_patches = (padded_h // patch_size) * (padded_w // patch_size)
    expected = n_patches * patch_size * patch_size
    if genome.size != expected:
        raise ValueError(f"Genome length {genome.size} != expected {expected}")
    patches = genome.reshape(n_patches, patch_size, patch_size)
    grid = patches.reshape(padded_h // patch_size, padded_w // patch_size, patch_size, patch_size)
    image = grid.transpose(0, 2, 1, 3).reshape(padded_h, padded_w)
    return np.clip(image[:h, :w], 0.0, 1.0).astype(np.float32)


def _mate_patchwise(ind1: list[float], ind2: list[float], patch_len: int) -> tuple[list[float], list[float]]:
    n_patches = len(ind1) // patch_len
    for p in range(n_patches):
        if random.random() < 0.5:
            s = p * patch_len
            e = s + patch_len
            ind1[s:e], ind2[s:e] = ind2[s:e], ind1[s:e]
    return ind1, ind2


def _mutate_patchwise(
    ind: list[float],
    patch_len: int,
    sigma: float,
    indpb: float,
) -> tuple[list[float]]:
    n_patches = len(ind) // patch_len
    for p in range(n_patches):
        if random.random() < indpb:
            s = p * patch_len
            e = s + patch_len
            for i in range(s, e):
                ind[i] = min(1.0, max(0.0, ind[i] + random.gauss(0.0, sigma)))
    return (ind,)


def optimize_ga(
    noisy_path: Path,
    out_dir: Path,
    *,
    clean_path: Path | None = None,
    generations: int = 40,
    population_size: int = 30,
    patch_size: int = 8,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.3,
    mutation_sigma: float = 0.05,
    loss_mode: str = "blind_ssim",
    alpha: float = 0.8,
    beta: float = 0.2,
    window_size: int = 11,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a patchwise GA evolutionary search and save the best result/metrics."""
    random.seed(seed)
    np.random.seed(seed)

    v = _load_gray_01(noisy_path)
    u = _load_gray_01(clean_path) if clean_path is not None else None
    if u is not None and u.shape != v.shape:
        raise ValueError("clean and noisy images must have same shape")
    if loss_mode == "hybrid_ssim_mse" and u is None:
        raise ValueError("hybrid_ssim_mse requires --clean/-u")

    base_genome = _image_to_patch_genome(v, patch_size)
    patch_len = patch_size * patch_size
    image_shape = tuple(v.shape)

    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    stats_dir = out_dir / "stats"
    img_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    try:
        creator.FitnessMax
    except AttributeError:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.GAIndividual
    except AttributeError:
        creator.create("GAIndividual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def init_individual() -> creator.GAIndividual:
        g = base_genome.copy()
        noise = np.random.normal(0.0, mutation_sigma, size=g.shape).astype(np.float32)
        g = np.clip(g + noise, 0.0, 1.0)
        return creator.GAIndividual(g.tolist())

    def evaluate(individual: list[float]) -> tuple[float]:
        u_hat = _patch_genome_to_image(np.asarray(individual, dtype=np.float32), image_shape, patch_size)
        loss, _metrics = evaluate_objective(
            u_hat,
            v=v,
            u=u,
            loss_mode=loss_mode,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
        )
        return (-float(loss),)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", _mate_patchwise, patch_len=patch_len)
    toolbox.register("mutate", _mutate_patchwise, patch_len=patch_len, sigma=mutation_sigma, indpb=0.2)

    pop = toolbox.population(n=population_size)
    history: list[dict[str, float | int]] = []

    for gen in range(1, generations + 1):
        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        best = tools.selBest(pop, 1)[0]
        best_img = _patch_genome_to_image(np.asarray(best, dtype=np.float32), image_shape, patch_size)
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
            "generation": gen,
            "best_fitness": float(best.fitness.values[0]),
            "loss": float(best_loss),
        }
        rec.update({k: float(vv) for k, vv in best_metrics.items()})
        history.append(rec)
        if gen == 1 or gen == generations or gen % max(1, generations // 10) == 0:
            print(f"gen {gen}/{generations} loss={rec['loss']:.6f}", end="")
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

        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        pop[:] = offspring

    best = tools.selBest(pop, 1)[0]
    best_img = _patch_genome_to_image(np.asarray(best, dtype=np.float32), image_shape, patch_size)
    out_tif = img_dir / "denoised.tif"
    save_uint8_grayscale(out_tif, (best_img * 255.0).clip(0, 255).astype(np.uint8))
    np.savez_compressed(
        out_dir / "ga_solution.npz",
        genome=np.asarray(best, dtype=np.float32),
        image_shape=np.asarray(image_shape, dtype=np.int64),
        patch_size=np.asarray([patch_size], dtype=np.int64),
        noisy_path=np.asarray([str(noisy_path.resolve())]),
        clean_path=np.asarray([str(clean_path.resolve()) if clean_path is not None else ""]),
        loss_mode=np.asarray([loss_mode]),
        alpha=np.asarray([alpha], dtype=np.float32),
        beta=np.asarray([beta], dtype=np.float32),
        window_size=np.asarray([window_size], dtype=np.int64),
    )

    final = {
        "mode": "single_image_ga",
        "noisy_path": str(noisy_path.resolve()),
        "clean_path": str(clean_path.resolve()) if clean_path is not None else None,
        "out_denoised": str(out_tif.resolve()),
        "loss_mode": loss_mode,
        "alpha": alpha,
        "beta": beta,
        "window_size": window_size,
        "population_size": population_size,
        "generations": generations,
        "crossover_prob": crossover_prob,
        "mutation_prob": mutation_prob,
        "mutation_sigma": mutation_sigma,
        "patch_size": patch_size,
        "history_tail": history[-5:] if len(history) > 5 else history,
    }
    if history:
        last = history[-1]
        for key in (
            "loss",
            "ssim_hat_vs_clean",
            "ssim_vs_clean",
            "ssim_hat_vs_noisy",
            "ssim_clean_vs_noisy",
            "mse_vs_clean",
            "psnr_vs_clean",
        ):
            if key in last:
                final[key if key != "loss" else "final_loss"] = last[key]
    with open(stats_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    with open(stats_dir / "history.jsonl", "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")
    return final


def infer_ga(
    checkpoint: Path,
    noisy_path: Path,
    out_dir: Path,
    *,
    clean_path: Path | None = None,
) -> dict[str, Any]:
    """Reconstruct the saved GA best individual and write metrics/image."""
    data = np.load(checkpoint, allow_pickle=True)
    genome = np.asarray(data["genome"], dtype=np.float32)
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
    u_hat = _patch_genome_to_image(genome, image_shape, patch_size)

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
        sub.add_parser("optimize-ga", help="Run patchwise genetic algorithm evolutionary search (single-image)")
        if sub is not None
        else argparse.ArgumentParser("optimize-ga")
    )
    p.add_argument("--noisy", "-v", required=True, type=Path, help="Noisy image v")
    p.add_argument("--clean", "-u", type=Path, default=None, help="Optional clean image u")
    p.add_argument("--out-dir", "-o", type=Path, default=Path("output/ga"))
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--population-size", type=int, default=30)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--crossover-prob", type=float, default=0.7)
    p.add_argument("--mutation-prob", type=float, default=0.3)
    p.add_argument("--mutation-sigma", type=float, default=0.05)
    p.add_argument("--loss-mode", choices=("blind_ssim", "hybrid_ssim_mse"), default="blind_ssim")
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--window-size", type=int, default=11)
    p.add_argument("--seed", type=int, default=42)
    return p


def add_infer_parser(sub: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    p = (
        sub.add_parser("infer-ga", help="Write GA denoised image from saved ga_solution.npz")
        if sub is not None
        else argparse.ArgumentParser("infer-ga")
    )
    p.add_argument("--checkpoint", "-c", required=True, type=Path, help="ga_solution.npz")
    p.add_argument("--noisy", "-v", required=True, type=Path, help="Noisy image v")
    p.add_argument("--clean", "-u", type=Path, default=None, help="Optional clean image u")
    p.add_argument("--out-dir", "-o", type=Path, default=Path("output/ga/infer"))
    return p


def optimize_cli(args: argparse.Namespace) -> None:
    optimize_ga(
        args.noisy,
        args.out_dir,
        clean_path=args.clean,
        generations=args.generations,
        population_size=args.population_size,
        patch_size=args.patch_size,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        mutation_sigma=args.mutation_sigma,
        loss_mode=args.loss_mode,
        alpha=args.alpha,
        beta=args.beta,
        window_size=args.window_size,
        seed=args.seed,
    )
    print(f"Done. Outputs under {args.out_dir}")


def infer_cli(args: argparse.Namespace) -> None:
    m = infer_ga(args.checkpoint, args.noisy, args.out_dir, clean_path=args.clean)
    print(json.dumps(m, indent=2))
