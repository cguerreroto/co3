"""Train autoencoder with pooled SSIM loss vs noisy observation v (single or multi-pair)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim

from image_enhancement.autoencoders.model import DenoisingAutoencoder
from image_enhancement.common.constraints import residual_energy_penalty
from image_enhancement.common.image_io import read_grayscale_01, save_uint8_grayscale
from image_enhancement.common.ssim_loss import PooledSSIMLoss, pooled_ssim
from image_enhancement.preprocessing.noisify_dir import iter_clean_images


def _load_gray_01(path: Path) -> np.ndarray:
    """Read image at ``path`` and return grayscale float array scaled to [0, 1]."""
    return read_grayscale_01(path)


def _to_tensor_hw1(arr: np.ndarray) -> torch.Tensor:
    """Convert a 2D numpy image (H, W) to a tensor shaped (1, 1, H, W)."""
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return t


def psnr_tensor(x: torch.Tensor, y: torch.Tensor, l: float = 1.0) -> float:
    """Compute PSNR in dB for tensors ``x`` and ``y`` using dynamic range ``l``."""
    mse = torch.mean((x - y) ** 2)
    if mse.item() <= 0:
        return float("inf")
    return float(10.0 * torch.log10((l * l) / mse))


def load_pairs_from_manifest(path: Path) -> list[tuple[Path, Path | None]]:
    """Load (noisy, clean) paths from JSON array file or JSONL (one object per line)."""
    path = Path(path)
    pairs: list[tuple[Path, Path | None]] = []
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON manifest must be a list of {noisy, clean?} objects")
        for row in data:
            noisy = Path(row["noisy"])
            clean = Path(row["clean"]) if row.get("clean") else None
            pairs.append((noisy, clean))
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                noisy = Path(row["noisy"])
                clean = Path(row["clean"]) if row.get("clean") else None
                pairs.append((noisy, clean))
    if not pairs:
        raise ValueError(f"Empty manifest: {path}")
    return pairs


def load_pairs_from_dirs(clean_dir: Path, noisy_dir: Path) -> list[tuple[Path, Path]]:
    """Match clean tree under ``clean_dir`` to parallel paths under ``noisy_dir``."""
    clean_dir = Path(clean_dir).resolve()
    noisy_dir = Path(noisy_dir).resolve()
    pairs: list[tuple[Path, Path]] = []
    for clean_path in iter_clean_images(clean_dir):
        rel = clean_path.relative_to(clean_dir)
        noisy_path = noisy_dir / rel
        if not noisy_path.is_file():
            # Support renamed outputs from preprocess noisify-dir:
            # slice_250.tif -> slice_noisy_250.tif
            stem = rel.stem
            if stem.startswith("slice_"):
                alt_name = "slice_noisy_" + stem[len("slice_") :] + rel.suffix
            else:
                alt_name = stem + "_noisy" + rel.suffix
            alt_path = noisy_dir / rel.with_name(alt_name)
            if alt_path.is_file():
                noisy_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Missing noisy image for {clean_path}: expected {noisy_path} or {alt_path}"
                )
        pairs.append((noisy_path, clean_path))
    if not pairs:
        raise FileNotFoundError(f"No clean images under {clean_dir}")
    return pairs


def filter_exclude_pairs(
    pairs: list[tuple[Path, Path | None]],
    exclude_file: Path | None,
) -> list[tuple[Path, Path | None]]:
    """Drop pairs whose noisy or clean path matches a line in ``exclude_file`` (basename or full path)."""
    if exclude_file is None:
        return pairs
    lines = {ln.strip() for ln in Path(exclude_file).read_text(encoding="utf-8").splitlines() if ln.strip()}
    out: list[tuple[Path, Path | None]] = []
    for noisy_p, clean_p in pairs:
        n_s = str(noisy_p.resolve())
        c_s = str(clean_p.resolve()) if clean_p else ""
        n_b = noisy_p.name
        c_b = clean_p.name if clean_p else ""
        skip = False
        for ln in lines:
            if ln in (n_s, c_s, n_b, c_b) or n_s.endswith(ln) or (c_s and c_s.endswith(ln)):
                skip = True
                break
        if not skip:
            out.append((noisy_p, clean_p))
    return out


def _resolve_epsilon(
    epsilon: float | None,
    noise_meta_path: Path | None,
    first_noisy: Path,
) -> float | None:
    if epsilon is not None:
        return epsilon
    meta = noise_meta_path
    if meta is None:
        cand = Path(str(first_noisy) + ".json")
        if cand.is_file():
            meta = cand
    if meta is None:
        return None
    with open(meta, encoding="utf-8") as f:
        m = json.load(f)
    n_pix = m.get("n_pixels")
    if n_pix is None:
        n_pix = 1
    return float(m.get("epsilon_hint", float(n_pix) * (float(m.get("sigma", 0.1)) ** 2)))


def _eval_aggregate(
    model: DenoisingAutoencoder,
    pairs: list[tuple[Path, Path | None]],
    dev: torch.device,
    window_size: int,
) -> dict[str, float] | None:
    """Average SSIM/PSNR vs clean over pairs that have ground truth."""
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for noisy_p, clean_p in pairs:
            if clean_p is None:
                continue
            v_np = _load_gray_01(noisy_p)
            u_np = _load_gray_01(clean_p)
            if u_np.shape != v_np.shape:
                continue
            v = _to_tensor_hw1(v_np).to(dev)
            u = _to_tensor_hw1(u_np).to(dev)
            u_hat = model(v)
            rows.append(
                {
                    "ssim_hat_vs_clean": float(
                        pooled_ssim(u_hat, u, window_size=window_size, L=1.0).detach().cpu()
                    ),
                    "ssim_hat_vs_noisy": float(
                        pooled_ssim(u_hat, v, window_size=window_size, L=1.0).detach().cpu()
                    ),
                    "ssim_clean_vs_noisy": float(
                        pooled_ssim(u, v, window_size=window_size, L=1.0).detach().cpu()
                    ),
                    "psnr_vs_clean": psnr_tensor(u_hat, u),
                }
            )
    if not rows:
        return None
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def train_multi(
    pairs: list[tuple[Path, Path | None]],
    out_dir: Path,
    *,
    val_pairs: list[tuple[Path, Path | None]] | None = None,
    noise_meta_path: Path | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    window_size: int = 11,
    lambda_residual: float = 0.0,
    epsilon: float | None = None,
    device: str | None = None,
    seed: int = 42,
    shuffle: bool = False,
) -> dict[str, Any]:
    """
    Train on multiple (noisy, clean?) pairs; one optimizer step per pair per epoch (stochastic over images).

    Averaged metrics per epoch over all pairs with clean reference.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not pairs:
        raise ValueError("pairs must be non-empty")

    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    stats_dir = out_dir / "stats"
    img_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    epsilon = _resolve_epsilon(epsilon, noise_meta_path, pairs[0][0])
    if lambda_residual > 0 and epsilon is None:
        raise ValueError("lambda_residual > 0 requires --epsilon or sidecar JSON with epsilon_hint")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DenoisingAutoencoder().to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = PooledSSIMLoss(window_size=window_size, L=1.0).to(dev)

    history: list[dict[str, float | int]] = []
    for ep in range(1, epochs + 1):
        model.train()
        order = list(range(len(pairs)))
        if shuffle:
            random.shuffle(order)
        epoch_loss = 0.0
        n_steps = 0
        for i in order:
            noisy_p, clean_p = pairs[i]
            v_np = _load_gray_01(noisy_p)
            v = _to_tensor_hw1(v_np).to(dev)
            opt.zero_grad()
            u_hat = model(v)
            loss = criterion(u_hat, v)
            total = loss
            if lambda_residual > 0 and epsilon is not None:
                total = total + lambda_residual * residual_energy_penalty(v, u_hat, epsilon)
            total.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu())
            n_steps += 1

        avg_loss = epoch_loss / max(1, n_steps)
        rec: dict[str, float | int] = {"epoch": int(ep), "loss": float(avg_loss)}
        agg = _eval_aggregate(model, pairs, dev, window_size)
        if agg is not None:
            rec.update({k: float(v) for k, v in agg.items()})
            if "ssim_hat_vs_clean" in rec:
                rec["ssim_vs_clean"] = float(rec["ssim_hat_vs_clean"])
        if val_pairs:
            val_agg = _eval_aggregate(model, val_pairs, dev, window_size)
            if val_agg is not None:
                for k, v in val_agg.items():
                    rec[f"val_{k}"] = float(v)
        history.append(rec)
        if ep == 1 or ep == epochs or ep % max(1, epochs // 10) == 0:
            print(f"epoch {ep}/{epochs} mean_loss={rec['loss']:.6f}", end="")
            if "ssim_hat_vs_clean" in rec:
                print(
                    f" mean SSIM(û,u)={rec['ssim_hat_vs_clean']:.4f} SSIM(û,v)={rec['ssim_hat_vs_noisy']:.4f}",
                    end="",
                )
                print(f" PSNR={rec['psnr_vs_clean']:.2f} dB", end="")
            print()

    model.eval()
    torch.save(model.state_dict(), out_dir / "autoencoder.pt")
    first_noisy, first_clean = pairs[0]
    v_np = _load_gray_01(first_noisy)
    v = _to_tensor_hw1(v_np).to(dev)
    with torch.no_grad():
        u_hat = model(v)
    sample_path = img_dir / "denoised_sample.tif"
    den_u8 = (u_hat.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    save_uint8_grayscale(sample_path, den_u8)

    final: dict[str, Any] = {
        "mode": "multi",
        "n_train_pairs": len(pairs),
        "epochs": epochs,
        "window_size": window_size,
        "lambda_residual": lambda_residual,
        "epsilon": epsilon,
        "final_loss": history[-1]["loss"] if history else None,
        "train_pairs_head": [{"noisy": str(a), "clean": str(b) if b else None} for a, b in pairs[:5]],
        "out_denoised_sample": str(sample_path.resolve()),
        "history_tail": history[-5:] if len(history) > 5 else history,
    }
    last = history[-1] if history else {}
    for k in ("psnr_vs_clean", "ssim_hat_vs_clean", "ssim_vs_clean", "ssim_hat_vs_noisy", "ssim_clean_vs_noisy"):
        if k in last:
            final[k] = last[k]
    if "ssim_hat_vs_clean" in last:
        final["ssim_vs_clean"] = last.get("ssim_hat_vs_clean")

    with open(stats_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    with open(stats_dir / "history.jsonl", "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    return final


def train(
    noisy_path: Path,
    out_dir: Path,
    *,
    clean_path: Path | None = None,
    noise_meta_path: Path | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    window_size: int = 11,
    lambda_residual: float = 0.0,
    epsilon: float | None = None,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train denoising AE on a single noisy input ``noisy_path`` and save artifacts under ``out_dir``.

    Delegates to ``train_multi`` with one pair.
    """
    meta = noise_meta_path
    if meta is None and epsilon is None:
        cand = Path(str(noisy_path) + ".json")
        if cand.is_file():
            meta = cand

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    stats_dir = out_dir / "stats"
    img_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    v_np = _load_gray_01(noisy_path)
    v = _to_tensor_hw1(v_np)
    u: torch.Tensor | None = None
    if clean_path is not None:
        u_np = _load_gray_01(clean_path)
        u = _to_tensor_hw1(u_np)
        if u.shape != v.shape:
            raise ValueError("clean and noisy images must have the same shape for evaluation")

    epsilon = _resolve_epsilon(epsilon, meta, noisy_path)
    if lambda_residual > 0 and epsilon is None:
        raise ValueError("lambda_residual > 0 requires --epsilon or --noise-meta with epsilon_hint")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    v = v.to(dev)

    model = DenoisingAutoencoder().to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = PooledSSIMLoss(window_size=window_size, L=1.0).to(dev)

    u_dev = u.to(dev) if u is not None else None
    ssim_clean_vs_noisy: float | None = None
    if u_dev is not None:
        with torch.no_grad():
            ssim_clean_vs_noisy = float(
                pooled_ssim(u_dev, v, window_size=window_size, L=1.0).detach().cpu()
            )

    history: list[dict[str, float | int]] = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        u_hat = model(v)
        loss = criterion(u_hat, v)
        total = loss
        if lambda_residual > 0 and epsilon is not None:
            total = total + lambda_residual * residual_energy_penalty(v, u_hat, epsilon)
        total.backward()
        opt.step()

        rec: dict[str, float | int] = {"epoch": int(ep), "loss": float(loss.detach().cpu())}
        if u_dev is not None:
            with torch.no_grad():
                ssim_hat_u = float(
                    pooled_ssim(u_hat, u_dev, window_size=window_size, L=1.0).detach().cpu()
                )
                ssim_hat_v = float(
                    pooled_ssim(u_hat, v, window_size=window_size, L=1.0).detach().cpu()
                )
                rec["ssim_hat_vs_clean"] = ssim_hat_u
                rec["ssim_vs_clean"] = ssim_hat_u
                rec["ssim_hat_vs_noisy"] = ssim_hat_v
                rec["psnr_vs_clean"] = psnr_tensor(u_hat, u_dev)
                if ep == 1:
                    rec["ssim_clean_vs_noisy"] = ssim_clean_vs_noisy
        history.append(rec)
        if ep == 1 or ep == epochs or ep % max(1, epochs // 10) == 0:
            print(f"epoch {ep}/{epochs} loss={rec['loss']:.6f}", end="")
            if u_dev is not None:
                print(
                    f" SSIM(û,u)={rec['ssim_hat_vs_clean']:.4f} SSIM(û,v)={rec['ssim_hat_vs_noisy']:.4f}",
                    end="",
                )
                print(f" PSNR(u,û)={rec['psnr_vs_clean']:.2f} dB", end="")
            print()

    model.eval()
    with torch.no_grad():
        u_hat = model(v)
    out_tif = img_dir / "denoised.tif"
    den_u8 = (u_hat.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    save_uint8_grayscale(out_tif, den_u8)
    torch.save(model.state_dict(), out_dir / "autoencoder.pt")

    final: dict[str, Any] = {
        "mode": "single",
        "epochs": epochs,
        "window_size": window_size,
        "lambda_residual": lambda_residual,
        "epsilon": epsilon,
        "final_loss": history[-1]["loss"] if history else None,
        "noisy_path": str(noisy_path.resolve()),
        "out_denoised": str(out_tif.resolve()),
        "history_tail": history[-5:] if len(history) > 5 else history,
    }
    if u is not None:
        last = history[-1]
        final["psnr_vs_clean"] = last.get("psnr_vs_clean")
        final["ssim_hat_vs_clean"] = last.get("ssim_hat_vs_clean")
        final["ssim_vs_clean"] = last.get("ssim_vs_clean")
        final["ssim_hat_vs_noisy"] = last.get("ssim_hat_vs_noisy")
        if ssim_clean_vs_noisy is not None:
            final["ssim_clean_vs_noisy"] = ssim_clean_vs_noisy
        if clean_path is not None:
            final["clean_path"] = str(clean_path.resolve())
    with open(stats_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    with open(stats_dir / "history.jsonl", "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    return final


def infer_ae(
    checkpoint: Path,
    noisy_path: Path,
    out_dir: Path,
    *,
    clean_path: Path | None = None,
    window_size: int = 11,
    device: str | None = None,
) -> dict[str, Any]:
    """Load weights, denoise one noisy image, write TIFF and optional metrics vs clean."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DenoisingAutoencoder().to(dev)
    state = torch.load(checkpoint, map_location=dev)
    model.load_state_dict(state)
    model.eval()

    v_np = _load_gray_01(noisy_path)
    v = _to_tensor_hw1(v_np).to(dev)
    with torch.no_grad():
        u_hat = model(v)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / "denoised.tif"
    den_u8 = (u_hat.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    save_uint8_grayscale(out_tif, den_u8)

    metrics: dict[str, Any] = {
        "noisy_path": str(noisy_path.resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "out_denoised": str(out_tif.resolve()),
    }
    if clean_path is not None:
        u_np = _load_gray_01(clean_path)
        if u_np.shape != v_np.shape:
            raise ValueError("clean and noisy must have the same shape")
        u = _to_tensor_hw1(u_np).to(dev)
        metrics["ssim_hat_vs_clean"] = float(
            pooled_ssim(u_hat, u, window_size=window_size, L=1.0).detach().cpu()
        )
        metrics["ssim_hat_vs_noisy"] = float(
            pooled_ssim(u_hat, v, window_size=window_size, L=1.0).detach().cpu()
        )
        metrics["ssim_clean_vs_noisy"] = float(
            pooled_ssim(u, v, window_size=window_size, L=1.0).detach().cpu()
        )
        metrics["psnr_vs_clean"] = psnr_tensor(u_hat, u)
        metrics["clean_path"] = str(clean_path.resolve())
    with open(out_dir / "infer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def add_train_parser(sub: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Create and return the CLI parser for the ``train-ae`` command."""
    p = (
        sub.add_parser("train-ae", help="Train SSIM-only denoising autoencoder (single or multi-pair)")
        if sub is not None
        else argparse.ArgumentParser("train-ae")
    )
    p.add_argument("--noisy", "-v", type=Path, default=None, help="Noisy observation v (single mode)")
    p.add_argument("--clean", "-u", type=Path, help="Ground-truth u for eval metrics only (single mode)")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON or JSONL list of {noisy, clean} paths per line (multi mode)",
    )
    p.add_argument(
        "--clean-dir",
        type=Path,
        default=None,
        help="Clean tree root; use with --noisy-dir (mirrored paths)",
    )
    p.add_argument(
        "--noisy-dir",
        type=Path,
        default=None,
        help="Noisy tree root; use with --clean-dir",
    )
    p.add_argument(
        "--exclude-from-train",
        type=Path,
        default=None,
        help="Text file: one path fragment per line to drop from training pairs (hold-out slices)",
    )
    p.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Optional JSON/JSONL of pairs for validation metrics each epoch",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle training pair order each epoch (multi mode)",
    )
    p.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=Path("output/ae"),
        help="Output root (images/, stats/, checkpoint)",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window-size", type=int, default=11)
    p.add_argument("--lambda-residual", type=float, default=0.0, help="Weight for ||v-û||^2 soft penalty")
    p.add_argument("--epsilon", type=float, default=None, help="Residual budget (or use sidecar JSON)")
    p.add_argument("--noise-meta", type=Path, default=None, help="JSON from noisify (epsilon_hint)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def add_infer_parser(sub: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """CLI for ``infer-ae``."""
    p = (
        sub.add_parser("infer-ae", help="Denoise one image using a trained autoencoder.pt")
        if sub is not None
        else argparse.ArgumentParser("infer-ae")
    )
    p.add_argument("--checkpoint", "-c", required=True, type=Path, help="autoencoder.pt")
    p.add_argument("--noisy", "-v", required=True, type=Path, help="Noisy input v")
    p.add_argument("--clean", "-u", type=Path, help="Optional ground truth for metrics")
    p.add_argument("--out-dir", "-o", type=Path, default=Path("output/ae/infer"), help="Output directory")
    p.add_argument("--window-size", type=int, default=11)
    p.add_argument("--device", type=str, default=None)
    return p


def train_cli(args: argparse.Namespace) -> None:
    """Execute AE training from parsed CLI arguments and print completion summary."""
    val_pairs: list[tuple[Path, Path | None]] | None = None
    if args.val_manifest is not None:
        val_pairs = load_pairs_from_manifest(args.val_manifest)

    cdir, ndir = args.clean_dir, args.noisy_dir
    if (cdir is None) != (ndir is None):
        raise SystemExit("Provide both --clean-dir and --noisy-dir together, or neither")

    if args.manifest is not None:
        pairs = load_pairs_from_manifest(args.manifest)
        pairs = filter_exclude_pairs(pairs, args.exclude_from_train)
        if not pairs:
            raise SystemExit("No training pairs left after filtering (check --exclude-from-train)")
        train_multi(
            pairs,
            args.out_dir,
            val_pairs=val_pairs,
            noise_meta_path=args.noise_meta,
            epochs=args.epochs,
            lr=args.lr,
            window_size=args.window_size,
            lambda_residual=args.lambda_residual,
            epsilon=args.epsilon,
            device=args.device,
            seed=args.seed,
            shuffle=args.shuffle,
        )
    elif args.clean_dir is not None and args.noisy_dir is not None:
        pairs = load_pairs_from_dirs(args.clean_dir, args.noisy_dir)
        pairs = filter_exclude_pairs(pairs, args.exclude_from_train)
        if not pairs:
            raise SystemExit("No training pairs left after filtering (check --exclude-from-train)")
        train_multi(
            pairs,
            args.out_dir,
            val_pairs=val_pairs,
            noise_meta_path=args.noise_meta,
            epochs=args.epochs,
            lr=args.lr,
            window_size=args.window_size,
            lambda_residual=args.lambda_residual,
            epsilon=args.epsilon,
            device=args.device,
            seed=args.seed,
            shuffle=args.shuffle,
        )
    else:
        if args.noisy is None:
            raise SystemExit("Provide --noisy/-v (single), or --manifest, or --clean-dir with --noisy-dir")
        train(
            args.noisy,
            args.out_dir,
            clean_path=args.clean,
            noise_meta_path=args.noise_meta,
            epochs=args.epochs,
            lr=args.lr,
            window_size=args.window_size,
            lambda_residual=args.lambda_residual,
            epsilon=args.epsilon,
            device=args.device,
            seed=args.seed,
        )
    print(f"Done. Outputs under {args.out_dir}")


def infer_cli(args: argparse.Namespace) -> None:
    """Run inference CLI."""
    m = infer_ae(
        args.checkpoint,
        args.noisy,
        args.out_dir,
        clean_path=args.clean,
        window_size=args.window_size,
        device=args.device,
    )
    print(json.dumps(m, indent=2))


def main() -> None:
    """Standalone entrypoint for this module."""
    parser = add_train_parser()
    train_cli(parser.parse_args())


if __name__ == "__main__":
    main()
