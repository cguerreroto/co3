"""Train autoencoder with pooled SSIM loss vs noisy observation v."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import optim

from image_enhancement.autoencoders.model import DenoisingAutoencoder
from image_enhancement.common.constraints import residual_energy_penalty
from image_enhancement.common.ssim_loss import PooledSSIMLoss, pooled_ssim
from image_enhancement.common.image_io import read_grayscale_01, save_uint8_grayscale


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
) -> dict:
    """
    Train denoising AE on noisy input ``noisy_path`` and save artifacts under ``out_dir``.

    Inputs: noisy/clean paths and training hyperparameters.
    Output: dictionary with final metrics and output paths.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    stats_dir = out_dir / "stats"
    img_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    v_np = _load_gray_01(noisy_path)
    v = _to_tensor_hw1(v_np)
    if clean_path is not None:
        u_np = _load_gray_01(clean_path)
        u = _to_tensor_hw1(u_np)
        if u.shape != v.shape:
            raise ValueError("clean and noisy images must have the same shape for evaluation")
    else:
        u = None

    if noise_meta_path is not None and epsilon is None:
        with open(noise_meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        epsilon = float(meta.get("epsilon_hint", meta.get("n_pixels", v_np.size) * meta.get("sigma", 0.1) ** 2))

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

    metrics_path = stats_dir / "metrics.json"
    final: dict[str, object] = {
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
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    with open(stats_dir / "history.jsonl", "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    return final


def add_train_parser(sub: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Create and return the CLI parser for the ``train-ae`` command."""
    p = (
        sub.add_parser("train-ae", help="Train SSIM-only denoising autoencoder")
        if sub is not None
        else argparse.ArgumentParser("train-ae")
    )
    p.add_argument("--noisy", "-v", required=True, type=Path, help="Noisy observation v (TIFF)")
    p.add_argument("--clean", "-u", type=Path, help="Ground-truth u for eval metrics only")
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
    p.add_argument("--epsilon", type=float, default=None, help="Residual budget (or use --noise-meta)")
    p.add_argument("--noise-meta", type=Path, default=None, help="JSON from noisify (epsilon_hint)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def train_cli(args: argparse.Namespace) -> None:
    """Execute AE training from parsed CLI arguments and print completion summary."""
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


def main() -> None:
    """Standalone entrypoint for this module."""
    parser = add_train_parser()
    train_cli(parser.parse_args())


if __name__ == "__main__":
    main()
