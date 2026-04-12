"""Single-tile particle swarm optimisation for patch-wise denoising."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from image_enhancement.common.objectives import evaluate_objective


def run_tile_pso(
    v_crop: np.ndarray,
    u_crop: np.ndarray | None,
    *,
    patch: int,
    loss_mode: str,
    alpha: float,
    beta: float,
    window_size: int,
    device: torch.device | None,
    swarm_size: int,
    iterations: int,
    inertia: float,
    cognitive: float,
    social: float,
    velocity_max: float,
    init_noise: float,
    seed: int | None,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """
    Optimise one patch x patch crop with PSO.

    Returns (best_u_hat, best_loss, history_rows) where best_u_hat has
    shape (patch, patch) and values in [0, 1].
    """
    rng = np.random.default_rng(seed)

    n = patch * patch
    v_flat = np.asarray(v_crop, dtype=np.float32).reshape(-1)
    if u_crop is not None:
        u_crop = np.asarray(u_crop, dtype=np.float32)

    def _loss(flat: np.ndarray) -> float:
        arr = np.clip(flat.reshape(patch, patch), 0.0, 1.0).astype(np.float32)
        loss, _ = evaluate_objective(
            arr,
            v=v_crop.astype(np.float32),
            u=u_crop,
            loss_mode=loss_mode,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
            device=device,
        )
        return float(loss)

    # ------------------------------------------------------------------ init
    positions = np.repeat(v_flat[None, :], swarm_size, axis=0).astype(np.float32)
    positions += rng.uniform(-init_noise, init_noise, size=(swarm_size, n)).astype(np.float32)
    positions = np.clip(positions, 0.0, 1.0)

    velocities = rng.uniform(-velocity_max, velocity_max, size=(swarm_size, n)).astype(np.float32)

    personal_best_pos = positions.copy()
    personal_best_loss = np.array([_loss(p) for p in positions], dtype=np.float32)

    best_idx = int(np.argmin(personal_best_loss))
    global_best_pos = personal_best_pos[best_idx].copy()
    global_best_loss = float(personal_best_loss[best_idx])

    history: list[dict[str, Any]] = [{"iteration": 0, "best_loss": global_best_loss}]

    # ------------------------------------------------------------------ loop
    for it in range(1, iterations + 1):
        r1 = rng.random(size=(swarm_size, n)).astype(np.float32)
        r2 = rng.random(size=(swarm_size, n)).astype(np.float32)

        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best_pos - positions)
            + social * r2 * (global_best_pos[None, :] - positions)
        )
        velocities = np.clip(velocities, -velocity_max, velocity_max)
        positions = np.clip(positions + velocities, 0.0, 1.0).astype(np.float32)

        for i in range(swarm_size):
            loss_i = _loss(positions[i])
            if loss_i < personal_best_loss[i]:
                personal_best_loss[i] = loss_i
                personal_best_pos[i] = positions[i].copy()
                if loss_i < global_best_loss:
                    global_best_loss = loss_i
                    global_best_pos = positions[i].copy()

        history.append({"iteration": it, "best_loss": global_best_loss})

    best_u_hat = np.clip(global_best_pos.reshape(patch, patch), 0.0, 1.0).astype(np.float32)
    return best_u_hat, global_best_loss, history
