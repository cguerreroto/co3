"""Single-tile genetic algorithm for patch-wise denoising using deap (does not import global ga_runner)."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from deap import base, creator, tools

from image_enhancement.common.objectives import evaluate_objective


def _ensure_deap_types() -> None:
    if not hasattr(creator, "FitnessMinPatchwise"):
        creator.create("FitnessMinPatchwise", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualPatchwise"):
        creator.create("IndividualPatchwise", list, fitness=creator.FitnessMinPatchwise)


def run_tile_ga(
    v_crop: np.ndarray,
    u_crop: np.ndarray | None,
    *,
    patch: int,
    loss_mode: str,
    alpha: float,
    beta: float,
    window_size: int,
    device: torch.device | None,
    population: int,
    generations: int,
    cxpb: float,
    mutpb: float,
    seed: int | None,
    indpb: float = 0.2,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """
    Optimize one BxB patch; returns (best_u_hat, best_loss, history_rows).
    """
    _ensure_deap_types()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = patch * patch
    v_crop = np.asarray(v_crop, dtype=np.float32)
    if u_crop is not None:
        u_crop = np.asarray(u_crop, dtype=np.float32)

    history: list[dict[str, Any]] = []

    def evaluate_ind(ind: list[float]) -> tuple[float]:
        arr = np.asarray(ind, dtype=np.float32).reshape(patch, patch)
        arr = np.clip(arr, 0.0, 1.0)
        loss, _metrics = evaluate_objective(
            arr,
            v=v_crop,
            u=u_crop,
            loss_mode=loss_mode,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
            device=device,
        )
        return (float(loss),)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.IndividualPatchwise,
        lambda: creator.IndividualPatchwise([random.random() for _ in range(n)]),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.08, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_ind)

    def init_from_noisy() -> creator.IndividualPatchwise:
        flat = v_crop.reshape(-1).astype(np.float64).tolist()
        return creator.IndividualPatchwise(flat)

    pop = [init_from_noisy()]
    for _ in range(population - 1):
        pop.append(creator.IndividualPatchwise([random.random() for _ in range(n)]))
    invalid = [p for p in pop if not p.fitness.valid]
    fits = toolbox.map(toolbox.evaluate, invalid)
    for ind, fit in zip(invalid, fits):
        ind.fitness.values = fit

    gen0_best = min(float(ind.fitness.values[0]) for ind in pop)
    best_loss = gen0_best
    best_ind = list(tools.selBest(pop, 1)[0])
    history.append({"generation": -1, "best_loss": gen0_best})

    for gen in range(generations):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                mutant[:] = [float(np.clip(float(g), 0.0, 1.0)) for g in mutant]
                del mutant.fitness.values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        pop[:] = offspring
        gen_best = min((float(ind.fitness.values[0]), list(ind)) for ind in pop)
        gl, gbest = gen_best
        if gl < best_loss:
            best_loss = gl
            best_ind = gbest
        history.append({"generation": gen, "best_loss": gl})

    if best_ind is None:
        best_ind = list(pop[0])
        best_loss = float(evaluate_ind(best_ind)[0])

    out = np.asarray(best_ind, dtype=np.float32).reshape(patch, patch)
    out = np.clip(out, 0.0, 1.0)
    return out, float(best_loss), history
