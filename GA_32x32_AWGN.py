import os
import time
import json
import tracemalloc
from pathlib import Path

import numpy as np
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error
from skimage.restoration import estimate_sigma


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = 32
RANDOM_SEED = 42

# Genetic algorithm parameters
POP_SIZE = 12
N_GENERATIONS = 15
ELITE_SIZE = 2
TOURNAMENT_SIZE = 3

# Pixel-wise mutation parameters
MUTATION_RATE = 0.03
MUTATION_STD = 0.03

# Residual constraint parameter:
# epsilon_patch = EPSILON_FACTOR * n_pixels * sigma_patch^2
# Start with 1.0 for a strict interpretation of the text.
# You can later try 1.2 or 1.5 if it is too restrictive.
EPSILON_FACTOR = 1.5

INPUT_DIR = "output_images"
NOISY_BASENAME = "noisy_image"
CLEAN_BASENAME = "clean_image"

OUTPUT_IMAGE_NAME = "ga_pixelwise_ssim_residual_patch32.png"
OUTPUT_METRICS_NAME = "ga_pixelwise_ssim_residual_patch32_metrics.json"


# ============================================================
# IMAGE I/O
# ============================================================

def find_image_file(input_dir, base_name):
    """
    Find an image file in the given directory using a base name.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    allowed_ext = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

    for ext in allowed_ext:
        candidate = input_path / f"{base_name}{ext}"
        if candidate.exists():
            return str(candidate)

    for f in input_path.iterdir():
        if f.is_file() and f.stem == base_name:
            return str(f)
        if f.is_file() and f.name.startswith(base_name):
            return str(f)

    raise FileNotFoundError(f"No file found for base name '{base_name}' in {input_dir}")


def load_grayscale_image(image_path):
    """
    Load an image as grayscale float image in [0, 1].
    """
    image = io.imread(image_path, as_gray=True)
    image = img_as_float(image)
    image = np.clip(image, 0.0, 1.0)
    return image


def save_image(image_path, image):
    """
    Save a float image in [0, 1] as uint8.
    """
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    io.imsave(image_path, image_uint8)


# ============================================================
# PATCH UTILITIES
# ============================================================

def pad_image_to_multiple(image, patch_size=32):
    """
    Pad the image so both dimensions are multiples of patch_size.
    Reflection padding is used.
    """
    h, w = image.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    return padded, h, w


def extract_non_overlapping_patches(image, patch_size=32):
    """
    Extract non-overlapping patches.
    Returns:
        patches, n_rows, n_cols
    """
    h, w = image.shape
    n_rows = h // patch_size
    n_cols = w // patch_size

    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            patch = image[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size
            ]
            patches.append(patch.copy())

    return patches, n_rows, n_cols


def reconstruct_from_patches(patches, n_rows, n_cols, patch_size=32):
    """
    Reconstruct an image from non-overlapping patches.
    """
    h = n_rows * patch_size
    w = n_cols * patch_size
    image = np.zeros((h, w), dtype=np.float64)

    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            image[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size
            ] = patches[idx]
            idx += 1

    return image


# ============================================================
# CHROMOSOME UTILITIES
# ============================================================

def patch_to_vector(patch):
    """
    Convert a 2D patch into a 1D chromosome.
    """
    return patch.reshape(-1).astype(np.float64)


def vector_to_patch(vector, patch_size=32):
    """
    Convert a 1D chromosome back into a 2D patch.
    """
    return vector.reshape(patch_size, patch_size)


def clip_pixels(vector):
    """
    Enforce box constraints 0 <= pixel <= 1.
    """
    return np.clip(vector, 0.0, 1.0)


# ============================================================
# RESIDUAL CONSTRAINT
# ============================================================

def estimate_patch_noise_sigma(noisy_patch):
    """
    Estimate local noise sigma for one patch.
    """
    sigma = float(estimate_sigma(noisy_patch, channel_axis=None, average_sigmas=True))
    return max(sigma, 1e-8)


def compute_patch_epsilon(noisy_patch, epsilon_factor=1.0):
    """
    Compute epsilon for the residual constraint:
        ||v_patch - u_hat_patch||_2^2 <= epsilon

    Following the project text, epsilon is taken of order n * sigma^2.
    """
    sigma_patch = estimate_patch_noise_sigma(noisy_patch)
    n_pixels = noisy_patch.size
    epsilon = epsilon_factor * n_pixels * (sigma_patch ** 2)
    return float(epsilon), float(sigma_patch)


def residual_energy(candidate_patch, noisy_patch):
    """
    Compute squared L2 residual energy.
    """
    diff = noisy_patch - candidate_patch
    return float(np.sum(diff ** 2))


# ============================================================
# FITNESS FUNCTION
# ============================================================

def patch_fitness_ssim_with_residual_constraint(
    candidate_vector,
    noisy_patch,
    epsilon_patch,
    patch_size=32
):
    """
    Compute fitness for one candidate patch.

    Objective:
        maximize SSIM(candidate_patch, noisy_patch)

    Constraint:
        ||noisy_patch - candidate_patch||_2^2 <= epsilon_patch

    The GA is still a maximizer. If the residual constraint is violated,
    the candidate receives a strong penalty.
    """
    candidate_patch = vector_to_patch(candidate_vector, patch_size=patch_size)
    score = float(ssim(candidate_patch, noisy_patch, data_range=1.0))

    resid = residual_energy(candidate_patch, noisy_patch)

    if resid <= epsilon_patch:
        feasible = True
        fitness = score
    else:
        feasible = False
        # Strong penalty for infeasible candidates
        # The farther beyond epsilon, the worse the fitness.
        violation_ratio = (resid - epsilon_patch) / max(epsilon_patch, 1e-12)
        fitness = score - 10.0 * violation_ratio

    return float(fitness), float(score), float(resid), feasible


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_population_from_noisy_patch(
    noisy_patch,
    pop_size,
    rng,
    epsilon_patch,
    init_std=0.02,
    max_attempts=30
):
    """
    Initialize the population around the noisy patch.

    The first individual is the noisy patch itself, which is always feasible
    because its residual energy is 0.

    Other individuals are generated near the noisy patch and accepted if they
    satisfy the residual constraint. If that fails repeatedly, the candidate is
    projected back toward the noisy patch.
    """
    base_vector = patch_to_vector(noisy_patch)
    population = [base_vector.copy()]

    while len(population) < pop_size:
        candidate = None

        for _ in range(max_attempts):
            trial = base_vector + rng.normal(0.0, init_std, size=base_vector.shape)
            trial = clip_pixels(trial)

            trial_patch = vector_to_patch(trial, patch_size=noisy_patch.shape[0])
            resid = residual_energy(trial_patch, noisy_patch)

            if resid <= epsilon_patch:
                candidate = trial
                break

        if candidate is None:
            # Fallback: project a perturbed candidate back toward the noisy patch
            trial = base_vector + rng.normal(0.0, init_std, size=base_vector.shape)
            trial = clip_pixels(trial)

            alpha = 0.5
            for _ in range(20):
                projected = alpha * trial + (1.0 - alpha) * base_vector
                projected = clip_pixels(projected)

                projected_patch = vector_to_patch(projected, patch_size=noisy_patch.shape[0])
                resid = residual_energy(projected_patch, noisy_patch)

                if resid <= epsilon_patch:
                    candidate = projected
                    break

                alpha *= 0.5

            if candidate is None:
                candidate = base_vector.copy()

        population.append(candidate)

    return population


# ============================================================
# GA OPERATORS
# ============================================================

def tournament_selection(population, fitnesses, rng, k=3):
    """
    Select one individual using tournament selection.
    """
    idxs = rng.choice(len(population), size=k, replace=False)
    best_idx = idxs[np.argmax([fitnesses[i] for i in idxs])]
    return population[best_idx].copy()


def crossover(parent1, parent2, rng):
    """
    Arithmetic crossover in pixel space.
    """
    alpha = rng.uniform(0.0, 1.0, size=parent1.shape)
    child = alpha * parent1 + (1.0 - alpha) * parent2
    return clip_pixels(child)


def project_to_residual_constraint(candidate_vector, noisy_patch, epsilon_patch):
    """
    Project a candidate toward the noisy patch until the residual constraint
    is satisfied.
    """
    base_vector = patch_to_vector(noisy_patch)
    candidate = candidate_vector.copy()

    candidate_patch = vector_to_patch(candidate, patch_size=noisy_patch.shape[0])
    resid = residual_energy(candidate_patch, noisy_patch)
    if resid <= epsilon_patch:
        return candidate

    alpha = 0.5
    for _ in range(30):
        projected = alpha * candidate + (1.0 - alpha) * base_vector
        projected = clip_pixels(projected)

        projected_patch = vector_to_patch(projected, patch_size=noisy_patch.shape[0])
        resid = residual_energy(projected_patch, noisy_patch)

        if resid <= epsilon_patch:
            return projected

        alpha *= 0.5

    return base_vector.copy()


def mutate(individual, rng, noisy_patch, epsilon_patch, mutation_rate=0.03, mutation_std=0.03):
    """
    Mutate an individual by adding Gaussian noise to a subset of pixels,
    then project back to the feasible set if needed.
    """
    child = individual.copy()

    mutation_mask = rng.random(size=child.shape) < mutation_rate
    if np.any(mutation_mask):
        child[mutation_mask] += rng.normal(
            0.0,
            mutation_std,
            size=np.sum(mutation_mask)
        )

    child = clip_pixels(child)
    child = project_to_residual_constraint(child, noisy_patch, epsilon_patch)
    return child


def evaluate_population(population, noisy_patch, epsilon_patch, patch_size=32):
    """
    Evaluate all individuals in the population on one patch.
    """
    fitnesses = []
    ssim_scores = []
    residuals = []
    feasible_flags = []

    for individual in population:
        fitness, ssim_score, resid, feasible = patch_fitness_ssim_with_residual_constraint(
            candidate_vector=individual,
            noisy_patch=noisy_patch,
            epsilon_patch=epsilon_patch,
            patch_size=patch_size
        )
        fitnesses.append(fitness)
        ssim_scores.append(ssim_score)
        residuals.append(resid)
        feasible_flags.append(feasible)

    return (
        np.array(fitnesses, dtype=np.float64),
        np.array(ssim_scores, dtype=np.float64),
        np.array(residuals, dtype=np.float64),
        np.array(feasible_flags, dtype=bool)
    )


# ============================================================
# PATCH-LEVEL PIXEL-WISE GA
# ============================================================

def genetic_optimize_patch_pixels_with_residual(
    noisy_patch,
    rng,
    patch_size=32,
    pop_size=12,
    n_generations=15,
    elite_size=2,
    tournament_size=3,
    mutation_rate=0.03,
    mutation_std=0.03,
    epsilon_factor=1.0
):
    """
    Optimize one patch directly in pixel space with:
    - SSIM-only objective
    - residual constraint
    - box constraints on pixel intensities
    """
    epsilon_patch, sigma_patch = compute_patch_epsilon(
        noisy_patch,
        epsilon_factor=epsilon_factor
    )

    population = initialize_population_from_noisy_patch(
        noisy_patch=noisy_patch,
        pop_size=pop_size,
        rng=rng,
        epsilon_patch=epsilon_patch
    )

    n_evaluations = 0
    best_global_fitness = -np.inf
    best_global_individual = None
    best_global_ssim = None
    best_global_residual = None

    for _ in range(n_generations):
        fitnesses, ssim_scores, residuals, feasible_flags = evaluate_population(
            population=population,
            noisy_patch=noisy_patch,
            epsilon_patch=epsilon_patch,
            patch_size=patch_size
        )
        n_evaluations += len(population)

        order = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in order]
        fitnesses = fitnesses[order]
        ssim_scores = ssim_scores[order]
        residuals = residuals[order]
        feasible_flags = feasible_flags[order]

        if fitnesses[0] > best_global_fitness:
            best_global_fitness = float(fitnesses[0])
            best_global_individual = population[0].copy()
            best_global_ssim = float(ssim_scores[0])
            best_global_residual = float(residuals[0])

        next_population = []

        # Elitism
        for i in range(elite_size):
            next_population.append(population[i].copy())

        # Generate rest of next generation
        while len(next_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, rng, k=tournament_size)
            parent2 = tournament_selection(population, fitnesses, rng, k=tournament_size)

            child = crossover(parent1, parent2, rng)
            child = mutate(
                child,
                rng,
                noisy_patch=noisy_patch,
                epsilon_patch=epsilon_patch,
                mutation_rate=mutation_rate,
                mutation_std=mutation_std
            )

            next_population.append(child)

        population = next_population

    best_patch = vector_to_patch(best_global_individual, patch_size=patch_size)

    info = {
        "best_patch_ssim_to_noisy": float(best_global_ssim),
        "best_patch_residual_energy": float(best_global_residual),
        "epsilon_patch": float(epsilon_patch),
        "estimated_sigma_patch": float(sigma_patch),
        "n_evaluations": int(n_evaluations)
    }

    return best_patch, info


# ============================================================
# FULL IMAGE PIPELINE
# ============================================================

def run_pixelwise_ga_on_image_with_residual(
    noisy_image,
    patch_size=32,
    pop_size=12,
    n_generations=15,
    elite_size=2,
    tournament_size=3,
    mutation_rate=0.03,
    mutation_std=0.03,
    epsilon_factor=1.0,
    seed=42
):
    """
    Run patch-wise pixel optimization with SSIM-only objective
    and residual constraint on the full image.
    """
    rng = np.random.default_rng(seed)

    padded_image, original_h, original_w = pad_image_to_multiple(
        noisy_image,
        patch_size=patch_size
    )

    patches, n_rows, n_cols = extract_non_overlapping_patches(
        padded_image,
        patch_size=patch_size
    )

    reconstructed_patches = []
    patch_ssims = []
    patch_residuals = []
    patch_epsilons = []
    patch_sigmas = []
    total_evaluations = 0

    for patch_idx, noisy_patch in enumerate(patches, start=1):
        best_patch, info = genetic_optimize_patch_pixels_with_residual(
            noisy_patch=noisy_patch,
            rng=rng,
            patch_size=patch_size,
            pop_size=pop_size,
            n_generations=n_generations,
            elite_size=elite_size,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            mutation_std=mutation_std,
            epsilon_factor=epsilon_factor
        )

        reconstructed_patches.append(best_patch)
        patch_ssims.append(info["best_patch_ssim_to_noisy"])
        patch_residuals.append(info["best_patch_residual_energy"])
        patch_epsilons.append(info["epsilon_patch"])
        patch_sigmas.append(info["estimated_sigma_patch"])
        total_evaluations += info["n_evaluations"]

        print(
            f"[Patch {patch_idx:03d}/{len(patches)}] "
            f"ssim={info['best_patch_ssim_to_noisy']:.6f}  "
            f"residual={info['best_patch_residual_energy']:.6f} / "
            f"epsilon={info['epsilon_patch']:.6f}"
        )

    reconstructed = reconstruct_from_patches(
        reconstructed_patches,
        n_rows=n_rows,
        n_cols=n_cols,
        patch_size=patch_size
    )

    reconstructed = reconstructed[:original_h, :original_w]
    reconstructed = np.clip(reconstructed, 0.0, 1.0)

    summary = {
        "n_patches": len(patches),
        "patch_size": patch_size,
        "avg_patch_ssim_to_noisy": float(np.mean(patch_ssims)),
        "std_patch_ssim_to_noisy": float(np.std(patch_ssims)),
        "avg_patch_residual_energy": float(np.mean(patch_residuals)),
        "avg_patch_epsilon": float(np.mean(patch_epsilons)),
        "avg_estimated_sigma_patch": float(np.mean(patch_sigmas)),
        "total_evaluations": int(total_evaluations)
    }

    return reconstructed, summary


# ============================================================
# FINAL REFERENCE-BASED EVALUATION
# ============================================================

def compute_reference_metrics(reference_image, test_image, label):
    """
    Compute reference-based metrics using the clean image.
    """
    return {
        f"ssim_{label}": float(ssim(reference_image, test_image, data_range=1.0)),
        f"psnr_{label}": float(psnr(reference_image, test_image, data_range=1.0)),
        f"mse_{label}": float(mean_squared_error(reference_image, test_image)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    tracemalloc.start()
    t0 = time.perf_counter()

    # Load noisy image used during optimization
    noisy_path = find_image_file(INPUT_DIR, NOISY_BASENAME)
    noisy_image = load_grayscale_image(noisy_path)

    print(f"Noisy image loaded: {noisy_path}")
    print(f"Noisy image shape : {noisy_image.shape}")

    # Run pixel-wise GA with SSIM-only objective + residual constraint
    denoised_image, summary = run_pixelwise_ga_on_image_with_residual(
        noisy_image=noisy_image,
        patch_size=PATCH_SIZE,
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        elite_size=ELITE_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        mutation_rate=MUTATION_RATE,
        mutation_std=MUTATION_STD,
        epsilon_factor=EPSILON_FACTOR,
        seed=RANDOM_SEED
    )

    # Runtime and memory
    elapsed = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Diagnostics relative to noisy image
    ssim_denoised_vs_noisy = float(ssim(denoised_image, noisy_image, data_range=1.0))
    residual_energy_global = float(np.sum((noisy_image - denoised_image) ** 2))

    # Load clean image only for final evaluation
    clean_path = find_image_file(INPUT_DIR, CLEAN_BASENAME)
    clean_image = load_grayscale_image(clean_path)

    print(f"Clean image loaded: {clean_path}")
    print(f"Clean image shape : {clean_image.shape}")

    if clean_image.shape != noisy_image.shape:
        raise ValueError(
            f"Shape mismatch: clean image shape = {clean_image.shape}, "
            f"noisy image shape = {noisy_image.shape}"
        )

    metrics_noisy_vs_clean = compute_reference_metrics(
        clean_image,
        noisy_image,
        label="noisy_vs_clean"
    )

    metrics_denoised_vs_clean = compute_reference_metrics(
        clean_image,
        denoised_image,
        label="denoised_vs_clean"
    )

    ssim_gain = (
        metrics_denoised_vs_clean["ssim_denoised_vs_clean"]
        - metrics_noisy_vs_clean["ssim_noisy_vs_clean"]
    )
    psnr_gain = (
        metrics_denoised_vs_clean["psnr_denoised_vs_clean"]
        - metrics_noisy_vs_clean["psnr_noisy_vs_clean"]
    )
    mse_gain = (
        metrics_noisy_vs_clean["mse_noisy_vs_clean"]
        - metrics_denoised_vs_clean["mse_denoised_vs_clean"]
    )

    # Save outputs
    output_image_path = os.path.join(INPUT_DIR, OUTPUT_IMAGE_NAME)
    output_metrics_path = os.path.join(INPUT_DIR, OUTPUT_METRICS_NAME)

    save_image(output_image_path, denoised_image)

    metrics = {
        "method": "pixel-wise genetic algorithm with SSIM objective and residual constraint",
        "faithful_to_project_text": True,
        "input_images": {
            "noisy_image": noisy_path,
            "clean_image": clean_path
        },
        "image_shape": list(noisy_image.shape),
        "patch_size": PATCH_SIZE,
        "ga_parameters": {
            "population_size": POP_SIZE,
            "n_generations": N_GENERATIONS,
            "elite_size": ELITE_SIZE,
            "tournament_size": TOURNAMENT_SIZE,
            "mutation_rate": MUTATION_RATE,
            "mutation_std": MUTATION_STD,
            "epsilon_factor": EPSILON_FACTOR,
            "seed": RANDOM_SEED
        },
        "optimization_summary": summary,
        "no_reference_diagnostics": {
            "ssim_denoised_vs_noisy": ssim_denoised_vs_noisy,
            "global_residual_energy": residual_energy_global
        },
        "reference_comparison": {
            **metrics_noisy_vs_clean,
            **metrics_denoised_vs_clean,
            "ssim_gain": float(ssim_gain),
            "psnr_gain_db": float(psnr_gain),
            "mse_gain": float(mse_gain)
        },
        "runtime_seconds": float(elapsed),
        "memory_peak_mb": float(peak_mem / (1024 ** 2))
    }

    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Print final summary
    print("\n========== FINAL RESULTS ==========")
    print(f"Denoised image saved to : {output_image_path}")
    print(f"Metrics saved to        : {output_metrics_path}")
    print(f"Total runtime           : {elapsed:.3f} s")
    print(f"Peak memory             : {peak_mem / (1024 ** 2):.3f} MB")
    print(f"Total evaluations       : {summary['total_evaluations']}")

    print("\n----- Diagnostics relative to noisy image -----")
    print(f"SSIM(denoised, noisy)   : {ssim_denoised_vs_noisy:.6f}")
    print(f"Global residual energy  : {residual_energy_global:.6f}")

    print("\n----- Residual-constraint summary -----")
    print(f"Average patch epsilon   : {summary['avg_patch_epsilon']:.6f}")
    print(f"Average patch residual  : {summary['avg_patch_residual_energy']:.6f}")
    print(f"Average patch sigma     : {summary['avg_estimated_sigma_patch']:.6f}")

    print("\n----- Final comparison with clean image -----")
    print(f"SSIM(noisy, clean)      : {metrics_noisy_vs_clean['ssim_noisy_vs_clean']:.6f}")
    print(f"PSNR(noisy, clean)      : {metrics_noisy_vs_clean['psnr_noisy_vs_clean']:.6f} dB")
    print(f"MSE(noisy, clean)       : {metrics_noisy_vs_clean['mse_noisy_vs_clean']:.8f}")

    print(f"SSIM(denoised, clean)   : {metrics_denoised_vs_clean['ssim_denoised_vs_clean']:.6f}")
    print(f"PSNR(denoised, clean)   : {metrics_denoised_vs_clean['psnr_denoised_vs_clean']:.6f} dB")
    print(f"MSE(denoised, clean)    : {metrics_denoised_vs_clean['mse_denoised_vs_clean']:.8f}")

    print("\n----- Improvement after denoising -----")
    print(f"SSIM gain               : {ssim_gain:+.6f}")
    print(f"PSNR gain               : {psnr_gain:+.6f} dB")
    print(f"MSE reduction           : {mse_gain:+.8f}")


if __name__ == "__main__":
    main()