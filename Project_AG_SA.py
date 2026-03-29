import os
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.optimize import dual_annealing

from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import pygad


# =========================================================
# 1. DATA
# =========================================================
def load_example_image():
    """
    Returns a grayscale image scaled in [0, 1].
    """
    image = img_as_float(data.camera())
    return image


def add_noise(image, noise_var=0.01, seed=42):
    """
    Add Gaussian noise to the clean image.
    0.005 low noise, 0.01 medium, 0.05 strong
    """
    noisy = random_noise(image, mode="gaussian", var=noise_var, seed=seed)
    return np.clip(noisy, 0.0, 1.0)


def save_image(image, filepath):
    """
    Save one grayscale image to disk.
    """
    plt.imsave(filepath, image, cmap="gray", vmin=0.0, vmax=1.0)


# =========================================================
# 2. DENOISING MODEL
# =========================================================
def denoise_gaussian(noisy_image, sigma):
    """
    Gaussian denoising with parameter sigma.
    """
    sigma = float(np.clip(sigma, 0.1, 3.0))
    denoised = gaussian_filter(noisy_image, sigma=sigma)
    return np.clip(denoised, 0.0, 1.0)


# =========================================================
# 3. OBJECTIVE / METRICS
# =========================================================
def objective_sigma(sigma, noisy_image, clean_image):
    """
    Objective to minimize:
        1 - SSIM(denoised, clean)
    """
    sigma = float(np.atleast_1d(sigma)[0])
    denoised = denoise_gaussian(noisy_image, sigma)
    score_ssim = ssim(clean_image, denoised, data_range=1.0)
    return 1.0 - score_ssim


def evaluate_solution(sigma, noisy_image, clean_image):
    """
    Return a dictionary of useful metrics for one sigma.
    """
    sigma = float(np.atleast_1d(sigma)[0])
    denoised = denoise_gaussian(noisy_image, sigma)
    score_ssim = ssim(clean_image, denoised, data_range=1.0)
    score_psnr = psnr(clean_image, denoised, data_range=1.0)
    mse = np.mean((clean_image - denoised) ** 2)

    return {
        "sigma": sigma,
        "ssim": score_ssim,
        "psnr": score_psnr,
        "mse": mse,
        "image": denoised,
    }


# =========================================================
# 4. GENETIC ALGORITHM WITH PYGAD
# =========================================================
def run_genetic_algorithm(noisy_image, clean_image,
                          num_generations=30,
                          sol_per_pop=20,
                          num_parents_mating=8,
                          mutation_percent_genes=30,
                          random_seed=42):
    """
    Run GA with PyGAD.
    PyGAD maximizes a fitness function, so we maximize SSIM
    (equivalent to minimizing 1 - SSIM).
    """
    history_best_fitness = []

    def fitness_func(ga_instance, solution, solution_idx):
        sigma = float(solution[0])
        denoised = denoise_gaussian(noisy_image, sigma)
        return ssim(clean_image, denoised, data_range=1.0)

    def on_generation(ga_instance):
        best_fitness = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness
        )[1]
        history_best_fitness.append(best_fitness)

    gene_space = [{"low": 0.1, "high": 3.0}]

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=1,
        gene_space=gene_space,
        init_range_low=0.1,
        init_range_high=3.0,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=mutation_percent_genes,
        random_seed=random_seed,
        on_generation=on_generation,
        suppress_warnings=True
    )

    start = time.perf_counter()
    ga_instance.run()
    elapsed = time.perf_counter() - start

    solution, solution_fitness, _ = ga_instance.best_solution()
    best_sigma = float(solution[0])

    result = evaluate_solution(best_sigma, noisy_image, clean_image)
    result["time_sec"] = elapsed
    result["history"] = history_best_fitness
    result["method"] = "Genetic Algorithm (PyGAD)"

    return result


# =========================================================
# 5. SIMULATED ANNEALING
# =========================================================
def run_simulated_annealing(noisy_image, clean_image, maxiter=100, seed=42):
    """
    Run simulated annealing using scipy.optimize.dual_annealing.
    """
    bounds = [(0.1, 3.0)]

    start = time.perf_counter()
    result_sa = dual_annealing(
        func=lambda x: objective_sigma(x, noisy_image, clean_image),
        bounds=bounds,
        maxiter=maxiter,
        seed=seed
    )
    elapsed = time.perf_counter() - start

    best_sigma = float(result_sa.x[0])

    result = evaluate_solution(best_sigma, noisy_image, clean_image)
    result["time_sec"] = elapsed
    result["history"] = None
    result["method"] = "Simulated Annealing"

    return result


# =========================================================
# 6. VISUALIZATION
# =========================================================
def plot_results(clean_image, noisy_image, result_ga, result_sa):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(clean_image, cmap="gray")
    axes[0].set_title("Clean image")
    axes[0].axis("off")

    axes[1].imshow(noisy_image, cmap="gray")
    axes[1].set_title("Noisy image")
    axes[1].axis("off")

    axes[2].imshow(result_ga["image"], cmap="gray")
    axes[2].set_title(
        f"GA\nsigma={result_ga['sigma']:.3f}\nSSIM={result_ga['ssim']:.4f}"
    )
    axes[2].axis("off")

    axes[3].imshow(result_sa["image"], cmap="gray")
    axes[3].set_title(
        f"SA\nsigma={result_sa['sigma']:.3f}\nSSIM={result_sa['ssim']:.4f}"
    )
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def plot_ga_history(history):
    if history is None or len(history) == 0:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (SSIM)")
    plt.title("GA convergence")
    plt.tight_layout()
    plt.show()


# =========================================================
# 7. MAIN EXPERIMENT
# =========================================================
def main():
    noise_var = 0.01
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    clean_image = load_example_image()
    noisy_image = add_noise(clean_image, noise_var=noise_var, seed=42)

    # Save clean and noisy images in separate files
    clean_path = os.path.join(output_dir, "clean_image.png")
    noisy_path = os.path.join(output_dir, "noisy_image.png")
    save_image(clean_image, clean_path)
    save_image(noisy_image, noisy_path)

    print(f"Clean image saved to : {clean_path}")
    print(f"Noisy image saved to : {noisy_path}")

    print("Running Genetic Algorithm...")
    result_ga = run_genetic_algorithm(
        noisy_image,   # ordre corrigé
        clean_image,   # ordre corrigé
        num_generations=30,
        sol_per_pop=20,
        num_parents_mating=8,
        mutation_percent_genes=30,
        random_seed=42
    )

    print("Running Simulated Annealing...")
    result_sa = run_simulated_annealing(
        noisy_image,   # ordre corrigé
        clean_image,   # ordre corrigé
        maxiter=100,
        seed=42
    )

    print("\n=== RESULTS ===")
    for result in [result_ga, result_sa]:
        print(f"\nMethod: {result['method']}")
        print(f"Best sigma : {result['sigma']:.4f}")
        print(f"SSIM       : {result['ssim']:.6f}")
        print(f"PSNR       : {result['psnr']:.4f}")
        print(f"MSE        : {result['mse']:.6f}")
        print(f"Time (s)   : {result['time_sec']:.4f}")

    plot_results(clean_image, noisy_image, result_ga, result_sa)
    plot_ga_history(result_ga["history"])


if __name__ == "__main__":
    main()