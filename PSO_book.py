# PSO for Perceptual Image Denoising — SSIM-only formulation
# L(u_hat) = 1 - S(u_hat, v) = (1/P) sum_p (1 - SSIM(R_p u_hat, R_p v))
# subject to: 0 <= u_hat_k <= 1  (6a),  ||v - u_hat||^2 <= epsilon  (6b)

# libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import util
from PIL import Image
import pyswarms
import time
import multiprocess as multiprocessing

# parameters
patch_shape = (16, 16)

n_particles = 20
n_iter = 200
init_noise = 0.1

w_inertia = 0.7
c1_cognitive = 1.5
c2_social = 1.5

# sigma known from noisify preprocessing (0-255 scale → normalize to [0,1])
sigma = 25 / 255

# set to False to disable constraint (6b)
use_residual_constraint = True


# --- objective and projection ---

def objective(u_patch, v_patch):
    ssim_patch = ssim(u_patch, v_patch, data_range=1.0, full=False, gradient=False)
    return 1.0 - float(ssim_patch)

def project_l2_ball(particles, v_flat, eps_patch):
    # project particles violating ||v - x||_2 <= sqrt(eps_patch) back onto the ball surface
    if not use_residual_constraint:
        return particles
    diff = particles - v_flat
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    r = np.sqrt(eps_patch)
    mask = (norms.squeeze() > r)
    particles[mask] = v_flat + diff[mask] * (r / norms[mask])
    return particles


#====================================================================================
# --- manual PSO ---

def initialize_swarm(v_patch):
    v_patch_flat = v_patch.flatten()
    # initialize particles near the noisy patch with small perturbations for better convergence
    u_particles = v_patch_flat + np.random.uniform(-init_noise, init_noise, size=(n_particles, v_patch_flat.size))
    u_particles = np.clip(u_particles, 0, 1)
    u_velocity = np.random.uniform(-0.1, 0.1, size=(n_particles, v_patch_flat.size))

    u_patches = u_particles.reshape(n_particles, *patch_shape)
    scores = np.array([objective(u_patch, v_patch) for u_patch in u_patches], dtype=float)

    u_best_p = u_particles.copy()
    u_best_scores = scores.copy()
    g_best_p = u_best_p[np.argmin(u_best_scores)]
    g_best_scores = np.min(u_best_scores)

    return u_particles, u_velocity, u_best_p, u_best_scores, g_best_p, g_best_scores


# no random seed, as PSO stochastic by design
# clamp velocity to [-0.1, 0.1] to prevent particles overshooting the [0,1] pixel range
# clip positions to [0, 1] pixel range; project onto L2-ball for constraint (6b)
def pso_loop(v_patch, u_particles, u_velocity, u_best_p, u_best_scores, g_best_p, g_best_scores):
    v_flat = v_patch.flatten()
    K = v_flat.size
    eps_patch = (K / n_pixels) * epsilon

    for n in range(0, n_iter):
        r1 = np.random.uniform(size=(n_particles, K))
        r2 = np.random.uniform(size=(n_particles, K))

        u_velocity = w_inertia*u_velocity + c1_cognitive*r1*(u_best_p - u_particles) + c2_social*r2*(g_best_p - u_particles)
        u_velocity = np.clip(u_velocity, -0.1, 0.1)

        u_particles = u_particles + u_velocity
        u_particles = project_l2_ball(u_particles, v_flat, eps_patch)  # constraint (6b)
        u_particles = np.clip(u_particles, 0, 1)                       # constraint (6a)

        u_patches = u_particles.reshape(n_particles, *patch_shape)
        new_scores = np.array([objective(u_patch, v_patch) for u_patch in u_patches], dtype=float)

        improved = new_scores < u_best_scores
        u_best_scores[improved] = new_scores[improved]
        u_best_p[improved] = u_particles[improved]

        if new_scores.min() < g_best_scores:
            g_best_scores = new_scores.min()
            g_best_p = u_best_p[np.argmin(new_scores)]

    return g_best_scores, g_best_p


def denoise_patch_manual(v_patch):
    u_particles, u_velocity, u_best_p, u_best_scores, g_best_p, g_best_scores = initialize_swarm(v_patch)
    g_best_scores, g_best_p = pso_loop(v_patch, u_particles, u_velocity, u_best_p, u_best_scores, g_best_p, g_best_scores)
    return g_best_p.reshape(patch_shape)


# batch process image patches
# stitch patches back: list → (n_patches, 16, 16) → (n_rows, n_cols, 16, 16) →
# transpose to (n_rows, 16, n_cols, 16) so pixels of the same row are adjacent → full image
def denoise_image_manual(im_n):
    patches_im_n = util.view_as_blocks(im_n, patch_shape)
    v_patches = patches_im_n.reshape(-1, *patch_shape)

    with multiprocessing.Pool() as pool:
        denoised_patches = pool.map(denoise_patch_manual, v_patches)

    denoised_patches_arr = np.array(denoised_patches)
    n_rows = im_n.shape[0] // patch_shape[0]
    n_cols = im_n.shape[1] // patch_shape[1]
    den_reshaped = denoised_patches_arr.reshape(n_rows, n_cols, patch_shape[0], patch_shape[1])
    den_final = den_reshaped.transpose(0, 2, 1, 3).reshape(im_n.shape)
    return den_final


#====================================================================================
# --- pyswarms PSO ---

def denoise_patch_pyswarms(v_patch):
    v_patch_flat = v_patch.flatten()
    K = v_patch_flat.size
    eps_patch = (K / n_pixels) * epsilon

    # initialize particles near the noisy patch with small perturbations for better convergence
    init_pos = v_patch_flat + np.random.uniform(-init_noise, init_noise, size=(n_particles, K))
    init_pos = np.clip(init_pos, 0, 1)
    # project initial positions so swarm starts feasible w.r.t. constraint (6b)
    init_pos = project_l2_ball(init_pos, v_patch_flat, eps_patch)
    init_pos = np.clip(init_pos, 0, 1)

    def wrapper(swarm):
        return np.array([objective(swarm[s].reshape(patch_shape), v_patch) for s in range(len(swarm))])

    bounds = (np.zeros(K), np.ones(K))
    swarm_options = {"c1": c1_cognitive, "c2": c2_social, "w": w_inertia}
    optimizer = pyswarms.single.GlobalBestPSO(n_particles, dimensions=K, options=swarm_options, bounds=bounds, velocity_clamp=(-0.1, 0.1), init_pos=init_pos)
    best_cost, best_pos = optimizer.optimize(wrapper, iters=n_iter, verbose=False)

    return best_pos.reshape(patch_shape)


def denoise_image_pyswarms(im_n):
    patches_im_n = util.view_as_blocks(im_n, patch_shape)
    v_patches = patches_im_n.reshape(-1, *patch_shape)

    with multiprocessing.Pool() as pool:
        denoised_patches = pool.map(denoise_patch_pyswarms, v_patches)

    denoised_patches_arr = np.array(denoised_patches)
    n_rows = im_n.shape[0] // patch_shape[0]
    n_cols = im_n.shape[1] // patch_shape[1]
    den_reshaped = denoised_patches_arr.reshape(n_rows, n_cols, patch_shape[0], patch_shape[1])
    den_final = den_reshaped.transpose(0, 2, 1, 3).reshape(im_n.shape)
    return den_final


#====================================================================================
# --- metrics ---

def objective_full(u, v):
    patches_u = util.view_as_blocks(u, patch_shape).reshape(-1, *patch_shape)
    patches_v = util.view_as_blocks(v, patch_shape).reshape(-1, *patch_shape)
    ssim_pooled = np.mean([ssim(pu, pv, data_range=1.0) for pu, pv in zip(patches_u, patches_v)])
    return 1.0 - ssim_pooled


# --- convergence experiment ---

def pso_with_history(v_patch, n_iterations):
    v_flat = v_patch.flatten()
    K = v_flat.size
    eps_patch = (K / n_pixels) * epsilon

    u_particles, u_velocity, u_best_p, u_best_scores, g_best_p, g_best_scores = initialize_swarm(v_patch)
    history = [g_best_scores]
    for n in range(0, n_iterations):
        r1 = np.random.uniform(size=(n_particles, K))
        r2 = np.random.uniform(size=(n_particles, K))
        u_velocity = w_inertia*u_velocity + c1_cognitive*r1*(u_best_p - u_particles) + c2_social*r2*(g_best_p - u_particles)
        u_velocity = np.clip(u_velocity, -0.1, 0.1)
        u_particles = u_particles + u_velocity
        u_particles = project_l2_ball(u_particles, v_flat, eps_patch)
        u_particles = np.clip(u_particles, 0, 1)
        u_patches = u_particles.reshape(n_particles, *patch_shape)
        new_scores = np.array([objective(up, v_patch) for up in u_patches], dtype=float)
        improved = new_scores < u_best_scores
        u_best_scores[improved] = new_scores[improved]
        u_best_p[improved] = u_particles[improved]
        if new_scores.min() < g_best_scores:
            g_best_scores = new_scores.min()
            g_best_p = u_best_p[np.argmin(new_scores)]
        history.append(g_best_scores)
    return np.array(history), g_best_p.reshape(patch_shape)


#====================================================================================
def main():
    global n_iter, n_pixels, epsilon

    # --- 1. preprocessing ---
    with Image.open("slice_noisy.png") as im:
        im_grey = im.convert("L")
    print(f"Image format, size, color mode: {im_grey.format, im_grey.size, im_grey.mode}")
    im_n = np.array(im_grey) / 255

    # pad to nearest multiple of patch_shape if needed
    h, w = im_n.shape
    h_pad = int(np.ceil(h / patch_shape[0])) * patch_shape[0]
    w_pad = int(np.ceil(w / patch_shape[1])) * patch_shape[1]
    pad_h = h_pad - h
    pad_w = w_pad - w

    if pad_h > 0 or pad_w > 0:
        im_n = np.pad(im_n, ((0, pad_h), (0, pad_w)), mode="reflect")
        print(f"Padded from ({h}, {w}) → {im_n.shape}")
    else:
        print(f"Image shape: {im_n.shape} — no padding needed")

    # residual energy bound: use original (unpadded) pixel count
    n_pixels = h * w
    epsilon = n_pixels * sigma**2
    print(f"sigma={sigma:.4f}, n_pixels={n_pixels}, epsilon={epsilon:.2f}")

    # divide image into patches
    patches_im_n = util.view_as_blocks(im_n, patch_shape)
    print(f"Shape of patched image: {patches_im_n.shape}")

    v_patch = patches_im_n[10, 10]
    print(f"Patch shape: {v_patch.shape}")

    # visualize image and selected patch
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    axes[0].imshow(im_n, cmap="gray")
    axes[0].set_title("Noisy image", fontsize=8)
    axes[0].axis("off")
    axes[1].imshow(v_patch, cmap="gray")
    axes[1].set_title("Patch [10, 10]", fontsize=8)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    # --- 2. run manual PSO ---
    t0 = time.time()
    denoised_manual = denoise_image_manual(im_n)
    t_manual = time.time() - t0
    print(f"Manual PSO: {t_manual:.1f} s")

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    axes[0].imshow(im_n, cmap="gray")
    axes[0].set_title("Noisy image", fontsize=8)
    axes[0].axis("off")
    axes[1].imshow(denoised_manual, cmap="gray")
    axes[1].set_title(f"Denoised: manual ({t_manual:.1f} s)", fontsize=8)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    # --- 3. run pyswarms PSO ---
    t0 = time.time()
    denoised_psw = denoise_image_pyswarms(im_n)
    t_psw = time.time() - t0
    print(f"pyswarms PSO: {t_psw:.1f} s")

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(im_n, cmap="gray")
    axes[0].set_title("Noisy image", fontsize=8)
    axes[0].axis("off")
    axes[1].imshow(denoised_manual, cmap="gray")
    axes[1].set_title(f"Denoised: manual ({t_manual:.1f} s)", fontsize=8)
    axes[1].axis("off")
    axes[2].imshow(denoised_psw, cmap="gray")
    axes[2].set_title(f"Denoised: pyswarms ({t_psw:.1f} s)", fontsize=8)
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

    # --- 4. quantitative metrics ---
    images = {
        "Noisy":        im_n,
        "Manual PSO":   denoised_manual,
        "pyswarms PSO": denoised_psw,
    }

    print(f"{'Image':<18} {'Mean':>6} {'Std':>6} {'Residual':>12} {'<= eps?':>8} {'Objective L':>12}")
    print("-" * 68)
    for name, img in images.items():
        mean_val = img.mean()
        std_val = img.std()
        resid = np.sum((im_n - img)**2)
        feasible = resid <= epsilon
        obj_val = objective_full(img, im_n) if name != "Noisy" else 0.0
        print(f"{name:<18} {mean_val:>6.4f} {std_val:>6.4f} {resid:>12.2f} {str(feasible):>8} {obj_val:>12.4f}")

    print(f"\nepsilon = {epsilon:.2f}  (sigma={sigma:.4f}, n_pixels={n_pixels})")

    # --- 5. single-patch convergence experiment ---
    cx, cy = im_n.shape[0] // 2, im_n.shape[1] // 2
    v_patch_test = im_n[cx:cx + patch_shape[0], cy:cy + patch_shape[1]]

    iter_counts = [200, 500, 1000]
    histories = {}
    for n in iter_counts:
        hist, _ = pso_with_history(v_patch_test, n)
        histories[n] = hist
        print(f"n_iter={n:4d}  final L={hist[-1]:.6f}")

    plt.figure(figsize=(8, 4))
    for n, hist in histories.items():
        plt.plot(hist, label=f"n_iter={n}")
    plt.xlabel("Iteration")
    plt.ylabel("Global best score")
    plt.title("Single-patch convergence — SSIM-only loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 6. full image with n_iter=1000 ---
    n_iter = 1000

    t0 = time.time()
    denoised_1000 = denoise_image_manual(im_n)
    t_1000 = time.time() - t0
    print(f"Manual PSO n_iter=1000: {t_1000:.1f} s")

    print(f"\n{'n_iter':>8} {'Residual':>12} {'<= eps?':>8} {'Objective L':>12}")
    print("-" * 44)
    for label, img in [("200", denoised_manual), ("1000", denoised_1000)]:
        resid = np.sum((im_n - img)**2)
        obj_val = objective_full(img, im_n)
        print(f"{label:>8} {resid:>12.2f} {str(resid <= epsilon):>8} {obj_val:>12.4f}")

    n_iter = 200  # reset

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(im_n, cmap="gray")
    axes[0].set_title("Noisy", fontsize=8)
    axes[0].axis("off")
    axes[1].imshow(denoised_manual, cmap="gray")
    axes[1].set_title("Manual n_iter=200", fontsize=8)
    axes[1].axis("off")
    axes[2].imshow(denoised_1000, cmap="gray")
    axes[2].set_title(f"Manual n_iter=1000 ({t_1000:.0f} s)", fontsize=8)
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
