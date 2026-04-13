# Image enhancement (SSIM denoising)

Python package for pooled SSIM denoising experiments: NIfTI-to-image preprocessing, AWGN noisification, optional resize for genetic algorithm (GA) or particle swarm optimization (PSO) branches, and a convolutional autoencoder trained to minimize $1 - \mathcal{S}(\hat{u}, v)$, where $\mathcal{S}$ is the mean local SSIM between the denoised estimate $\hat{u}$ and the noisy observation $v$.

## Objectives

(shared across methods, selected with `--loss-mode`.)

1. Blind objective. Minimize loss derived from pooled SSIM between $\hat{u}$ and the observation $v$ (equivalently $1 - \mathcal{S}(\hat{u},v)$ on the relevant domain).
2. Supervised hybrid objective. When paired $(u,v)$ are available, minimize $\alpha\bigl(1 - \mathcal{S}(\hat{u}, u)\bigr) + \beta\,\mathrm{MSE}(\hat{u}, u)$ (`hybrid_ssim_mse`), combining SSIM and MSE against clean $u$.

## Search geometry

(how $\hat{u}$ is built and optimized; this is independent of which objective you select.)

| Geometry | Description | GA-related CLI |
|----------|-------------|------------------|
| (A) Global | One search over the entire image at once (single genome or swarm for the full field). | `optimize-ga`, `infer-ga` |
| (B) Patch-wise | Independent optimization on each overlapping tile, then overlap blending into $\hat{u}_{\mathrm{final}}$. | `optimize-ga-patchwise`, `infer-ga-patchwise` |

PSO (`optimize-pso` / `infer-pso`) uses the same objectives and (A) global geometry as `optimize-ga`. The autoencoder (`train-ae` / `infer-ae`) is a learned global mapper; it is not the patch-wise GA path.

## Setup

### 1. Clone or open the repo and create a virtual environment (venv)

```bash
cd /<path_to>/co3            # modify <path_to> according to your actual path to this repository
python3 -m venv .co3
source .co3/bin/activate   # Windows: .co3\Scripts\activate
```

Keep the venv activated whenever you work on this project (each new terminal: `cd .../co3` and `source .co3/bin/activate` again).

### 2. Install this project and its dependencies (recommended)

Run exactly this from the repository root (the folder that contains `pyproject.toml`):

```bash
pip install --upgrade pip
pip install -e . # Also: python3 -m pip install --user -e .
```

The trailing `.` in `pip install -e .` means “install from the current directory.” Do not omit it.

What this does:

- `pyproject.toml` is only a config file: you never “install” or “run” it by name. `pip` reads it automatically when you run `pip install -e .` here.
- That command installs third-party libraries listed under `dependencies` in `pyproject.toml` (`numpy`, `nibabel`, `tifffile`, `imageio`, `torch`) and also installs this package (`image_enhancement`) in editable mode (`-e`): your live code under `src/` is used immediately, and you can run `python -m image_enhancement.main` without setting `PYTHONPATH`.

### 3. Optional: `requirements.txt` alone

`requirements.txt` lists the same kinds of dependencies for tools that expect a plain list. It does not register this project as a package.

- You do not need two separate install steps if you already ran `pip install -e .`, which covers dependencies from `pyproject.toml`.
- If you prefer to install only libraries first: `pip install -r requirements.txt`, then you still need `pip install -e .` so Python can `import image_enhancement` and find the CLI. Otherwise use `PYTHONPATH=src` every time (not recommended).

### 4. Check

```bash
python -m image_enhancement.main --help
# optional shortcut (same program), if pip put it on your PATH:
image-enhancement --help
```

## Data layout

- Place inputs under `assets/` (local only; image extensions may be git-ignored).
- Outputs default to `output/ae/`, `output/ga/`, `output/ga_patchwise/`, `output/pso/` for autoencoder, global GA, patch-wise GA, and PSO runs respectively (you may use any directory names; these are typical).

## Commands

In the script blocks below, comments use two levels: major topics use a line of `#` plus asterisks (10 characters wider than the former `# === ... ===` line), with the title centered; numbered subsections use `#` plus dots (same width), with the subsection title centered on the middle line.

```bash
# ******************************************************
# ******** convert nifti to tiff (jpg/jpeg/png) ********
# ******************************************************

# Single slice from NIfTI (axis 1 is often coronal in RAS; inspect with --output-dir first). Output can be .tif/.tiff/.png/.jpg/.jpeg.
image-enhancement preprocess nifti-to-tiff -i assets/volume.nii.gz -o assets/slice.tif --axis 1 --index 257

# Or export all slices along an axis for browsing
image-enhancement preprocess nifti-to-tiff -i assets/volume.nii.gz --output-dir assets/slices --axis 1

# Autoencoders: Only slices 250–350 (inclusive), or many volumes via glob (each under output-root/<stem>/)
image-enhancement preprocess nifti-to-tiff -i assets/volume.nii.gz --output-dir assets/slices_vol --axis 1 --index-start 250 --index-end 350

image-enhancement preprocess nifti-to-tiff --input-glob "assets/*.nii.gz" --output-root assets/slices_by_volume --output-dir assets/_argparse_dummy --axis 1 --index-start 250 --index-end 350


# *************************************
# ******** Preprocess: noisify ********
# *************************************

# AWGN for one image (u -> v) + JSON sidecar (sigma, epsilon_hint, ...).  Input and output can be .tif/.tiff/.png/.jpg/.jpeg.
image-enhancement preprocess noisify -i assets/slice.tif -o assets/slice_noisy.tif --sigma 25

# AWGN for an entire folder tree: mirror all images and keep relative paths
# assets/slices_vol/...      -> assets/slices_noise_vol/...
# assets/slices_by_volume/... -> assets/slices_noise_by_volume/...
# filenames are renamed like: slice_257.tif -> slice_noisy_257.tif (+ sidecar .json)
image-enhancement preprocess noisify-dir --clean-dir assets/slices_vol --noisy-dir assets/slices_noise_vol --sigma 25

image-enhancement preprocess noisify-dir --clean-dir assets/slices_by_volume --noisy-dir assets/slices_noise_by_volume --sigma 25


# ************************************
# ******** Preprocess: resize ********
# ************************************

# Resize for GA/PSO branch (proportional, max side 128). Input and output can be .tif/.tiff/.png/.jpg/.jpeg.
image-enhancement preprocess resize -i assets/slice.tif -o assets/slice_small.tif --max-side 128

# Note: resize creates u_small only; run preprocess noisify on assets/slice_small.tif with --sigma to produce v_small.

image-enhancement preprocess resize -i assets/slice.tif -o assets/slice_small.tif --max-side 128 \
&& image-enhancement preprocess noisify -i assets/slice_small.tif -o assets/slice_small_noisy.tif --sigma 25


# ****************************************
# ******** Autoencoders: training ********
# ****************************************

#.......................................
#            1. Single image            
#.......................................

# One noisy/clean pair only
# Option A. Default mode: blind_ssim
# optimize 1 - SSIM(û,v)
image-enhancement train-ae -v assets/slices_noise_by_volume/volume/slice_noisy_257.tif -u assets/slices_by_volume/volume/slice_257.tif -o output/ae --epochs 200 --loss-mode blind_ssim

# Option B. Hybrid supervised mode
# optimize alpha*(1-SSIM(û,u)) + beta*MSE(û,u)
image-enhancement train-ae -v assets/slices_noise_by_volume/volume/slice_noisy_257.tif -u assets/slices_by_volume/volume/slice_257.tif -o output/ae_hybrid --epochs 200 \
  --loss-mode hybrid_ssim_mse --alpha 0.8 --beta 0.2

# Optional residual-energy soft penalty for the same single image
image-enhancement train-ae -v assets/slices_noise_by_volume/volume/slice_noisy_257.tif -u assets/slices_by_volume/volume/slice_257.tif -o output/ae \
  --lambda-residual 1e-4 --noise-meta assets/slices_noise_by_volume/volume/slice_noisy_257.tif.json

#.......................................
#    2. Group of images in one folder   
#.......................................

# Example: assets/slices_vol <-> assets/slices_noise_vol
# One optimization step per image per epoch; metrics are epoch means
# Option A. Default mode: blind_ssim
image-enhancement train-ae --clean-dir assets/slices_vol --noisy-dir assets/slices_noise_vol -o output/ae --epochs 50 --shuffle --loss-mode blind_ssim

# Option B. Hybrid supervised mode
# optimize alpha*(1-SSIM(û,u)) + beta*MSE(û,u)
image-enhancement train-ae --clean-dir assets/slices_vol --noisy-dir assets/slices_noise_vol -o output/ae_hybrid \
  --epochs 50 --shuffle --loss-mode hybrid_ssim_mse --alpha 0.8 --beta 0.2

#.......................................
# 3. Group of images in multiple folders
#.......................................

# Example: assets/slices_by_volume/<volume_name>/... <-> assets/slices_noise_by_volume/<volume_name>/...
# Option A. Default mode: blind_ssim
image-enhancement train-ae --clean-dir assets/slices_by_volume --noisy-dir assets/slices_noise_by_volume -o output/ae --epochs 50 --shuffle --loss-mode blind_ssim

# Option B. Hybrid supervised mode
# optimize alpha*(1-SSIM(û,u)) + beta*MSE(û,u)
image-enhancement train-ae --clean-dir assets/slices_by_volume --noisy-dir assets/slices_noise_by_volume -o output/ae_hybrid \
  --epochs 50 --shuffle --loss-mode hybrid_ssim_mse --alpha 0.8 --beta 0.2

# Or JSONL manifest: one {"noisy":"...","clean":"..."} per line. Hold out a slice by listing it in exclude.txt
image-enhancement train-ae --manifest train_pairs.jsonl --exclude-from-train exclude.txt -o output/ae


# *************************************
# ******** Autoencoders: infer ********
# *************************************

#.......................................
#           1. Single image                                                  
#.......................................

# Option A. Checkpoint trained with blind_ssim
image-enhancement infer-ae -c output/ae/autoencoder.pt -v assets/slices_noise_by_volume/volume/slice_noisy_257.tif -u assets/slices_by_volume/volume/slice_257.tif -o output/ae/infer_holdout

# Option B. Checkpoint trained with hybrid_ssim_mse
image-enhancement infer-ae -c output/ae_hybrid/autoencoder.pt -v assets/slices_noise_by_volume/volume/slice_noisy_257.tif -u assets/slices_by_volume/volume/slice_257.tif -o output/ae_hybrid/infer_holdout

#.......................................
#   2. Group of images in one folder     
#.......................................
# if training used assets/slices_vol and assets/slices_noise_vol  

# Option A. Checkpoint trained with blind_ssim
image-enhancement infer-ae -c output/ae/autoencoder.pt -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/ae/infer_holdout

# Option B. Checkpoint trained with hybrid_ssim_mse
image-enhancement infer-ae -c output/ae_hybrid/autoencoder.pt -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/ae_hybrid/infer_holdout

#.......................................
# 3. Group of images in multiple folders 
#.......................................
# if training used assets/slices_by_volume and assets/slices_noise_by_volume

# Option A. Checkpoint trained with blind_ssim
image-enhancement infer-ae -c output/ae/autoencoder.pt -v assets/slices_noise_by_volume/volume/slice_noisy_257.tif -u assets/slices_by_volume/volume/slice_257.tif -o output/ae/infer_holdout

# Option B. Checkpoint trained with hybrid_ssim_mse
image-enhancement infer-ae -c output/ae_hybrid/autoencoder.pt -v assets/slices_noise_by_volume/volume/slice_noisy_257.tif -u assets/slices_by_volume/volume/slice_257.tif -o output/ae_hybrid/infer_holdout
```

Equivalent module invocation:

```bash
python -m image_enhancement.main preprocess noisify -i ... -o ... --sigma 25
python -m image_enhancement.main train-ae -v ... -o output/ae
```

### CLI help (`argparse`)

Run these in the terminal to print usage and every flag for that command:

```bash
python -m image_enhancement.main --help
python -m image_enhancement.main preprocess --help
python -m image_enhancement.main train-ae --help
python -m image_enhancement.main infer-ae --help
python -m image_enhancement.main optimize-ga --help
python -m image_enhancement.main infer-ga --help
python -m image_enhancement.main optimize-pso --help
python -m image_enhancement.main infer-pso --help
python -m image_enhancement.main optimize-ga-patchwise --help
python -m image_enhancement.main infer-ga-patchwise --help
python -m image_enhancement.main optimize-pso-patchwise --help
python -m image_enhancement.main infer-pso-patchwise --help
```

Preprocess subcommands each have their own options:

```bash
python -m image_enhancement.main preprocess nifti-to-tiff --help
python -m image_enhancement.main preprocess noisify --help
python -m image_enhancement.main preprocess noisify-dir --help
python -m image_enhancement.main preprocess resize --help
```

If the `image-enhancement` script is on your `PATH` (after `pip install -e .`), you can swap `python -m image_enhancement.main` for `image-enhancement` in the commands above (for example `image-enhancement preprocess --help`).

---
### `train-ae` outputs and metrics

Training uses the same objective families as in the introduction: (1) `blind_ssim` (pooled SSIM vs. $v$) and (2) `hybrid_ssim_mse` (vs.\ clean $u$ when available). The search is neither (A) evolutionary nor (B) patch-wise GA: it is gradient descent on a global convolutional autoencoder that maps $v \mapsto \hat{u}$ for the full image.

`--loss-mode blind_ssim` minimizes pooled SSIM loss between $\hat{u}$ and $v$ (blind objective). `--loss-mode hybrid_ssim_mse` minimizes `alpha*(1-SSIM(û,u)) + beta*MSE(û,u)` and therefore requires clean references (`-u`, `--clean-dir` + `--noisy-dir`, or manifest entries with `clean`). Single-image mode (`-v` / `--noisy`): one row per epoch in `history.jsonl`. Multi-pair mode (`--manifest` or `--clean-dir` + `--noisy-dir`): each epoch performs one optimization step per training pair (order optionally shuffled with `--shuffle`); logged `loss` is the mean training loss over pairs; SSIM/PSNR/MSE columns are means over all pairs with clean references. Validation pairs (`--val-manifest`) add `val_*` columns. Outputs include `autoencoder.pt`, `stats/metrics.json`, and `images/denoised_sample.tif` (first manifest pair after training). For an unbiased test slice, exclude it from training (`--exclude-from-train`) and run `infer-ae`.

When you pass `--clean` / `-u` (single mode) or use multi-pair mode with clean references, logs and `output/ae/stats/metrics.json` include full-reference evaluation (not part of the loss):

| Field | Meaning |
| ----- | ------- |
| `ssim_hat_vs_clean` | Pooled SSIM $(\hat{u},u)$. Primary quality vs ground truth |
| `ssim_vs_clean` | Same value as `ssim_hat_vs_clean` (backward-compatible alias) |
| `ssim_hat_vs_noisy` | Pooled SSIM $(\hat{u},v)$. Similarity to the noisy input |
| `ssim_clean_vs_noisy` | Pooled SSIM $(u,v)$. Noise difficulty baseline (also on epoch 1 in `history.jsonl`) |
| `mse_vs_clean` | MSE $(\hat{u},u)$. Raw pixelwise error to the clean target |
| `psnr_vs_clean` | PSNR $(u,\hat{u})$. Classical baseline (monotone in MSE) |
| `runtime_sec` | Wall-clock runtime for the full training or inference run |
| `peak_rss_bytes` | Peak resident process memory observed during the run |

`metrics.json` also records `loss_mode`, `alpha`, `beta`, `window_size`, `lambda_residual`, `epsilon`, paths, `runtime_sec`, and `peak_rss_bytes`. In blind mode, PSNR is often enough as a classical scalar alongside SSIM; in hybrid experiments it is useful to inspect raw `mse_vs_clean` directly because it is part of the optimized objective.

---
### `optimize-ga` / `infer-ga`: (A) Global search geometry

This is the global GA: one evolutionary loop over a single whole-image chromosome (the image is stored as a sequence of small patches in one genome, but fitness is evaluated on the full reconstructed $\hat{u}$ vs. $v$ / $u$). It is intentionally narrower than AE:

- single-image denoising first
- outputs under e.g.\ `output/ga/`

Objective (1) blind vs. (2) hybrid (see table at the top):

```bash
# (A) Global GA, objective (1): blind pooled SSIM relative to v
image-enhancement optimize-ga -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/ga --generations 40 --population-size 30 --patch-size 8 --loss-mode blind_ssim

# (A) Global GA, objective (2): hybrid SSIM+MSE relative to u (pass clean image with -u)
image-enhancement optimize-ga -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/ga_hybrid --generations 40 --population-size 30 --patch-size 8 --loss-mode hybrid_ssim_mse --alpha 0.8 --beta 0.2
```

Inference (rebuild $\hat{u}$ from `ga_solution.npz` and evaluate):

```bash
# (A) Global inference after a blind run
image-enhancement infer-ga -c output/ga/ga_solution.npz -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/ga/infer_holdout

# (A) Global inference after a hybrid run
image-enhancement infer-ga -c output/ga_hybrid/ga_solution.npz -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/ga_hybrid/infer_holdout
```

GA metrics mirror the AE metrics as much as possible: `ssim_hat_vs_clean`, `ssim_hat_vs_noisy`, `ssim_clean_vs_noisy`, `mse_vs_clean`, `psnr_vs_clean`, `runtime_sec`, and `peak_rss_bytes`, plus GA-specific settings such as `patch_size`, population size, generations, crossover probability, and mutation probability.

---
### `optimize-ga-patchwise` / `infer-ga-patchwise`: (B) Patch-wise search geometry

(B) Patch-wise search runs independent GA on each spatial tile (same objective per crop as above), then overlap blending with Hann, triangular, Tukey, or flat weights into a full $\hat{u}_{\mathrm{final}}$. Final reporting uses the fused image against full-image $v$ (and $u$ if provided). This complements (A) `optimize-ga`; it is not a replacement. Use both to compare global and patch-wise search under objectives (1) and (2). Typical output directory: `output/ga_patchwise/`.

Objective (1) blind vs. (2) hybrid (same flags as global GA; `--patch-size` here is the tile side $B$, not the global GA patch genome tile):

```bash
# (B) Patch-wise GA, objective (1): blind per-tile fit to noisy crop; full-image metrics on fused û
image-enhancement optimize-ga-patchwise -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/ga_patchwise \
  --loss-mode blind_ssim --patch-size 64 --stride 32 --generations 30 --population 40

# (B) Patch-wise GA, objective (2): hybrid (pass clean image with -u for aligned tile crops)
image-enhancement optimize-ga-patchwise -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/ga_patchwise_hybrid \
  --loss-mode hybrid_ssim_mse --alpha 0.8 --beta 0.2 --patch-size 64 --stride 32

# (B) Patch-wise inference: reload fused û and recompute metrics
image-enhancement infer-ga-patchwise -c output/ga_patchwise/ga_patchwise_solution.npz -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/ga_patchwise/infer_holdout
```

Artifacts: `ga_patchwise_solution.npz`, `stats/metrics.json`, `stats/history.jsonl`, `images/denoised.tif`.

---
### `optimize-pso` / `infer-pso`: (A) Global search geometry (same as `optimize-ga`)

PSO mirrors global GA (`optimize-ga`): one swarm over a whole-image candidate, objectives (1) and (2) as above, not the (B) patch-wise tile-and-blend path.

- single-image denoising first
- same patch-tiled encoding of the full image as global GA
- outputs under e.g.\ `output/pso/`

```bash
# (A) Global PSO, objective (1): blind
image-enhancement optimize-pso -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/pso --iterations 60 --swarm-size 24 --patch-size 8 --loss-mode blind_ssim

# (A) Global PSO, objective (2): hybrid
image-enhancement optimize-pso -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/pso_hybrid --iterations 60 --swarm-size 24 --patch-size 8 --loss-mode hybrid_ssim_mse --alpha 0.8 --beta 0.2
```

Inference:

```bash
# (A) Global PSO inference, blind checkpoint
image-enhancement infer-pso -c output/pso/pso_solution.npz -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/pso/infer_holdout

# (A) Global PSO inference, hybrid checkpoint
image-enhancement infer-pso -c output/pso_hybrid/pso_solution.npz -v assets/slices_noise_vol/slice_noisy_257.tif -u assets/slices_vol/slice_257.tif -o output/pso_hybrid/infer_holdout
```

PSO metrics mirror GA and AE as closely as possible: `ssim_hat_vs_clean`, `ssim_hat_vs_noisy`, `ssim_clean_vs_noisy`, `mse_vs_clean`, `psnr_vs_clean`, `runtime_sec`, and `peak_rss_bytes`, plus PSO-specific settings such as `patch_size`, swarm size, iterations, inertia, cognitive coefficient, social coefficient, initialization noise, and maximum velocity.

---
### `optimize-pso-patchwise` / `infer-pso-patchwise`: (B) Patch-wise search geometry (PSO)

(B) Patch-wise PSO mirrors `optimize-ga-patchwise`: independent PSO swarm per spatial tile, then overlap blending with Hann (default), triangular, Tukey, or flat weights into a full $\hat{u}_{\mathrm{final}}$. Use `--blend` to switch blending methods and compare metrics across runs. Typical output directory: `output/pso_patchwise/`.

Objective (1) blind vs. (2) hybrid (same flags as global PSO; `--patch-size` here is the tile side $B$; use `--blend` to change the blending method):

```bash
# (B) Patch-wise PSO, objective (1): blind, default Hann blending
image-enhancement optimize-pso-patchwise -v assets/slice_noisy.tif -u assets/slice.tif -o output/pso_patchwise \
  --loss-mode blind_ssim --patch-size 32 --stride 16 --swarm-size 20 --iterations 50

# (B) Patch-wise PSO, objective (2): hybrid SSIM+MSE
image-enhancement optimize-pso-patchwise -v assets/slice_noisy.tif -u assets/slice.tif -o output/pso_patchwise_hybrid \
  --loss-mode hybrid_ssim_mse --alpha 0.8 --beta 0.2 --patch-size 32 --stride 16 --swarm-size 20 --iterations 50

# (B) Patch-wise PSO with different blending modes: change --blend to compare hann / triangular / tukey
image-enhancement optimize-pso-patchwise -v assets/slice_noisy.tif -u assets/slice.tif -o output/pso_patchwise_hann     --blend hann       --patch-size 32 --stride 16 --swarm-size 20 --iterations 50
image-enhancement optimize-pso-patchwise -v assets/slice_noisy.tif -u assets/slice.tif -o output/pso_patchwise_tri      --blend triangular --patch-size 32 --stride 16 --swarm-size 20 --iterations 50
image-enhancement optimize-pso-patchwise -v assets/slice_noisy.tif -u assets/slice.tif -o output/pso_patchwise_tukey    --blend tukey      --patch-size 32 --stride 16 --swarm-size 20 --iterations 50
# (B) Patch-wise inference: reload fused û and recompute metrics
image-enhancement infer-pso-patchwise -c output/pso_patchwise/pso_patchwise_solution.npz -v assets/slice_noisy.tif -u assets/slice.tif -o output/pso_patchwise/infer_holdout
```

Artifacts: `pso_patchwise_solution.npz`, `stats/metrics.json`, `stats/history.jsonl`, `images/denoised.tif`.

## Package layout

- `src/image_enhancement/preprocessing/`: NIfTI export (slice ranges, optional glob), single-file and batch (`noisify-dir`) AWGN, resize
- `src/image_enhancement/common/`: pooled SSIM loss, shared objectives, performance tracking, constraints
- `src/image_enhancement/autoencoders/`: `model.py`, `training.py` (blind SSIM or hybrid SSIM+MSE)
- `src/image_enhancement/genetic_algorithm/`: global whole-image GA (`ga_runner.py`, `optimize-ga`, `infer-ga`)
- `src/image_enhancement/ga_patchwise/`: local per-tile GA with overlap blending (`optimize-ga-patchwise`, `infer-ga-patchwise`)
- `src/image_enhancement/particle_swarm_opt/`: global whole-image PSO (`pso_runner.py`, `optimize-pso`, `infer-pso`)
- `src/image_enhancement/pso_patchwise/`: local per-tile PSO with overlap blending (`tile_pso.py`, `patchwise_pso_runner.py`, `optimize-pso-patchwise`, `infer-pso-patchwise`)

Autoencoder checkpoints and logs are written under `output/ae/images/`, `output/ae/stats/` (`metrics.json`, `history.jsonl`), and `output/ae/autoencoder.pt`.
