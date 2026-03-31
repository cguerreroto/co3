# Image enhancement (SSIM denoising)

Python package for pooled SSIM denoising experiments: NIfTI-to-image preprocessing, AWGN noisification, optional resize for genetic algorithm (GA) or particle swarm optimization (PSO) branches, and a convolutional autoencoder trained to minimize $1 - \mathcal{S}(\hat{u}, v)$, where $\mathcal{S}$ is the mean local SSIM between the denoised estimate $\hat{u}$ and the noisy observation $v$.

## Setup

### 1. Clone or open the repo and create a virtual environment (venv)

```bash
cd /<path_to>/co3            # modify <path_to> according to your actual path to this repository
python -m venv .co3
source .co3/bin/activate   # Windows: .co3\Scripts\activate
```

Keep the venv activated whenever you work on this project (each new terminal: `cd .../co3` and `source .co3/bin/activate` again).

### 2. Install this project and its dependencies (recommended)

Run exactly this from the repository root (the folder that contains `pyproject.toml`):

```bash
pip install --upgrade pip
pip install -e .
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
- Outputs default to `output/ae/`, `output/ga/`, `output/pso/` for autoencoder, genetic algorithm, and particle swarm runs respectively.

## Commands

```bash
# ----------------------------------------------- convert nifti to tiff (jpg/jpeg/png) -----------------------------------------------

# Single slice from NIfTI (axis 1 is often coronal in RAS; inspect with --output-dir first). Output can be .tif/.tiff/.png/.jpg/.jpeg.
image-enhancement preprocess nifti-to-tiff -i assets/volume.nii.gz -o assets/slice.tif --axis 1 --index 257

# Or export all slices along an axis for browsing
image-enhancement preprocess nifti-to-tiff -i assets/volume.nii.gz --output-dir assets/slices --axis 1

# Autoencoders: Only slices 250–350 (inclusive), or many volumes via glob (each under output-root/<stem>/)
image-enhancement preprocess nifti-to-tiff -i assets/volume.nii.gz --output-dir assets/slices_vol --axis 1 --index-start 250 --index-end 350

image-enhancement preprocess nifti-to-tiff --input-glob "assets/*.nii.gz" --output-root assets/slices_by_volume --output-dir assets/_argparse_dummy --axis 1 --index-start 250 --index-end 350


# -------------------------------------------------------- Preprocess: noisify --------------------------------------------------------

# AWGN for one image (u -> v) + JSON sidecar (sigma, epsilon_hint, ...).  Input and output can be .tif/.tiff/.png/.jpg/.jpeg.
image-enhancement preprocess noisify -i assets/slice.tif -o assets/slice_noisy.tif --sigma 25

# AWGN for an entire folder tree: mirror all images and keep relative paths
# assets/slices_vol/...      -> assets/slices_noise_vol/...
# assets/slices_by_volume/... -> assets/slices_noise_by_volume/...
# filenames are renamed like: slice_257.tif -> slice_noisy_257.tif (+ sidecar .json)
image-enhancement preprocess noisify-dir --clean-dir assets/slices_vol --noisy-dir assets/slices_noise_vol --sigma 25
image-enhancement preprocess noisify-dir --clean-dir assets/slices_by_volume --noisy-dir assets/slices_noise_by_volume --sigma 25


# -------------------------------------------------------- Preprocess: resize --------------------------------------------------------

# Resize for GA/PSO branch (proportional, max side 128). Input and output can be .tif/.tiff/.png/.jpg/.jpeg.
image-enhancement preprocess resize -i assets/slice.tif -o assets/slice_small.tif --max-side 128

# Note: resize creates u_small only; run preprocess noisify on assets/slice_small.tif with --sigma to produce v_small.

image-enhancement preprocess resize -i assets/slice.tif -o assets/slice_small.tif --max-side 128 \
&& image-enhancement preprocess noisify -i assets/slice_small.tif -o assets/slice_small_noisy.tif --sigma 25


# ------------------------------------------------------- Autoencoders: training -------------------------------------------------------

# Train autoencoder on noisy TIFF; optional clean TIFF for eval metrics
image-enhancement train-ae -v assets/slice_noisy.tif -u assets/slice.tif -o output/ae --epochs 200

# Optional residual-energy soft penalty (discourages identity hat(u)=v); epsilon ~ n*sigma^2 from noisify JSON
image-enhancement train-ae -v assets/slice_noisy.tif -u assets/slice.tif -o output/ae \
  --lambda-residual 1e-4 --noise-meta assets/slice_noisy.tif.json

# Multi-pair training (mirrored clean/noisy directories) — one Adam step per image per epoch; metrics are epoch means
image-enhancement train-ae --clean-dir assets/slices_by_volume --noisy-dir assets/slices_noise_by_volume -o output/ae --epochs 50 --shuffle

# Or JSONL manifest: one {"noisy":"...","clean":"..."} per line. Hold out a slice by listing it in exclude.txt
image-enhancement train-ae --manifest train_pairs.jsonl --exclude-from-train exclude.txt -o output/ae


# ------------------------------------------------------- Autoencoders: infer  -------------------------------------------------------

# After training, denoise one held-out noisy slice (writes denoised.tif + infer_metrics.json)
# Example using generated folders (replace volume/slice as needed):
image-enhancement infer-ae -c output/ae/autoencoder.pt -v assets/slices_noise_by_volume/coronacases_002/slice_noisy_257.tif -u assets/slices_by_volume/coronacases_002/slice_257.tif -o output/ae/infer_holdout
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
```

Preprocess subcommands each have their own options:

```bash
python -m image_enhancement.main preprocess nifti-to-tiff --help
python -m image_enhancement.main preprocess noisify --help
python -m image_enhancement.main preprocess noisify-dir --help
python -m image_enhancement.main preprocess resize --help
```

If the `image-enhancement` script is on your `PATH` (after `pip install -e .`), you can swap `python -m image_enhancement.main` for `image-enhancement` in the commands above (for example `image-enhancement preprocess --help`).

### `train-ae` outputs and metrics

Training minimizes pooled SSIM loss between $\hat{u}$ and $v$ (blind objective). **Single-image mode** (`-v` / `--noisy`): one row per epoch in `history.jsonl`. **Multi-pair mode** (`--manifest` or `--clean-dir` + `--noisy-dir`): each epoch performs one optimization step per training pair (order optionally shuffled with `--shuffle`); logged `loss` is the **mean** training loss over pairs; SSIM/PSNR columns are **means over all pairs with clean references**. Validation pairs (`--val-manifest`) add `val_*` columns. Outputs include `autoencoder.pt`, `stats/metrics.json`, and `images/denoised_sample.tif` (first manifest pair after training). For an unbiased test slice, exclude it from training (`--exclude-from-train`) and run **`infer-ae`**.

When you pass `--clean` / `-u` (single mode) or use multi-pair mode with clean references, logs and `output/ae/stats/metrics.json` include full-reference evaluation (not part of the loss):

| Field | Meaning |
| ----- | ------- |
| `ssim_hat_vs_clean` | Pooled SSIM $(\hat{u},u)$. Primary quality vs ground truth |
| `ssim_vs_clean` | Same value as `ssim_hat_vs_clean` (backward-compatible alias) |
| `ssim_hat_vs_noisy` | Pooled SSIM $(\hat{u},v)$. Similarity to the noisy input |
| `ssim_clean_vs_noisy` | Pooled SSIM $(u,v)$. Noise difficulty baseline (also on epoch 1 in `history.jsonl`) |
| `psnr_vs_clean` | PSNR $(u,\hat{u})$. Classical baseline (monotone in MSE) |

`metrics.json` also records `window_size`, `lambda_residual`, `epsilon`, and paths. PSNR is sufficient if you want a classical scalar alongside SSIM; raw MSE is redundant with PSNR on the same pair unless you need it for statistics.

## Package layout

- `src/image_enhancement/preprocessing/`: NIfTI export (slice ranges, optional glob), single-file and batch (`noisify-dir`) AWGN, resize
- `src/image_enhancement/common/`: pooled SSIM loss, constraints
- `src/image_enhancement/autoencoders/`: `model.py`, `training.py` (SSIM loss vs $v$)

Autoencoder checkpoints and logs are written under `output/ae/images/`, `output/ae/stats/` (`metrics.json`, `history.jsonl`), and `output/ae/autoencoder.pt`.
