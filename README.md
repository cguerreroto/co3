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
# Single slice from NIfTI (axis 1 is often coronal in RAS; inspect with --output-dir first). Output can be .tif/.tiff/.png/.jpg/.jpeg.
image-enhancement preprocess nifti-to-tiff -i assets/volume.nii.gz -o assets/slice.tif --axis 1 --index 257

# Or export all slices along an axis for browsing
image-enhancement preprocess nifti-to-tiff -i assets/volume.nii.gz --output-dir assets/slices --axis 1

# AWGN: clean u to noisy v + JSON sidecar (sigma, epsilon_hint, …). Input and output can be .tif/.tiff/.png/.jpg/.jpeg.
image-enhancement preprocess noisify -i assets/slice.tif -o assets/slice_noisy.tif --sigma 25

# Resize for GA/PSO branch (proportional, max side 128). Input and output can be .tif/.tiff/.png/.jpg/.jpeg.
image-enhancement preprocess resize -i assets/slice.tif -o assets/slice_small.tif --max-side 128

# Note: resize creates u_small only; run preprocess noisify on assets/slice_small.tif with --sigma to produce v_small.

image-enhancement preprocess resize -i assets/slice.tif -o assets/slice_small.tif --max-side 128 \
&& image-enhancement preprocess noisify -i assets/slice_small.tif -o assets/slice_small_noisy.tif --sigma 25
```

Equivalent module invocation:

```bash
python -m image_enhancement.main preprocess noisify -i ... -o ... --sigma 25
```

### CLI help (`argparse`)

Run these in the terminal to print usage and every flag for that command:

```bash
python -m image_enhancement.main --help
python -m image_enhancement.main preprocess --help
```

Preprocess subcommands each have their own options:

```bash
python -m image_enhancement.main preprocess nifti-to-tiff --help
python -m image_enhancement.main preprocess noisify --help
python -m image_enhancement.main preprocess resize --help
```

If the `image-enhancement` script is on your `PATH` (after `pip install -e .`), you can swap `python -m image_enhancement.main` for `image-enhancement` in the commands above (for example `image-enhancement preprocess --help`).
