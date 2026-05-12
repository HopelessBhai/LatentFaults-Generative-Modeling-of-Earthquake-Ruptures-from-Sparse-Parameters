# latent-faults-slipgen

<p align="center">
  <b>latent-faults-slipgen</b><br/>
  Latent-space surrogate model for stochastic earthquake slip generation from sparse source parameters
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Repo-latent--faults--slipgen-111827?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-111827?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Interface-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-Seismology-0ea5e9?style=for-the-badge" />
</p>

---

## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Quick Verification Test](#quick-verification-test)
- [Pipeline Runbook](#pipeline-runbook)
  - [Stage 1 — VQ-VAE Training](#stage-1--vq-vae-training)
  - [Stage 2 — Hyperparameter Search (optional)](#stage-2--hyperparameter-search-optional)
  - [Stage 3 — Mapper and Decoder Training](#stage-3--mapper-and-decoder-training)
  - [Stage 4 — Batch Inference and Evaluation](#stage-4--batch-inference-and-evaluation)
- [Interactive App](#interactive-app)
- [Script Reference (Inputs, Outputs, Options)](#script-reference-inputs-outputs-options)
- [Artifacts Map](#artifacts-map)
- [Repository Layout](#repository-layout)
- [Reproducibility Notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository implements a two-stage deep generative pipeline that learns to produce spatially
distributed earthquake slip fields from sparse geophysical source parameters:

1. **Representation learning** — a Vector-Quantised Variational AutoEncoder (VQ-VAE) compresses
   50 × 50 normalised slip images into a discrete latent space of size 13 × 13 × 16.
2. **Conditional generation** — a feed-forward mapping network (FFN) predicts the latent code
   from sparse source parameters; a CLIP-style contrastive alignment loss enforces structural
   consistency between predicted and reference latent codes; the VQ-VAE decoder then
   reconstructs the slip field.

The code accompanies the manuscript:

> Nayak, M., Goswami, A., Neelamraju, P. M., and Raghukanth, S. T. G.  
> *Latent Faults: A Latent-Space Surrogate Model for Stochastic Earthquake Slip Generation
> from Sparse Source Parameters.*  
> Computers & Geosciences (under review).

### Headline results

| Metric | Reported value |
|---|---|
| Dataset | 200 SRCMOD events, Mw 5.8–9.2 |
| Grid standardisation | 50 × 50 cells |
| Codebook size K | 512 |
| 2D PSD correlation (proposed vs. reference) | 0.93 |
| Radial PSD correlation | 0.96 |
| Inference time per realisation | ~10–20 ms (GPU) |

---

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Any CUDA-capable NVIDIA GPU | NVIDIA RTX 3090 (24 GB VRAM) |
| GPU VRAM | 4 GB | 8 GB |
| CPU RAM | 8 GB | 16 GB |
| Disk space | 2 GB (code + models) | 10 GB (code + models + dataset) |

> **CPU-only mode**: All scripts fall back to CPU automatically when no GPU is available,
> but training will be significantly slower.

### Software

| Component | Version |
|---|---|
| Operating system | Linux (recommended), macOS, Windows |
| Python | 3.9 or later |
| CUDA (optional) | 11.8 or later |
| cuDNN (optional) | 8.6 or later |

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/HopelessBhai/latent-faults-slipgen.git
cd latent-faults-slipgen
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users**: the `requirements.txt` installs the CUDA-enabled build of PyTorch.
> If a different CUDA version is needed, install PyTorch separately first:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```

---

## Data Requirements

This project uses the **SRCMOD Finite-Source Rupture Model Database** as its training data
source. SRCMOD is a publicly available repository maintained by ETH Zurich.

### Obtaining the data

1. Visit the SRCMOD database:
   [https://www.seismo.ethz.ch/static/srcmod/Homepage.html](https://www.seismo.ethz.ch/static/srcmod/Homepage.html)
2. Download `.fsp` files for the desired earthquake events.
3. Preprocess the `.fsp` files to extract slip grids and source parameters (preprocessing
   notebooks are excluded from this release to keep the repository concise; the pipeline
   starts from already-extracted data as described below).

### Required input files

| Path | Description |
|---|---|
| `Dataset/text_vec.npy` | NumPy dict mapping event key → 16-dim feature vector |
| `Dataset/filtered_images_train/` | Preprocessed 50 × 50 PNG slip images (training) |
| `Dataset/filtered_images_test/` | Preprocessed 50 × 50 PNG slip images (test) |
| `assets/dz.json` | JSON mapping event key → Dz value (subfault width, km) |

### Feature vector format (`text_vec.npy`)

Each entry in `text_vec.npy` is a 16-dimensional array with the following ordered fields:

| Index | Name | Description | Unit |
|---|---|---|---|
| 0 | LAT | Epicentre latitude | degrees |
| 1 | LON | Epicentre longitude | degrees |
| 2 | DEP | Hypocentre depth | km |
| 3 | STRK | Fault strike angle | degrees |
| 4 | DIP | Fault dip angle | degrees |
| 5 | RAKE | Rake angle | degrees |
| 6 | LEN_f | Fault length | km |
| 7 | WID | Fault width | km |
| 8 | Htop | Depth to top of rupture | km |
| 9 | HypX | Hypocenter along-strike position | km |
| 10 | HypZ | Hypocenter along-dip position | km |
| 11 | Nx | Number of subfaults along strike | — |
| 12 | Nz | Number of subfaults along dip | — |
| 13 | Dx | Subfault length | km |
| 14 | Dz | Subfault width | km |
| 15 | Mw | Moment magnitude | — |

### Image file naming convention

Training images must follow the pattern:

```
interpolated_slip_image_<event_key>.fsp.png
```

where `<event_key>` matches keys in `text_vec.npy` and `dz.json`.

---

## Quick Verification Test

The following command verifies that the installation is correct and the core model
architecture is functional. **No dataset is required.**

```bash
python - <<'EOF'
import torch
import numpy as np
from train_vqvae import VQVAE

print("=== latent-faults-slipgen — installation check ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Test VQ-VAE with synthetic random slip maps
model = VQVAE(latent_dim=16, num_embeddings=512).to(device)
x = torch.rand(4, 1, 50, 50).to(device)          # 4 synthetic 50x50 slip maps
recon, vq_loss = model(x)
assert recon.shape == (4, 1, 50, 50), "Unexpected reconstruction shape"
print(f"VQ-VAE:  input {tuple(x.shape)} -> recon {tuple(recon.shape)}, VQ loss={vq_loss.item():.4f}")

# 2. Test LatentNN mapping network
from latent_mapper import LatentNN
mapper = LatentNN(input_dim=16, hidden_dims=[256], output_dim=2704, dropout_prob=0.1).to(device)
params = torch.rand(4, 16).to(device)              # 4 synthetic feature vectors
latent_pred = mapper(params)
assert latent_pred.shape == (4, 2704), "Unexpected latent shape"
print(f"Mapper:  input {tuple(params.shape)} -> latent {tuple(latent_pred.shape)}")

print("=== All checks passed. Installation is verified. ===")
EOF
```

Expected output:

```
=== latent-faults-slipgen — installation check ===
Device: cuda            # or "cpu"
VQ-VAE:  input (4, 1, 50, 50) -> recon (4, 1, 50, 50), VQ loss=0.xxxx
Mapper:  input (4, 16) -> latent (4, 2704)
=== All checks passed. Installation is verified. ===
```

---

## Pipeline Runbook

### Stage 1 — VQ-VAE Training

**Script:** `train_vqvae.py`

```bash
python train_vqvae.py
```

**What it does:**
1. Loads all PNG images from `Dataset/filtered_images_train/`.
2. Trains the VQ-VAE (encoder + codebook + decoder) to reconstruct 50 × 50 slip maps.
3. Extracts and saves latent embeddings for all training images.

**Key configurable constants (edit at the top of the script):**

| Constant | Default | Description |
|---|---|---|
| `heatmap_dir` | `Dataset/filtered_images_train` | Training image directory |
| `latent_dim` | 16 | Number of latent channels |
| `num_embeddings` | 512 | Codebook size K |
| `epochs` | 1000 | Maximum training epochs |
| `lr` | 1e-4 | Learning rate |
| `val_split` | 0.2 | Validation fraction |
| `batch_size` | 16 | Batch size |

**Outputs:**

| Path | Description |
|---|---|
| `models/vqvae_finetuned.pth` | Best model weights (lowest validation loss) |
| `embeddings/image_latents.pkl` | Dict: event key → flattened latent vector (2704-dim) |
| `plots/loss_plot_train_vqvae.png` | Training and validation loss curves |

---

### Stage 2 — Hyperparameter Search (optional)

**Script:** `tune_mapper.py`

```bash
python tune_mapper.py
```

**What it does:**  
Runs an Optuna hyperparameter search over the mapping network's learning rate, dropout
probability, L1 regularisation coefficient, and hidden layer sizes. Uses the VQ-VAE
trained in Stage 1 (must be available at `models/vqvae_finetuned.pth`).

**Key configurable constants:**

| Constant | Default | Description |
|---|---|---|
| `TEXT_EMBED_PATH` | `Dataset/text_vec.npy` | Source parameter vectors |
| `IMAGE_DIR` | `Dataset/filtered_images_train` | Training images |
| `BATCH_SIZE` | 32 | Batch size |
| `TEST_SPLIT` | 0.2 | Validation fraction |
| `n_trials` | 100 | Number of Optuna trials |
| `tuning_epochs` | 10000 | Max epochs per trial |

**Outputs:**

| Path | Description |
|---|---|
| `models/best_hyperparams.json` | Best hyperparameters found by Optuna |

> **Note:** This stage is optional. If you prefer to use fixed hyperparameters, you can
> manually create or edit `models/best_hyperparams.json`. A template is provided at
> `models/fixed_hyperparams_manual.json`.

---

### Stage 3 — Mapper and Decoder Training

**Script:** `train_mapper_decoder.py`

```bash
python train_mapper_decoder.py
```

**What it does:**  
Trains the feed-forward mapping network (FFN) and fine-tunes the VQ-VAE decoder jointly.
The VQ-VAE encoder and codebook are frozen. The training objective combines:
- MSE reconstruction loss in the spatial domain.
- CLIP-style contrastive alignment loss in the latent domain.
- L1 weight regularisation on the mapping network.

**Key configurable constants:**

| Constant | Default | Description |
|---|---|---|
| `text_embed_path` | `Dataset/text_vec.npy` | Source parameter vectors |
| `image_dir` | `Dataset/filtered_images_train` | Training images |
| `OUTPUT_DIM` | 2704 | Latent output dimension (16 × 13 × 13) |
| `TEST_SPLIT` | 0.2 | Validation fraction |
| `N_EPOCHS` | 10000 | Maximum training epochs |
| `PATIENCE` | 2000 | Early-stopping patience |

Hyperparameters are loaded from `models/best_hyperparams.json` (produced by Stage 2 or
supplied manually).

**Outputs:**

| Path | Description |
|---|---|
| `models/latent_model.pth` | Best mapping network weights |
| `models/decoder_model.pth` | Best decoder weights |
| `scaler_x.pkl` | Fitted StandardScaler for the input features |
| `plots/loss_plot_pipeline.png` | Training and validation loss curves |

---

### Stage 4 — Batch Inference and Evaluation

**Script:** `run_inference.py`

```bash
python run_inference.py
```

**What it does:**  
Iterates over all images in `Dataset/filtered_images_test/`, generates a predicted slip
field for each event using the trained pipeline, and saves slip arrays and diagnostic plots.

**Expected inputs:**

| Path | Description |
|---|---|
| `Dataset/text_vec.npy` | Feature vectors (must include test events) |
| `Dataset/filtered_images_test/` | Ground-truth test images |
| `models/latent_model.pth` | Trained mapping network |
| `models/decoder_model.pth` | Trained decoder |
| `scaler_x.pkl` | Feature scaler from Stage 3 |
| `models/best_hyperparams.json` | Hyperparameter config |
| `assets/dz.json` | Dz values for slip-unit conversion |

**Outputs:**

| Path | Description |
|---|---|
| `Dataset/predicted_images_LAT_LON/` | Predicted slip images (PNG) |
| `Dataset/slip_arrays_inference/` | Predicted slip arrays (`.npy`) |
| `error_metrics/<key>_error_metrics.json` | Per-event error metrics |
| `test_metrics.json` | Aggregate metrics across the test set |

**Output metrics (per event and aggregate):**

| Key | Description |
|---|---|
| `max_error` | Maximum absolute error (slip units) |
| `mean_error` | Mean absolute error (slip units) |
| `min_error` | Minimum absolute error |
| `std_error` | Standard deviation of the error |

---

## Interactive App

**Script:** `interactive_slip_app.py`

```bash
streamlit run interactive_slip_app.py
```

Opens a browser-based GUI where you can:

1. Adjust moment magnitude (Mw), fault strike, dip, and rake with sliders.
2. Set additional parameters (latitude, longitude, depth, subfault geometry).
3. Optionally fix a random seed for reproducible stochastic sampling.
4. Enable Dz scaling to display the slip field in physical units (metres).
5. View the generated slip map with along-strike / down-dip axis labels.

**Required files for the app:**

| Path | Description |
|---|---|
| `Dataset/text_vec.npy` | Used to infer input dimensionality and slider ranges |
| `models/latent_model.pth` | Mapping network weights |
| `models/decoder_model.pth` | Decoder weights |
| `models/best_hyperparams.json` | Architecture config |
| `scaler_x.pkl` | Feature scaler |
| `assets/dz.json` | Dz values (optional; enables physical slip units) |

**App parameters explained:**

| Parameter | Description |
|---|---|
| Mw | Moment magnitude (1–10). Controls rupture dimensions via scaling laws. |
| STRK | Fault strike angle (degrees from North, 0–360). |
| DIP | Fault dip angle (degrees from horizontal, 0–90). |
| RAKE | Rake angle (degrees, −180 to 180; describes slip direction). |
| LAT / LON | Epicentre geographic coordinates. |
| DEP | Hypocentre depth (km). |
| Nx / Nz / Dx | Subfault discretisation parameters. |
| Dz | Subfault width (km); used for slip-unit conversion when enabled. |
| Random seed | Seeds the truncated-normal sampling for derived parameters. |

> **Note:** Fault length, fault width, hypocenter position, and top-of-rupture depth are
> computed automatically from Mw using empirical scaling relations and statistical sampling.

---

## Script Reference (Inputs, Outputs, Options)

### `train_vqvae.py`

| Item | Details |
|---|---|
| **Input** | PNG images in `Dataset/filtered_images_train/` |
| **Output** | `models/vqvae_finetuned.pth`, `embeddings/image_latents.pkl` |
| **Imports** | `VQVAE`, `VectorQuantizer`, `ImageDataset`, `fine_tune_vqvae`, `extract_latents` |
| **Key classes** | `VQVAE(latent_dim, num_embeddings, beta)`: full model; `VectorQuantizer(num_embeddings, embedding_dim, beta)`: codebook module |

### `latent_mapper.py`

| Item | Details |
|---|---|
| **Input** | `Dataset/text_vec.npy` (features), `embeddings/image_latents.pkl` (targets) |
| **Output** | `models/latent_mapper.pth` (standalone mapper, if used directly) |
| **Key classes** | `LatentNN(input_dim, hidden_dims, output_dim, dropout_prob)`: the FFN mapper |
| **Key functions** | `prepare_dataloaders(...)`, `train(...)`, `evaluate(...)` |

### `decoder.py`

| Item | Details |
|---|---|
| **Input** | Latent embedding tensor of shape [B, 2704] |
| **Output** | Reconstructed slip image of shape [B, 1, 50, 50] |
| **Key classes** | `Decoder(model_weights_path, data_dir, device)`: wraps VQVAE decoder; exposes `forward()`, `visualize_prediction()`, `get_lat_lon_from_image()` |

### `train_mapper_decoder.py`

| Item | Details |
|---|---|
| **Input** | `Dataset/text_vec.npy`, `Dataset/filtered_images_train/`, `embeddings/image_latents.pkl` |
| **Output** | `models/latent_model.pth`, `models/decoder_model.pth`, `scaler_x.pkl` |
| **Key functions** | `prepare_dataloaders(...)`, `train(latent, decoder, ...)`, `evaluate(latent, decoder, ...)` |

### `tune_mapper.py`

| Item | Details |
|---|---|
| **Input** | Same as `train_mapper_decoder.py` plus trained VQ-VAE |
| **Output** | `models/best_hyperparams.json` |
| **Options** | `n_trials` in `main()` sets the number of Optuna trials |

### `run_inference.py`

| Item | Details |
|---|---|
| **Input** | Trained model weights, `Dataset/filtered_images_test/`, `Dataset/text_vec.npy` |
| **Output** | Slip PNG images, `.npy` arrays, error metric JSONs |
| **Key classes** | `Inference(latent_model_path, decoder_model_path)`: loads and exposes `generate(event_key, actual_image_path, save_path)` |

### `interactive_slip_app.py`

| Item | Details |
|---|---|
| **Input** | Trained model weights, `Dataset/text_vec.npy`, `scaler_x.pkl` |
| **Output** | Browser-rendered slip map; optional slip array download |
| **Run** | `streamlit run interactive_slip_app.py` |

---

## Artifacts Map

| Artifact | Path | Produced by |
|---|---|---|
| VQ-VAE weights | `models/vqvae_finetuned.pth` | `train_vqvae.py` |
| Image latent dictionary | `embeddings/image_latents.pkl` | `train_vqvae.py` |
| Mapping network weights | `models/latent_model.pth` | `train_mapper_decoder.py` |
| Decoder weights | `models/decoder_model.pth` | `train_mapper_decoder.py` |
| Input feature scaler | `scaler_x.pkl` | `train_mapper_decoder.py` |
| Best hyperparameters | `models/best_hyperparams.json` | `tune_mapper.py` |
| Inference slip arrays | `Dataset/slip_arrays_inference/*.npy` | `run_inference.py` |
| Error metrics | `error_metrics/*.json`, `test_metrics.json` | `run_inference.py` |
| Loss plots | `plots/*.png` | training scripts |

---

## Repository Layout

```
latent-faults-slipgen/
│
├── assets/
│   ├── dz.json                        # Event key -> Dz value map
│   ├── normalizing_slip_range.npy     # Per-event slip normalisation range
│   └── utils.py                       # Shared utilities (losses, plotting, slip scaling)
│
├── models/
│   ├── best_hyperparams.json          # Hyperparameters from Optuna search
│   ├── fixed_hyperparams.json         # Alternative fixed hyperparameters
│   └── fixed_hyperparams_manual.json  # Manually tuned hyperparameter template
│
├── report_pics/                       # Figures used in this README
│   ├── full_pipeline.png
│   ├── VQVAE.png
│   ├── clip_loss.png
│   └── sim_combined.png
│
├── train_vqvae.py          # Stage 1: VQ-VAE training + latent extraction
├── latent_mapper.py        # Stage 2/3: FFN mapper definition + standalone training
├── train_mapper_decoder.py # Stage 3: Joint mapper + decoder training (main pipeline)
├── tune_mapper.py          # Stage 2: Optuna hyperparameter search
├── run_inference.py        # Stage 4: Batch inference + evaluation
├── decoder.py              # Decoder wrapper with visualisation utilities
├── interactive_slip_app.py # Streamlit interactive slip map generator
│
├── requirements.txt
├── LICENSE
└── README.md
```

> `Dataset/`, `embeddings/`, `scaler_x.pkl`, `models/*.pth`, and
> analysis notebooks are excluded from version control via `.gitignore`
> to keep the repository size manageable.

---

## Reproducibility Notes

### Reproducing the paper results

The paper reports results on 200 SRCMOD finite-fault rupture models. Full reproduction
requires:

1. Download `.fsp` files for events with `5.8 ≤ Mw ≤ 9.2` from SRCMOD.
2. Run the preprocessing pipeline (to extract feature vectors into `text_vec.npy`
   and normalised PNG slip images).
3. Run Stages 1–4 in order with the hyperparameters from `models/best_hyperparams.json`.

Key hyperparameters reported in the paper and used by this code:

| Hyperparameter | Value |
|---|---|
| Latent dimension (h × w × c) | 13 × 13 × 16 |
| Codebook size K | 512 |
| Commitment weight β | 0.25 |
| Contrastive temperature τ | 0.07 |
| Learning rate | 1 × 10⁻³ |
| Batch size | 16 |
| Optimiser | Adam |
| Training epochs | 40 (convergence) |
| Regularisation coefficient λ | 0.25 |

### Key consistency requirements

- **`scaler_x.pkl` must match the model weights**: the scaler is fit on the training split
  used for Stage 3. Using a scaler from a different run will produce incorrect results.
- **Event-key naming must be consistent** across `text_vec.npy`, image filenames,
  and `assets/dz.json`.
- **`assets/normalizing_slip_range.npy`** is required by `assets/utils.py` for the
  `pixels_to_slip()` function. This file encodes the per-event min/max slip values
  needed to convert normalised pixel intensities back to physical slip (metres).
- **Codebook size consistency**: `train_vqvae.py`, `decoder.py`, and any saved weights
  must all use the same `num_embeddings` value (512 for paper results). Loading
  weights trained with a different codebook size will raise a shape mismatch error.

---

## Architecture Diagrams

<p align="center">
  <img src="report_pics/full_pipeline.png" alt="End-to-end latent-fault generation pipeline" width="96%">
</p>

<p align="center">
  <img src="report_pics/VQVAE.png" alt="VQ-VAE architecture" width="96%">
</p>

<details>
<summary><b>Contrastive alignment loss (CLIP-style)</b></summary>
<br/>
<p align="center">
  <img src="report_pics/clip_loss.png" alt="Contrastive alignment" width="78%">
</p>
</details>

<details>
<summary><b>Representative generated slip maps</b></summary>
<br/>
<p align="center">
  <img src="report_pics/sim_combined.png" alt="Qualitative slip map results" width="96%">
</p>
</details>

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: Dataset/text_vec.npy` | Dataset not yet prepared | Follow the [Data Requirements](#data-requirements) section |
| `RuntimeError: Error(s) in loading state_dict` | Model weights and architecture mismatch | Ensure `num_embeddings` is identical in training and inference (default: 512) |
| Very few matched samples during training | Key mismatch between `text_vec.npy` and image filenames | Verify naming convention: `interpolated_slip_image_<key>.fsp.png` |
| Slip values look incorrect or out of range | Wrong `Dz` value or missing `normalizing_slip_range.npy` | Check `assets/dz.json` keys and `assets/normalizing_slip_range.npy` |
| Unstable training or NaN loss | Learning rate too high or batch size too small | Use `tune_mapper.py` to find stable hyperparameters; or reduce `lr` in `models/best_hyperparams.json` |
| `ModuleNotFoundError: No module named 'assets'` | Script run from wrong directory | Run all scripts from the repository root: `cd latent-faults-slipgen && python train_vqvae.py` |
| Streamlit app crashes on startup | Missing model weights or `Dataset/text_vec.npy` | Complete Stages 1–3 before launching the app |

---

## Data Source

- SRCMOD finite-fault database:
  [https://www.seismo.ethz.ch/static/srcmod/Homepage.html](https://www.seismo.ethz.ch/static/srcmod/Homepage.html)

---

## Citation

If you use this code, please cite:

```bibtex
@article{nayak2025latentfaults,
  title   = {Latent Faults: A Latent-Space Surrogate Model for Stochastic
             Earthquake Slip Generation from Sparse Source Parameters},
  author  = {Nayak, Manish and Goswami, Atmadip and
             Neelamraju, Pavan Mohan and Raghukanth, S. T. G.},
  journal = {Computers \& Geosciences},
  year    = {2025},
  note    = {Under review}
}
```

Repository: [https://github.com/HopelessBhai/latent-faults-slipgen](https://github.com/HopelessBhai/latent-faults-slipgen)

---

## License

This project is released under the **MIT License**. See `LICENSE` for full terms.
