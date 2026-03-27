# Spectral Denoising using ResNetThreshold1D

This project implements a **1D Residual Shrinkage Network** (`ResNetThreshold1D`) for denoising spectral data (FTIR). The model combines residual connections with learnable per-channel soft-thresholding activations, which naturally suppress noise-like coefficients while preserving meaningful spectral features.

## About The Project

This project provides a complete pipeline for training and evaluating a 1D ResNet with learnable soft-thresholding for spectral denoising. The key innovation over a plain ResNet is the **soft-threshold activation** — each residual block learns its own shrinkage threshold per channel, making the network especially effective at separating signal from noise without a U-Net encoder–decoder structure.

### Architecture highlights

- **`SoftThreshold`** — learnable per-channel soft-thresholding replaces ReLU in every residual block
- **`ResidualShrinkageBlock`** — two Conv1d layers with BatchNorm and soft-thresholding, plus identity skip connection
- **`ResNetThreshold1D`** — head conv → N residual shrinkage blocks → tail conv; uses residual learning (`output = noisy − predicted_noise`)

### Key pipeline components

| Script / Module | Purpose |
| :--- | :--- |
| `models/resnet_threshold.py` | Model definition (ResNetThreshold1D) |
| `scripts/train_resnet_threshold.py` | Train the model |
| `scripts/evaluate_resnet_threshold.py` | Evaluate the trained model |
| `scripts/Downstream Task Evaluation.py` | RF classifier comparison (denoising methods) |
| `scripts/baseline.py` | ASLS baseline correction |
| `scripts/make_pairs.py` | Build clean/noisy .npy pairs |
| `scripts/load_qc.py` | QC-checked dataset loader with train/test split |
| `scripts/augment_data.py` | On-the-fly data augmentation utilities |
| `scripts/create_noisy_data.py` | Generate synthetic noisy spectra |
| `scripts/metrics.py` | General metric helpers (PSNR, SSIM, NRMSE) |
| `scripts/spectral_metrics.py` | Spectral-specific metrics (peak shift, FWHM, band area) |
| `scripts/classifier_evaluate.py` | Random Forest training and evaluation helpers |
| `scripts/audit_trail.py` | Timestamped audit logging |
| `scripts/visualize_results.py` | Overlay plot of raw / corrected / denoised spectra |

## Getting Started

### Prerequisites

`conda` (Anaconda or Miniconda) with Python 3.10.

### Installation

1. **Clone the repo**
   ```sh
   git clone https://github.com/nabhya8013/spectral_denoise.git
   cd spectral_denoise
   ```
2. **Create and activate the Conda environment**
   ```sh
   conda create -n spectral_env python=3.10
   conda activate spectral_env
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## Data Preparation

Place raw `.txt` spectra (two-column: wavenumber, intensity) in `data/raw/` and
baseline-corrected files in `data/processed/`. Then run:

```sh
python scripts/baseline.py          # baseline-correct raw spectra
python scripts/make_pairs.py        # create clean/noisy .npy pairs  → data/pairs/
python scripts/create_noisy_data.py # add synthetic noise to clean spectra
```

## Training

```sh
python scripts/train_resnet_threshold.py
```

The script will:

- Load all `_clean.npy` / `_noisy.npy` pairs from `data/pairs/`
- Apply on-the-fly augmentation (noise, baseline shifts, spikes)
- Train `ResNetThreshold1D` with a hybrid MSE + cosine-similarity loss
- Save the trained model to `models/resnet_threshold1d.pth`
- Save evaluation metrics to `results/resnet_threshold_metrics.json`

## Evaluation

```sh
python scripts/evaluate_resnet_threshold.py
```

Loads the trained model, runs inference on the held-out 20 % validation split,
and reports:

| Metric | Description |
| :--- | :--- |
| Mean MSE | Mean squared error (lower is better) |
| Mean PSNR | Peak signal-to-noise ratio in dB (higher is better) |
| Mean SSIM | Structural similarity index (higher is better) |
| Mean Corr | Pearson correlation (higher is better) |
| Overall Quality | Weighted composite score (%) |

Results are saved to `results/resnet_threshold_metrics.json`.

## Inference example

```python
import torch
import numpy as np
from scipy.signal import resample
from models.resnet_threshold import ResNetThreshold1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetThreshold1D(in_channels=1, hidden_channels=64, num_blocks=8).to(device)
model.load_state_dict(torch.load("models/resnet_threshold1d.pth", map_location=device, weights_only=True))
model.eval()

raw_spectrum = np.loadtxt("path/to/your/spectrum.txt")[:, 1]  # intensity column
resampled = resample(raw_spectrum, 1024).astype(np.float32)
noisy_tensor = torch.from_numpy(resampled).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    denoised_tensor = model(noisy_tensor)

denoised_spectrum = denoised_tensor.cpu().squeeze().numpy()
```

## Downstream Evaluation

Run `scripts/Downstream Task Evaluation.py` to compare the ResNetThreshold1D denoiser
against Savitzky–Golay and wavelet baselines using a downstream Random Forest classifier.
Labels must be available in `data/dataset/` (produced by `load_qc.py`).
