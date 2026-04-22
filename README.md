# Spectral Denoising using ResUNet1D

This project implements a **1D Residual U-Net** (`ResUNet1D`) tailored for achieving high-precision denoising of spectral data (e.g., FTIR spectra). The model goes beyond generic denoising by incorporating architectural enhancements and spectral-specific loss functions designed comprehensively to prevent peak positional shifts and prioritize feature fidelity in spectral ranges (like the fingerprint region).

## About The Project

This pipeline trains and evaluates a single-stage `ResUNet1D` to separate spectral signal from noise. The key focus here is avoiding the "peak shift" and "peak rounding" artifacts commonly seen in standard MSE-driven regression, specifically achieving a high-quality reconstruction in complex, feature-rich spectral datasets.

### Architecture Highlights

- **`MultiScaleContext1D`** — captures local and regional structural relationships simultaneously across multiple kernel dilations/scales.
- **`AttentionGate1D (Skip Gates)`** — selectively passes spatial features through skip connections based on decoder gating signals.
- **`SqueezeExcite1D`** — per-channel attention within residual blocks to calibrate channel relevance.
- **`ResidualConvBlock1D`** — heavily incorporates GelU activations, group normalization, and skip connections for stable deep feature propagation.
- **`AdaptiveSpikeSuppressor & Local Refiners`** — handle sudden artifact spikes and post-hoc local details.

### Advanced Spectral Loss Toolkit

Housed in `spectral_losses.py`, the model optimizes against a vast suite of metrics:
- **Peak Alignment Loss** ensures reconstructed absorption peaks center precisely on the target.
- **Fingerprint-specific L1/MSE/D1** strongly enforces accuracy in the critical 104-726 fingerprint index range.
- **Curvature (D2) and Slope (D1)** derivatives alongside FFT loss match the overall topology and high-frequency details.
- **Valley/Peak Amplitudes** penalize "undershooting" deep troughs to preserve sharp peak shapes.

## Key Pipeline Components

| Script / Module | Purpose |
| :--- | :--- |
| `models/resunet1d.py` | Model definition (ResUNet1D) |
| `scripts/train_resunet.py` | Configure and train the model using composite spectral losses |
| `scripts/evaluate_resunet.py` | Evaluate the trained model precisely over valid holdout sets |
| `scripts/baseline.py` | Generate baseline-corrected spectra from raw inputs |
| `scripts/make_pairs.py` | Transform raw/processed data into matched `.npy` clean/noisy pairs |
| `scripts/pair_data.py` | Defines PyTorch `FTIRPairsDataset` loaders and splitting utilities |
| `scripts/augment_data.py` | On-the-fly comprehensive augmentations (e.g., elastic warp, shifts, spikes) |
| `scripts/spectral_losses.py` | Defines `CompositeSpectralLoss` incorporating ~20 tunable objectives |
| `scripts/spectral_eval_utils.py` | General calculation handlers for spectral metrics (PSNR, RMSE, Shifts) |
| `scripts/export_eval_plots.py` | Generates visual overlays mapping clean vs noisy vs denoised signals |
| `run_pipeline.sh` | Bash script running the full end-to-end task |

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

## Workflow & Pipeline

End-to-End execution:

```sh
./run_pipeline.sh
```

### 1. Data Preparation

Place raw `.txt` files in `data/raw/` and baseline-corrected equivalent sets in `data/processed/`. The `make_pairs.py` script matches and pairs these sets into input-output pair files:

```sh
python scripts/baseline.py
python scripts/make_pairs.py
```

### 2. Training `ResUNet1D`

```sh
python scripts/train_resunet.py
```

The script will:
- Load matched clean/noisy `.npy` representations robustly.
- Dynamically augment spectral instances enforcing dataset shift invariance.
- Manage intricate `CompositeSpectralLoss` optimizations emphasizing metrics specifically requested via ENV variables (see `review.md` documentation templates).
- Export model weights to `models/resunet1d_single_stage_final.pth` (or best based on configuration).

### 3. Evaluation and Visualization

```sh
python scripts/evaluate_resunet.py
python scripts/export_eval_plots.py
```

Load the best-trained parameters and measure performance over the test splits. Target scores track properties such as global PSNR, SSIM, structural structural peak position correlations, Noise Floor variations, and Fingerprint Region Specific Root Mean Squared Error (FP RMSE).

## Inference Example

```python
import torch
import numpy as np
from scipy.signal import resample
from models.resunet1d import ResUNet1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResUNet1D(
    base_channels=64, 
    use_skip_gates=True, 
    use_se=True, 
    use_multiscale_context=True
).to(device)

model.load_state_dict(torch.load("models/resunet1d_single_stage_final.pth", map_location=device, weights_only=True))
model.eval()

# Assume resampled (N=1868) raw intensity input vector
noisy_tensor = torch.from_numpy(raw_spectrum).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    denoised_tensor = model(noisy_tensor)

denoised_spectrum = denoised_tensor.cpu().squeeze().numpy()
```
