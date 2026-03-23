"""
Training script for the ResNetThreshold1D spectral denoising model.

Usage (from the project root):
    python scripts/train_resnet_threshold.py

The script expects clean/noisy .npy pair files under ``data/pairs``,
named ``<basename>_clean.npy`` and ``<basename>_noisy.npy``.
Trained weights are saved to ``models/resnet_threshold1d.pth`` and
evaluation metrics are written to ``results/resnet_threshold_metrics.json``.
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from scipy.signal import resample

# Allow running from either the project root or the scripts/ sub-directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from models.resnet_threshold import ResNetThreshold1D
from augment_data import add_noise, add_baseline, add_spikes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PAIRS_DIR = os.path.join(_project_root, "data", "pairs")
TARGET_LEN = 1024
BATCH_SIZE = 8
EPOCHS = 100
LR = 5e-4
HIDDEN_CHANNELS = 64
NUM_BLOCKS = 8
MODEL_PATH = os.path.join(_project_root, "models", "resnet_threshold1d.pth")
RESULTS_PATH = os.path.join(_project_root, "results", "resnet_threshold_metrics.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FTIRPairsDataset(Dataset):
    """Loads clean/noisy .npy pairs and optionally applies on-the-fly augmentation."""

    def __init__(self, clean_files, noisy_files, augment=True, target_len=TARGET_LEN):
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.augment = augment
        self.target_len = target_len

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean = np.load(self.clean_files[idx]).astype(np.float32)
        noisy = np.load(self.noisy_files[idx]).astype(np.float32)

        if len(clean) != self.target_len:
            clean = resample(clean, self.target_len).astype(np.float32)
        if len(noisy) != self.target_len:
            noisy = resample(noisy, self.target_len).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                noisy = add_noise(noisy, noise_level=0.02)
            if np.random.rand() < 0.3:
                noisy = add_baseline(noisy, coeff=0.0005)
            if np.random.rand() < 0.2:
                noisy = add_spikes(noisy, num_spikes=3)

        clean_t = torch.from_numpy(clean).unsqueeze(0)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)
        return noisy_t, clean_t


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
class HybridLoss(nn.Module):
    """Weighted combination of MSE and cosine-similarity loss.

    Args:
        alpha: Weight for the MSE term (default 0.8).
        beta:  Weight for the cosine-similarity term (default 0.2).
    """

    def __init__(self, alpha: float = 0.8, beta: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        cos_sim = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        return self.alpha * mse_loss + self.beta * (1.0 - cos_sim)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def compute_metrics(output: np.ndarray, clean: np.ndarray):
    """Return (mse, psnr, ssim, corr) between output and clean spectra."""
    o, c = _normalize(output), _normalize(clean)
    mse_val = float(np.mean((o - c) ** 2))
    psnr_val = float(20.0 * np.log10(1.0 / (np.sqrt(mse_val) + 1e-8)))
    ssim_val = float(ssim(o, c, data_range=1.0))
    corr_val = float(pearsonr(o.flatten(), c.flatten())[0])
    return mse_val, psnr_val, ssim_val, corr_val


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, verbose=True
    )
    criterion = HybridLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                val_loss += criterion(output, clean).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_and_save(model, val_loader):
    """Evaluate model on the validation loader and save metrics to JSON."""
    model.eval()
    mse_list, psnr_list, ssim_list, corr_list = [], [], [], []

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            for o, c in zip(output.cpu().numpy(), clean.cpu().numpy()):
                m = compute_metrics(o.squeeze(), c.squeeze())
                mse_list.append(m[0])
                psnr_list.append(m[1])
                ssim_list.append(m[2])
                corr_list.append(m[3])

    mean_mse = float(np.mean(mse_list))
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_corr = float(np.mean(corr_list))

    # Weighted quality score (same weighting scheme as original pipeline)
    weights = {"mse": 0.2, "psnr": 0.3, "ssim": 0.3, "corr": 0.2}
    mse_quality = 100.0 / (1.0 + mean_mse)
    psnr_quality = min(mean_psnr / 50.0 * 100.0, 100.0)
    ssim_quality = mean_ssim * 100.0
    corr_quality = mean_corr * 100.0
    overall_quality = (
        weights["mse"] * mse_quality
        + weights["psnr"] * psnr_quality
        + weights["ssim"] * ssim_quality
        + weights["corr"] * corr_quality
    )

    print("\nEvaluation Results on Validation Set:")
    print(f"  Mean MSE : {mean_mse:.6f}")
    print(f"  Mean PSNR: {mean_psnr:.2f} dB")
    print(f"  Mean SSIM: {mean_ssim:.4f}")
    print(f"  Mean Corr: {mean_corr:.4f}")
    print(f"  Overall Quality Score: {overall_quality:.2f}%")

    metrics = {
        "model": "ResNetThreshold1D",
        "mean_mse": mean_mse,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "mean_corr": mean_corr,
        "mse_quality": mse_quality,
        "psnr_quality": psnr_quality,
        "ssim_quality": ssim_quality,
        "corr_quality": corr_quality,
        "overall_quality": overall_quality,
        "weights": weights,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {RESULTS_PATH}")
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Collect pair files
    clean_files = sorted(
        [os.path.join(PAIRS_DIR, f) for f in os.listdir(PAIRS_DIR) if f.endswith("_clean.npy")]
    )
    noisy_files = sorted(
        [os.path.join(PAIRS_DIR, f) for f in os.listdir(PAIRS_DIR) if f.endswith("_noisy.npy")]
    )
    if not clean_files:
        raise FileNotFoundError(
            f"No clean/noisy .npy pairs found in {PAIRS_DIR}. "
            "Run make_pairs.py and create_noisy_data.py first."
        )

    train_c, val_c, train_n, val_n = train_test_split(
        clean_files, noisy_files, test_size=0.2, random_state=42
    )

    train_ds = FTIRPairsDataset(train_c, train_n, augment=True)
    val_ds = FTIRPairsDataset(val_c, val_n, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ResNetThreshold1D(
        in_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        num_blocks=NUM_BLOCKS,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"ResNetThreshold1D — {num_params / 1e6:.2f}M parameters")

    train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    evaluate_and_save(model, val_loader)
