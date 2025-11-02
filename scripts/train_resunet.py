import os
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
from augment_data import add_noise, add_baseline, add_spikes

# -------------------
# Settings
# -------------------
pairs_dir = "data/pairs"
target_len = 1024
batch_size = 8
model_path = "models/resunet1d.pth"
results_path = "results/eval_metrics.json"

# Device selection (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# -------------------
# Dataset
# -------------------
class FTIRPairsDataset(Dataset):
    def __init__(self, clean_files, noisy_files, augment=True, target_len=1024):
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.augment = augment
        self.target_len = target_len

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean = np.load(self.clean_files[idx])
        noisy = np.load(self.noisy_files[idx])

        # Resample spectra to the target length
        if len(clean) != self.target_len:
            clean = resample(clean, self.target_len)
        if len(noisy) != self.target_len:
            noisy = resample(noisy, self.target_len)
        
        # Using the standard, stable augmentation
        if self.augment:
            if np.random.rand() < 0.5:
                noisy = add_noise(noisy, noise_level=0.02)
            if np.random.rand() < 0.3:
                noisy = add_baseline(noisy, coeff=0.0005)
            if np.random.rand() < 0.2:
                noisy = add_spikes(noisy, num_spikes=3)

        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)
        return noisy, clean

# -------------------
# ResUNet1D Model
# -------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class ResUNet1D(nn.Module):
    # --- Using the best performing deeper model ---
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)
        self.up2 = nn.ConvTranspose1d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch*8, base_ch*4)
        self.up1 = nn.ConvTranspose1d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch*4, base_ch*2)
        self.up0 = nn.ConvTranspose1d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(base_ch*2, base_ch)
        self.final = nn.Conv1d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d2 = self.up2(b); d2 = torch.cat([d2, e3], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e2], dim=1); d1 = self.dec1(d1)
        d0 = self.up0(d1); d0 = torch.cat([d0, e1], dim=1); d0 = self.dec0(d0)
        out = self.final(d0)
        return out

# -------------------
# Loss Function
# -------------------
class HybridLoss(nn.Module):
    # Using the best performing loss weights
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__(); self.alpha, self.beta, self.mse = alpha, beta, nn.MSELoss()
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        pred_flat, target_flat = pred.view(pred.size(0), -1), target.view(target.size(0), -1)
        cos_sim = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        ssim_loss = 1 - cos_sim
        return self.alpha * mse_loss + self.beta * ssim_loss

# -------------------
# Training Function
# -------------------
def train_model(model, train_loader, val_loader, epochs=100, lr=5e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    criterion = HybridLoss()
    for epoch in range(1, epochs+1):
        model.train(); train_loss = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad(); output = model(noisy); loss = criterion(output, clean); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval(); val_loss = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy); loss = criterion(output, clean); val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

# -------------------
# Helper functions
# -------------------
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def compute_metrics(o, c):
    o_norm, c_norm = normalize(o), normalize(c)
    mse_val = np.mean((o_norm - c_norm) ** 2)
    psnr_val = 20 * np.log10(1.0 / (np.sqrt(mse_val) + 1e-8))
    ssim_val = ssim(o_norm, c_norm, data_range=1.0)
    corr_val, _ = pearsonr(o_norm.flatten(), c_norm.flatten())
    return mse_val, psnr_val, ssim_val, corr_val

# ------------------------
# Evaluation Function
# ------------------------
def evaluate_model():
    print("\nStarting evaluation...")
    model = ResUNet1D(base_ch=64).to(device)
    # Load the model with weights_only=True for safety
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded from {model_path} ({num_params/1e6:.2f}M parameters)")

    clean_files = sorted([os.path.join(pairs_dir, f) for f in os.listdir(pairs_dir) if f.endswith("clean.npy")])
    noisy_files = sorted([os.path.join(pairs_dir, f) for f in os.listdir(pairs_dir) if f.endswith("noisy.npy")])
    _, val_c, _, val_n = train_test_split(clean_files, noisy_files, test_size=0.2, random_state=42)
    val_dataset = FTIRPairsDataset(val_c, val_n, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    mse_list, psnr_list, ssim_list, corr_list = [], [], [], []

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            out_np, clean_np = output.cpu().numpy(), clean.cpu().numpy()
            for o, c in zip(out_np, clean_np):
                metrics = compute_metrics(o.squeeze(), c.squeeze())
                mse_list.append(metrics[0]); psnr_list.append(metrics[1]); ssim_list.append(metrics[2]); corr_list.append(metrics[3])

    mean_mse, mean_psnr, mean_ssim, mean_corr = np.mean(mse_list), np.mean(psnr_list), np.mean(ssim_list), np.mean(corr_list)
    mse_quality, psnr_quality = 100 / (1 + mean_mse), min(mean_psnr / 50 * 100, 100)
    ssim_quality, corr_quality = mean_ssim * 100, mean_corr * 100
    weights = {"mse": 0.2, "psnr": 0.3, "ssim": 0.3, "corr": 0.2}
    overall_quality = (weights["mse"] * mse_quality + weights["psnr"] * psnr_quality + weights["ssim"] * ssim_quality + weights["corr"] * corr_quality)

    print("\n Evaluation Results on Validation Set:")
    print(f"Mean MSE:   {mean_mse:.6f}\nMean PSNR:  {mean_psnr:.2f} dB\nMean SSIM:  {mean_ssim:.4f}\nMean Corr:  {mean_corr:.4f}")
    print(f"Overall Quality Score: {overall_quality:.2f}%")

    metrics = { 
        "mean_mse": float(mean_mse), 
        "mean_psnr": float(mean_psnr), 
        "mean_ssim": float(mean_ssim), 
        "mean_corr": float(mean_corr), 
        "overall_quality": float(overall_quality),
        "optimal_baseline_params": {"lam": 1e4, "p": 0.315}
    }
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f: json.dump(metrics, f, indent=4)
    print(f"\n📂 Metrics saved to {results_path}")

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    clean_files = sorted([os.path.join(pairs_dir, f) for f in os.listdir(pairs_dir) if f.endswith("clean.npy")])
    noisy_files = sorted([os.path.join(pairs_dir, f) for f in os.listdir(pairs_dir) if f.endswith("noisy.npy")])
    train_c, val_c, train_n, val_n = train_test_split(clean_files, noisy_files, test_size=0.2, random_state=42)
    train_dataset = FTIRPairsDataset(train_c, train_n, augment=True)
    val_dataset = FTIRPairsDataset(val_c, val_n, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Use the final optimized model settings
    model = ResUNet1D(base_ch=64).to(device)
    train_model(model, train_loader, val_loader, epochs=100, lr=5e-4)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n Final Model saved to {model_path}")
    
    evaluate_model()