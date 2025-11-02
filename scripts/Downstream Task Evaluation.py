import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- 1. CORRECT MODEL DEFINITION (Copied from scripts/train_resunet.py) ---

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

# --- 2. DENOISING HELPER FUNCTIONS ---

def denoise_unet(noisy_data_np, model, device):
    noisy_tensor = torch.tensor(noisy_data_np, dtype=torch.float32).to(device)
    if noisy_tensor.dim() == 2:
        noisy_tensor = noisy_tensor.unsqueeze(1)
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)
    return denoised_tensor.cpu().squeeze(1).numpy()

def denoise_sg(noisy_data_np):
    window_length, polyorder = 21, 3
    if noisy_data_np.ndim == 1:
        noisy_data_np = noisy_data_np.reshape(1, -1)
    return np.array([savgol_filter(spec, window_length, polyorder) for spec in noisy_data_np]).squeeze()

def denoise_wavelet(noisy_data_np):
    if noisy_data_np.ndim == 1:
        noisy_data_np = noisy_data_np.reshape(1, -1)
    denoised_list = []
    for spec in noisy_data_np:
        coeffs = pywt.wavedec(spec, "sym8", mode="per", level=1)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(spec)))
        coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        denoised_list.append(pywt.waverec(coeffs_thresh, "sym8", mode="per"))
    return np.array(denoised_list).squeeze()

# --- 3. CLASSIFIER HELPER FUNCTIONS ---

def train_rf(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf

def eval_rf(clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    return acc, f1

# --- 4. MAIN EXECUTION ---

def run_downstream_evaluation():
    print("--- 1. Setting up paths and device ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = os.getcwd() 
    dataset_dir = os.path.join(project_root, "data/dataset")
    model_path = os.path.join(project_root, "models/resunet1d.pth")

    print(f"\n--- 2. Load Model ---")
    try:
        unet_model = ResUNet1D(base_ch=64).to(device)
        unet_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        unet_model.eval()
        print(f"✅ U-Net model loaded successfully from {model_path}.")
    except Exception as e:
        print(f"ERROR: Could not load model. {e}")
        return

    print(f"\n--- 3. Load Data ---")
    try:
        X_val = np.load(os.path.join(dataset_dir, "X_test.npy"))
        Y_val = np.load(os.path.join(dataset_dir, "Y_test.npy"))
        X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
        Y_train = np.load(os.path.join(dataset_dir, "Y_train.npy"))
        print(f"✅ Loaded data from {dataset_dir}.")

        # --- UPDATED: Load REAL Labels ---
        y_labels_val = np.load(os.path.join(dataset_dir, "Z_test_labels.npy"))
        y_labels_train = np.load(os.path.join(dataset_dir, "Z_train_labels.npy"))
        print("✅ Loaded REAL classification labels.")
        # ----------------------------------

    except FileNotFoundError as e:
        print(f"ERROR: Data file or Label file not found. {e}")
        print("Please run the updated 'load_qc.py' script first.")
        return
    
    # --- FIX: CROP DATA FOR U-NET COMPATIBILITY ---
    TARGET_LENGTH = 1864 
    print(f"\n--- Cropping data from {X_train.shape[1]} to {TARGET_LENGTH} for U-Net compatibility ---")
    X_train = X_train[:, :TARGET_LENGTH]
    Y_train = Y_train[:, :TARGET_LENGTH]
    X_val = X_val[:, :TARGET_LENGTH]
    Y_val = Y_val[:, :TARGET_LENGTH]
    print(f"✅ Data cropped successfully. New shape: {X_train.shape}")
    
    print("\n--- 4. Denoise Datasets ---")
    try:
        # Denoise Validation Set
        denoised_unet_val = denoise_unet(X_val, unet_model, device)
        denoised_sg_val = denoise_sg(X_val)
        denoised_wavelet_val = denoise_wavelet(X_val)
        print("✅ Denoised validation set (U-Net, SG, Wavelet).")

        # Denoise Training Set
        denoised_unet_train = denoise_unet(X_train, unet_model, device)
        denoised_sg_train = denoise_sg(X_train)
        denoised_wavelet_train = denoise_wavelet(X_train)
        print("✅ Denoised training set (U-Net, SG, Wavelet).")
    
    except RuntimeError as e:
        print(f"\n--- !!! ERROR during U-Net Denoising: {e} ---")
        return

    print("\n--- 5. Train and Evaluate Classifiers ---")
    
    clf_noisy = train_rf(X_train, y_labels_train)
    clf_clean = train_rf(Y_train, y_labels_train)
    clf_unet = train_rf(denoised_unet_train, y_labels_train)
    clf_sg = train_rf(denoised_sg_train, y_labels_train)
    clf_wavelet = train_rf(denoised_wavelet_train, y_labels_train)
    print("✅ Classifiers trained.")

    acc_noisy, f1_noisy = eval_rf(clf_noisy, X_val, y_labels_val)
    acc_clean, f1_clean = eval_rf(clf_clean, Y_val, y_labels_val)
    acc_unet, f1_unet = eval_rf(clf_unet, denoised_unet_val, y_labels_val)
    acc_sg, f1_sg = eval_rf(clf_sg, denoised_sg_val, y_labels_val)
    acc_wavelet, f1_wavelet = eval_rf(clf_wavelet, denoised_wavelet_val, y_labels_val)
    print("✅ Classifiers evaluated.")

    print("\n--- 6. Downstream Classification Performance (Random Forest) ---")
    print("| Input Data Source | Accuracy (↑) | F1-Score (weighted) (↑) |")
    print("| :--- | :---: | :---: |")
    print(f"| Noisy (Baseline)  | {acc_noisy:.2%}      | {f1_noisy:.4f}                  |")
    print(f"| Savitzky-Golay    | {acc_sg:.2%}      | {f1_sg:.4f}                  |")
    print(f"| Wavelet           | {acc_wavelet:.2%}      | {f1_wavelet:.4f}                  |")
    print(f"| **1D U-Net (Ours)** | **{acc_unet:.2%}** | **{f1_unet:.4f}** |")
    print(f"| Clean Target      | {acc_clean:.2%}      | {f1_clean:.4f}                  |")
    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == 'scripts':
        os.chdir('..')
        print(f"Changed directory to project root: {os.getcwd()}")
        
    run_downstream_evaluation()