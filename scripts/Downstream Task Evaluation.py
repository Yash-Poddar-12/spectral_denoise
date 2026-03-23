import os
import sys
import numpy as np
import torch
from scipy.signal import savgol_filter
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Allow running from either the project root or the scripts/ sub-directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.resnet_threshold import ResNetThreshold1D

# --- 1. DENOISING HELPER FUNCTIONS ---

def denoise_resnet(noisy_data_np, model, device):
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

# --- 2. CLASSIFIER HELPER FUNCTIONS ---

def train_rf(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf

def eval_rf(clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    return acc, f1

# --- 3. MAIN EXECUTION ---

def run_downstream_evaluation():
    print("--- 1. Setting up paths and device ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = _project_root
    dataset_dir = os.path.join(project_root, "data/dataset")
    model_path = os.path.join(project_root, "models/resnet_threshold1d.pth")

    print(f"\n--- 2. Load Model ---")
    try:
        resnet_model = ResNetThreshold1D(in_channels=1, hidden_channels=64, num_blocks=8).to(device)
        resnet_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        resnet_model.eval()
        print(f"ResNetThreshold1D model loaded successfully from {model_path}.")
    except Exception as e:
        print(f"ERROR: Could not load model. {e}")
        return

    print(f"\n--- 3. Load Data ---")
    try:
        X_val = np.load(os.path.join(dataset_dir, "X_test.npy"))
        Y_val = np.load(os.path.join(dataset_dir, "Y_test.npy"))
        X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
        Y_train = np.load(os.path.join(dataset_dir, "Y_train.npy"))
        print(f"Loaded data from {dataset_dir}.")

        y_labels_val = np.load(os.path.join(dataset_dir, "Z_test_labels.npy"))
        y_labels_train = np.load(os.path.join(dataset_dir, "Z_train_labels.npy"))
        print("Loaded real classification labels.")

    except FileNotFoundError as e:
        print(f"ERROR: Data file or Label file not found. {e}")
        print("Please run the updated 'load_qc.py' script first.")
        return

    print("\n--- 4. Denoise Datasets ---")
    try:
        denoised_resnet_val = denoise_resnet(X_val, resnet_model, device)
        denoised_sg_val = denoise_sg(X_val)
        denoised_wavelet_val = denoise_wavelet(X_val)
        print("Denoised validation set (ResNetThreshold, SG, Wavelet).")

        denoised_resnet_train = denoise_resnet(X_train, resnet_model, device)
        denoised_sg_train = denoise_sg(X_train)
        denoised_wavelet_train = denoise_wavelet(X_train)
        print("Denoised training set (ResNetThreshold, SG, Wavelet).")

    except RuntimeError as e:
        print(f"\n--- ERROR during ResNetThreshold Denoising: {e} ---")
        return

    print("\n--- 5. Train and Evaluate Classifiers ---")

    clf_noisy = train_rf(X_train, y_labels_train)
    clf_clean = train_rf(Y_train, y_labels_train)
    clf_resnet = train_rf(denoised_resnet_train, y_labels_train)
    clf_sg = train_rf(denoised_sg_train, y_labels_train)
    clf_wavelet = train_rf(denoised_wavelet_train, y_labels_train)
    print("Classifiers trained.")

    acc_noisy, f1_noisy = eval_rf(clf_noisy, X_val, y_labels_val)
    acc_clean, f1_clean = eval_rf(clf_clean, Y_val, y_labels_val)
    acc_resnet, f1_resnet = eval_rf(clf_resnet, denoised_resnet_val, y_labels_val)
    acc_sg, f1_sg = eval_rf(clf_sg, denoised_sg_val, y_labels_val)
    acc_wavelet, f1_wavelet = eval_rf(clf_wavelet, denoised_wavelet_val, y_labels_val)
    print("Classifiers evaluated.")

    print("\n--- 6. Downstream Classification Performance (Random Forest) ---")
    print("| Input Data Source              | Accuracy (↑) | F1-Score (weighted) (↑) |")
    print("| :---                           | :---:        | :---:                   |")
    print(f"| Noisy (Baseline)               | {acc_noisy:.2%}      | {f1_noisy:.4f}                  |")
    print(f"| Savitzky-Golay                 | {acc_sg:.2%}      | {f1_sg:.4f}                  |")
    print(f"| Wavelet                        | {acc_wavelet:.2%}      | {f1_wavelet:.4f}                  |")
    print(f"| **ResNetThreshold1D (Ours)**   | **{acc_resnet:.2%}** | **{f1_resnet:.4f}** |")
    print(f"| Clean Target                   | {acc_clean:.2%}      | {f1_clean:.4f}                  |")
    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == 'scripts':
        os.chdir('..')
        print(f"Changed directory to project root: {os.getcwd()}")

    run_downstream_evaluation()
