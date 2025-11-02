import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Paths
pairs_dir = "data/pairs"
dataset_dir = "data/dataset"
labels_file = "data/labels.csv"
os.makedirs(dataset_dir, exist_ok=True)

def normalize_spectrum(spectrum, method="zscore"):
    """Normalize spectrum to help model training."""
    if method == "zscore":
        mean = spectrum.mean()
        std = spectrum.std()
        return (spectrum - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = spectrum.min()
        max_val = spectrum.max()
        return (spectrum - min_val) / (max_val - min_val + 1e-8)
    else:
        return spectrum

# --- Load Classification Labels ---
try:
    labels_df = pd.read_csv(labels_file)
    label_map = dict(zip(labels_df['basename'].astype(str), labels_df['class_label']))
    print(f"✅ Loaded {len(label_map)} labels from {labels_file}")
except FileNotFoundError:
    print(f"❌ ERROR: Label file not found at {labels_file}.")
    exit()
except Exception as e:
    print(f"❌ ERROR: Could not read labels file. {e}")
    exit()
# ---------------------------------------

# Get list of clean files
clean_files = sorted([f for f in os.listdir(pairs_dir) if f.endswith("_clean.npy")])

X_noisy = []
Y_clean = []
Z_labels = []

print(f"Found {len(clean_files)} clean/noisy pairs. Matching with labels...")

# --- NEW: Add variable to store the expected shape ---
first_shape = None
# ----------------------------------------------------

for fname_c in clean_files:
    # --- UPDATED: Ensure basename is a string for lookup ---
    base = fname_c.replace("_clean.npy", "")
    fname_n = f"{base}_noisy.npy"
    
    if base not in label_map:
        print(f"⚠️ Warning: No label found for sample '{base}'. Skipping this file.")
        continue

    clean_path = os.path.join(pairs_dir, fname_c)
    noisy_path = os.path.join(pairs_dir, fname_n)

    try:
        clean_y = np.load(clean_path)
        noisy_y = np.load(noisy_path)
    except FileNotFoundError:
        print(f"⚠️ Warning: Skipping '{base}' (missing noisy or clean file).")
        continue

    if clean_y.shape != noisy_y.shape:
        print(f"❌ Length mismatch in {base}: clean={clean_y.shape}, noisy={noisy_y.shape}")
        continue

    # --- NEW: Check for uniform shape across all files ---
    if first_shape is None:
        first_shape = clean_y.shape
        print(f"INFO: Setting expected data shape to {first_shape} (based on file '{base}')")
    
    if clean_y.shape != first_shape:
        print(f"❌ Shape mismatch in {base}: Expected {first_shape} but got {clean_y.shape}. Skipping.")
        continue
    # ----------------------------------------------------

    # Normalize spectra
    clean_y = normalize_spectrum(clean_y, method="zscore")
    noisy_y = normalize_spectrum(noisy_y, method="zscore")

    # Append data and label
    Y_clean.append(clean_y)
    X_noisy.append(noisy_y)
    Z_labels.append(label_map[base])

# --- (QC plot logic removed) ---

# Convert to arrays
X_noisy = np.array(X_noisy, dtype=np.float32)
Y_clean = np.array(Y_clean, dtype=np.float32)
Z_labels = np.array(Z_labels, dtype=np.int64)

print(f"\nFinal dataset shapes after label matching and shape check:")
print(f"X_noisy: {X_noisy.shape}")
print(f"Y_clean: {Y_clean.shape}")
print(f"Z_labels: {Z_labels.shape}")

if len(X_noisy) == 0:
    print("❌ ERROR: No data was loaded. Check label/filename matches or shape mismatches.")
    exit()

# Train/test split
X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
    X_noisy, Y_clean, Z_labels, test_size=0.2, random_state=42, stratify=Z_labels
)

print(f"\nTrain: X={X_train.shape}, Y={Y_train.shape}, Z={Z_train.shape}")
print(f"Test:  X={X_test.shape}, Y={Y_test.shape}, Z={Z_test.shape}")

# Save
np.save(os.path.join(dataset_dir, "X_train.npy"), X_train)
np.save(os.path.join(dataset_dir, "Y_train.npy"), Y_train)
np.save(os.path.join(dataset_dir, "X_test.npy"), X_test)
np.save(os.path.join(dataset_dir, "Y_test.npy"), Y_test)
np.save(os.path.join(dataset_dir, "Z_train_labels.npy"), Z_train)
np.save(os.path.join(dataset_dir, "Z_test_labels.npy"), Z_test)

print(f"✅ Saved normalized + split dataset (and labels) to {dataset_dir}")