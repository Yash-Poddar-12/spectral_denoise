import os
import numpy as np

# Paths
clean_dir = "data/processed"
noisy_dir = "data/raw"
pairs_dir = "data/pairs"

os.makedirs(pairs_dir, exist_ok=True)

# Get file lists
clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.txt')])
noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.txt')])

# Ensure file lists are matched correctly
clean_basenames = {os.path.splitext(f)[0] for f in clean_files}
noisy_basenames = {os.path.splitext(f)[0] for f in noisy_files}

common_basenames = sorted(list(clean_basenames.intersection(noisy_basenames)))

if len(common_basenames) == 0:
    raise FileNotFoundError("No matching files found between raw and processed directories.")

print(f"Found {len(common_basenames)} matching raw/clean file pairs.")

for base in common_basenames:
    clean_path = os.path.join(clean_dir, f"{base}.txt")
    noisy_path = os.path.join(noisy_dir, f"{base}.txt")
    
    # Skip header row (assumes first row has text)
    clean = np.loadtxt(clean_path, skiprows=1)[:, 1]
    noisy = np.loadtxt(noisy_path, skiprows=1)[:, 1]

    # Save as .npy pair
    np.save(os.path.join(pairs_dir, f"{base}_clean.npy"), clean)
    np.save(os.path.join(pairs_dir, f"{base}_noisy.npy"), noisy)

print(f"✅ Saved {len(common_basenames)} pairs to {pairs_dir}")