import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Build paired .npy spectra from clean and raw .txt files.")
    parser.add_argument("--clean-dir", default="data/processed", help="Directory with cleaned/processed .txt spectra")
    parser.add_argument("--noisy-dir", default="data/raw", help="Directory with raw/noisy .txt spectra")
    parser.add_argument("--pairs-dir", default="data/pairs", help="Output directory for paired .npy files")
    args = parser.parse_args()

    clean_dir = args.clean_dir
    noisy_dir = args.noisy_dir
    pairs_dir = args.pairs_dir

    os.makedirs(pairs_dir, exist_ok=True)

    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith(".txt")])
    noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith(".txt")])

    clean_basenames = {os.path.splitext(f)[0] for f in clean_files}
    noisy_basenames = {os.path.splitext(f)[0] for f in noisy_files}
    common_basenames = sorted(list(clean_basenames.intersection(noisy_basenames)))

    if len(common_basenames) == 0:
        raise FileNotFoundError("No matching files found between raw and processed directories.")

    print(f"Found {len(common_basenames)} matching raw/clean file pairs.")

    for base in common_basenames:
        clean_path = os.path.join(clean_dir, f"{base}.txt")
        noisy_path = os.path.join(noisy_dir, f"{base}.txt")

        clean = np.loadtxt(clean_path, skiprows=1)[:, 1]
        noisy = np.loadtxt(noisy_path, skiprows=1)[:, 1]

        np.save(os.path.join(pairs_dir, f"{base}_clean.npy"), clean)
        np.save(os.path.join(pairs_dir, f"{base}_noisy.npy"), noisy)

    print(f"Saved {len(common_basenames)} pairs to {pairs_dir}")


if __name__ == "__main__":
    main()
