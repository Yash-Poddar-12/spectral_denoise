import os
import numpy as np
from pybaselines.whittaker import asls
import matplotlib.pyplot as plt

def process_file(file_path, output_dir, plot=False):
    # Load 2-column txt, skipping the header row
    data = np.loadtxt(file_path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"File {file_path} does not have 2 columns")

    x, y = data[:, 0], data[:, 1]

    # Apply baseline correction
    baseline, _ = asls(y, lam=1e4, p=0.309)
    corrected = y - baseline

    # Save corrected spectrum
    out_file = os.path.join(output_dir, os.path.basename(file_path))
    np.savetxt(out_file, np.column_stack((x, corrected)),
               header="Wavenumber Intensity(corrected)", comments="")

    # Optional plot for inspection
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label="Original")
        plt.plot(x, baseline, label="Baseline (asls)")
        plt.plot(x, corrected, label="Corrected")
        plt.title(os.path.basename(file_path))
        plt.xlabel("Wavenumber")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()

def batch_process(input_dir="data/raw", output_dir="data/processed", plot=False):
    os.makedirs(output_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    for fname in txt_files:
        file_path = os.path.join(input_dir, fname)
        try:
            process_file(file_path, output_dir, plot=plot)
            print(f"✔ Processed {fname}")
        except Exception as e:
            print(f"⚠ Error processing {fname}: {e}")

if __name__ == "__main__":
    batch_process(plot=False)