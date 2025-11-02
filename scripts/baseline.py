import os
import numpy as np
from pybaselines.whittaker import asls
import matplotlib.pyplot as plt

def process_file(file_path, output_dir, plot=False):
    """
    Loads a two-column spectrum file, applies the optimized ASLS baseline correction,
    and saves the corrected spectrum.
    """
    # Load 2-column txt, skipping the header row
    try:
        data = np.loadtxt(file_path, skiprows=1)
    except (ValueError, IndexError):
        # Fallback for files that might not have a header
        data = np.loadtxt(file_path)
        
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"File {file_path} is not in the expected 2-column format.")

    x, y = data[:, 0], data[:, 1]

    # --- FINAL OPTIMIZED PARAMETERS that achieved 97.15% ---
    baseline, _ = asls(y, lam=1e4, p=0.315)
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
    """Processes all .txt files in the input directory."""
    os.makedirs(output_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Starting baseline correction for {len(txt_files)} files...")
    for fname in txt_files:
        file_path = os.path.join(input_dir, fname)
        try:
            process_file(file_path, output_dir, plot=plot)
            print(f"✔ Processed {fname}")
        except Exception as e:
            print(f"⚠ Error processing {fname}: {e}")
    print("✅ Baseline correction complete.")

if __name__ == "__main__":
    batch_process(plot=False)