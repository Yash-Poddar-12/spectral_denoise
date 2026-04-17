import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from models.resunet1d import ResUNet1D
from spectral_eval_utils import evaluate_spectrum

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", str(Path(_project_root) / "results" / "resunet_single_stage_final.json")))


def _load_config_defaults():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


_CFG = _load_config_defaults()

BASE_CHANNELS = int(os.getenv("BASE_CHANNELS", str(_CFG.get("base_channels", 64))))
RESIDUAL_LEARNING = os.getenv("RESIDUAL_LEARNING", str(_CFG.get("residual_learning", True))).lower() in ("1", "true", "yes")
NORM_TYPE = os.getenv("NORM_TYPE", str(_CFG.get("norm_type", "group")))
USE_SKIP_GATES = os.getenv("USE_SKIP_GATES", str(_CFG.get("use_skip_gates", False))).lower() in ("1", "true", "yes")
USE_SE = os.getenv("USE_SE", str(_CFG.get("use_se", True))).lower() in ("1", "true", "yes")
USE_INPUT_MEDIAN = os.getenv("USE_INPUT_MEDIAN", str(_CFG.get("use_input_median", True))).lower() in ("1", "true", "yes")
INPUT_MEDIAN_KERNEL = int(os.getenv("INPUT_MEDIAN_KERNEL", str(_CFG.get("input_median_kernel", 5))))
INPUT_MEDIAN_BLEND = float(os.getenv("INPUT_MEDIAN_BLEND", str(_CFG.get("input_median_blend", 0.15))))
USE_SPIKE_SUPPRESSOR = os.getenv("USE_SPIKE_SUPPRESSOR", str(_CFG.get("use_spike_suppressor", False))).lower() in ("1", "true", "yes")
SPIKE_SUPPRESSOR_THRESHOLD = float(os.getenv("SPIKE_SUPPRESSOR_THRESHOLD", "2.5"))
SPIKE_SUPPRESSOR_BLEND = float(os.getenv("SPIKE_SUPPRESSOR_BLEND", "0.85"))
SPIKE_SUPPRESSOR_EDGE_POINTS = int(os.getenv("SPIKE_SUPPRESSOR_EDGE_POINTS", "96"))
SPIKE_SUPPRESSOR_EDGE_GAIN = float(os.getenv("SPIKE_SUPPRESSOR_EDGE_GAIN", "1.8"))
USE_MULTISCALE_CONTEXT = os.getenv("USE_MULTISCALE_CONTEXT", str(_CFG.get("use_multiscale_context", False))).lower() in ("1", "true", "yes")
USE_DETAIL_HEAD = os.getenv("USE_DETAIL_HEAD", str(_CFG.get("use_detail_head", False))).lower() in ("1", "true", "yes")
USE_POSITIONAL_BIAS = os.getenv("USE_POSITIONAL_BIAS", str(_CFG.get("use_positional_bias", False))).lower() in ("1", "true", "yes")
USE_DERIVATIVE_BIAS = os.getenv("USE_DERIVATIVE_BIAS", str(_CFG.get("use_derivative_bias", False))).lower() in ("1", "true", "yes")
USE_LOCAL_REFINER = os.getenv("USE_LOCAL_REFINER", str(_CFG.get("use_local_refiner", False))).lower() in ("1", "true", "yes")
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(_project_root) / "models" / "resunet1d_single_stage_final.pth")))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_txt_spectrum(path: Path):
    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} is not a valid 2-column txt spectrum")
    return data[:, 0].astype(np.float32), data[:, 1].astype(np.float32)


def normalize_joint_zscore(clean: np.ndarray, noisy: np.ndarray):
    stacked = np.concatenate([clean, noisy], axis=0)
    mean = float(stacked.mean())
    std = float(stacked.std()) + 1e-8
    return (clean - mean) / std, (noisy - mean) / std, mean, std


def create_model(device: torch.device) -> ResUNet1D:
    model = ResUNet1D(
        base_channels=BASE_CHANNELS,
        residual_learning=RESIDUAL_LEARNING,
        norm_type=NORM_TYPE,
        use_skip_gates=USE_SKIP_GATES,
        use_se=USE_SE,
        use_input_median=USE_INPUT_MEDIAN,
        input_median_kernel=INPUT_MEDIAN_KERNEL,
        input_median_blend=INPUT_MEDIAN_BLEND,
        use_spike_suppressor=USE_SPIKE_SUPPRESSOR,
        spike_suppressor_threshold=SPIKE_SUPPRESSOR_THRESHOLD,
        spike_suppressor_blend=SPIKE_SUPPRESSOR_BLEND,
        spike_suppressor_edge_points=SPIKE_SUPPRESSOR_EDGE_POINTS,
        spike_suppressor_edge_gain=SPIKE_SUPPRESSOR_EDGE_GAIN,
        use_multiscale_context=USE_MULTISCALE_CONTEXT,
        use_detail_head=USE_DETAIL_HEAD,
        use_positional_bias=USE_POSITIONAL_BIAS,
        use_derivative_bias=USE_DERIVATIVE_BIAS,
        use_local_refiner=USE_LOCAL_REFINER,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def annotate_metrics(ax, metrics: dict, title: str):
    lines = [title]
    for key in (
        "psnr",
        "ssim",
        "corr",
        "noise_floor_std",
        "rmse_fingerprint",
        "peak_amp_mae",
        "peak_shift_mean_abs",
    ):
        value = metrics.get(key)
        if isinstance(value, float) and np.isnan(value):
            continue
        lines.append(f"{key}: {value:.4f}" if isinstance(value, (float, np.floating)) else f"{key}: {value}")
    ax.text(
        1.01,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f8f8f8", "edgecolor": "#cccccc"},
    )


def save_before_plot(sample_id: str, x: np.ndarray, noisy: np.ndarray, clean: np.ndarray, metrics: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=180)
    ax.plot(x, noisy, color="#767676", linewidth=1.0, alpha=0.90, label="Before: raw/noisy input")
    ax.set_title(f"Before Denoising: sample {sample_id}")
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Intensity")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")
    info = {
        "raw_mean": float(np.mean(noisy)),
        "raw_std": float(np.std(noisy)),
        "corrected_target_mean": float(np.mean(clean)),
        "corrected_target_std": float(np.std(clean)),
    }
    annotate_metrics(ax, {**metrics, **info}, "Raw input plot\nmetrics vs corrected target")
    plt.tight_layout(rect=[0.0, 0.0, 0.83, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_after_plot(
    sample_id: str,
    x: np.ndarray,
    raw_input: np.ndarray,
    denoised_full_scale: np.ndarray,
    denoised_corrected: np.ndarray,
    baseline_est: np.ndarray,
    clean: np.ndarray,
    metrics: dict,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=180)
    ax.plot(x, raw_input, color="#9a9a9a", linewidth=0.9, alpha=0.70, label="Raw input")
    ax.plot(x, denoised_full_scale, color="#0f766e", linewidth=1.2, alpha=0.95, label="After: denoised output")
    ax.plot(
        x,
        clean + baseline_est,
        color="#2563eb",
        linewidth=1.0,
        alpha=0.90,
        linestyle="--",
        label="Clean target + estimated baseline",
    )
    ax.set_title(f"After Denoising: sample {sample_id}")
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Intensity")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")
    info = {
        "output_mean": float(np.mean(denoised_full_scale)),
        "output_std": float(np.std(denoised_full_scale)),
        "corrected_output_mean": float(np.mean(denoised_corrected)),
        "corrected_output_std": float(np.std(denoised_corrected)),
        "corrected_target_mean": float(np.mean(clean)),
        "corrected_target_std": float(np.std(clean)),
    }
    annotate_metrics(ax, {**metrics, **info}, "After plot in raw scale\nmetrics vs corrected target")
    plt.tight_layout(rect=[0.0, 0.0, 0.83, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def export_sample(sample_id: str, output_dir: Path, model=None, device=None):
    raw_path = Path(_project_root) / "data" / "raw" / f"{sample_id}.txt"
    clean_path = Path(_project_root) / "data" / "processed" / f"{sample_id}.txt"
    if not raw_path.exists() or not clean_path.exists():
        raise FileNotFoundError(f"Missing raw/processed txt for sample {sample_id}")

    x_raw, raw_y = load_txt_spectrum(raw_path)
    x_clean, clean_y = load_txt_spectrum(clean_path)
    if len(x_raw) != len(x_clean) or not np.allclose(x_raw, x_clean, atol=1e-5):
        raise ValueError(f"Wavenumber axis mismatch for sample {sample_id}")

    clean_norm, noisy_norm, mean, std = normalize_joint_zscore(clean_y, raw_y)

    if device is None:
        device = get_device()
    if model is None:
        model = create_model(device)
    noisy_tensor = torch.from_numpy(noisy_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred_norm = model(noisy_tensor).detach().cpu().squeeze().numpy()

    denoised_corrected = pred_norm * std + mean
    baseline_est = raw_y - clean_y
    denoised_full_scale = denoised_corrected + baseline_est
    before_metrics = evaluate_spectrum(raw_y, clean_y)
    after_metrics = evaluate_spectrum(denoised_corrected, clean_y)

    before_path = output_dir / f"{sample_id}_before.png"
    after_path = output_dir / f"{sample_id}_after.png"
    save_before_plot(sample_id, x_raw, raw_y, clean_y, before_metrics, before_path)
    save_after_plot(
        sample_id,
        x_raw,
        raw_y,
        denoised_full_scale,
        denoised_corrected,
        baseline_est,
        clean_y,
        after_metrics,
        after_path,
    )

    return before_path, after_path, before_metrics, after_metrics


def export_all_samples(output_dir: Path):
    raw_dir = Path(_project_root) / "data" / "raw"
    clean_dir = Path(_project_root) / "data" / "processed"
    raw_ids = {p.stem for p in raw_dir.glob("*.txt")}
    clean_ids = {p.stem for p in clean_dir.glob("*.txt")}
    sample_ids = sorted(raw_ids & clean_ids, key=lambda x: int(x) if x.isdigit() else x)
    if not sample_ids:
        raise FileNotFoundError("No matching raw/processed sample ids found")

    device = get_device()
    model = create_model(device)
    rows = []
    for idx, sample_id in enumerate(sample_ids, start=1):
        before_path, after_path, before_metrics, after_metrics = export_sample(
            sample_id,
            output_dir,
            model=model,
            device=device,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "before_plot": str(before_path),
                "after_plot": str(after_path),
                "before_psnr": before_metrics["psnr"],
                "after_psnr": after_metrics["psnr"],
                "before_ssim": before_metrics["ssim"],
                "after_ssim": after_metrics["ssim"],
                "before_corr": before_metrics["corr"],
                "after_corr": after_metrics["corr"],
                "before_noise_floor_std": before_metrics["noise_floor_std"],
                "after_noise_floor_std": after_metrics["noise_floor_std"],
            }
        )
        if idx % 50 == 0 or idx == len(sample_ids):
            print(f"Exported {idx}/{len(sample_ids)} samples")

    index_path = output_dir / "evaluation_plot_index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "before_plot",
                "after_plot",
                "before_psnr",
                "after_psnr",
                "before_ssim",
                "after_ssim",
                "before_corr",
                "after_corr",
                "before_noise_floor_std",
                "after_noise_floor_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return index_path, len(sample_ids)


def main():
    parser = argparse.ArgumentParser(description="Export before/after evaluation plots for one sample.")
    parser.add_argument("--sample", default="365", help="Sample basename without extension, e.g. 365")
    parser.add_argument("--all", action="store_true", help="Export before/after plots for all matching samples")
    parser.add_argument(
        "--output-dir",
        default=str(Path(_project_root) / "results" / "evaluation_plots"),
        help="Directory for output plot files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.all:
        index_path, count = export_all_samples(output_dir)
        print(f"Index CSV: {index_path}")
        print(f"Total samples exported: {count}")
        return

    before_path, after_path, before_metrics, after_metrics = export_sample(args.sample, output_dir)
    print(f"Before plot: {before_path}")
    print(f"After plot:  {after_path}")
    print("Before metrics:", before_metrics)
    print("After metrics:", after_metrics)


if __name__ == "__main__":
    main()
