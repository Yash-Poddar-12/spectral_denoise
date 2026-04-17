import math
from typing import Dict

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim


WN_MIN = 399.264912
WN_MAX = 4000.364384


def normalize_to_unit(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def compute_basic_metrics(output: np.ndarray, clean: np.ndarray):
    o, c = normalize_to_unit(output), normalize_to_unit(clean)
    mse_val = float(np.mean((o - c) ** 2))
    psnr_val = float(20.0 * np.log10(1.0 / (np.sqrt(mse_val) + 1e-8)))
    ssim_val = float(ssim(o, c, data_range=1.0))
    corr_val = float(pearsonr(o.flatten(), c.flatten())[0])
    return mse_val, psnr_val, ssim_val, corr_val


def quality_score(mean_mse: float, mean_psnr: float, mean_ssim: float, mean_corr: float):
    weights = {"mse": 0.2, "psnr": 0.3, "ssim": 0.3, "corr": 0.2}
    mse_quality = 100.0 / (1.0 + mean_mse)
    psnr_quality = min(mean_psnr / 50.0 * 100.0, 100.0)
    ssim_quality = mean_ssim * 100.0
    corr_quality = mean_corr * 100.0
    overall = (
        weights["mse"] * mse_quality
        + weights["psnr"] * psnr_quality
        + weights["ssim"] * ssim_quality
        + weights["corr"] * corr_quality
    )
    return overall, mse_quality, psnr_quality, ssim_quality, corr_quality, weights


def build_wavenumber_axis(length: int) -> np.ndarray:
    return np.linspace(WN_MIN, WN_MAX, length, dtype=np.float32)


def region_masks(length: int) -> Dict[str, np.ndarray]:
    axis = build_wavenumber_axis(length)
    return {
        "boundary_low": axis <= 500.0,
        "fingerprint": (axis >= 600.0) & (axis <= 1800.0),
        "midband": (axis > 1800.0) & (axis < 2800.0),
        "high_wavenumber": axis >= 2800.0,
    }


def rmse_region(output: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    diff = output[mask] - clean[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def baseline_mask(clean: np.ndarray) -> np.ndarray:
    grad = np.abs(np.gradient(clean))
    amp = np.abs(clean)
    grad_thr = np.percentile(grad, 45)
    amp_thr = np.percentile(amp, 45)
    mask = (grad <= grad_thr) & (amp <= amp_thr)
    if mask.sum() < max(16, clean.size // 20):
        mask = grad <= np.percentile(grad, 60)
    return mask


def noise_floor_std(output: np.ndarray, clean: np.ndarray) -> float:
    mask = baseline_mask(clean)
    return float(np.std(output[mask] - clean[mask]))


def peak_fidelity_metrics(output: np.ndarray, clean: np.ndarray) -> Dict[str, float]:
    prominence = max(np.std(clean) * 0.25, 0.25)
    peaks, _ = find_peaks(clean, prominence=prominence, distance=max(4, clean.size // 80))
    if peaks.size == 0:
        return {
            "num_peaks": 0.0,
            "peak_amp_mae": math.nan,
            "peak_shift_mean_abs": math.nan,
        }

    amp_errors = []
    shift_errors = []
    for peak_idx in peaks:
        left = max(0, peak_idx - 6)
        right = min(clean.size, peak_idx + 7)
        local_output = output[left:right]
        if local_output.size == 0:
            continue
        pred_idx = left + int(np.argmax(local_output))
        amp_errors.append(abs(float(output[pred_idx] - clean[peak_idx])))
        shift_errors.append(abs(pred_idx - peak_idx))

    if not amp_errors:
        return {
            "num_peaks": float(peaks.size),
            "peak_amp_mae": math.nan,
            "peak_shift_mean_abs": math.nan,
        }

    return {
        "num_peaks": float(peaks.size),
        "peak_amp_mae": float(np.mean(amp_errors)),
        "peak_shift_mean_abs": float(np.mean(shift_errors)),
    }


def evaluate_spectrum(output: np.ndarray, clean: np.ndarray) -> Dict[str, float]:
    mse_val, psnr_val, ssim_val, corr_val = compute_basic_metrics(output, clean)
    masks = region_masks(len(clean))
    peak_metrics = peak_fidelity_metrics(output, clean)
    return {
        "mse": mse_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "corr": corr_val,
        "noise_floor_std": noise_floor_std(output, clean),
        "rmse_boundary_low": rmse_region(output, clean, masks["boundary_low"]),
        "rmse_fingerprint": rmse_region(output, clean, masks["fingerprint"]),
        "rmse_midband": rmse_region(output, clean, masks["midband"]),
        "rmse_high_wavenumber": rmse_region(output, clean, masks["high_wavenumber"]),
        **peak_metrics,
    }


def aggregate_spectrum_metrics(metrics_list):
    keys = metrics_list[0].keys()
    out = {}
    for key in keys:
        values = np.array([m[key] for m in metrics_list], dtype=np.float64)
        if np.all(np.isnan(values)):
            out[f"mean_{key}"] = math.nan
            out[f"std_{key}"] = math.nan
        else:
            out[f"mean_{key}"] = float(np.nanmean(values))
            out[f"std_{key}"] = float(np.nanstd(values))
    return out
