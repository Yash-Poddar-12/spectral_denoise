import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from models.resunet1d import ResUNet1D
from pair_data import FTIRPairsDataset, denormalize_output, get_pair_file_lists, split_pair_files
from spectral_eval_utils import aggregate_spectrum_metrics, evaluate_spectrum, quality_score

CONFIG_PATH = os.getenv(
    "CONFIG_PATH",
    os.path.join(_project_root, "results", "resunet_single_stage_final.json"),
)


def _load_config_defaults():
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


_CFG = _load_config_defaults()

PAIRS_DIR = os.getenv("PAIRS_DIR", os.path.join(_project_root, "data", "pairs"))
TARGET_LEN = int(os.getenv("TARGET_LEN", "1868"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
NORMALIZATION = os.getenv("NORMALIZATION", "joint_zscore")
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
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(_project_root, "models", "resunet1d_single_stage_best.pth"))
RESULTS_PATH = os.getenv(
    "RESULTS_PATH",
    os.path.join(_project_root, "results", "resunet_single_stage_eval.json"),
)


def get_device() -> torch.device:
    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    if torch.cuda.is_available():
        return torch.device("cuda")
    if force_cuda:
        raise RuntimeError(
            "FORCE_CUDA=1 but torch.cuda.is_available() is False. "
            "Fix NVIDIA driver/runtime, then retry."
        )
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def selection_score(metrics: dict) -> float:
    overall, _, _, _, _, _ = quality_score(
        metrics["mean_mse"],
        metrics["mean_psnr"],
        metrics["mean_ssim"],
        metrics["mean_corr"],
    )

    def safe(key: str) -> float:
        value = metrics.get(key, 0.0)
        if isinstance(value, float) and np.isnan(value):
            return 0.0
        return float(value)

    return overall - 6.0 * safe("mean_noise_floor_std") - 1.0 * safe("mean_peak_amp_mae") - 0.15 * safe("mean_peak_shift_mean_abs")


def create_model() -> ResUNet1D:
    return ResUNet1D(
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
    )


def evaluate_batch_outputs(outputs, clean_targets):
    metrics = [evaluate_spectrum(o.squeeze(), c.squeeze()) for o, c in zip(outputs, clean_targets)]
    aggregate = aggregate_spectrum_metrics(metrics)
    overall, mse_q, psnr_q, ssim_q, corr_q, weights = quality_score(
        aggregate["mean_mse"],
        aggregate["mean_psnr"],
        aggregate["mean_ssim"],
        aggregate["mean_corr"],
    )
    aggregate["overall_quality"] = overall
    aggregate["mse_quality"] = mse_q
    aggregate["psnr_quality"] = psnr_q
    aggregate["ssim_quality"] = ssim_q
    aggregate["corr_quality"] = corr_q
    aggregate["weights"] = weights
    aggregate["selection_score"] = selection_score(aggregate)
    return aggregate


def run_evaluation():
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = create_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded from {MODEL_PATH} ({num_params / 1e6:.2f}M parameters)")

    clean_files, noisy_files = get_pair_file_lists(PAIRS_DIR)
    _, val_c, _, val_n = split_pair_files(clean_files, noisy_files, test_size=0.2, seed=42)

    val_ds = FTIRPairsDataset(
        val_c,
        val_n,
        target_len=TARGET_LEN,
        normalization=NORMALIZATION,
        augment_fn=None,
        cache_in_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    noisy_outputs = []
    model_outputs = []
    clean_targets = []
    sample_index = 0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            output_np = output.cpu().numpy()
            batch_size = output_np.shape[0]
            for batch_offset in range(batch_size):
                clean_raw, noisy_raw = val_ds.get_raw_pair(sample_index + batch_offset)
                noisy_outputs.append(noisy_raw)
                model_outputs.append(
                    denormalize_output(
                        output_np[batch_offset].squeeze(),
                        clean_raw,
                        noisy_raw,
                        normalization=NORMALIZATION,
                    )
                )
                clean_targets.append(clean_raw)
            sample_index += batch_size

    noisy_metrics = evaluate_batch_outputs(noisy_outputs, clean_targets)
    model_metrics = evaluate_batch_outputs(model_outputs, clean_targets)

    comparison = {
        "overall_quality_gain": model_metrics["overall_quality"] - noisy_metrics["overall_quality"],
        "selection_score_gain": model_metrics["selection_score"] - noisy_metrics["selection_score"],
        "noise_floor_delta": noisy_metrics["mean_noise_floor_std"] - model_metrics["mean_noise_floor_std"],
        "peak_amp_mae_delta": noisy_metrics["mean_peak_amp_mae"] - model_metrics["mean_peak_amp_mae"],
        "fingerprint_rmse_delta": noisy_metrics["mean_rmse_fingerprint"] - model_metrics["mean_rmse_fingerprint"],
    }

    print("\nValidation Comparison:")
    print(f"  Noisy overall quality : {noisy_metrics['overall_quality']:.2f}%")
    print(f"  Model overall quality : {model_metrics['overall_quality']:.2f}%")
    print(f"  Quality gain          : {comparison['overall_quality_gain']:.2f}")
    print(f"  Noisy selection score : {noisy_metrics['selection_score']:.3f}")
    print(f"  Model selection score : {model_metrics['selection_score']:.3f}")
    print(f"  Noise-floor delta     : {comparison['noise_floor_delta']:.4f}")
    print(f"  Peak-amp-MAE delta    : {comparison['peak_amp_mae_delta']:.4f}")
    print(f"  Fingerprint RMSE delta: {comparison['fingerprint_rmse_delta']:.4f}")

    metrics = {
        "model": "ResUNet1D",
        "stage": "single_stage",
        "model_path": MODEL_PATH,
        "pairs_dir": PAIRS_DIR,
        "target_len": TARGET_LEN,
        "normalization": NORMALIZATION,
        "residual_learning": RESIDUAL_LEARNING,
        "norm_type": NORM_TYPE,
        "use_skip_gates": USE_SKIP_GATES,
        "use_se": USE_SE,
        "use_input_median": USE_INPUT_MEDIAN,
        "input_median_kernel": INPUT_MEDIAN_KERNEL,
        "input_median_blend": INPUT_MEDIAN_BLEND,
        "use_spike_suppressor": USE_SPIKE_SUPPRESSOR,
        "spike_suppressor_threshold": SPIKE_SUPPRESSOR_THRESHOLD,
        "spike_suppressor_blend": SPIKE_SUPPRESSOR_BLEND,
        "spike_suppressor_edge_points": SPIKE_SUPPRESSOR_EDGE_POINTS,
        "spike_suppressor_edge_gain": SPIKE_SUPPRESSOR_EDGE_GAIN,
        "use_multiscale_context": USE_MULTISCALE_CONTEXT,
        "use_detail_head": USE_DETAIL_HEAD,
        "use_positional_bias": USE_POSITIONAL_BIAS,
        "use_derivative_bias": USE_DERIVATIVE_BIAS,
        "use_local_refiner": USE_LOCAL_REFINER,
        "noisy_input_metrics": noisy_metrics,
        "model_metrics": model_metrics,
        "comparison": comparison,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {RESULTS_PATH}")
    return metrics


if __name__ == "__main__":
    run_evaluation()
