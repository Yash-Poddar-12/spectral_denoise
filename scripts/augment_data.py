"""
Advanced data augmentation for FTIR spectral denoising.
v2 — Adds physically-motivated augmentations for 95%+ quality target.

New augmentations:
  - Pink (1/f) noise: realistic sensor/instrument drift
  - Spectral shift: calibration drift invariance
  - Elastic warping: peak shape robustness
  - Mixup helper: smoother decision boundaries
"""

import numpy as np
import os
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Original augmentations (retained)
# ---------------------------------------------------------------------------

def add_noise(intensity, noise_level=0.01):
    """Add Gaussian white noise with random level in [0, noise_level]."""
    random_noise_level = np.random.uniform(0, noise_level)
    noise = np.random.normal(0, random_noise_level, intensity.shape)
    return intensity + noise


def add_baseline(intensity, coeff=0.0001):
    """Add a quadratic baseline drift."""
    trend = coeff * np.linspace(-1, 1, len(intensity)) ** 2
    return intensity + trend


def add_spikes(intensity, num_spikes=5, spike_height=0.5):
    """Add random cosmic-ray-style spikes."""
    corrupted = intensity.copy()
    random_num_spikes = np.random.randint(0, num_spikes + 1)
    if random_num_spikes > 0:
        indices = np.random.choice(len(intensity), random_num_spikes, replace=False)
        for i in indices:
            corrupted[i] += spike_height * np.random.uniform(0.5, 1.0)
    return corrupted


def add_detector_glitches(
    intensity,
    max_glitches=3,
    amp_std_range=(4.0, 10.0),
    edge_bias=True,
    edge_fraction=0.08,
    max_width=3,
):
    """Add sharp detector glitches, biased toward spectral edges when requested."""
    corrupted = intensity.copy()
    n = len(intensity)
    signal_std = max(float(np.std(intensity)), 1e-3)
    num_glitches = np.random.randint(0, max_glitches + 1)
    if num_glitches == 0:
        return corrupted.astype(np.float32)

    edge_count = max(4, int(n * edge_fraction))
    for _ in range(num_glitches):
        if edge_bias and np.random.rand() < 0.70:
            if np.random.rand() < 0.75:
                center = np.random.randint(0, edge_count)
            else:
                center = np.random.randint(n - edge_count, n)
        else:
            center = np.random.randint(0, n)

        width = np.random.randint(1, max_width + 1)
        sign = -1.0 if np.random.rand() < 0.65 else 1.0
        amplitude = sign * np.random.uniform(*amp_std_range) * signal_std
        left = max(0, center - width // 2)
        right = min(n, left + width)
        corrupted[left:right] += amplitude

    return corrupted.astype(np.float32)


# ---------------------------------------------------------------------------
# NEW: Pink (1/f) noise — characteristic of spectrometer sensor drift
# ---------------------------------------------------------------------------

def add_pink_noise(intensity, noise_level=0.02):
    """Add 1/f (pink) noise — more realistic than white noise for instruments.

    Pink noise has equal power per *octave*, meaning low-frequency drift
    dominates, which matches real spectrometer behaviour.
    """
    n = len(intensity)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = 1.0  # avoid division by zero at DC

    # 1/f amplitude envelope
    pink_filter = 1.0 / np.sqrt(freqs)

    # Generate white noise in frequency domain, shape with 1/f
    white = np.random.randn(n)
    white_fft = np.fft.rfft(white)
    pink_fft = white_fft * pink_filter
    pink = np.fft.irfft(pink_fft, n=n)

    # Normalise to desired noise level
    pink = pink / (np.std(pink) + 1e-8) * noise_level
    return (intensity + pink).astype(np.float32)


# ---------------------------------------------------------------------------
# NEW: Spectral shift — simulates wavenumber calibration drift
# ---------------------------------------------------------------------------

def spectral_shift(intensity, max_shift=5, shift=None):
    """Circularly shift the spectrum by a small random amount.

    Simulates slight calibration drifts in the wavenumber axis.
    Uses interpolation for sub-pixel shifts.
    """
    if shift is None:
        shift = np.random.uniform(-max_shift, max_shift)
    n = len(intensity)
    indices = np.arange(n, dtype=np.float64) - shift
    # Clamp to valid range and interpolate
    indices = np.clip(indices, 0, n - 1)
    return np.interp(indices, np.arange(n), intensity).astype(np.float32)


# ---------------------------------------------------------------------------
# NEW: Elastic warping — non-uniform stretching along wavenumber axis
# ---------------------------------------------------------------------------

def elastic_warp_1d(intensity, sigma=4.0, alpha=4.0, displacement=None):
    """Apply smooth elastic deformation along the spectral axis.

    Creates a smooth random displacement field (Gaussian-filtered),
    then resamples the signal. This makes peaks slightly broader/narrower.

    Args:
        intensity: 1D signal array.
        sigma: Gaussian filter width for smoothing the displacement field.
        alpha: Magnitude of the displacement.
    """
    n = len(intensity)
    # Random displacement field, smoothed to be physically plausible
    if displacement is None:
        displacement = np.random.randn(n)
    displacement = gaussian_filter1d(displacement, sigma) * alpha
    new_indices = np.arange(n, dtype=np.float64) + displacement
    new_indices = np.clip(new_indices, 0, n - 1)
    return np.interp(new_indices, np.arange(n), intensity).astype(np.float32)


# ---------------------------------------------------------------------------
# NEW: Intensity scaling — simulates varying sample concentration
# ---------------------------------------------------------------------------

def intensity_scale(intensity, scale_range=(0.8, 1.2)):
    """Randomly scale the overall intensity to simulate concentration changes."""
    scale = np.random.uniform(*scale_range)
    return (intensity * scale).astype(np.float32)


def add_heteroscedastic_noise(intensity, base_level=0.003, signal_scale=0.015):
    """Noise level increases with local signal magnitude."""
    local_scale = base_level + signal_scale * np.abs(intensity - intensity.mean()) / (np.std(intensity) + 1e-8)
    noise = np.random.normal(0.0, local_scale, size=intensity.shape)
    return (intensity + noise).astype(np.float32)


def add_low_frequency_drift(intensity, strength=0.02):
    """Smooth low-frequency drift approximating instrument baseline wander."""
    x = np.linspace(0.0, 1.0, len(intensity), dtype=np.float32)
    phase = np.random.uniform(0, 2 * np.pi)
    drift = strength * (
        0.65 * np.sin(2 * np.pi * x + phase)
        + 0.35 * np.sin(4 * np.pi * x + phase / 2.0)
    )
    return (intensity + drift).astype(np.float32)


# ---------------------------------------------------------------------------
# Legacy convenience wrapper
# ---------------------------------------------------------------------------

def augment_spectrum(intensity, apply_noise=True, apply_baseline=True, apply_spikes=True):
    """Apply augmentation pipeline to one spectrum."""
    augmented = intensity.copy()
    if apply_noise:
        augmented = add_noise(augmented)
    if apply_baseline:
        augmented = add_baseline(augmented)
    if apply_spikes:
        augmented = add_spikes(augmented)
    return augmented


class SpectralAugmenter:
    """Epoch-aware augmentation schedule for denoising pairs."""

    def __init__(self):
        self.progress = 0.0
        self.shift_prob = float(os.getenv("AUG_SHIFT_PROB", "0.08"))
        self.shift_max = float(os.getenv("AUG_SHIFT_MAX", "0.8"))
        self.warp_prob = float(os.getenv("AUG_WARP_PROB", "0.05"))
        self.warp_sigma = float(os.getenv("AUG_WARP_SIGMA", "6.0"))
        self.warp_alpha = float(os.getenv("AUG_WARP_ALPHA", "0.6"))

    def set_epoch(self, epoch: int, total_epochs: int):
        self.progress = min(max((epoch - 1) / max(total_epochs - 1, 1), 0.0), 1.0)

    def __call__(self, clean: np.ndarray, noisy: np.ndarray):
        clean = clean.astype(np.float32, copy=True)
        noisy = noisy.astype(np.float32, copy=True)

        # Stronger augmentation early, gentler late for curriculum SNR training.
        curriculum = 1.0 - 0.55 * self.progress

        if np.random.rand() < 0.70:
            noisy = add_heteroscedastic_noise(
                noisy,
                base_level=0.003 * curriculum,
                signal_scale=0.018 * curriculum,
            )
        if np.random.rand() < 0.45:
            noisy = add_pink_noise(noisy, noise_level=np.random.uniform(0.003, 0.010) * curriculum)
        if np.random.rand() < 0.35:
            noisy = add_low_frequency_drift(noisy, strength=np.random.uniform(0.004, 0.020) * curriculum)
        if np.random.rand() < 0.25:
            noisy = add_baseline(noisy, coeff=np.random.uniform(0.00003, 0.00016) * curriculum)
        if np.random.rand() < 0.12:
            noisy = add_spikes(
                noisy,
                num_spikes=np.random.randint(1, 4),
                spike_height=np.random.uniform(0.05, 0.20) * curriculum,
            )
        if np.random.rand() < 0.30:
            noisy = add_detector_glitches(
                noisy,
                max_glitches=np.random.randint(1, 4),
                amp_std_range=(4.0 * curriculum, 12.0 * curriculum),
                edge_bias=True,
                edge_fraction=0.14,
                max_width=3,
            )
        if self.shift_prob > 0.0 and np.random.rand() < self.shift_prob:
            shift = np.random.uniform(-self.shift_max, self.shift_max) * curriculum
            clean = spectral_shift(clean, max_shift=self.shift_max, shift=shift)
            noisy = spectral_shift(noisy, max_shift=self.shift_max, shift=shift)
        if self.warp_prob > 0.0 and np.random.rand() < self.warp_prob:
            displacement = np.random.randn(clean.shape[0]).astype(np.float32)
            clean = elastic_warp_1d(clean, sigma=self.warp_sigma, alpha=self.warp_alpha * curriculum, displacement=displacement)
            noisy = elastic_warp_1d(noisy, sigma=self.warp_sigma, alpha=self.warp_alpha * curriculum, displacement=displacement)

        return clean.astype(np.float32), noisy.astype(np.float32)
