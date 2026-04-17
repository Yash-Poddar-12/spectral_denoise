# Spectral Denoising Reset And Retraining Plan

## Goal
Build a denoiser that is publishable for FTIR-like 1D spectra:
- no boundary spikes
- no hallucinated peaks
- lower baseline noise
- preserved peak position, width, and amplitude
- reproducible evaluation with physics-aware metrics and ablations

## Diagnosis From Current Outputs
The current two-stage outputs show four failure modes:
1. boundary artifact near the low-wavenumber edge
2. overshoot / amplitude explosion on some spectra
3. jagged fingerprint-region artifacts
4. over-smoothing of real spectral structure

Those failure modes point to a combination of:
- poor boundary handling
- decoder upsampling artifacts
- objective mismatch (MSE-style reconstruction alone)
- training noise that does not match instrument noise
- skip connections passing noise too directly into the decoder

## Ranked Fix Order

### Phase 0: Reset Baseline
Keep only a minimal baseline set for restart:
- `models/resnet_threshold1d.pth`
- `models/resunet1d.pth`
- `models/resunet1d_best.pth`
- `models/resunet1d_best_v2.pth`
- `models/two_stage_refiner_best.pth`
- source files under `models/` and `scripts/`

Everything else from probes, sweeps, push variants, and temporary plotting should stay deleted.

### Phase 1: Immediate Non-Retraining Fixes
Apply these before the next training run.

1. Reflection padding everywhere in 1D conv stacks.
- Replace zero-padding behavior with explicit reflection padding.
- Target files: `models/resunet1d.py`, `models/two_stage_refiner.py`, any future 1D blocks.
- Expected effect: remove edge spike near ~400 cm^-1.

2. Decoder upsampling fix.
- Replace `ConvTranspose1d` with `Upsample(mode='linear' or 'nearest') + Conv1d`.
- Expected effect: reduce checkerboard / zigzag decoder artifacts.

3. S-G fallback blend for inference only.
- Add optional post-model blend for robustness checks:
  - `output = alpha * model_output + (1 - alpha) * savgol(output)`
- Use only as a benchmark / fail-safe, not as the publishable primary method.
- Expected effect: quick reduction of oscillatory artifacts while retraining work proceeds.

### Phase 2: First Retraining Run
This is the highest-priority training change set.

1. Residual learning.
- Train the model to predict noise/residual, not the clean signal directly.
- Formula:
  - `pred_clean = noisy - pred_noise`
- Expected effect: stop hallucinating structure in flat regions.

2. Composite loss.
- Replace plain reconstruction loss with:
  - `L = w_l1 * L1(clean, pred)`
  - `+ w_d1 * L1(d/dx clean, d/dx pred)`
  - `+ w_d2 * L1(d2/dx2 clean, d2/dx2 pred)`
  - `+ w_tv * TV(pred)`
  - `+ w_fft * FFT_magnitude_loss(clean, pred)`
  - `+ w_amp * peak_amplitude_penalty(clean, pred)`
- Recommended starting weights:
  - `w_l1 = 1.0`
  - `w_d1 = 0.15`
  - `w_d2 = 0.10`
  - `w_tv = 0.01`
  - `w_fft = 0.05`
  - `w_amp = 0.10`
- Expected effect: preserve shape, suppress jaggedness, control overshoot, preserve peak heights.

3. Per-spectrum normalization.
- Keep normalization individual per spectrum.
- Save mean/std for inversion and analysis.
- Do not rely on batch-global intensity statistics.

4. Normalization layers.
- Replace or ablate `BatchNorm1d` with `InstanceNorm1d` or `GroupNorm`.
- Reason: batch norm can distort relative amplitude structure in spectral data.

### Phase 3: Architecture Upgrade
After Phase 2 stabilizes training, improve the network rather than stacking more checkpoints.

1. Gated skip connections.
- Add attention gates on encoder-to-decoder skip paths.
- Reason: skip connections are currently leaking high-frequency noise into the decoder.

2. Channel recalibration.
- Add lightweight SE-style blocks or channel attention inside encoder/decoder blocks.
- Reason: emphasize chemically meaningful bands, suppress nuisance channels.

3. Preserve 1D spectral locality.
- Use larger receptive field selectively with dilated convolutions in bottleneck blocks.
- Do not over-deepen blindly.

4. Keep the model single-stage first.
- Do not begin with a new two-stage system.
- First produce one strong single-stage publishable model.
- Only revisit a second stage if it gives a measurable, reproducible gain on peak metrics.

### Phase 4: Data Pipeline Rebuild
Current model quality is limited by synthetic-pair realism.

1. Instrument-matched noise synthesis.
Generate noisy training targets using a mixture of:
- additive Gaussian noise
- heteroscedastic noise proportional to signal level
- baseline drift / low-frequency wander
- sparse spike artifacts
- slight line broadening / resolution mismatch if present in raw data

2. Curriculum SNR training.
Train in stages:
- early epochs: moderate/high noise to learn denoising
- later epochs: near-clean spectra matching observed real SNR
This reduces aggressive smoothing on already-clean spectra.

3. Region-aware sampling.
During training, overweight examples or loss contributions from:
- fingerprint region
- known peak-dense regions
- boundary regions
This keeps the model from optimizing only easy flat baselines.

4. Validation split discipline.
Use patient/sample-level separation if applicable.
Avoid leakage across train/val/test from related spectra.

### Phase 5: Replace Evaluation
The existing evaluation is not publication-grade on its own.

Use three physics-aware primary metrics in addition to L1 / PSNR / SSIM:

1. Noise-floor metric.
- Standard deviation in baseline-only windows.
- Goal: lower than raw, without systematic distortion.

2. Peak fidelity metric.
For matched peaks, measure:
- center shift
- amplitude error
- width error
- area error

3. Region-wise RMSE / MAE.
Report separately for:
- boundary region
- fingerprint region
- smooth baseline region
- high-wavenumber region

Add these secondary metrics:
- Pearson correlation
- derivative correlation
- spectral angle mapper
- residual power spectrum ratio

### Phase 6: Publication Protocol
A publishable result needs more than one best-case figure.

1. Lock a held-out test set.
- Never tune on it.

2. Run ablations.
At minimum:
- baseline ResUNet
- + reflection padding
- + residual learning
- + composite loss
- + better normalization layer
- + gated skips / SE blocks
- + realistic noise curriculum

3. Report mean +/- std across multiple seeds.
- Minimum 3 runs, ideally 5.

4. Include failure cases.
- Show cases where raw is already clean.
- Show cases with dense peaks.
- Show edge-region behavior.

5. Compare against classical methods.
- Savitzky-Golay
- wavelet denoising
- optional ALS + smoothing baseline

## Concrete Repo Changes For Next Coding Pass

### `models/resunet1d.py`
- add explicit reflection padding wrapper block
- replace `ConvTranspose1d` decoder with `Upsample + Conv1d`
- switch to residual-noise prediction mode
- replace `BatchNorm1d` with `InstanceNorm1d` or `GroupNorm`
- add optional gated skip connection path
- add optional SE block path

### `scripts/train_resunet.py`
- implement composite loss
- save normalization stats if needed for inversion
- add curriculum scheduling for synthetic noise severity
- log new metrics per region
- save full config JSON with seed, loss weights, architecture flags

### `scripts/augment_data.py`
- rebuild noise simulator with instrument-matched components
- add spike artifacts and baseline drift
- support SNR curriculum bands

### `scripts/evaluate_resunet.py`
- add physics-aware metrics
- export per-region metrics
- export representative qualitative cases automatically

### New utility modules recommended
- `scripts/losses.py`
- `scripts/noise_model.py`
- `scripts/peak_metrics.py`
- `scripts/region_metrics.py`

## Recommended Next Training Sequence

### Run A: Stabilization
- reflection padding
- upsample+conv decoder
- residual learning
- composite loss
- instance/group norm

Success criteria:
- edge spike removed
- no oscillatory baseline artifacts
- lower baseline noise than raw
- no peak overshoot on inspected cases

### Run B: Spectral-preservation upgrade
- Run A + gated skips
- Run A + SE blocks

Success criteria:
- improved peak fidelity without noise-floor regression

### Run C: Data realism upgrade
- Run B + instrument-matched noise
- Run B + curriculum SNR

Success criteria:
- model behaves correctly on already-clean raw spectra
- fewer over-smoothing failures

## What Not To Do Next
- do not keep multiplying checkpoint variants without a fixed ablation table
- do not optimize only PSNR / SSIM
- do not trust raw-vs-output visuals alone
- do not retain transposed convolutions if artifacts persist
- do not publish without peak/baseline region metrics and seed variance

## Definition Of "Better Model"
The next model should only replace the current baseline if it satisfies all of:
- zero visible edge artifact on representative samples
- lower baseline noise-floor than raw and current baseline model
- peak-center error below a fixed threshold
- lower region-wise error in fingerprint region
- no amplitude blow-up on clean-ish spectra
- stable performance across multiple seeds

## Immediate Next Coding Task
Implement Run A only.
That is the shortest path to a materially better model:
- reflection padding
- upsample+conv decoder
- residual prediction
- composite loss
- instance/group norm

Do not add the full attention/two-stage stack until Run A is stable and measured.
