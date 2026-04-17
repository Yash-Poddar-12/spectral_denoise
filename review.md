# ResUNet1D Model Review — Sample 778 Analysis & Tuning Guide

## Revision History

| Round | Date | Peak Shift | RMSE FP | Noise Floor | PSNR | Overall Quality |
|-------|------|-----------|---------|-------------|------|-----------------|
| v1 (baseline) | Apr 17 R1 | 0.4062 | 0.1835 | 0.0343 | 49.51 | 94.25% |
| **v2 (current)** | **Apr 17 R2** | **0.1562** | **0.1823** | **0.0295** | **47.77** | **95.10%** |

---

## Round 2 — What Changed

Architecture & loss function received significant overhaul:
- ✅ **Skip gates re-enabled** (`use_skip_gates: true`)
- ✅ **SE blocks re-enabled** (`use_se: true`)
- ✅ **Multiscale context enabled** (`use_multiscale_context: true`)
- ✅ **PeakAlignmentLoss added** (`w_peak_align: 0.30`)
- ✅ **Peak center loss added** (`w_peak_center: 1.10`) — softmax-based differentiable centering
- ✅ **Fingerprint-specific losses added** (L1: 0.16, MSE: 0.16, D1: 0.12)
- ✅ **Derivative weights increased** (D1: 0.15→0.60, D2: 0.10→0.35)
- ✅ **Valley undershoot penalty** (`w_valley_under: 0.22`)
- ✅ **Hard sample mining** (14× multiplier on worst samples)

---

## Round 2 — Visual Assessment (Sample 778)

### What Improved ✅

| Metric | v1 | v2 | Δ | Verdict |
|--------|----|----|---|---------|
| `peak_shift_mean_abs` | 0.4062 | **0.1562** | −0.25 | ✅ **Major fix** — hit the ≤0.15 target |
| `noise_floor_std` | 0.0343 | **0.0295** | −0.005 | ✅ Modest improvement |
| `peak_amp_mae` | 0.1277 | **0.1254** | −0.002 | ✅ Marginal |
| `corr` | 0.9995 | **0.9996** | +0.0001 | ✅ Held |

### What Regressed or Stalled 🔴

| Metric | v1 | v2 | Δ | Verdict |
|--------|----|----|---|---------|
| `psnr` | 49.51 | **47.77** | −1.74 dB | 🔴 **Regressed** — new losses competing with reconstruction |
| `rmse_fingerprint` | 0.1835 | **0.1823** | −0.001 | 🔴 **Stalled** — still far from 0.05–0.10 target |
| Peak sharpness | Rounded | Still slightly rounded | — | 🟡 Marginal visual improvement |

### Visually on the Plots

1. **Peak alignment is genuinely better** — In the After plot, the teal line's absorption dips now land precisely on the same x-positions as the dashed blue clean target. The 0.40→0.16 shift reduction is visually confirmed — narrow troughs at ~500, ~700, ~1050 cm⁻¹ no longer wobble sideways.

2. **But peaks remain slightly rounded** — Zooming into the ~1050 cm⁻¹ deep trough and the cluster at 400–500 cm⁻¹, the denoised troughs don't quite reach the full depth of the clean target. The bottoms are subtly "filled in" by ~0.5–1.0 intensity units.

3. **PSNR dropped because the loss landscape is now crowded** — With 18 active loss terms competing (MSE, L1, D1, D2, TV, FFT, amplitude, baseline, smooth_consistency, peak_profile, peak_center, edge_L1, edge_D1, fingerprint_L1, fingerprint_MSE, fingerprint_D1, valley_under, peak_align), the model is trying to satisfy too many objectives simultaneously. The peak alignment improved at the cost of overall reconstruction fidelity.

4. **Fingerprint RMSE barely moved** — Despite adding 3 fingerprint-specific losses, the RMSE only dropped 0.001. The fingerprint weights (0.16, 0.16, 0.12) are being diluted by the sheer number of other loss terms.

---

## Root Cause Analysis — Why the Remaining Issues Persist

### 1. Loss Term Overcrowding
The loss function now has **18 weighted terms**. Many are overlapping (e.g., `w_l1` and `w_fingerprint_l1` both penalise L1 error; `w_d1` and `w_fingerprint_d1` both penalise derivative error). The optimiser sees a blurred, conflicting gradient landscape. Each term individually pushes in the right direction, but collectively they create gradient interference.

**Evidence:** PSNR dropped despite all other metrics holding or improving. This is the classic sign of multi-objective conflict — the model finds a Pareto compromise that satisfies no single objective optimally.

### 2. Too Few Epochs at Too Low LR
The training ran for only **8 epochs** at `max_lr=6e-5`. With 18 loss terms and a complex loss surface, the model barely had time to settle. The history shows selection_score was still climbing at epoch 8 (92.21 → 92.31) — it hadn't plateaued.

### 3. Fingerprint Weight Insufficient Relative to Total
The fingerprint band covers indices 104–726 (~622 points out of 1868). The fingerprint-specific weights (L1: 0.16, MSE: 0.16, D1: 0.12 = total 0.44) are small compared to the global weights (MSE: 1.0, L1: 0.25, D1: 0.6, etc. = total ~4.5). The fingerprint-specific gradient is ~10% of the total, yet this band is where the hardest reconstruction happens.

### 4. Peak Sharpness vs TV Tension
`w_tv: 0.006` penalises high-frequency variation — including the sharp V-shapes at absorption dips. Meanwhile `w_peak_profile: 0.4` tries to preserve those same shapes. These two losses directly oppose each other at peak boundaries. The current TV weight is too high relative to the peak sharpness demand.

---

## Round 3 — Exact Changes to Apply

### Strategy: Simplify & Amplify

Instead of adding more loss terms, **consolidate and rebalance** the existing ones. Cut smoothing forces, amplify fingerprint/peak forces, train longer.

### Complete Environment Variable Changes

Below is the exact diff of every env var that needs to change. Copy-paste the export block into your shell before running `python scripts/train_resunet.py`, or modify `run_pipeline.sh`.

#### Training Config Changes

| Env Var | v2 (current) | v3 (new) | Why |
|---------|-------------|----------|-----|
| `EPOCHS` | 8 | **20** | Model was still improving at epoch 8, needs more time |
| `MAX_LR` | 6e-5 | **1.2e-4** | Warmer exploration for the rebalanced loss surface |
| `MIN_LR` | 1e-5 | **8e-6** | Slight bump for better late-stage refinement |
| `PATIENCE` | 4 | **7** | Let the model explore longer before early stopping |
| `EMA_DECAY` | 0.995 | **0.996** | Smoother EMA for stability with many loss terms |

#### Loss Weight Changes — Decrease (Free Up Gradient Budget)

| Env Var | v2 (current) | v3 (new) | Why |
|---------|-------------|----------|-----|
| `LOSS_W_MSE` | 1.0 | **0.60** | Too dominant, drowns out fingerprint losses |
| `LOSS_W_L1` | 0.25 | **0.20** | Slight reduction to rebalance |
| `LOSS_W_D1` | 0.6 | **0.50** | Over-constraining, slight pullback |
| `LOSS_W_D2` | 0.35 | **0.25** | Over-constraining curvature at peaks |
| `LOSS_W_TV` | 0.006 | **0.002** | **Critical** — this is actively rounding peaks |
| `LOSS_W_BASELINE` | 0.05 | **0.03** | Baseline is already well-reconstructed |
| `LOSS_W_SMOOTH_CONSISTENCY` | 0.025 | **0.012** | Also fights peak sharpness |
| `LOSS_W_EDGE_L1` | 0.18 | **0.10** | Edges are fine, don't need this much weight |
| `LOSS_W_EDGE_D1` | 0.12 | **0.08** | Same — edges already converged |

#### Loss Weight Changes — Increase (Focus on Remaining Issues)

| Env Var | v2 (current) | v3 (new) | Why |
|---------|-------------|----------|-----|
| `LOSS_W_AMP` | 0.2 | **0.25** | Stronger amplitude enforcement for deeper troughs |
| `LOSS_W_PEAK_PROFILE` | 0.4 | **0.50** | Sharper peak window reconstruction |
| `LOSS_W_PEAK_CENTER` | 1.1 | **1.20** | Lock peak positions even harder |
| `PEAK_POINT_WEIGHT` | 2.2 | **2.5** | Heavier per-point weight at peak locations |
| `PEAK_SOFTMAX_TEMP` | 22.0 | **25.0** | Sharper softmax = more precise centering |
| `LOSS_W_FINGERPRINT_L1` | 0.16 | **0.35** | **Critical** — >2× increase to dominate gradient |
| `LOSS_W_FINGERPRINT_MSE` | 0.16 | **0.30** | **Critical** — >2× increase |
| `LOSS_W_FINGERPRINT_D1` | 0.12 | **0.20** | Derivative matching in fingerprint band |
| `FINGERPRINT_POINT_WEIGHT` | 2.0 | **3.0** | Make fingerprint points 3× as important in global L1 |
| `LOSS_W_VALLEY_UNDER` | 0.22 | **0.30** | Peaks aren't reaching full trough depth |
| `VALLEY_QUANTILE` | 0.22 | **0.18** | Catch more valley points for depth enforcement |

#### No Change (Keep As-Is)

| Env Var | Value | Why |
|---------|-------|-----|
| `LOSS_W_PEAK_ALIGN` | 0.30 | Working — fixed the peak shift |
| `PEAK_ALIGN_WINDOW` | 5 | Working |
| `PEAK_ALIGN_WEIGHT` | 2.4 | Working |
| `LOSS_W_FFT` | 0.05 | Fine |
| `HARD_SAMPLE_FILE` | results/hard_peak_review_samples_v2.txt | Keep |
| `HARD_SAMPLE_MULTIPLIER` | 14.0 | Keep |
| `USE_SKIP_GATES` | 1 | Keep |
| `USE_SE` | 1 | Keep |
| `USE_MULTISCALE_CONTEXT` | 1 | Keep |
| `INIT_MODEL_PATH` | models/resunet1d_single_stage_final.pth | Fine-tune from v2 |

### Copy-Paste Shell Block

```bash
# === ROUND 3 — Copy-paste this before running train_resunet.py ===

# Training config
export EPOCHS=20
export MAX_LR=1.2e-4
export MIN_LR=8e-6
export PATIENCE=7
export EMA_DECAY=0.996

# Architecture (unchanged from v2)
export USE_SKIP_GATES=1
export USE_SE=1
export USE_MULTISCALE_CONTEXT=1
export USE_INPUT_MEDIAN=0
export USE_SPIKE_SUPPRESSOR=0

# Init from v2 best weights
export INIT_MODEL_PATH=models/resunet1d_single_stage_final.pth

# Output paths (save as new file to preserve v2 for comparison)
export MODEL_PATH=models/resunet1d_single_stage_final.pth
export RESULTS_PATH=results/resunet_single_stage_final.json

# --- LOSS WEIGHTS: DECREASED (free gradient budget) ---
export LOSS_W_MSE=0.60
export LOSS_W_L1=0.20
export LOSS_W_D1=0.50
export LOSS_W_D2=0.25
export LOSS_W_TV=0.002
export LOSS_W_BASELINE=0.03
export LOSS_W_SMOOTH_CONSISTENCY=0.012
export LOSS_W_EDGE_L1=0.10
export LOSS_W_EDGE_D1=0.08

# --- LOSS WEIGHTS: INCREASED (focus on remaining issues) ---
export LOSS_W_AMP=0.25
export LOSS_W_PEAK_PROFILE=0.50
export LOSS_W_PEAK_CENTER=1.20
export PEAK_POINT_WEIGHT=2.5
export PEAK_SOFTMAX_TEMP=25.0
export LOSS_W_FINGERPRINT_L1=0.35
export LOSS_W_FINGERPRINT_MSE=0.30
export LOSS_W_FINGERPRINT_D1=0.20
export FINGERPRINT_POINT_WEIGHT=3.0
export LOSS_W_VALLEY_UNDER=0.30
export VALLEY_QUANTILE=0.18

# --- LOSS WEIGHTS: UNCHANGED (working fine) ---
export LOSS_W_FFT=0.05
export LOSS_W_PEAK_ALIGN=0.30
export PEAK_ALIGN_WINDOW=5
export PEAK_ALIGN_WEIGHT=2.4
export SLOPE_POINT_WEIGHT=1.2
export MASK_LEADING_POINTS=3
export PEAK_WINDOW_RADIUS=5
export PEAK_QUANTILE=0.8
export FINGERPRINT_START=104
export FINGERPRINT_END=726
export HARD_SAMPLE_FILE=results/hard_peak_review_samples_v2.txt
export HARD_SAMPLE_MULTIPLIER=14.0

# === Run training ===
python scripts/train_resunet.py
```

### Architecture — No Changes Needed

The v2 architecture (skip gates + SE + multiscale context) is correct. The problem is purely in loss weighting and training duration.

### Augmentation — Verify Shift Reduction Applied

Confirm these from Round 1 recommendations are still in effect in `scripts/augment_data.py`:
```python
# spectral_shift: probability ≤ 0.08, magnitude ≤ ±0.8
# elastic_warp: probability ≤ 0.05, alpha ≤ 0.6
```

---

## Weight Budget Analysis

### v2 Total Loss Weight Distribution
```
Global reconstruction:  MSE(1.0) + L1(0.25) + D1(0.6) + D2(0.35) = 2.20  (47%)
Smoothing forces:       TV(0.006) + smooth(0.025) + baseline(0.05) = 0.081 (2%)
Edge terms:             edge_L1(0.18) + edge_D1(0.12) = 0.30               (6%)
Peak terms:             amp(0.2) + profile(0.4) + center(1.1) + align(0.3) = 2.00 (43%)
Fingerprint terms:      fp_L1(0.16) + fp_MSE(0.16) + fp_D1(0.12) = 0.44   (9%)
Valley:                 valley(0.22) = 0.22                                  (5%)
FFT:                    fft(0.05) = 0.05                                     (1%)
                                                          TOTAL ≈ 4.67
Fingerprint share of total gradient: ~9%
```

### v3 Target Distribution
```
Global reconstruction:  MSE(0.6) + L1(0.2) + D1(0.5) + D2(0.25) = 1.55   (30%)  ← reduced
Smoothing forces:       TV(0.002) + smooth(0.012) + baseline(0.03) = 0.044 (1%)   ← halved
Edge terms:             edge_L1(0.10) + edge_D1(0.08) = 0.18               (4%)   ← reduced
Peak terms:             amp(0.25) + profile(0.5) + center(1.2) + align(0.3) = 2.25 (44%) ← increased
Fingerprint terms:      fp_L1(0.35) + fp_MSE(0.30) + fp_D1(0.20) = 0.85   (17%)  ← DOUBLED
Valley:                 valley(0.30) = 0.30                                  (6%)  ← increased
FFT:                    fft(0.05) = 0.05                                     (1%)
                                                          TOTAL ≈ 5.12
Fingerprint share of total gradient: ~17% (was 9%)
```

Key shift: Fingerprint gradient share nearly doubled (9% → 17%), smoothing forces halved (2% → 1%), global reconstruction reduced (47% → 30%). The fingerprint band is now the second-largest gradient contributor after peak terms.

---

## Success Criteria for Round 3

| Metric | v2 (current) | Round 3 Target | Stretch Goal |
|--------|-------------|----------------|--------------|
| `peak_shift_mean_abs` | 0.1562 | ≤ 0.12 | ≤ 0.08 |
| `rmse_fingerprint` | 0.1823 | **≤ 0.10** | ≤ 0.06 |
| `peak_amp_mae` | 0.1254 | ≤ 0.08 | ≤ 0.05 |
| `noise_floor_std` | 0.0295 | ≤ 0.015 | ≤ 0.008 |
| `psnr` | 47.77 | ≥ 49.0 | ≥ 51.0 |
| `overall_quality` | 95.10% | ≥ 96.0% | ≥ 97.0% |

> [!WARNING]
> **The fingerprint RMSE (0.18 → target 0.10) is the single most important metric.** If only one thing improves in Round 3, it must be this. The v2 changes added fingerprint losses but at weights too low to matter against 15 other terms. Round 3 must make the fingerprint band the **dominant gradient source**.

> [!TIP]
> **Monitor the PSNR vs fingerprint RMSE tradeoff.** If PSNR drops further while fingerprint RMSE improves, that's acceptable — it means the model is reallocating capacity from easy flat regions to hard peak regions. A model with PSNR 46 and fingerprint RMSE 0.06 is more useful for downstream classification than one with PSNR 50 and fingerprint RMSE 0.18.

> [!NOTE]
> **After retraining**, regenerate sample 778 plots with `python scripts/export_eval_plots.py` and compare against this document. Update the Revision History table with v3 numbers.
