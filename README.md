# NIFTY Options IV Surface Reconstruction

Reconstructing a complete, arbitrage-consistent implied volatility surface from sparse real-world NIFTY options market data using Akima spline interpolation and log-moneyness smile fitting.

---

## Overview

Options markets frequently have missing implied volatility (IV) data — not every strike trades at every timestamp. This project solves that problem for **NIFTY 27 JAN 26** options (both CE and PE) by interpolating and extrapolating IV values across strikes using the shape of the volatility smile, conditioned on the underlying price at each point in time.

The result is a fully filled IV surface suitable for downstream use in derivatives pricing, risk management, or model calibration.

---

## The Problem

Given a time-series dataset of NIFTY options IVs across many strikes, a large fraction of entries are `NaN` — the option either didn't trade or data wasn't captured. Naive forward-fill or median imputation ignores the structural shape of the volatility smile. This project instead reconstructs missing IVs using the observed smile at each timestamp.

---

## Methodology

### 1. Log-Moneyness Parameterization

Strikes are transformed to **log-moneyness** (`log(K / S)`) before interpolation, which naturally centers the smile around the ATM strike and makes the interpolation geometry strike-invariant.

### 2. Akima Spline Interpolation (Interior Points)

For strikes that fall within the range of observed data, [Akima1DInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html) is used. Akima splines avoid the oscillation artifacts of cubic splines while preserving local shape — critical for the asymmetric, skewed nature of volatility smiles.

### 3. Blended Edge Extrapolation (Wing Strikes)

For strikes outside the observed range (the wings), a blend of:
- Akima derivative at the boundary point
- Linear regression slope over the 3 nearest observed points

is used with an exponential damping term to prevent unbounded extrapolation:

```
v = v_boundary + slope * (1 - exp(-10 * dist)) / 10
```

This produces smooth, bounded IV estimates even at far OTM strikes.

### 4. Fallback Chain

If interpolation fails for any point, the following fallback cascade is applied:
1. Last known IV for that strike (temporal carry-forward)
2. Median IV of observed strikes at that timestamp
3. Global fallback of `0.12` (12% IV floor)

All final values are clipped to a minimum of `0.001` to prevent non-positive IVs.

---

## Validation — Leave-One-Out Cross Validation (LOOCV)

`loocv.py` validates the interpolation approach by systematically hiding one observed IV at a time and measuring how well the model recovers it from the remaining observations.

Metrics reported:
- **Overall MSE** — across all held-out points
- **Interior MSE** — strikes within the observed range (interpolation regime)
- **Edge MSE** — strikes outside the observed range (extrapolation regime)
- **High IV MSE** — rows where median IV > 0.5 (high volatility regimes)
- **Low IV MSE** — rows where median IV ≤ 0.5 (normal regimes)

---

## Project Structure

```
.
├── dataset.csv              # Raw input — NIFTY options IV data with missing values
├── filled_dataset.csv       # Output — fully reconstructed IV surface
├── submission.csv           # Kaggle-style submission: (id, value) for missing entries
├── solve_iv_surface.py      # Main reconstruction pipeline
├── loocv.py                 # Leave-one-out cross-validation script
└── submission-converter.ipynb  # Notebook to reformat output for submission
```

---

## Installation

```bash
git clone https://github.com/Hiralllll/NIFTY-Options-IV-Surface-Reconstruction.git
cd NIFTY-Options-IV-Surface-Reconstruction
pip install pandas numpy scipy
```

---

## Usage

### Step 1 — Reconstruct the IV Surface

```bash
python solve_iv_surface.py
```

Reads `dataset.csv`, fills all missing IV values, and writes:
- `filled_dataset.csv` — complete dataset with no NaNs
- `submission.csv` — only the filled entries in `(id, value)` format

### Step 2 — Validate with LOOCV

```bash
python loocv.py
```

Prints MSE breakdown across interior, edge, high-IV, and low-IV regimes.

---

## Dataset Format

`dataset.csv` is expected to have the following structure:

| Column | Description |
|---|---|
| `datetime` | Timestamp of the observation |
| `underlying_price` | NIFTY spot price at that timestamp |
| `NIFTY27JAN26<strike>CE` | Implied volatility for a Call at that strike |
| `NIFTY27JAN26<strike>PE` | Implied volatility for a Put at that strike |

Missing values are represented as `NaN`.

---

## Key Design Decisions

**Why Akima over cubic splines?** Cubic splines minimize global curvature, which can produce oscillations between sparse observations. Akima splines are locally determined and handle irregular spacing without overfitting — important for real market data where observed strikes are unevenly distributed.

**Why log-moneyness?** The volatility smile is better behaved as a function of `log(K/S)` than absolute strike. It naturally normalizes for changes in the underlying level across time.

**Why blended extrapolation?** Pure linear extrapolation diverges; pure Akima derivative extrapolation can be noisy at the boundary. The 70/30 blend (linear / Akima) with exponential damping gives stable, realistic wing estimates.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| SciPy (`Akima1DInterpolator`) | Spline interpolation |
| Jupyter Notebook | Submission formatting |
