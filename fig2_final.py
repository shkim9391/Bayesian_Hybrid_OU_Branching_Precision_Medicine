#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2 (Bayesian), fully reproducible:

Inputs
------
1) Clinical metadata:
   kmt2a_clinical_data.xlsx

2) Bayesian OU–Branching posterior:
   posterior.nc (ArviZ InferenceData with r_raw, theta_raw, sigma_raw)

Outputs
-------
1) fig2_group_param_table_with_hdi.csv
   Columns:
     Group, Theta_Mean, Sigma_Mean, R_Mean, DriftToNoise,
     r_hdi_low, r_hdi_high, dtn_hdi_low, dtn_hdi_high

2) Figure2_Bayesian_composite_CI_exact.{png,pdf}
   Two-panel Nature-style figure:
     A: Mean r with 95% HDI
     B: Drift-to-noise (σ²/θ) with 95% HDI
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------
# 0. Paths – EDIT THESE IF YOUR LAYOUT CHANGES
# ---------------------------------------------------------------------
PROJECT_ROOT = Path("/Bayesian_Hybrid_OU_Branching_Precision_Medicine")
FIG2_DIR = PROJECT_ROOT / "Figure 2"
OUT_DIR = FIG2_DIR / "out_bayes_real"

CLIN_PATH = FIG2_DIR / "kmt2a_clinical_data.xlsx"
IDATA_PATH = OUT_DIR / "posterior.nc"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not CLIN_PATH.exists():
    raise FileNotFoundError(f"Clinical metadata not found at:\n  {CLIN_PATH}")

if not IDATA_PATH.exists():
    raise FileNotFoundError(f"posterior.nc not found at:\n  {IDATA_PATH}")

# ---------------------------------------------------------------------
# 1. Load clinical metadata, derive group labels
# ---------------------------------------------------------------------
# The Excel file has 3 header lines before the actual header row,
# so we use header=3 to get columns: Patient_ID, Disease, Group, ...
clin = pd.read_excel(CLIN_PATH, header=3)

# Unique group labels present in the cohort
clin_groups = sorted(clin["Group"].dropna().unique().tolist())
print("Groups in clinical metadata:", clin_groups)

# IMPORTANT:
# The OU–Branching model used a specific group order when indexing
# r_raw, theta_raw, sigma_raw. We encode that explicitly here.
group_order = [
    "Early",
    "Early/refractory",
    "Late",
    "Remission",
    "Very early",
    "Very early/refractory",
]

# Basic consistency check (optional, but nice for reviewers)
missing_in_clin = [g for g in group_order if g not in clin_groups]
if missing_in_clin:
    print("WARNING: These groups are not found in clinical metadata:", missing_in_clin)

n_groups = len(group_order)

# ---------------------------------------------------------------------
# 2. Load posterior samples from OU–Branching model
# ---------------------------------------------------------------------
idata = az.from_netcdf(IDATA_PATH)

# We assume posterior variables have shape (..., n_groups) with
# the last dimension aligned to `group_order`.
r_raw = idata.posterior["r_raw"].values       # e.g. (chains, draws, groups)
theta_raw = idata.posterior["theta_raw"].values
sigma_raw = idata.posterior["sigma_raw"].values

# Flatten all non-group dims into "samples" and keep groups as columns
r = r_raw.reshape(-1, n_groups)       # (n_samples, n_groups)
theta = theta_raw.reshape(-1, n_groups)
sigma = sigma_raw.reshape(-1, n_groups)

print("Posterior shapes (samples × groups):")
print("  r     :", r.shape)
print("  theta :", theta.shape)
print("  sigma :", sigma.shape)

# ---------------------------------------------------------------------
# 3. Highest-posterior-density intervals (HPDI/HDI)
# ---------------------------------------------------------------------
def hdi_95(array_2d, cred_mass=0.95):
    """
    Exact highest-posterior-density interval (HPDI) for each column.
    array_2d: shape (n_samples, n_groups)
    """
    arr = np.asarray(array_2d)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (samples × groups), got {arr.shape}")

    n, k = arr.shape
    sorted_samples = np.sort(arr, axis=0)

    interval_idx_inc = int(np.floor(cred_mass * n))
    if interval_idx_inc < 1:
        raise ValueError("Not enough samples to compute HDI")

    n_intervals = n - interval_idx_inc
    interval_width = sorted_samples[interval_idx_inc:, :] - sorted_samples[:n_intervals, :]

    min_idx = np.argmin(interval_width, axis=0)
    idx_upper = min_idx + interval_idx_inc

    lower = sorted_samples[min_idx, np.arange(k)]
    upper = sorted_samples[idx_upper, np.arange(k)]

    return lower, upper

# ---------------------------------------------------------------------
# 4. Group-level summaries: means + HDIs
# ---------------------------------------------------------------------
# Means
theta_mean = theta.mean(axis=0)
sigma_mean = sigma.mean(axis=0)
r_mean = r.mean(axis=0)

# Drift-to-noise σ²/θ
dtn = sigma**2 / theta
dtn_mean = dtn.mean(axis=0)

# HDIs for r and drift-to-noise
r_low, r_high = hdi_95(r)
dtn_low, dtn_high = hdi_95(dtn)

# ---------------------------------------------------------------------
# 5. Assemble group summary table
# ---------------------------------------------------------------------
summary_df = pd.DataFrame({
    "Group": group_order,
    "Theta_Mean": theta_mean,
    "Sigma_Mean": sigma_mean,
    "R_Mean": r_mean,
    "DriftToNoise": dtn_mean,
    "r_hdi_low": r_low,
    "r_hdi_high": r_high,
    "dtn_hdi_low": dtn_low,
    "dtn_hdi_high": dtn_high,
})

table_path = OUT_DIR / "fig2_group_param_table_with_hdi.csv"
summary_df.to_csv(table_path, index=False)
print("Saved group parameter table to:\n  ", table_path)

# ---------------------------------------------------------------------
# 6. Plot Figure 2 – Nature-style minimal aesthetic
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

x = np.arange(n_groups)
width = 0.6

dark_blue = "#1f77b4"
light_blue = "#c7dcef"

# ---------------------- Panel A: mean r -------------------------------
ax1.bar(x, r_mean, width=width, color=dark_blue, zorder=2)

for i in range(n_groups):
    rect = Rectangle(
        (x[i] - width / 2, r_low[i]),
        width,
        r_high[i] - r_low[i],
        facecolor=light_blue,
        edgecolor="none",
        zorder=1,
    )
    ax1.add_patch(rect)

ax1.axhline(0, color="black", linewidth=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(group_order, rotation=35, ha="right")
ax1.set_ylabel("Mean r")
ax1.set_title("A.  Mean r by group (95% HDI)", fontweight="bold", fontsize=10)

# ----------------- Panel B: drift-to-noise σ²/θ ----------------------
ax2.bar(x, dtn_mean, width=width, color=dark_blue, zorder=2)

for i in range(n_groups):
    rect = Rectangle(
        (x[i] - width / 2, dtn_low[i]),
        width,
        dtn_high[i] - dtn_low[i],
        facecolor=light_blue,
        edgecolor="none",
        zorder=1,
    )
    ax2.add_patch(rect)

ax2.set_xticks(x)
ax2.set_xticklabels(group_order, rotation=35, ha="right")
ax2.set_ylabel("Drift-to-noise (σ²/θ)")
ax2.set_title("B.  Drift-to-noise by group (95% HDI)", fontweight="bold", fontsize=10)

out_png = OUT_DIR / "Figure2_Bayesian_composite_CI_exact.png"
out_pdf = OUT_DIR / "Figure2_Bayesian_composite_CI_exact.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")

print("Saved Figure 2 to:\n  ", out_png, "\n  ", out_pdf)
