#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#figs8_remission_summary.py for Supplementary Figure S8 and Table S2

import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# This script assumes it lives inside:
# /Users/seung-hwan.kim/Desktop/Bayesian_Hybrid_OU_Branching_Precision_Medicine/FigS8
BASE_DIR = Path(__file__).resolve().parent

REMISSION_PTS = ["P1", "P2", "P7", "P8", "P15",
                 "P16", "P19", "P26", "P27", "P29", "P108"]

OUT_DIR = BASE_DIR / "FigS8_Remission_Summary"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load longitudinal series
# ---------------------------------------------------------------------
series_path = BASE_DIR / "series_auto.csv"
series = pd.read_csv(series_path)

# Sanity check: expecting columns Patient_ID, series, t, value
expected_cols = {"Patient_ID", "series", "t", "value"}
if not expected_cols.issubset(series.columns):
    raise ValueError(
        f"Expected columns {expected_cols} in series_auto.csv, "
        f"found {list(series.columns)}"
    )

# ---------------------------------------------------------------------
# 2. Extract posterior trajectories & parameter summaries
# ---------------------------------------------------------------------
traj_dict = {}   # {patient: (t, posterior_mean_traj)}
params = []      # list of dicts with patient, mu, theta, sigma

for pid in REMISSION_PTS:
    nc_path = BASE_DIR / f"{pid}_ou_posterior.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"Posterior file not found for {pid}: {nc_path}")

    print(f"[INFO] Loading posterior for {pid} from {nc_path}")
    idata = az.from_netcdf(nc_path)

    # Timepoints for trait series (x) for this patient
    sub = series.query("Patient_ID == @pid and series == 'x'").sort_values("t")
    t = sub["t"].to_numpy(dtype=float)
    if t.size == 0:
        print(f"[WARN] No x-series found for {pid}, skipping.")
        continue

    # Flatten posterior draws
    mu_draws = idata.posterior["mu"].values.reshape(-1)
    theta_draws = idata.posterior["theta"].values.reshape(-1)
    sigma_draws = idata.posterior["sigma"].values.reshape(-1)

    # Store posterior means for violin plots
    params.append(dict(
        patient=pid,
        mu=mu_draws.mean(),
        theta=theta_draws.mean(),
        sigma=sigma_draws.mean()
    ))

    # -----------------------------------------------------------------
    # Compute an approximate posterior mean trajectory
    # -----------------------------------------------------------------
    # If a predicted trajectory is already stored (e.g. 'x_pred' or 'x_ppc'),
    # use it. Otherwise, approximate via the analytic OU mean.
    posterior = idata.posterior

    if "x_pred" in posterior.data_vars:
        x_draws = posterior["x_pred"].values.reshape(-1, t.size)
        m_traj = x_draws.mean(axis=0)
    elif "x_ppc" in posterior.data_vars:
        x_draws = posterior["x_ppc"].values.reshape(-1, t.size)
        m_traj = x_draws.mean(axis=0)
    else:
        # Approximate: use OU mean formula for each posterior draw and average.
        # X_t = mu + (x0 - mu) * exp(-theta * dt)
        print(f"[INFO] Approximating OU mean trajectory analytically for {pid}")
        x0 = sub["value"].iloc[0]
        t0 = t[0]

        n_draws = min(500, mu_draws.size)
        idx = np.random.choice(mu_draws.size, size=n_draws, replace=False)

        traj_samples = np.zeros((n_draws, t.size), dtype=float)
        for k, j in enumerate(idx):
            mu_j = float(mu_draws[j])
            theta_j = float(theta_draws[j])
            # Build trajectory deterministically from OU mean
            vals = np.zeros_like(t, dtype=float)
            vals[0] = x0
            for i in range(1, t.size):
                dt = t[i] - t[i-1]
                vals[i] = mu_j + (vals[i-1] - mu_j) * np.exp(-theta_j * dt)
            traj_samples[k, :] = vals

        m_traj = traj_samples.mean(axis=0)

    traj_dict[pid] = (t, m_traj)

# Convert parameter list to DataFrame
params_df = pd.DataFrame(params)
params_df.to_csv(OUT_DIR / "remission_param_summary.csv", index=False)
print(f"[INFO] Saved parameter summary to {OUT_DIR/'remission_param_summary.csv'}")

# Optionally, quick check plot of averaged trajectory
# (You can customize or replace this with full publication-quality figure.)

# ---------------------------------------------------------------------
# Example: averaged remission trajectory (Panel A prototype)
# ---------------------------------------------------------------------
all_times = np.linspace(0, max(max(t) for t, _ in traj_dict.values()), 100)
interp_trajs = []

for pid, (t, m_traj) in traj_dict.items():
    interp = np.interp(all_times, t, m_traj)
    interp_trajs.append(interp)

interp_trajs = np.vstack(interp_trajs)
mean_traj = interp_trajs.mean(axis=0)
q_low, q_high = np.percentile(interp_trajs, [2.5, 97.5], axis=0)

fig, ax = plt.subplots(figsize=(6,4))
for row in interp_trajs:
    ax.plot(all_times, row, color="lightcoral", alpha=0.2, lw=1)
ax.plot(all_times, mean_traj, color="red", lw=2, label="Remission mean trajectory")
ax.fill_between(all_times, q_low, q_high, color="lightblue", alpha=0.4,
                label="Remission 95% band")
ax.set_xlabel("time (y)")
ax.set_ylabel("normalized trait")
ax.set_title("Remission-group OU mean trajectories")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "remission_mean_trajectory.png", dpi=300)
plt.close(fig)

print(f"[INFO] Saved example trajectory plot to {OUT_DIR/'remission_mean_trajectory.png'}")
