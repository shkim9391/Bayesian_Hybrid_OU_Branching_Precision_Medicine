#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#make_S9_table.py

import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path

# ------------ CONFIG ---------------
DATA_CSV = Path("phase_ou_data.csv")
IN_NETCDF = Path("ou_phase_model_posteriors.nc")
OUT_CSV = Path("S9_parameter_table.csv")


# Softplus (same as used in the model)
def softplus(z):
    return np.log1p(np.exp(z))


# ------------ MAIN -----------------
def main():
    print("Loading posterior...")
    trace = az.from_netcdf(IN_NETCDF)
    post = trace.posterior

    df = pd.read_csv(DATA_CSV)
    df["patient_idx"] = df["patient_id"].astype("category").cat.codes
    patients = df["patient_id"].astype("category").cat.categories
    id_to_idx = {p: i for i, p in enumerate(patients)}

    # Only 1 phase used (induction)
    phase_idx = 0

    # Extract raw parameters
    theta_0_raw = post["theta_0_raw"].values
    sigma_0_raw = post["sigma_0_raw"].values
    beta_p  = post["beta_p"].values[:, :, phase_idx]
    gamma_p = post["gamma_p"].values[:, :, phase_idx]
    v_i = post["v_i"].values
    w_i = post["w_i"].values

    # Global transforms
    theta_0 = softplus(theta_0_raw) + 1e-3
    sigma_0 = softplus(sigma_0_raw) + 1e-3

    # Collect rows
    rows = []

    def summarize(arr):
        arr = arr.ravel()
        return {
            "mean": np.mean(arr),
            "sd": np.std(arr),
            "hdi_50_low": np.percentile(arr, 25),
            "hdi_50_high": np.percentile(arr, 75),
            "hdi_95_low": np.percentile(arr, 2.5),
            "hdi_95_high": np.percentile(arr, 97.5),
        }

    # Loop over P15 and P28
    for pid in ["P15", "P28"]:
        pidx = id_to_idx[pid]

        # Patient-level θ and σ
        theta_raw = theta_0_raw + beta_p + v_i[:, :, pidx]
        theta = softplus(theta_raw) + 1e-3

        sigma_log = np.log(sigma_0) + gamma_p + w_i[:, :, pidx]
        sigma = np.exp(sigma_log) + 1e-3

        theta_stats = summarize(theta)
        sigma_stats = summarize(sigma)

        # Add to rows
        rows.append({
            "patient": pid,
            "parameter": "theta (mean-reversion)",
            **theta_stats
        })

        rows.append({
            "patient": pid,
            "parameter": "sigma (volatility)",
            **sigma_stats
        })

    # Output table
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
