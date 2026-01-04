#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#phase_ou_plots_P15_P28.py

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import pytensor.tensor as pt

# ---------- CONFIG ----------
DATA_CSV = Path("phase_ou_data.csv")
IN_NETCDF = Path("ou_phase_model_posteriors.nc")
FIG_DIR = Path("FigS9_P15_P28")
FIG_DIR.mkdir(exist_ok=True)


def softplus(z):
    return np.log1p(np.exp(z))


def main():
    print("Loading data and trace...")
    df = pd.read_csv(DATA_CSV)
    trace = az.from_netcdf(IN_NETCDF)

    # Map patient indices
    df["patient_idx"] = df["patient_id"].astype("category").cat.codes
    patient_index_to_id = dict(
        enumerate(df["patient_id"].astype("category").cat.categories)
    )
    id_to_patient_idx = {v: k for k, v in patient_index_to_id.items()}

    # Only 1 phase in this FigS9 run
    phase_idx = 0

    # Extract raw parameters from trace
    post = trace.posterior

    mu_0 = post["mu_0"].values                # (chain, draw)
    theta_0_raw = post["theta_0_raw"].values  # (chain, draw)
    sigma_0_raw = post["sigma_0_raw"].values  # (chain, draw)

    alpha_p = post["alpha_p"].values[:, :, phase_idx]   # (chain, draw)
    beta_p  = post["beta_p"].values[:, :, phase_idx]    # (chain, draw)
    gamma_p = post["gamma_p"].values[:, :, phase_idx]   # (chain, draw)

    u_i = post["u_i"].values                  # (chain, draw, patient)
    v_i = post["v_i"].values                  # (chain, draw, patient)
    w_i = post["w_i"].values                  # (chain, draw, patient)

    # Global transformed parameters
    theta_0 = softplus(theta_0_raw) + 1e-3
    sigma_0 = softplus(sigma_0_raw) + 1e-3

    # Helper: compute patient-level θ and σ for a given patient id
    def patient_params(patient_id):
        pidx = id_to_patient_idx[patient_id]

        u = u_i[:, :, pidx]
        v = v_i[:, :, pidx]
        w = w_i[:, :, pidx]

        # θ: softplus(theta_0_raw + beta_p + v)
        theta_raw = theta_0_raw + beta_p + v
        theta = softplus(theta_raw) + 1e-3

        # σ: log-scale
        sigma_log = np.log(sigma_0) + gamma_p + w
        sigma = np.exp(sigma_log) + 1e-3

        return theta.ravel(), sigma.ravel()

    # Compute for P15 and P28
    print("Computing posterior θ, σ for P15 and P28...")
    theta_P15, sigma_P15 = patient_params("P15")
    theta_P28, sigma_P28 = patient_params("P28")

    # ------------- Plot: θ and σ histograms -------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # θ
    ax = axes[0]
    ax.hist(theta_P15, bins=50, density=True, alpha=0.5, label="P15")
    ax.hist(theta_P28, bins=50, density=True, alpha=0.5, label="P28")
    ax.set_xlabel("θ (mean-reversion)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior θ – induction")
    ax.legend()

    # σ
    ax = axes[1]
    ax.hist(sigma_P15, bins=50, density=True, alpha=0.5, label="P15")
    ax.hist(sigma_P28, bins=50, density=True, alpha=0.5, label="P28")
    ax.set_xlabel("σ (volatility)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior σ – induction")
    ax.legend()

    fig.suptitle("P15 vs P28 – OU parameters (induction)")
    fig.tight_layout()

    outpath = FIG_DIR / "FigS9_P15_P28_induction_theta_sigma.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    print("Saved Fig S9 to:", outpath)


if __name__ == "__main__":
    main()
