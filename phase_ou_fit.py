#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 06:45:06 2025

@author: seung-hwan.kim
"""

#phase_ou_fit.py

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path
import pytensor.tensor as pt

def softplus(z):
    # simple softplus, good enough for our scale
    return pt.log1p(pt.exp(z))


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DATA_CSV   = Path("phase_ou_data.csv")          # <-- change if needed
OUT_NETCDF = Path("ou_phase_model_posteriors.nc")
SUMMARY_CSV = Path("phase_ou_summary.csv")


# ---------------------------------------------------------
# DATA PREP
# ---------------------------------------------------------
def prepare_phase_data(df):
    """
    Expects columns:
        patient_id, phase, t, phase_time, x, event
    """
    df = df.copy()

    # Ensure correct sorting
    df = df.sort_values(["patient_id", "phase", "t"]).reset_index(drop=True)

    # Integer indices for patient and phase
    df["patient_idx"] = df["patient_id"].astype("category").cat.codes
    df["phase_idx"]   = df["phase"].astype("category").cat.codes

    patient_index_to_id = dict(enumerate(df["patient_id"].astype("category").cat.categories))
    phase_index_to_label = dict(enumerate(df["phase"].astype("category").cat.categories))

    return df, patient_index_to_id, phase_index_to_label


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
def build_phase_aware_ou_model(df):
    """
    Hierarchical OU parameters by patient + phase,
    plus exponential time-to-relapse hazard depending on OU state.
    """
    import pytensor.tensor as pt
    import numpy as np

    patient_idx = df["patient_idx"].values
    phase_idx   = df["phase_idx"].values
    t           = df["t"].values
    x           = df["x"].values
    event       = df["event"].values.astype(int)
    phase_time  = df["phase_time"].values

    N_patients = df["patient_idx"].nunique()
    N_phases   = df["phase_idx"].nunique()

    with pm.Model() as model:

        # -------------------------------------------------
        # GROUP-LEVEL PRIORS
        # -------------------------------------------------
        mu_0 = pm.Normal("mu_0", 0, 1)

        # work on unconstrained scale, enforce positivity via softplus
        theta_0_raw = pm.Normal("theta_0_raw", 0, 1.0)
        sigma_0_raw = pm.Normal("sigma_0_raw", 0, 1.0)

        theta_0 = pm.Deterministic("theta_0", softplus(theta_0_raw) + 1e-3)
        sigma_0 = pm.Deterministic("sigma_0", softplus(sigma_0_raw) + 1e-3)

        # -------------------------------------------------
        # PHASE-LEVEL EFFECTS
        # -------------------------------------------------
        alpha_p = pm.Normal("alpha_p", 0, 0.5, shape=N_phases)   # μ shift per phase
        beta_p  = pm.Normal("beta_p", 0, 0.5, shape=N_phases)   # θ shift (raw)
        gamma_p = pm.Normal("gamma_p", 0, 0.5, shape=N_phases)  # log-scale σ shift

        # -------------------------------------------------
        # PATIENT-LEVEL RANDOM EFFECTS
        # -------------------------------------------------
        u_i = pm.Normal("u_i", 0, 0.3, shape=N_patients)   # μ RE
        v_i = pm.Normal("v_i", 0, 0.3, shape=N_patients)   # θ RE (raw)
        w_i = pm.Normal("w_i", 0, 0.3, shape=N_patients)   # σ RE

        # -------------------------------------------------
        # OBSERVATION-SPECIFIC PARAMETERS
        # -------------------------------------------------
        # raw theta on R, then softplus to (0, +inf)
        theta_raw = theta_0_raw + beta_p[phase_idx] + v_i[patient_idx]
        theta_obs = softplus(theta_raw) + 1e-3

        mu_obs = mu_0 + alpha_p[phase_idx] + u_i[patient_idx]

        sigma_log = pt.log(sigma_0) + gamma_p[phase_idx] + w_i[patient_idx]
        sigma_obs = pt.exp(sigma_log) + 1e-3

        # -------------------------------------------------
        # OU TRANSITION LIKELIHOOD
        # -------------------------------------------------
        t_prev = t[:-1]
        t_next = t[1:]
        x_prev = x[:-1]
        x_next = x[1:]

        mu_prev    = mu_obs[:-1]
        theta_prev = theta_obs[:-1]
        sigma_prev = sigma_obs[:-1]

        # clamp dt to avoid zeros
        dt = t_next - t_prev
        dt = pt.clip(dt, 1e-4, np.inf)

        # OU analytic transition
        mean_next = x_prev * pt.exp(-theta_prev * dt) + \
                    mu_prev * (1 - pt.exp(-theta_prev * dt))

        var_next  = (sigma_prev**2) / (2 * theta_prev) * \
                    (1 - pt.exp(-2 * theta_prev * dt))

        # clamp variance to avoid 0 or negative numerical issues
        var_next = pt.clip(var_next, 1e-6, np.inf)

        pm.Normal(
            "x_next",
            mu=mean_next,
            sigma=pt.sqrt(var_next),
            observed=x_next
        )

        # -------------------------------------------------
        # STATE-DEPENDENT HAZARD (EXPONENTIAL SURVIVAL)
        # -------------------------------------------------
        h0_phase_raw = pm.Normal("h0_phase_raw", 0, 1.0, shape=N_phases)
        h0_phase = pm.Deterministic("h0_phase", softplus(h0_phase_raw) + 1e-6)

        eta = pm.Normal("eta", 0, 1.0)

        hazard = h0_phase[phase_idx] * pt.exp(eta * x)

        # Exponential survival:
        #   logp = event * (log λ - λ t) + (1 - event) * (-λ t)
        logp_surv = event * (pt.log(hazard) - hazard * phase_time) + \
                    (1 - event) * (-hazard * phase_time)

        pm.Potential("surv_like", logp_surv.sum())

    return model

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("Loading data from:", DATA_CSV)
    df = pd.read_csv(DATA_CSV)

    df, patient_index_to_id, phase_index_to_label = prepare_phase_data(df)

    print("N patients:", len(patient_index_to_id))
    print("N phases:", len(phase_index_to_label))

    print("Building model...")
    model = build_phase_aware_ou_model(df)

    print("Sampling...")
    with model:
        trace = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.92,
            random_seed=42
        )

    print("Saving NetCDF to:", OUT_NETCDF)
    az.to_netcdf(trace, OUT_NETCDF)

    print("Saving summary to:", SUMMARY_CSV)
    summary_df = az.summary(trace)
    summary_df.to_csv(SUMMARY_CSV)

    print("Done.")


if __name__ == "__main__":
    main()