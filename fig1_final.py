#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 1 – OU calibration method comparison (KMT2A longitudinal cohort)

From-scratch pipeline:

Inputs
------
1) kmt2a_longitudinal_clean.xlsx
   Columns (case-insensitive): Patient, Group, Day, ...
   Group is mapped to ordinal stage codes (very early → late).

Outputs
-------
1) ou_methods_comparison_real.csv
   Per-patient OU parameter estimates for each calibration method:
   columns: Patient_ID, method, mu, theta, sigma

2) Figure1_calimethod_comparison_95CI.png / .pdf
   Cohort-level mean ± 95% CI for mu, theta, sigma across methods.

The script is self-contained and uses pure-Python Metropolis for the
Bayesian fits to maximize reproducibility.
"""

from pathlib import Path
import os

os.environ["PYTENSOR_FLAGS"] = (
    "optimizer=None,linker=py,cxx=,device=cpu,exception_verbosity=high"
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import pymc as pm
import arviz as az
from pytensor.compile import Mode

# ---------------------------------------------------------------------
# Paths – EDIT THESE IF YOUR LAYOUT CHANGES
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(
    "/Bayesian_Hybrid_OU_Branching_Precision_Medicine"
)
FIG1_DIR = PROJECT_ROOT / "Figure 1"
OUT_DIR = FIG1_DIR / "out_compare_real"

EXCEL_PATH = PROJECT_ROOT / "kmt2a_longitudinal_clean.xlsx"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not EXCEL_PATH.exists():
    raise FileNotFoundError(f"Longitudinal Excel not found at:\n  {EXCEL_PATH}")

# ---------------------------------------------------------------------
# PyMC / PyTensor mode (pure Python for reproducibility)
# ---------------------------------------------------------------------
PURE_PY_MODE = Mode(linker="py", optimizer=None)

# ---------------------------------------------------------------------
# OU helpers
# ---------------------------------------------------------------------
def ou_transition_moments(x0, mu, theta, sigma, dt):
    """Mean and variance of OU transition X(t+dt) | X(t)=x0."""
    m = mu + (x0 - mu) * np.exp(-theta * dt)
    v = (sigma**2) / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * dt))
    return m, v


def ou_mle_fit(t, x):
    """OU MLE via Gaussian transition likelihood."""
    dt = np.diff(t)
    x_prev, x_next = x[:-1], x[1:]

    def nll(params):
        mu, theta, sigma = params
        theta = max(theta, 1e-6)
        sigma = max(sigma, 1e-6)
        m, v = ou_transition_moments(x_prev, mu, theta, sigma, dt)
        return 0.5 * np.sum(np.log(2 * np.pi * v) + (x_next - m) ** 2 / v)

    res = minimize(
        nll,
        x0=[float(np.mean(x)), 0.5, 0.5],
        bounds=[(-10, 10), (1e-6, 10.0), (1e-6, 10.0)],
        method="L-BFGS-B",
    )
    return res.x if res.success else [np.nan, np.nan, np.nan]


def ou_mom_estimator(t, x):
    """Method-of-moments estimator for OU (may fail for sparse/low-variance data)."""
    dx = np.diff(x)
    vx = np.var(x)
    if vx <= 1e-12 or len(t) < 3:
        raise ValueError("MoM not identifiable (low variance or <3 points).")
    cov_xdx = np.cov(x[:-1], dx)[0, 1]
    theta = -cov_xdx / vx
    mu = np.mean(x)
    sigma = np.std(dx - (-theta) * (x[:-1] - mu)) / np.sqrt(2)
    return mu, max(theta, 1e-4), sigma


def impute_points_from_bayes(t_obs, x_obs, idata, t_new):
    """OU-consistent midpoint imputation using Bayesian posterior draws."""
    mu = idata.posterior["mu"].values.reshape(-1)
    th = idata.posterior["theta"].values.reshape(-1)
    sg = idata.posterior["sigma"].values.reshape(-1)
    x_imp = []
    for tn in t_new:
        j = np.where(t_obs <= tn)[0].max()
        x0, t0 = float(x_obs[j]), float(t_obs[j])
        dt = tn - t0
        means = mu + (x0 - mu) * np.exp(-th * dt)
        x_imp.append(float(means.mean()))
    return np.array(x_imp)


def impute_points_from_mle(t_obs, x_obs, mu, th, sg, t_new):
    """OU-consistent midpoint imputation using MLE parameters."""
    x_imp = []
    for tn in t_new:
        j = np.where(t_obs <= tn)[0].max()
        x0, t0 = float(x_obs[j]), float(t_obs[j])
        m, _ = ou_transition_moments(x0, mu, th, sg, tn - t0)
        x_imp.append(float(m))
    return np.array(x_imp)


def run_mom_on_augmented(t_obs, x_obs, t_new, x_new):
    """Run MoM on original + imputed time points."""
    t_all = np.concatenate([t_obs, t_new])
    x_all = np.concatenate([x_obs, x_new])
    order = np.argsort(t_all)
    t_all = t_all[order]
    x_all = x_all[order]
    mu_mom, th_mom, sg_mom = ou_mom_estimator(t_all, x_all)
    return dict(mu=mu_mom, theta=th_mom, sigma=sg_mom)


# ---------------------------------------------------------------------
# Per-patient comparison
# ---------------------------------------------------------------------
def compare_patient(
    pid,
    sub,
    impute_midpoints=1,
    impute_source="bayes",
    n_draws=1000,
    n_tune=1000,
):
    """
    Fit OU to a single patient with:
      - MLE
      - Bayesian Metropolis (pure Python)
      - MoM (raw or imputed if <3 points)
    """
    t = sub["t"].to_numpy(dtype=float)
    x = sub["x"].to_numpy(dtype=float)

    # sort by time
    order = np.argsort(t)
    t, x = t[order], x[order]

    out = []

    # collapse duplicate times: keep modal stage at that time
    if np.any(np.diff(t) == 0):
        uniq_t = np.unique(t)
        x_collapsed = []
        for tt in uniq_t:
            vals = x[t == tt]
            mode_val = pd.Series(vals).mode().iloc[0]
            x_collapsed.append(mode_val)
        t = uniq_t
        x = np.asarray(x_collapsed, dtype=float)

    # --- MLE ---
    mu_mle, th_mle, sg_mle = ou_mle_fit(t, x)
    out.append(
        dict(Patient_ID=pid, method="MLE", mu=mu_mle, theta=th_mle, sigma=sg_mle)
    )

    # --- Bayesian (Metropolis) ---
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        theta = pm.HalfNormal("theta", 1)
        sigma = pm.HalfNormal("sigma", 1)

        dt = np.diff(t)
        eps = 1e-9
        for i in range(1, len(t)):
            m, v = ou_transition_moments(float(x[i - 1]), mu, theta, sigma, float(dt[i - 1]))
            sd = pm.math.sqrt(pm.math.maximum(v, eps))
            pm.Normal(f"x_{i}", mu=m, sigma=sd, observed=float(x[i]))

        idata = pm.sample(
            draws=n_draws,
            tune=n_tune,
            chains=1,
            step=pm.Metropolis(),
            cores=1,
            progressbar=True,
            compile_kwargs={"mode": PURE_PY_MODE},
        )

    summary = az.summary(idata, var_names=["mu", "theta", "sigma"])
    mu_b, th_b, sg_b = summary["mean"].values
    out.append(
        dict(Patient_ID=pid, method="Bayesian", mu=mu_b, theta=th_b, sigma=sg_b)
    )

    # --- MoM ---
    if len(t) >= 3:
        try:
            mu_m, th_m, sg_m = ou_mom_estimator(t, x)
            out.append(
                dict(Patient_ID=pid, method="MoM", mu=mu_m, theta=th_m, sigma=sg_m)
            )
        except Exception:
            out.append(
                dict(Patient_ID=pid, method="MoM", mu=np.nan, theta=np.nan, sigma=np.nan)
            )
    else:
        # MoM with midpoint imputation
        if impute_midpoints > 0:
            t_new = np.linspace(float(t.min()), float(t.max()), impute_midpoints + 2)[
                1:-1
            ]
            if impute_source == "bayes":
                x_new = impute_points_from_bayes(t, x, idata, t_new)
            else:
                x_new = impute_points_from_mle(t, x, mu_mle, th_mle, sg_mle, t_new)
            try:
                est = run_mom_on_augmented(t, x, t_new, x_new)
                out.append(
                    dict(Patient_ID=pid, method="MoM (imputed)", **est)
                )
            except Exception:
                out.append(
                    dict(
                        Patient_ID=pid,
                        method="MoM (imputed)",
                        mu=np.nan,
                        theta=np.nan,
                        sigma=np.nan,
                    )
                )
        else:
            out.append(
                dict(Patient_ID=pid, method="MoM", mu=np.nan, theta=np.nan, sigma=np.nan)
            )

    return out


# ---------------------------------------------------------------------
# Load longitudinal Excel and tidy into (Patient_ID, t, x)
# ---------------------------------------------------------------------
GROUP_MAP = {
    "very early": 1,
    "very early/refractory": 2,
    "early": 3,
    "remission": 4,
    "late": 5,
}


def load_longitudinal_xlsx(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)

    # normalize headers (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    needed = ["patient", "group", "day"]
    if not all(k in cols for k in needed):
        raise ValueError("Input must include columns: Patient, Group, Day")

    d = df.rename(
        columns={
            cols["patient"]: "Patient",
            cols["group"]: "Group",
            cols["day"]: "Day",
        }
    ).copy()

    # coerce Day to numeric
    d["Day"] = pd.to_numeric(d["Day"], errors="coerce")

    # map textual Group to ordinal codes
    g = d["Group"].astype(str).str.strip().str.lower()
    g = g.replace(
        {
            "very early or refractory": "very early/refractory",
            "very-early": "very early",
            "very early refractory": "very early/refractory",
            "early/refractory": "very early/refractory",
        }
    )
    d["x"] = g.map(GROUP_MAP)

    d = d.dropna(subset=["Day", "x"]).copy()

    d["Patient_ID"] = d["Patient"].astype(str)
    d["t"] = d["Day"].astype(float)
    d["x"] = d["x"].astype(float)

    # collapse duplicate time points within patient
    d = (
        d.groupby(["Patient_ID", "t"], as_index=False)
        .agg(
            x=("x", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        )
    )

    d = d.sort_values(["Patient_ID", "t"]).reset_index(drop=True)
    return d[["Patient_ID", "t", "x"]]


# ---------------------------------------------------------------------
# Plot Figure 1 – cohort-level mean ± 95% CI
# ---------------------------------------------------------------------
def plot_figure1(df: pd.DataFrame, out_dir: Path) -> None:
    params = ["mu", "theta", "sigma"]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0), dpi=300, constrained_layout=True)

    for j, p in enumerate(params):
        ax = axs[j]

        d = df.dropna(subset=[p]).copy()
        if d.empty:
            ax.set_visible(False)
            continue

        g = d.groupby("method")[p]
        stats = g.agg(["mean", "count", "std"])
        stats = stats[stats["count"] > 0]
        if stats.empty:
            ax.set_visible(False)
            continue

        mean = stats["mean"]
        sem = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
        yerr = 1.96 * sem

        methods = mean.index.tolist()
        x = np.arange(len(methods))

        ax.bar(x, mean.values, yerr=yerr.values, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylabel(p)
        ax.set_title(p, fontweight="bold", fontsize=10)

    out_png = out_dir / "Figure1_calimethod_comparison_95CI.png"
    out_pdf = out_dir / "Figure1_calimethod_comparison_95CI.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("[+] Saved Figure 1 →")
    print("    ", out_png)
    print("    ", out_pdf)


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def main():
    # 1) Load tidy longitudinal series
    series = load_longitudinal_xlsx(EXCEL_PATH)

    # keep only patients with ≥2 distinct time points
    series = (
        series.groupby("Patient_ID")
        .filter(lambda s: s["t"].nunique() >= 2)
        .copy()
    )
    print(
        f"[INFO] Loaded {series['Patient_ID'].nunique()} patients / {len(series)} rows."
    )

    # 2) Fit each patient
    rows = []
    for pid, sub in series.groupby("Patient_ID"):
        rows.extend(
            compare_patient(
                pid,
                sub,
                impute_midpoints=1,
                impute_source="bayes",
            )
        )

    # 3) Save parameter table
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "ou_methods_comparison_real.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved OU calibration table → {csv_path}")

    # 4) Plot Figure 1
    plot_figure1(df, OUT_DIR)

    print("\n[SUMMARY] Parameter coverage by method:")
    print(
        df.groupby("method")[["mu", "theta", "sigma"]]
        .apply(lambda g: g.notna().mean())
        .round(2)
    )


if __name__ == "__main__":
    main()
