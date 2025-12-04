#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 18:38:37 2025

@author: seung-hwan.kim
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3 – Patient-level OU calibration and clonal dynamics

Inputs
------
1) Longitudinal "Series" workbook:
   kmt2a_longitudinal_clean.xlsx, with a sheet named "Series" that contains
   at least the following columns (case-insensitive):

   - Patient_ID : patient identifier (e.g. "P15")
   - series     : label for the type of series; expected values:
                  "x" for average trait, "n" for clone count
   - t          : time in years (float)
   - value      : numeric value of the series at time t

Outputs
-------
1) out_fig3/P15_ou_posterior.nc
   ArviZ InferenceData with OU posterior samples for the trait series.

2) out_fig3/Figure3_P15_pub.png / .pdf
   Three-panel Nature-style figure:
     A. Avg trait vs time with posterior mean and 95% predictive band
     B. Clone count vs time (step plot)
     C. Trait vs clone count scatter

To change the patient or file locations, edit the constants in the
"User parameters" section below.
"""

from pathlib import Path
import os

os.environ["PYTENSOR_FLAGS"] = (
    "optimizer=None,linker=py,cxx=,device=cpu,exception_verbosity=high"
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az
from pytensor.compile import Mode

# ---------------------------------------------------------------------
# User parameters – EDIT HERE IF NEEDED
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(
    "/Bayesian_Hybrid_OU_Branching_Precision_Medicine"
)
FIG3_DIR = PROJECT_ROOT / "Figure 3"
OUT_DIR = FIG3_DIR / "out_fig3"

EXCEL_PATH = PROJECT_ROOT / "kmt2a_longitudinal_clean.xlsx"
SERIES_SHEET = "Series"

PATIENT_ID = "P15"          # patient to plot
N_DRAWS = 2000              # posterior draws
N_TUNE = 2000               # tuning steps
N_OVERLAY = 80              # number of posterior mean trajectories to overlay
N_PPC = 500                 # posterior predictive sample paths for PPC band
RANDOM_SEED = 123           # for reproducibility

OUT_DIR.mkdir(parents=True, exist_ok=True)

PURE_PY_MODE = Mode(linker="py", optimizer=None)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_series(excel_path: Path, sheet_name: str = "Series") -> pd.DataFrame:
    """Load tidy longitudinal series from an Excel sheet.

    Expected columns (case-insensitive):
      Patient_ID, series, t, value
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Series workbook not found at:\n  {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # strip whitespace from column names
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    lower_cols = {c.lower(): c for c in df.columns}
    required = ["patient_id", "series", "t", "value"]
    if not all(k in lower_cols for k in required):
        raise ValueError(
            f"Sheet '{sheet_name}' must contain columns: Patient_ID, series, t, value "
            f"(case-insensitive). Found: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            lower_cols["patient_id"]: "Patient_ID",
            lower_cols["series"]: "series",
            lower_cols["t"]: "t",
            lower_cols["value"]: "value",
        }
    )

    # Basic checks similar to the standalone checking script
    nulls = df[["Patient_ID", "series", "t", "value"]].isna().any(axis=1).sum()
    dups = df.duplicated(subset=["Patient_ID", "series", "t"]).sum()
    bad_series = sorted(set(df["series"]) - {"x", "n"})

    print("✅ Loaded Series sheet from", excel_path)
    print("  Patients:", df["Patient_ID"].nunique())
    print("  Rows:", len(df))
    print("  Rows with NaN in required cols:", nulls)
    print("  Duplicate (Patient_ID, series, t):", dups)
    if bad_series:
        print("  ⚠️ Unexpected series labels found:", bad_series)

    # ensure each patient has at least two x(t) points for OU
    need2 = (
        df.query("series=='x'")
        .groupby("Patient_ID")["t"]
        .nunique()
        .reset_index(name="nx")
    )
    poor = need2.loc[need2["nx"] < 2, "Patient_ID"].tolist()
    print("  Patients with <2 distinct x(t):", poor if poor else "None")

    return df


def ou_transition_moments(x0, mu, theta, sigma, dt):
    """OU transition mean and variance after lag dt."""
    m = mu + (x0 - mu) * np.exp(-theta * dt)
    v = (sigma**2) / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * dt))
    return m, v


def fit_ou_bayes(t, x, draws=N_DRAWS, tune=N_TUNE, seed=RANDOM_SEED):
    """Bayesian OU fit using pure-Python Metropolis."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 2)
        theta = pm.HalfNormal("theta", 1)
        sigma = pm.HalfNormal("sigma", 1)

        dt = np.diff(t)
        eps = 1e-9
        for i in range(1, len(t)):
            m, v = ou_transition_moments(float(x[i - 1]), mu, theta, sigma, float(dt[i - 1]))
            sd = pm.math.sqrt(pm.math.maximum(v, eps))
            pm.Normal(f"x_{i}", mu=m, sigma=sd, observed=float(x[i]))

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=1,
            step=pm.Metropolis(),
            cores=1,
            random_seed=seed,
            progressbar=True,
            compile_kwargs={"mode": PURE_PY_MODE},
        )

    return idata


def posterior_summary(idata, cred_mass: float = 0.95):
    """
    Return posterior means and 95% HDIs for mu, theta, sigma.

    Uses ArviZ for the means and plain NumPy percentiles for HDIs
    to avoid xarray shape quirks.
    """
    # means from ArviZ
    summary = az.summary(idata, var_names=["mu", "theta", "sigma"])
    means = summary["mean"].values  # [mu_mean, theta_mean, sigma_mean]

    def hdi_np(values, prob=0.95):
        """Compute HDI via percentiles of flattened samples."""
        arr = np.asarray(values).reshape(-1)  # flatten chains × draws
        lower = np.percentile(arr, 100 * (1.0 - prob) / 2.0)
        upper = np.percentile(arr, 100 * (1.0 + prob) / 2.0)
        return lower, upper

    mu_low, mu_high = hdi_np(idata.posterior["mu"].values, cred_mass)
    th_low, th_high = hdi_np(idata.posterior["theta"].values, cred_mass)
    sg_low, sg_high = hdi_np(idata.posterior["sigma"].values, cred_mass)

    lower = np.array([mu_low, th_low, sg_low])
    upper = np.array([mu_high, th_high, sg_high])

    return means, lower, upper


def simulate_ppc_paths(t, x0, idata, n_paths=N_PPC, seed=RANDOM_SEED):
    """
    Simulate posterior predictive OU paths at observed times.

    Returns
    -------
    paths : array, shape (n_paths, n_times)
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    n_times = len(t)
    paths = np.zeros((n_paths, n_times), dtype=float)
    paths[:, 0] = x0

    mu_samples = idata.posterior["mu"].values.reshape(-1)
    th_samples = idata.posterior["theta"].values.reshape(-1)
    sg_samples = idata.posterior["sigma"].values.reshape(-1)
    n_post = mu_samples.size

    for k in range(n_paths):
        idx = rng.integers(0, n_post)
        mu = float(mu_samples[idx])
        th = float(th_samples[idx])
        sg = float(sg_samples[idx])

        x_curr = x0
        for i in range(1, n_times):
            dt = t[i] - t[i - 1]
            m, v = ou_transition_moments(x_curr, mu, th, sg, dt)
            x_curr = rng.normal(m, np.sqrt(max(v, 1e-9)))
            paths[k, i] = x_curr

    return paths


def ou_mean_trajectories(t, x0, idata, n_traj=N_OVERLAY, seed=RANDOM_SEED):
    """
    Compute OU mean trajectories for randomly chosen posterior draws.

    Returns
    -------
    overlay_means : array, shape (n_traj, n_times)
    mean_of_means : array, shape (n_times,)
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    n_times = len(t)

    mu_samples = idata.posterior["mu"].values.reshape(-1)
    th_samples = idata.posterior["theta"].values.reshape(-1)
    sg_samples = idata.posterior["sigma"].values.reshape(-1)
    n_post = mu_samples.size

    overlay_means = np.zeros((n_traj, n_times), dtype=float)

    for k in range(n_traj):
        idx = rng.integers(0, n_post)
        mu = float(mu_samples[idx])
        th = float(th_samples[idx])
        sg = float(sg_samples[idx])

        xs = np.zeros(n_times, dtype=float)
        xs[0] = x0
        for i in range(1, n_times):
            dt = t[i] - t[i - 1]
            m, v = ou_transition_moments(xs[i - 1], mu, th, sg, dt)
            xs[i] = m
        overlay_means[k] = xs

    mean_of_means = overlay_means.mean(axis=0)
    return overlay_means, mean_of_means


# ---------------------------------------------------------------------
# Figure 3 construction
# ---------------------------------------------------------------------
def make_figure3(
    series_df: pd.DataFrame,
    patient_id: str,
    out_dir: Path,
    out_stem: str = "Figure3_P15_pub",
):
    """
    Build the three-panel Figure 3 for a given patient.

    Parameters
    ----------
    series_df : DataFrame
        Full Series data for all patients.
    patient_id : str
        Patient identifier (e.g. "P15").
    out_dir : Path
        Directory where outputs will be written.
    out_stem : str
        Basename for figure files (PNG/PDF).
    """
    # --- subset data for requested patient ---
    sub = series_df.query("Patient_ID == @patient_id").copy()
    if sub.empty:
        raise ValueError(f"No rows found for Patient_ID={patient_id!r}.")

    # Trait series x(t)
    x_df = sub.query("series == 'x'").sort_values("t")
    if x_df.empty:
        raise ValueError(f"No 'x' series found for Patient_ID={patient_id!r}.")

    t_x = x_df["t"].to_numpy(float)
    y_x = x_df["value"].to_numpy(float)

    # Clone series n(t) – may be empty for some patients
    n_df = sub.query("series == 'n'").sort_values("t")
    t_n = n_df["t"].to_numpy(float) if not n_df.empty else np.array([])
    y_n = n_df["value"].to_numpy(float) if not n_df.empty else np.array([])

    # --- fit OU on trait series ---
    print(f"[INFO] Fitting OU model for patient {patient_id} (n={len(t_x)} time points).")
    idata = fit_ou_bayes(t_x, y_x)
    post_path = out_dir / f"{patient_id}_ou_posterior.nc"
    idata.to_netcdf(post_path)
    print("[+] Saved OU posterior to", post_path)

    means, lower, upper = posterior_summary(idata)
    mu_mean, th_mean, sg_mean = means
    mu_low, th_low, sg_low = lower
    mu_high, th_high, sg_high = upper

    # --- posterior predictive simulations at observed times ---
    ppc_paths = simulate_ppc_paths(t_x, x0=y_x[0], idata=idata)
    ppc_lower = np.percentile(ppc_paths, 2.5, axis=0)
    ppc_upper = np.percentile(ppc_paths, 97.5, axis=0)

    # coverage: fraction of observed y_x inside band
    cover = ((y_x >= ppc_lower) & (y_x <= ppc_upper)).mean()
    cover_str = f"PPC 95% cover: {int(round(cover * 100))}%"

    # OU mean trajectories
    overlay_means, mean_of_means = ou_mean_trajectories(t_x, x0=y_x[0], idata=idata)

    # -----------------------------------------------------------------
    # Plotting – Nature-style
    # -----------------------------------------------------------------
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

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(10.0, 3.0), dpi=300, constrained_layout=True
    )

    # ---------- Panel A: trait vs time with OU band & PPC ----------
    # PPC band
    ax1.fill_between(t_x, ppc_lower, ppc_upper, color="#c7dcef", alpha=0.8, label="PPC 95% band")

    # overlay individual mean trajectories
    for k in range(overlay_means.shape[0]):
        ax1.plot(t_x, overlay_means[k], color="red", alpha=0.05, linewidth=0.6)

    # posterior mean-of-means
    ax1.plot(t_x, mean_of_means, color="red", linewidth=2.0, label="Bayes mean")

    # observed points
    ax1.scatter(t_x, y_x, color="#1f77b4", s=30, zorder=3, label="obs")

    ax1.set_xlabel("time (y)")
    ax1.set_ylabel("avg trait")
    ax1.set_title(f"A. Avg trait vs time (Patient {patient_id})", fontweight="bold", fontsize=10)

    # inset with parameter summary
    text = (
        rf"$\mu = {mu_mean:.2f}$ [{mu_low:.2f}, {mu_high:.2f}]" + "\n"
        rf"$\theta = {th_mean:.2f}$ [{th_low:.2f}, {th_high:.2f}]" + "\n"
        rf"$\sigma = {sg_mean:.2f}$ [{sg_low:.2f}, {sg_high:.2f}]"
    )
    ax1.text(
        0.03,
        0.03,
        text,
        transform=ax1.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        va="bottom",
        ha="left",
    )

    # text with PPC coverage
    ax1.text(
        0.97,
        0.03,
        cover_str,
        transform=ax1.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        va="bottom",
        ha="right",
    )

    ax1.legend(frameon=False, loc="upper right")

    # ---------- Panel B: clone count vs time ----------
    if t_n.size > 0:
        ax2.step(t_n, y_n, where="post")
        ax2.scatter(t_n, y_n, s=25)
    ax2.set_xlabel("time (y)")
    ax2.set_ylabel("clones")
    ax2.set_title(f"B. Clone count vs time (Patient {patient_id})", fontweight="bold", fontsize=10)

    # ---------- Panel C: trait vs clone count ----------
    if t_n.size > 0:
        # align trait and clone counts at closest times
        # here we simply interpolate trait onto clone times
        trait_interp = np.interp(t_n, t_x, y_x)
        ax3.scatter(y_n, trait_interp, s=30)
        ax3.set_xlabel("clones")
        ax3.set_ylabel("avg trait")
    else:
        ax3.set_visible(False)

    ax3.set_title(f"C. Trait vs clone count (Patient {patient_id})", fontweight="bold")

    # save
    out_png = out_dir / f"{out_stem}.png"
    out_pdf = out_dir / f"{out_stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("[+] Saved Figure 3 →")
    print("    ", out_png)
    print("    ", out_pdf)


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def main():
    series_df = load_series(EXCEL_PATH, sheet_name=SERIES_SHEET)
    make_figure3(series_df, patient_id=PATIENT_ID, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
