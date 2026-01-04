#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#fig4_cohort_diagnostics.py

"""
Figure 4 – Cohort-level model diagnostics for Bayesian OU(-Branching) fits

Inputs
------
1) Longitudinal workbook:
   kmt2a_longitudinal_clean.xlsx
   - Sheet "Series" with columns (case-insensitive):
       Patient_ID, series, t, value
     where series = 'x' for trait and 'n' for clone count.

2) Clinical metadata:
   kmt2a_clinical_data.xlsx
   - Sheet "Clinical_Data" with header row at index 3 that includes
     Patient_ID and Group; this sheet defines the canonical patient
     cohort (including P28, P68, P135, P136).

3) Patient-level OU posterior files from Figure 3:
   Figure 3/out_fig3/<Patient_ID>_ou_posterior.nc
   Each is an ArviZ InferenceData with posterior variables:
     mu, theta, sigma

Outputs
-------
1) Figure_4/cohort_diagnostics.csv
   Columns:
     Patient_ID, n_timepoints, rmse, ppc_95, r2_bayes, mean_loglik

You can then merge clinical group labels into cohort_diagnostics.csv
to obtain kmt2a_cohort_diagnostics.csv for the R plotting script
fig4_model_diagnostics_bayesian_ou_branching_cohort.R.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az

# ---------------------------------------------------------------------
# Paths – EDIT THESE IF YOUR LAYOUT CHANGES
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(
    "/Users/seung-hwan.kim/Desktop/Bayesian_Hybrid_OU_Branching_Precision_Medicine"
)

EXCEL_PATH    = PROJECT_ROOT / "kmt2a_longitudinal_clean.xlsx"
CLINICAL_PATH = PROJECT_ROOT / "kmt2a_clinical_data.xlsx"
FIG3_OUT_DIR  = PROJECT_ROOT / "Figure 3" / "out_fig3"
FIG4_DIR      = PROJECT_ROOT / "Figure 4"
FIG4_DIR.mkdir(parents=True, exist_ok=True)

SERIES_SHEET = "Series"
OUT_CSV = FIG4_DIR / "cohort_diagnostics.csv"

N_PPC = 500        # number of posterior predictive paths per patient
RANDOM_SEED = 123  # for reproducibility


# ---------------------------------------------------------------------
# OU helpers
# ---------------------------------------------------------------------
def ou_transition_moments(x0, mu, theta, sigma, dt):
    """Mean and variance of OU transition X(t+dt) | X(t)=x0."""
    m = mu + (x0 - mu) * np.exp(-theta * dt)
    v = (sigma**2) / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * dt))
    return m, v


def simulate_ppc_paths(t, x0, idata, n_paths=N_PPC, seed=RANDOM_SEED):
    """
    Simulate posterior predictive OU paths at observed times.

    Returns
    -------
    paths : array shape (n_paths, n_times)
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


def mean_trajectory_from_posterior(t, x0, idata, n_traj=200, seed=RANDOM_SEED):
    """
    Posterior mean trajectory at observed times:
    average of OU mean paths over posterior draws.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    n_times = len(t)

    mu_samples = idata.posterior["mu"].values.reshape(-1)
    th_samples = idata.posterior["theta"].values.reshape(-1)
    sg_samples = idata.posterior["sigma"].values.reshape(-1)
    n_post = mu_samples.size

    n_traj = min(n_traj, n_post)
    idxs = rng.choice(n_post, size=n_traj, replace=False)

    means_all = np.zeros((n_traj, n_times), dtype=float)

    for j, idx in enumerate(idxs):
        mu = float(mu_samples[idx])
        th = float(th_samples[idx])
        sg = float(sg_samples[idx])

        xs = np.zeros(n_times, dtype=float)
        xs[0] = x0
        for i in range(1, n_times):
            dt = t[i] - t[i - 1]
            m, _ = ou_transition_moments(xs[i - 1], mu, th, sg, dt)
            xs[i] = m
        means_all[j] = xs

    return means_all.mean(axis=0)


def mean_loglik_per_transition(idata, t, x, n_draws=500, seed=RANDOM_SEED):
    """
    Approximate mean log-likelihood per transition E_theta[log p(x | theta)] / (n-1).
    Uses a subset of posterior draws for efficiency.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(t)
    if n < 2:
        return np.nan

    mu_samples = idata.posterior["mu"].values.reshape(-1)
    th_samples = idata.posterior["theta"].values.reshape(-1)
    sg_samples = idata.posterior["sigma"].values.reshape(-1)
    n_post = mu_samples.size

    n_draws = min(n_draws, n_post)
    idxs = rng.choice(n_post, size=n_draws, replace=False)

    ll_vals = []
    for idx in idxs:
        mu = float(mu_samples[idx])
        th = float(th_samples[idx])
        sg = float(sg_samples[idx])

        ll = 0.0
        for i in range(1, n):
            dt = t[i] - t[i - 1]
            m, v = ou_transition_moments(x[i - 1], mu, th, sg, dt)
            v = max(v, 1e-12)
            ll += -0.5 * (np.log(2 * np.pi * v) + (x[i] - m) ** 2 / v)
        ll_vals.append(ll)

    return float(np.mean(ll_vals) / (n - 1))


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_series(excel_path: Path, sheet_name: str = "Series") -> pd.DataFrame:
    """Load Series sheet and normalize columns."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Series workbook not found at:\n  {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    lower_cols = {c.lower(): c for c in df.columns}
    needed = ["patient_id", "series", "t", "value"]
    if not all(k in lower_cols for k in needed):
        raise ValueError(
            f"Sheet '{sheet_name}' must contain columns: "
            f"Patient_ID, series, t, value (case-insensitive). "
            f"Found: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            lower_cols["patient_id"]: "Patient_ID",
            lower_cols["series"]: "series",
            lower_cols["t"]: "t",
            lower_cols["value"]: "value",
        }
    )
    return df


def load_clinical_patients(clin_path: Path) -> list[str]:
    """
    Load canonical patient list from kmt2a_clinical_data.xlsx (Clinical_Data sheet).
    This ensures that patients like P28, P68, P135, P136 are always included.
    """
    if not clin_path.exists():
        raise FileNotFoundError(f"Clinical metadata not found at:\n  {clin_path}")

    # header row is at index 3
    df = pd.read_excel(clin_path, sheet_name="Clinical_Data", header=3)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    if "Patient_ID" not in df.columns:
        raise ValueError(
            f"'Clinical_Data' must contain Patient_ID column. Found: {list(df.columns)}"
        )

    patients = sorted(df["Patient_ID"].dropna().astype(str).unique())
    return patients


# ---------------------------------------------------------------------
# Main diagnostics loop
# ---------------------------------------------------------------------
def compute_cohort_diagnostics():
    series_df = load_series(EXCEL_PATH, sheet_name=SERIES_SHEET)
    clinical_patients = set(load_clinical_patients(CLINICAL_PATH))

    # patients with at least 1 x(t) observation
    trait_patients = set(
        series_df.query("series == 'x'")["Patient_ID"].dropna().astype(str).unique()
    )

    # patients for which an OU posterior exists
    posterior_patients = set()
    if FIG3_OUT_DIR.exists():
        for p in FIG3_OUT_DIR.glob("*_ou_posterior.nc"):
            name = p.name.split("_ou_posterior.nc")[0]
            posterior_patients.add(name)

    # Final cohort = union (clinical ∪ trait ∪ posterior)
    all_patients = sorted(clinical_patients | trait_patients | posterior_patients)

    print(f"[INFO] Cohort patients (union of clinical/series/posterior): {len(all_patients)}")
    print("       ", all_patients)

    rows = []

    for pid in all_patients:
        pid_str = str(pid)

        sub = (
            series_df.query("Patient_ID == @pid_str and series == 'x'")
            .sort_values("t")
            .copy()
        )

        if sub.empty:
            # No trait series at all
            n_t = 0
            rmse = ppc_95 = r2_bayes = mean_ll = np.nan
            rows.append(
                dict(
                    Patient_ID=pid_str,
                    n_timepoints=n_t,
                    rmse=rmse,
                    ppc_95=ppc_95,
                    r2_bayes=r2_bayes,
                    mean_loglik=mean_ll,
                )
            )
            continue

        t = sub["t"].to_numpy(float)
        x = sub["value"].to_numpy(float)
        n_t = len(t)

        post_file = FIG3_OUT_DIR / f"{pid_str}_ou_posterior.nc"
        if (n_t < 2) or (not post_file.exists()):
            # include patient, but diagnostics cannot be computed
            rmse = ppc_95 = r2_bayes = mean_ll = np.nan
            rows.append(
                dict(
                    Patient_ID=pid_str,
                    n_timepoints=n_t,
                    rmse=rmse,
                    ppc_95=ppc_95,
                    r2_bayes=r2_bayes,
                    mean_loglik=mean_ll,
                )
            )
            continue

        print(f"[INFO] Patient {pid_str}: n_timepoints={n_t}")

        idata = az.from_netcdf(post_file)

        # posterior predictive paths at observed times
        ppc_paths = simulate_ppc_paths(t, x0=x[0], idata=idata)

        # PPC 95% band & coverage
        lower = np.percentile(ppc_paths, 2.5, axis=0)
        upper = np.percentile(ppc_paths, 97.5, axis=0)
        cover = float(((x >= lower) & (x <= upper)).mean())

        # Bayesian R^2 using posterior predictive replicates
        var_y = np.var(x, ddof=1) if n_t > 1 else np.nan
        if n_t > 1 and var_y > 0:
            r2_vals = []
            for path in ppc_paths:
                resid = x - path
                r2 = 1.0 - np.var(resid, ddof=1) / var_y
                r2_vals.append(r2)
            r2_bayes = float(np.mean(r2_vals))
        else:
            r2_bayes = np.nan

        # posterior mean trajectory and RMSE
        mean_traj = mean_trajectory_from_posterior(t, x0=x[0], idata=idata)
        rmse = float(np.sqrt(np.mean((x - mean_traj) ** 2)))

        # mean log-likelihood per transition
        mean_ll = mean_loglik_per_transition(idata, t, x)

        rows.append(
            dict(
                Patient_ID=pid_str,
                n_timepoints=n_t,
                rmse=rmse,
                ppc_95=cover,
                r2_bayes=r2_bayes,
                mean_loglik=mean_ll,
            )
        )

    diag_df = pd.DataFrame(rows).sort_values("Patient_ID").reset_index(drop=True)
    diag_df.to_csv(OUT_CSV, index=False)
    print("\n[+] Saved cohort diagnostics to:")
    print("    ", OUT_CSV)
    return diag_df


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main():
    diag_df = compute_cohort_diagnostics()
    print("\n[SUMMARY]")
    print(diag_df.describe(include="all").T)


if __name__ == "__main__":
    main()
