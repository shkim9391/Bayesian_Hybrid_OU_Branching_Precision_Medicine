#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 13:11:58 2025

@author: seung-hwan.kim
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_fig2_posterior.py
---------------------
Generate Figure 2 posterior.nc (ArviZ InferenceData) from:
  (A) series_auto.csv        : Patient_ID, series{x,n}, t (years), value
  (B) patient_group.csv      : Patient_ID, Group

Writes (to --out_dir):
  - posterior.nc
  - posterior_summary.csv
  - group_order_used.csv

Posterior variables match Figure 2 plotter expectations:
  r_raw, theta_raw, sigma_raw  each has shape (n_groups,)

Run:
  python fit_fig2_posterior.py \
    --series_csv "/.../Figure_3/series_auto.csv" \
    --patient_group_csv "/.../Figure_2/patient_group.csv" \
    --out_dir "/.../Figure 2/out_bayes_real" \
    --draws 1000 --tune 1500 --chains 4 --target_accept 0.95
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 13


def read_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    need = {"Patient_ID", "series", "t", "value"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"[series_csv] Missing columns: {missing}. Found: {df.columns.tolist()}")

    df["Patient_ID"] = df["Patient_ID"].astype("string").str.strip()
    df["series"] = df["series"].astype("string").str.strip().str.lower()
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.loc[df["series"].isin(["x", "n"])].dropna(subset=["Patient_ID", "series", "t", "value"]).copy()
    df = df.sort_values(["Patient_ID", "series", "t"], kind="mergesort").reset_index(drop=True)
    return df


def read_patient_groups(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # allow common variants
    if "patient_id" in df.columns and "Patient_ID" not in df.columns:
        df = df.rename(columns={"patient_id": "Patient_ID"})
    if "patient" in df.columns and "Patient_ID" not in df.columns:
        df = df.rename(columns={"patient": "Patient_ID"})

    need = {"Patient_ID", "Group"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"[patient_group_csv] Missing columns: {missing}. Found: {df.columns.tolist()}")

    df["Patient_ID"] = df["Patient_ID"].astype("string").str.strip()
    df["Group"] = df["Group"].astype("string").str.strip()

    df = df.dropna(subset=["Patient_ID", "Group"]).drop_duplicates(subset=["Patient_ID"]).reset_index(drop=True)
    return df


def split_patient_series(series_df: pd.DataFrame, pid: str):
    dfx = series_df.query("Patient_ID == @pid and series == 'x'").copy()
    dfn = series_df.query("Patient_ID == @pid and series == 'n'").copy()
    tx = dfx["t"].to_numpy(dtype=float)
    xv = dfx["value"].to_numpy(dtype=float)
    tn = dfn["t"].to_numpy(dtype=float)
    nv = dfn["value"].to_numpy(dtype=float)
    return tx, xv, tn, nv


def fit_and_write(series_df: pd.DataFrame, pg_df: pd.DataFrame, out_dir: Path,
                  draws: int, tune: int, chains: int, target_accept: float):
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az

    out_dir.mkdir(parents=True, exist_ok=True)

    # Patient list and mapping
    patients = series_df["Patient_ID"].dropna().unique().tolist()

    pg = pg_df.set_index("Patient_ID").reindex(patients)
    if pg["Group"].isna().any():
        missing = pg.index[pg["Group"].isna()].tolist()
        raise ValueError(
            f"Missing Group labels for {len(missing)} patients in patient_group_csv. "
            f"Example: {missing[:10]}"
        )

    group_names = sorted(pg["Group"].unique().tolist())
    group_to_idx = {g: i for i, g in enumerate(group_names)}
    n_groups = len(group_names)

    patient_group_idx = {pid: group_to_idx[str(pg.loc[pid, "Group"])] for pid in patients}

    # Prefer numpyro sampler if available
    use_numpyro = False
    try:
        import jax  # noqa
        import numpyro  # noqa
        use_numpyro = True
    except Exception:
        use_numpyro = False

    eps = 1e-8

    with pm.Model() as model:
        G = n_groups

        # --- Raw group parameters (these names must exist for figure2.py)
        theta_raw = pm.Normal("theta_raw", 0.0, 1.0, shape=(G,))
        sigma_raw = pm.Normal("sigma_raw", 0.0, 1.0, shape=(G,))
        r_raw     = pm.Normal("r_raw",     0.0, 1.0, shape=(G,))

        # --- Transforms to constrained scales
        theta = pm.Deterministic("theta", 0.05 + pm.math.log1pexp(theta_raw))
        sigma = pm.Deterministic("sigma", 0.02 + 0.50 * pm.math.log1pexp(sigma_raw))
        r     = pm.Deterministic("r", 0.50 * r_raw)                                      # moderate

        # --- Global noise / dispersion
        tau_x = pm.HalfNormal("tau_x", 0.30)   # x obs noise
        phi   = pm.HalfNormal("phi",   2.00)   # NegBin alpha

        # --- Patient random effects
        mu_mu    = pm.Normal("mu_mu", 0.0, 1.0)
        mu_sigma = pm.HalfNormal("mu_sigma", 1.0)

        logn_mu    = pm.Normal("logn_mu", 0.0, 2.0)
        logn_sigma = pm.HalfNormal("logn_sigma", 1.0)

        for pid in patients:
            gix = patient_group_idx[pid]
            tx, xv, tn, nv = split_patient_series(series_df, pid)

            # patient set-point for OU
            z_mu = pm.Normal(f"z_mu_{pid}", 0.0, 1.0)
            mu_i = pm.Deterministic(f"mu_{pid}", mu_mu + mu_sigma * z_mu)

            # OU likelihood (exact transition)
            if len(tx) >= 2:
                x0 = pm.Normal(
                    f"x0_{pid}",
                    mu=mu_i,
                    sigma=pt.sqrt((sigma[gix] ** 2) / (2.0 * theta[gix]) + eps),
                )
                x_lat = [x0]
                for k in range(1, len(tx)):
                    dt = float(tx[k] - tx[k - 1])
                    m = mu_i + (x_lat[-1] - mu_i) * pt.exp(-theta[gix] * dt)
                    v = (sigma[gix] ** 2) / (2.0 * theta[gix]) * (1.0 - pt.exp(-2.0 * theta[gix] * dt)) + eps
                    xk = pm.Normal(f"x_{pid}_{k}", mu=m, sigma=pt.sqrt(v))
                    x_lat.append(xk)
                pm.Normal(f"x_obs_{pid}", mu=pt.stack(x_lat), sigma=tau_x, observed=xv)
            elif len(tx) == 1:
                pm.Normal(f"x_obs_{pid}", mu=mu_i, sigma=tau_x, observed=xv)

            # count layer (simple branching-like mean curve)
            # log mu_n(t) = logn0_i + r_g * t
            z_ln = pm.Normal(f"z_logn_{pid}", 0.0, 1.0)
            logn0_i = pm.Deterministic(f"logn0_{pid}", logn_mu + logn_sigma * z_ln)

            if len(tn) >= 1:
                mu_n = pt.exp(logn0_i + r[gix] * pt.as_tensor_variable(tn.astype(float)))
                mu_n = pt.clip(mu_n, 1e-6, 1e9)
                pm.NegativeBinomial(f"n_obs_{pid}", mu=mu_n, alpha=phi, observed=nv.astype(int))

        if use_numpyro:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=min(chains, 4),
                random_seed=SEED,
                target_accept=target_accept,
                nuts_sampler="numpyro",
                progressbar=True,
            )
        else:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=min(chains, 4),
                random_seed=SEED,
                target_accept=target_accept,
                progressbar=True,
            )

    # Write outputs
    nc_path = out_dir / "posterior.nc"
    az.to_netcdf(idata, nc_path)

    summ = az.summary(idata, var_names=["theta_raw", "sigma_raw", "r_raw", "theta", "sigma", "r", "tau_x", "phi"])
    summ.to_csv(out_dir / "posterior_summary.csv")

    pd.DataFrame({"Group": group_names}).to_csv(out_dir / "group_order_used.csv", index=False)

    print(f"[OK] wrote {nc_path}")
    print(f"[OK] wrote {out_dir / 'posterior_summary.csv'}")
    print(f"[OK] wrote {out_dir / 'group_order_used.csv'}")
    print(f"[OK] groups (order): {group_names}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series_csv", required=True)
    ap.add_argument("--patient_group_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.95)
    args = ap.parse_args()

    series_csv = Path(args.series_csv)
    pg_csv = Path(args.patient_group_csv)
    out_dir = Path(args.out_dir)

    if not series_csv.exists():
        raise FileNotFoundError(series_csv)
    if not pg_csv.exists():
        raise FileNotFoundError(pg_csv)

    series_df = read_series(series_csv)
    pg_df = read_patient_groups(pg_csv)

    fit_and_write(series_df, pg_df, out_dir, args.draws, args.tune, args.chains, args.target_accept)


if __name__ == "__main__":
    main()