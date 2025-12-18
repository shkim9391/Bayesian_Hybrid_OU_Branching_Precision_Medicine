#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 16:50:48 2025

@author: seung-hwan.kim
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ou_longitudinal_compare.py
OU calibration across longitudinal patients (KMT2A cohort)
- Input: kmt2a_longitudinal_samples.xlsx with columns: Patient, Group, Day, ...
- Output: out_compare_real/ou_methods_comparison_real.csv + plots + ranking
"""

import os
os.environ["PYTENSOR_FLAGS"] = "optimizer=None,linker=py,cxx=,device=cpu,exception_verbosity=high"

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import pymc as pm
import arviz as az
from pytensor.compile import Mode

PURE_PY_MODE = Mode(linker="py", optimizer=None)

# -----------------------------
# OU helpers
# -----------------------------
def ou_transition_moments(x0, mu, theta, sigma, dt):
    m = mu + (x0 - mu) * np.exp(-theta * dt)
    v = (sigma**2) / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * dt))
    return m, v

def ou_mle_fit(t, x):
    dt = np.diff(t)
    x_prev, x_next = x[:-1], x[1:]
    def nll(params):
        mu, theta, sigma = params
        theta = max(theta, 1e-6); sigma = max(sigma, 1e-6)
        m, v = ou_transition_moments(x_prev, mu, theta, sigma, dt)
        return 0.5 * np.sum(np.log(2*np.pi*v) + (x_next - m)**2 / v)
    res = minimize(nll, x0=[float(np.mean(x)), 0.5, 0.5],
                   bounds=[(-10,10), (1e-6,10.0), (1e-6,10.0)], method="L-BFGS-B")
    return res.x if res.success else [np.nan, np.nan, np.nan]

def ou_mom_estimator(t, x):
    dx = np.diff(x)
    vx = np.var(x)
    if vx <= 1e-12 or len(t) < 3:
        raise ValueError("MoM not identifiable (low variance or <3 points).")
    cov_xdx = np.cov(x[:-1], dx)[0, 1]
    theta = -cov_xdx / vx
    mu = np.mean(x)
    sigma = np.std(dx - (-theta)*(x[:-1] - mu)) / np.sqrt(2)
    return mu, max(theta, 1e-4), sigma

def impute_points_from_bayes(t_obs, x_obs, idata, t_new):
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
    x_imp = []
    for tn in t_new:
        j = np.where(t_obs <= tn)[0].max()
        x0, t0 = float(x_obs[j]), float(t_obs[j])
        m, _ = ou_transition_moments(x0, mu, th, sg, tn - t0)
        x_imp.append(float(m))
    return np.array(x_imp)

def run_mom_on_augmented(t_obs, x_obs, t_new, x_new):
    t_all = np.concatenate([t_obs, t_new])
    x_all = np.concatenate([x_obs, x_new])
    order = np.argsort(t_all)
    t_all = t_all[order]; x_all = x_all[order]
    mu_mom, th_mom, sg_mom = ou_mom_estimator(t_all, x_all)
    return dict(mu=mu_mom, theta=th_mom, sigma=sg_mom)

# -----------------------------
# Main comparison per patient
# -----------------------------
def compare_patient(pid, sub, impute_midpoints=1, impute_source="bayes",
                    n_draws=1000, n_tune=1000):
    t = sub["t"].to_numpy(dtype=float)
    x = sub["x"].to_numpy(dtype=float)

    # sort by time
    order = np.argsort(t)
    t, x = t[order], x[order]

    # ✅ add this:
    out = []   # <---- you use out.append(...) below; define it here

    # --- collapse duplicate times (dt==0 would kill the likelihood) ---
    # keep the modal stage at that time (if tie, take the first)
    if np.any(np.diff(t) == 0):
        uniq_t = np.unique(t)
        x_collapsed = []
        for tt in uniq_t:
            vals = x[t == tt]
            # modal value (numeric code for stage)
            mode_val = pd.Series(vals).mode().iloc[0]
            x_collapsed.append(mode_val)
        t = uniq_t
        x = np.asarray(x_collapsed, dtype=float)

    # MLE
    mu_mle, th_mle, sg_mle = ou_mle_fit(t, x)
    out.append(dict(Patient_ID=pid, method="MLE", mu=mu_mle, theta=th_mle, sigma=sg_mle))

    # Bayesian (Metropolis, pure python)
    with pm.Model() as model:
        mu    = pm.Normal("mu", 0, 1)
        theta = pm.HalfNormal("theta", 1)
        sigma = pm.HalfNormal("sigma", 1)
    
        dt = np.diff(t)
        eps = 1e-9  # tiny positive floor
        for i in range(1, len(t)):
            m, v = ou_transition_moments(float(x[i-1]), mu, theta, sigma, float(dt[i-1]))
            sd   = pm.math.sqrt(pm.math.maximum(v, eps))
            pm.Normal(f"x_{i}", mu=m, sigma=sd, observed=float(x[i]))
    
        idata = pm.sample(
            draws=n_draws, tune=n_tune, chains=1, step=pm.Metropolis(),
            cores=1, progressbar=True, compile_kwargs={"mode": PURE_PY_MODE}
        )
    summary = az.summary(idata, var_names=["mu","theta","sigma"])
    mu_b, th_b, sg_b = summary["mean"].values
    out.append(dict(Patient_ID=pid, method="Bayesian", mu=mu_b, theta=th_b, sigma=sg_b))

    # MoM (direct or with optional midpoint imputation if <3 time points)
    if len(t) >= 3:
        try:
            mu_m, th_m, sg_m = ou_mom_estimator(t, x)
            out.append(dict(Patient_ID=pid, method="MoM", mu=mu_m, theta=th_m, sigma=sg_m))
        except Exception:
            out.append(dict(Patient_ID=pid, method="MoM", mu=np.nan, theta=np.nan, sigma=np.nan))
    else:
        if impute_midpoints > 0:
            t_new = np.linspace(float(t.min()), float(t.max()), impute_midpoints+2)[1:-1]
            if impute_source == "bayes":
                x_new = impute_points_from_bayes(t, x, idata, t_new)
            else:
                x_new = impute_points_from_mle(t, x, mu_mle, th_mle, sg_mle, t_new)
            try:
                est = run_mom_on_augmented(t, x, t_new, x_new)
                out.append(dict(Patient_ID=pid, method="MoM (imputed)", **est))
            except Exception:
                out.append(dict(Patient_ID=pid, method="MoM (imputed)", mu=np.nan, theta=np.nan, sigma=np.nan))
        else:
            out.append(dict(Patient_ID=pid, method="MoM", mu=np.nan, theta=np.nan, sigma=np.nan))

    return out

# -----------------------------
# Ranking
# -----------------------------
def build_ranking(df, out_dir, params=("mu","theta","sigma"), w_acc=0.5, w_stab=0.3, w_cov=0.2):
    os.makedirs(out_dir, exist_ok=True)
    methods = sorted(df["method"].unique())
    rows = {m: {"acc":[], "stab":[], "cov":[]} for m in methods}

    # accuracy vs Bayes
    for p in params:
        pvt = df.pivot_table(index="Patient_ID", columns="method", values=p, aggfunc="first")
        bay = pvt.get("Bayesian", None)
        if bay is None: continue
        for m in pvt.columns:
            if m == "Bayesian": continue
            v = pvt[m]
            mask = bay.notna() & v.notna()
            mae = (v[mask] - bay[mask]).abs().mean() if mask.sum() > 0 else np.nan
            rows[m]["acc"].append(mae)
        # stability
        stab = df.groupby("method")[p].std(ddof=1).to_dict()
        for m in methods:
            rows[m]["stab"].append(stab.get(m, np.nan))
        # coverage
        total = df["Patient_ID"].nunique()
        got = df.dropna(subset=[p]).groupby("method")["Patient_ID"].nunique().to_dict()
        for m in methods:
            rows[m]["cov"].append(got.get(m, 0)/total)

    def minmax_inv(vals):  # lower better → 1 - scaled
        a = np.array(vals, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0: return None, None
        lo, hi = a.min(), a.max()
        if hi == lo: return lo, hi
        return lo, hi

    rank = []
    for m in methods:
        # accuracy score
        if rows[m]["acc"]:
            lo, hi = minmax_inv(rows[m]["acc"])
            if hi is None:
                acc_score = 0.5
            else:
                vals = np.array(rows[m]["acc"], float)
                acc_score = (1 - (np.nanmean(vals)-lo)/(hi-lo)) if hi>lo else 0.5
        else:
            acc_score = 0.0
        # stability score (lower std is better)
        if rows[m]["stab"]:
            lo, hi = minmax_inv(rows[m]["stab"])
            if hi is None:
                stab_score = 0.5
            else:
                vals = np.array(rows[m]["stab"], float)
                stab_score = (1 - (np.nanmean(vals)-lo)/(hi-lo)) if hi>lo else 0.5
        else:
            stab_score = 0.0
        # coverage score
        cov_score = float(np.nanmean(rows[m]["cov"])) if rows[m]["cov"] else 0.0

        composite = w_acc*acc_score + w_stab*stab_score + w_cov*cov_score
        rank.append(dict(method=m, accuracy_score=round(acc_score,3),
                         stability_score=round(stab_score,3),
                         coverage_score=round(cov_score,3),
                         composite_score=round(composite,3)))
    rank_df = pd.DataFrame(rank).sort_values("composite_score", ascending=False).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df)+1)
    rank_df = rank_df[["rank","method","composite_score","accuracy_score","stability_score","coverage_score"]]
    rank_df.to_csv(os.path.join(out_dir, "methods_ranking_table.csv"), index=False)
    return rank_df

# -----------------------------
# Plotting (across patients)
# -----------------------------
def plot_bars(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(1,3, figsize=(12,4), dpi=300)
    for j, p in enumerate(["mu","theta","sigma"]):
        if df[p].dropna().empty: axs[j].set_visible(False); continue
        g = df.groupby("method")[p]
        mean, sem = g.mean(), g.std(ddof=1)/np.sqrt(np.maximum(g.count(),1))
        yerr = (1.96*sem).fillna(0.0)
        axs[j].bar(mean.index, mean.values, yerr=yerr.values, capsize=4)
        axs[j].set_title(p); axs[j].set_ylabel(p)
    plt.tight_layout()
    f = os.path.join(out_dir, "Figure1_calimethod_comparison_95CI.png")
    plt.savefig(f, bbox_inches="tight"); plt.close()
    return f

def plot_ranking(rank_df, out_dir):
    methods = rank_df["method"].astype(str).tolist()
    comp = rank_df["composite_score"].values
    acc  = rank_df["accuracy_score"].values
    stab = rank_df["stability_score"].values
    cov  = rank_df["coverage_score"].values

    # composite
    plt.figure(figsize=(7.2,3.8), dpi=300)
    y = np.arange(len(methods))[::-1]
    plt.barh(y, comp[::-1])
    for i,v in enumerate(comp[::-1]): plt.text(v+0.01, y[i], f"{v:.2f}", va="center", fontsize=9)
    plt.yticks(y, methods[::-1]); plt.xlabel("Composite score (0–1)")
    plt.title("Calibration Methods – Composite Ranking")
    plt.xlim(0,1.05); plt.tight_layout()
    comp_png = os.path.join(out_dir, "Figure_Ranking_composite.png")
    plt.savefig(comp_png, bbox_inches="tight"); plt.close()

    # components
    plt.figure(figsize=(7.8,4.2), dpi=300)
    x = np.arange(len(methods)); w = 0.6
    plt.bar(x, acc, width=w, label="Accuracy", alpha=0.9)
    plt.bar(x, stab, width=w, bottom=acc, label="Stability", alpha=0.85)
    plt.bar(x, cov, width=w, bottom=acc+stab, label="Coverage", alpha=0.8)
    plt.xticks(x, methods); plt.ylabel("Score (0–1)")
    plt.title("Calibration Methods – Component Scores")
    plt.ylim(0, np.clip((acc+stab+cov).max()+0.1, 0, 3.0))
    plt.legend(frameon=False, ncols=3, bbox_to_anchor=(0.5,1.02), loc="lower center")
    plt.tight_layout()
    parts_png = os.path.join(out_dir, "Figure_Ranking_components.png")
    plt.savefig(parts_png, bbox_inches="tight"); plt.close()
    return comp_png, parts_png

# -----------------------------
# IO and driver
# -----------------------------
GROUP_MAP = {
    "very early": 1,
    "very early/refractory": 2,
    "early": 3,
    "remission": 4,
    "late": 5,
}

def load_longitudinal_xlsx(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)

    # ---- normalize headers (case-insensitive) ----
    cols = {c.lower(): c for c in df.columns}
    needed = ["patient", "group", "day"]
    if not all(k in cols for k in needed):
        raise ValueError("Input must include columns: Patient, Group, Day")

    d = df.rename(columns={
        cols["patient"]: "Patient",
        cols["group"]: "Group",
        cols["day"]: "Day",
    }).copy()

    # ---- clean & coerce types ----
    # Coerce Day to numeric; guard things like '-' with NaN
    d["Day"] = pd.to_numeric(d["Day"], errors="coerce")

    # Map textual Group to ordinal stage code x = {1..5}
    g = d["Group"].astype(str).str.strip().str.lower()
    # tolerate minor punctuation/spacing variants
    g = (g.replace({
        "very early or refractory": "very early/refractory",
        "very-early": "very early",
        "very early refractory": "very early/refractory",
        "early/refractory": "very early/refractory",   # adjust if you used this variant
    }))
    d["x"] = g.map(GROUP_MAP)

    # keep only valid rows
    d = d.dropna(subset=["Day", "x"]).copy()

    # canonical columns
    d["Patient_ID"] = d["Patient"].astype(str)
    d["t"] = d["Day"].astype(float)
    d["x"] = d["x"].astype(float)

    # ---- collapse duplicate time points within patient (dt==0 kills OU variance) ----
    # keep the modal stage at that time; if mode is empty, fall back to first
    d = (
        d.groupby(["Patient_ID", "t"], as_index=False)
         .agg(**{
             "x": ("x", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
         })
    )

    # sorted, tidy frame used by the rest of the pipeline
    d = d.sort_values(["Patient_ID", "t"]).reset_index(drop=True)
    return d[["Patient_ID", "t", "x"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", type=str, default="kmt2a_longitudinal_clean.xlsx",
                    help="Excel with Patient, Group, Day columns")
    ap.add_argument("--out", type=str, default="out_compare_real")
    ap.add_argument("--impute-midpoints", type=int, default=1,
                    help="Number of OU-consistent midpoints to add when a patient has <3 time points")
    ap.add_argument("--impute-source", type=str, choices=["bayes","mle"], default="bayes")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) Load tidy series from Excel
    series = load_longitudinal_xlsx(args.excel)

    # 2) Keep only patients with >= 2 distinct time points
    series = series.groupby("Patient_ID").filter(lambda s: s["t"].nunique() >= 2).copy()

    print(f"[INFO] Loaded {series['Patient_ID'].nunique()} patients / {len(series)} rows.")

    # 3) Fit each patient
    rows = []
    for pid, sub in series.groupby("Patient_ID"):
        rows.extend(compare_patient(
            pid, sub,
            impute_midpoints=args.impute_midpoints,
            impute_source=args.impute_source
        ))

    # 4) Save table + figures
    df = pd.DataFrame(rows)
    csv = os.path.join(args.out, "ou_methods_comparison_real.csv")
    df.to_csv(csv, index=False)
    print(f"[+] Saved table → {csv}")

    fig = plot_bars(df, args.out)
    print(f"[+] Saved Figure 1 → {fig}")

    rank_df = build_ranking(df, args.out)
    r1, r2 = plot_ranking(rank_df, args.out)
    print(f"[+] Saved ranking plots → {r1}, {r2}")

    print("\n[SUMMARY] Parameter coverage by method:")
    print(df.groupby("method")[["mu","theta","sigma"]].apply(lambda g: g.notna().mean()).round(2))

if __name__ == "__main__":
    main()