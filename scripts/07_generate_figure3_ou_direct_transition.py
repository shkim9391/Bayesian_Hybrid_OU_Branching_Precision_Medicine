from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = (
    Path.home() /scripts/ "07_generate_figure3_ou_direct_transition.py"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "Figure_3" / "out_ou_direct_transition"
DEFAULT_TRACE = DEFAULT_OUT_DIR / "ou_direct_transition_trace.nc"
DEFAULT_GROUPS = DEFAULT_OUT_DIR / "ou_direct_transition_group_order.csv"
DEFAULT_PATIENT_MAP = DEFAULT_OUT_DIR / "ou_direct_transition_patient_variable_map.csv"


def exact_hdi(samples_2d: np.ndarray, credible_mass: float = 0.95):
    arr = np.asarray(samples_2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected samples × groups array, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Nonfinite posterior values detected")
    n, k = arr.shape
    width_n = min(max(int(np.floor(credible_mass * n)), 1), n - 1)
    s = np.sort(arr, axis=0)
    widths = s[width_n:, :] - s[: n - width_n, :]
    idx = np.argmin(widths, axis=0)
    cols = np.arange(k)
    return s[idx, cols], s[idx + width_n, cols]


def posterior_matrix(idata, variable: str, n_groups: int) -> np.ndarray:
    if variable not in idata.posterior:
        raise KeyError(
            f"Posterior variable {variable!r} not found. Available: "
            f"{list(idata.posterior.data_vars)}"
        )
    arr = np.asarray(idata.posterior[variable].values, dtype=float)
    if arr.shape[-1] != n_groups:
        raise ValueError(
            f"{variable} group dimension={arr.shape[-1]}, expected {n_groups}"
        )
    return arr.reshape(-1, n_groups)


def summarize(samples, groups, parameter, credible_mass):
    low, high = exact_hdi(samples, credible_mass)
    return pd.DataFrame({
        "Group": groups,
        "parameter": parameter,
        "posterior_mean": samples.mean(axis=0),
        "posterior_median": np.median(samples, axis=0),
        "posterior_sd": samples.std(axis=0, ddof=1),
        "hdi_lower": low,
        "hdi_upper": high,
    })


def pairwise_table(samples, groups, parameter):
    rows = []
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            diff = samples[:, i] - samples[:, j]
            rows.append({
                "parameter": parameter,
                "Group_1": g1,
                "Group_2": g2,
                "P_Group_1_gt_Group_2": np.mean(diff > 0),
                "posterior_mean_difference": np.mean(diff),
                "posterior_median_difference": np.median(diff),
            })
    return pd.DataFrame(rows)


def patient_mu_table(idata, patient_map, credible_mass):
    required = {"Patient_ID", "Group", "group_index", "mu_variable"}
    missing = required - set(patient_map.columns)
    if missing:
        raise ValueError(f"Patient map missing columns: {sorted(missing)}")
    rows = []
    for row in patient_map.itertuples(index=False):
        var = str(row.mu_variable)
        if var not in idata.posterior:
            raise KeyError(f"Missing patient posterior variable: {var}")
        s = np.asarray(idata.posterior[var].values, dtype=float).reshape(-1)
        low, high = exact_hdi(s[:, None], credible_mass)
        rows.append({
            "Patient_ID": str(row.Patient_ID),
            "Group": str(row.Group),
            "group_index": int(row.group_index),
            "mu_variable": var,
            "posterior_mean_mu": np.mean(s),
            "posterior_median_mu": np.median(s),
            "posterior_sd_mu": np.std(s, ddof=1),
            "mu_hdi_lower": low[0],
            "mu_hdi_upper": high[0],
        })
    return pd.DataFrame(rows)


def add_panel(ax, samples, groups, ylabel, title, credible_mass):
    mean = samples.mean(axis=0)
    low, high = exact_hdi(samples, credible_mass)
    x = np.arange(len(groups))
    ax.bar(x, mean, width=0.62, color="#1f77b4", edgecolor="none", zorder=2)
    ax.errorbar(
        x, mean,
        yerr=np.vstack([mean - low, high - mean]),
        fmt="none", ecolor="black", elinewidth=1.1,
        capsize=3, capthick=1.0, zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=38, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--group_order", type=Path, default=DEFAULT_GROUPS)
    ap.add_argument("--patient_map", type=Path, default=DEFAULT_PATIENT_MAP)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--credible_mass", type=float, default=0.95)
    ap.add_argument("--digits", type=int, default=6)
    args = ap.parse_args()

    for path in (args.trace, args.group_order, args.patient_map):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0 < args.credible_mass < 1:
        raise ValueError("--credible_mass must be between 0 and 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    group_df = pd.read_csv(args.group_order)
    required = {"group_index", "Group"}
    missing = required - set(group_df.columns)
    if missing:
        raise ValueError(f"Group-order file missing columns: {sorted(missing)}")
    group_df = group_df.sort_values("group_index").reset_index(drop=True)
    expected = np.arange(len(group_df))
    if not np.array_equal(group_df["group_index"].to_numpy(int), expected):
        raise ValueError("group_index must be consecutive and start at zero")
    groups = group_df["Group"].astype(str).tolist()

    idata = az.from_netcdf(args.trace)
    theta = posterior_matrix(idata, "theta", len(groups))
    sigma = posterior_matrix(idata, "sigma", len(groups))
    if np.any(theta <= 0) or np.any(sigma <= 0):
        raise ValueError("theta and sigma must be strictly positive")

    dtn = sigma ** 2 / theta
    stationary_variance = sigma ** 2 / (2 * theta)

    group_summary = pd.concat([
        summarize(theta, groups, "theta", args.credible_mass),
        summarize(sigma, groups, "sigma", args.credible_mass),
        summarize(dtn, groups, "sigma_squared_over_theta", args.credible_mass),
        summarize(
            stationary_variance, groups,
            "stationary_variance_sigma_squared_over_2theta",
            args.credible_mass,
        ),
    ], ignore_index=True)
    float_cols = group_summary.select_dtypes(include="number").columns
    group_summary[float_cols] = group_summary[float_cols].round(args.digits)
    group_summary_path = args.out_dir / "Supplementary_Table_group_OU_parameter_summary.csv"
    group_summary.to_csv(group_summary_path, index=False)

    pairwise_specs = [
        (theta, "theta", "Supplementary_Table_pairwise_theta_probabilities.csv"),
        (sigma, "sigma", "Supplementary_Table_pairwise_sigma_probabilities.csv"),
        (dtn, "sigma_squared_over_theta", "Supplementary_Table_pairwise_drift_to_noise_probabilities.csv"),
    ]
    pairwise_paths = []
    for samples, name, filename in pairwise_specs:
        table = pairwise_table(samples, groups, name)
        nums = table.select_dtypes(include="number").columns
        table[nums] = table[nums].round(args.digits)
        path = args.out_dir / filename
        table.to_csv(path, index=False)
        pairwise_paths.append(path)

    patient_map = pd.read_csv(args.patient_map)
    patient_summary = patient_mu_table(idata, patient_map, args.credible_mass)
    nums = patient_summary.select_dtypes(include="number").columns
    patient_summary[nums] = patient_summary[nums].round(args.digits)
    patient_summary_path = args.out_dir / "Supplementary_Table_patient_equilibrium_states.csv"
    patient_summary.to_csv(patient_summary_path, index=False)

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
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), constrained_layout=True)
    add_panel(axes[0], theta, groups, r"Mean-reversion strength, $\theta$", "A.  Mean-reversion strength", args.credible_mass)
    add_panel(axes[1], sigma, groups, r"Diffusion scale, $\sigma$", "B.  Diffusion scale", args.credible_mass)
    add_panel(axes[2], dtn, groups, r"Drift-to-noise, $\sigma^2/\theta$", "C.  Drift-to-noise ratio", args.credible_mass)

    out_png = args.out_dir / "Figure3_OU_direct_transition_group_parameters.png"
    out_pdf = args.out_dir / "Figure3_OU_direct_transition_group_parameters.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved Figure 3:")
    print(f"     {out_png}")
    print(f"     {out_pdf}")
    print("[OK] Saved supplementary tables:")
    print(f"     {group_summary_path}")
    for path in pairwise_paths:
        print(f"     {path}")
    print(f"     {patient_summary_path}")


if __name__ == "__main__":
    main()
