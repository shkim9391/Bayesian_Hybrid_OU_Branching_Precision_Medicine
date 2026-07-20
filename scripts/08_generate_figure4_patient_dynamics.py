from __future__ import annotations

import argparse
import re
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil

SEED = 123

PROJECT_ROOT = (
    Path.home()
    /scripts/ "08_generate_figure4_patient_dynamics.py"
)

DEFAULT_SERIES = PROJECT_ROOT / "series_auto.csv"
POSTERIOR_DIR = PROJECT_ROOT / "Figure_3" / "out_ou_direct_transition"
DEFAULT_TRACE = POSTERIOR_DIR / "ou_direct_transition_trace.nc"
DEFAULT_GROUP_ORDER = POSTERIOR_DIR / "ou_direct_transition_group_order.csv"
DEFAULT_PATIENT_MAP = POSTERIOR_DIR / "ou_direct_transition_patient_variable_map.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "Figure_4" / "out_ou_direct_transition_patient_dynamics"


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
    return cleaned or "patient"


def exact_hdi_1d(samples: np.ndarray, credible_mass: float = 0.95) -> tuple[float, float]:
    values = np.asarray(samples, dtype=float).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Valid posterior samples are required for HDI calculation.")
    s = np.sort(values)
    width_n = min(max(int(np.floor(credible_mass * len(s))), 1), len(s) - 1)
    widths = s[width_n:] - s[: len(s) - width_n]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + width_n])


def read_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"Patient_ID", "series", "t", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Series file is missing columns: {sorted(missing)}")

    df["Patient_ID"] = df["Patient_ID"].astype("string").str.strip()
    df["series"] = df["series"].astype("string").str.strip().str.lower()
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.loc[df["series"].isin(["x", "n"])].dropna(
        subset=["Patient_ID", "series", "t", "value"]
    ).copy()

    duplicate_counts = df.groupby(["Patient_ID", "series", "t"]).size()
    duplicate_groups = int((duplicate_counts > 1).sum())
    if duplicate_groups:
        print(
            f"[INFO] Collapsing {duplicate_groups} duplicate "
            "Patient_ID-series-time combinations by mean."
        )

    return (
        df.groupby(["Patient_ID", "series", "t"], as_index=False)
        .agg(value=("value", "mean"))
        .sort_values(["Patient_ID", "series", "t"], kind="mergesort")
        .reset_index(drop=True)
    )


def load_group_order(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"group_index", "Group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Group-order file is missing columns: {sorted(missing)}")
    df = df.sort_values("group_index").reset_index(drop=True)
    expected = np.arange(len(df))
    if not np.array_equal(df["group_index"].to_numpy(dtype=int), expected):
        raise ValueError("group_index must be consecutive and begin at zero.")
    return df


def load_patient_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Patient_ID", "Group", "group_index", "mu_variable"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Patient-variable map is missing columns: {sorted(missing)}")
    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()
    return df


def posterior_vector(idata: az.InferenceData, variable: str) -> np.ndarray:
    if variable not in idata.posterior:
        raise KeyError(f"Posterior variable {variable!r} was not found.")
    return np.asarray(idata.posterior[variable].values, dtype=float).reshape(-1)


def posterior_group_matrix(idata: az.InferenceData, variable: str, n_groups: int) -> np.ndarray:
    if variable not in idata.posterior:
        raise KeyError(f"Posterior variable {variable!r} was not found.")
    values = np.asarray(idata.posterior[variable].values, dtype=float)
    if values.shape[-1] != n_groups:
        raise ValueError(f"{variable} has {values.shape[-1]} groups; expected {n_groups}.")
    return values.reshape(-1, n_groups)


def simulate_conditional_paths(
    times: np.ndarray,
    initial_value: float,
    mu_samples: np.ndarray,
    theta_samples: np.ndarray,
    sigma_samples: np.ndarray,
    tau_samples: np.ndarray,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Patient x-series times must be strictly increasing.")

    posterior_n = min(
        len(mu_samples), len(theta_samples), len(sigma_samples), len(tau_samples)
    )
    indices = rng.integers(0, posterior_n, size=n_paths)
    paths = np.empty((n_paths, len(times)), dtype=float)
    paths[:, 0] = float(initial_value)

    for path_index, posterior_index in enumerate(indices):
        mu = float(mu_samples[posterior_index])
        theta = float(theta_samples[posterior_index])
        sigma = float(sigma_samples[posterior_index])
        tau_x = float(tau_samples[posterior_index])
        current = float(initial_value)

        for time_index in range(1, len(times)):
            dt = float(times[time_index] - times[time_index - 1])
            decay = np.exp(-theta * dt)
            transition_mean = mu + (current - mu) * decay
            process_variance = (
                sigma**2 / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * dt))
            )
            total_variance = process_variance + tau_x**2
            current = rng.normal(
                transition_mean, np.sqrt(max(total_variance, 1e-12))
            )
            paths[path_index, time_index] = current

    return paths


def deterministic_mean_paths(
    times: np.ndarray,
    initial_value: float,
    mu_samples: np.ndarray,
    theta_samples: np.ndarray,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    posterior_n = min(len(mu_samples), len(theta_samples))
    sample_n = min(n_paths, posterior_n)
    indices = rng.choice(posterior_n, size=sample_n, replace=False)
    paths = np.empty((sample_n, len(times)), dtype=float)
    paths[:, 0] = float(initial_value)

    for path_index, posterior_index in enumerate(indices):
        mu = float(mu_samples[posterior_index])
        theta = float(theta_samples[posterior_index])
        current = float(initial_value)
        for time_index in range(1, len(times)):
            dt = float(times[time_index] - times[time_index - 1])
            current = mu + (current - mu) * np.exp(-theta * dt)
            paths[path_index, time_index] = current
    return paths


def make_patient_figure(
    patient_id: str,
    series_df: pd.DataFrame,
    idata: az.InferenceData,
    patient_record: pd.Series,
    theta_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    tau_samples: np.ndarray,
    out_dir: Path,
    out_stem: str,
    n_ppc: int,
    n_overlay: int,
    credible_mass: float,
    seed: int,
) -> dict:
    patient_data = series_df.loc[
        series_df["Patient_ID"].astype(str) == str(patient_id)
    ].copy()

    x_df = patient_data.loc[patient_data["series"] == "x"].sort_values("t")
    if x_df["t"].nunique() < 2:
        raise ValueError(f"Patient {patient_id} has fewer than two x time points.")

    times_x = x_df["t"].to_numpy(dtype=float)
    values_x = x_df["value"].to_numpy(dtype=float)

    n_df = patient_data.loc[patient_data["series"] == "n"].sort_values("t")
    times_n = n_df["t"].to_numpy(dtype=float)
    values_n = n_df["value"].to_numpy(dtype=float)

    group_index = int(patient_record["group_index"])
    group_name = str(patient_record["Group"])
    mu_variable = str(patient_record["mu_variable"])

    mu_samples = posterior_vector(idata, mu_variable)
    theta_samples = theta_matrix[:, group_index]
    sigma_samples = sigma_matrix[:, group_index]

    ppc_paths = simulate_conditional_paths(
        times_x,
        float(values_x[0]),
        mu_samples,
        theta_samples,
        sigma_samples,
        tau_samples,
        n_ppc,
        seed,
    )

    alpha = (1.0 - credible_mass) / 2.0
    ppc_lower = np.percentile(ppc_paths, 100.0 * alpha, axis=0)
    ppc_upper = np.percentile(ppc_paths, 100.0 * (1.0 - alpha), axis=0)
    ppc_median = np.median(ppc_paths, axis=0)

    mean_paths = deterministic_mean_paths(
        times_x,
        float(values_x[0]),
        mu_samples,
        theta_samples,
        n_overlay,
        seed + 1,
    )
    mean_path = np.mean(mean_paths, axis=0)
    coverage = float(np.mean((values_x >= ppc_lower) & (values_x <= ppc_upper)))

    mu_low, mu_high = exact_hdi_1d(mu_samples, credible_mass)
    theta_low, theta_high = exact_hdi_1d(theta_samples, credible_mass)
    sigma_low, sigma_high = exact_hdi_1d(sigma_samples, credible_mass)

    mu_mean = float(np.mean(mu_samples))
    theta_mean = float(np.mean(theta_samples))
    sigma_mean = float(np.mean(sigma_samples))

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

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)

    axes[0].fill_between(
        times_x,
        ppc_lower,
        ppc_upper,
        color="#c7dcef",
        alpha=0.9,
        label="95% posterior predictive band",
    )
    for path in mean_paths:
        axes[0].plot(times_x, path, color="#d62728", alpha=0.05, linewidth=0.6)
    axes[0].plot(
        times_x,
        mean_path,
        color="#d62728",
        linewidth=1.8,
        label="Posterior mean trajectory",
    )
    axes[0].plot(
        times_x,
        ppc_median,
        color="#9467bd",
        linewidth=1.0,
        linestyle="--",
        label="Predictive median",
    )
    axes[0].scatter(
        times_x,
        values_x,
        color="#1f77b4",
        s=28,
        zorder=4,
        label="Observed trait",
    )
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Average trait")
    axes[0].set_title(f"A  Trait dynamics ({patient_id})", loc="left", fontweight="bold")

    parameter_text = (
        rf"$\mu_i$ = {mu_mean:.2f} [{mu_low:.2f}, {mu_high:.2f}]" + "\n"
        rf"$\theta_{{{group_name}}}$ = {theta_mean:.2f} "
        rf"[{theta_low:.2f}, {theta_high:.2f}]" + "\n"
        rf"$\sigma_{{{group_name}}}$ = {sigma_mean:.2f} "
        rf"[{sigma_low:.2f}, {sigma_high:.2f}]" + "\n"
        f"PPC coverage = {100.0 * coverage:.0f}%"
    )
    axes[0].text(
        0.03,
        0.03,
        parameter_text,
        transform=axes[0].transAxes,
        fontsize=7.3,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "0.8",
        },
    )
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")

    if len(times_n) > 0:
        axes[1].step(times_n, values_n, where="post", color="#1f77b4", linewidth=1.4)
        axes[1].scatter(times_n, values_n, color="#1f77b4", s=26, zorder=3)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Clone-count series unavailable",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("Clone count")
    axes[1].set_title("B  Clone-count dynamics", loc="left", fontweight="bold")

    paired_n = 0
    correlation = np.nan
    if len(times_n) > 0:
        trait_at_clone_times = np.interp(times_n, times_x, values_x)
        axes[2].scatter(values_n, trait_at_clone_times, color="#1f77b4", s=30)
        paired_n = len(times_n)
        if (
            paired_n >= 3
            and np.std(values_n) > 0
            and np.std(trait_at_clone_times) > 0
        ):
            correlation = float(np.corrcoef(values_n, trait_at_clone_times)[0, 1])
        if np.isfinite(correlation):
            axes[2].text(
                0.04,
                0.95,
                f"Pearson r = {correlation:.2f}",
                transform=axes[2].transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
    else:
        axes[2].text(
            0.5,
            0.5,
            "Clone-count series unavailable",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
        )
    axes[2].set_xlabel("Clone count")
    axes[2].set_ylabel("Average trait")
    axes[2].set_title("C  Trait–clone relationship", loc="left", fontweight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{out_stem}.png"
    out_pdf = out_dir / f"{out_stem}.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "Patient_ID": patient_id,
        "Group": group_name,
        "group_index": group_index,
        "n_x_timepoints": int(len(times_x)),
        "n_x_transitions": int(len(times_x) - 1),
        "n_clone_timepoints": int(len(times_n)),
        "time_span_years": float(times_x[-1] - times_x[0]),
        "posterior_mean_mu": mu_mean,
        "mu_hdi_lower": mu_low,
        "mu_hdi_upper": mu_high,
        "posterior_mean_theta": theta_mean,
        "theta_hdi_lower": theta_low,
        "theta_hdi_upper": theta_high,
        "posterior_mean_sigma": sigma_mean,
        "sigma_hdi_lower": sigma_low,
        "sigma_hdi_upper": sigma_high,
        "ppc_coverage_percent": 100.0 * coverage,
        "trait_clone_paired_n": paired_n,
        "trait_clone_pearson_r": correlation,
        "figure_png": str(out_png),
        "figure_pdf": str(out_pdf),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 4 for P15 and corresponding patient-level "
            "OU direct-transition figures for all modeled patients."
        )
    )
    parser.add_argument("--series_csv", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--group_order", type=Path, default=DEFAULT_GROUP_ORDER)
    parser.add_argument("--patient_map", type=Path, default=DEFAULT_PATIENT_MAP)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--exemplar_patient", type=str, default="P15")
    parser.add_argument("--n_ppc", type=int, default=1000)
    parser.add_argument("--n_overlay", type=int, default=80)
    parser.add_argument("--credible_mass", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    for path in (args.series_csv, args.trace, args.group_order, args.patient_map):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.n_ppc < 10:
        raise ValueError("--n_ppc must be at least 10.")
    if args.n_overlay < 1:
        raise ValueError("--n_overlay must be at least 1.")
    if not 0.0 < args.credible_mass < 1.0:
        raise ValueError("--credible_mass must be between 0 and 1.")

    series_df = read_series(args.series_csv)
    group_order = load_group_order(args.group_order)
    patient_map = load_patient_map(args.patient_map)
    idata = az.from_netcdf(args.trace)

    n_groups = len(group_order)
    theta_matrix = posterior_group_matrix(idata, "theta", n_groups)
    sigma_matrix = posterior_group_matrix(idata, "sigma", n_groups)
    tau_samples = posterior_vector(idata, "tau_x")

    mapped_patients = set(patient_map["Patient_ID"].astype(str))
    series_patients = set(
        series_df.loc[series_df["series"] == "x", "Patient_ID"].astype(str)
    )
    eligible = sorted(mapped_patients & series_patients)
    eligible = [
        pid
        for pid in eligible
        if series_df.loc[
            (series_df["Patient_ID"].astype(str) == pid)
            & (series_df["series"] == "x"),
            "t",
        ].nunique()
        >= 2
    ]

    exemplar = str(args.exemplar_patient).strip()
    if exemplar not in eligible:
        raise ValueError(
            f"Exemplar patient {exemplar!r} is not eligible. "
            f"Eligible patients include: {eligible[:20]}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_patient_dir = args.out_dir / "all_patients"
    all_patient_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Generating patient dynamics figures for {len(eligible)} patients.")
    summary_rows = []

    for patient_index, patient_id in enumerate(eligible):
        patient_record = patient_map.loc[
            patient_map["Patient_ID"].astype(str) == patient_id
        ].iloc[0]
    
        filename_id = safe_filename(patient_id)
        patient_seed = args.seed + patient_index
    
        # Generate each patient exactly once
        summary = make_patient_figure(
            patient_id,
            series_df,
            idata,
            patient_record,
            theta_matrix,
            sigma_matrix,
            tau_samples,
            all_patient_dir,
            f"Figure4_{filename_id}_patient_dynamics",
            args.n_ppc,
            args.n_overlay,
            args.credible_mass,
            patient_seed,
        )
    
        summary_rows.append(summary)
    
        print(
            f"[OK] {patient_id}: "
            f"PPC coverage={summary['ppc_coverage_percent']:.1f}%"
        )
    
        # Copy the already-generated exemplar files
        if patient_id == exemplar:
            exemplar_png = (
                args.out_dir
                / f"Figure4_{filename_id}_patient_dynamics.png"
            )
    
            exemplar_pdf = (
                args.out_dir
                / f"Figure4_{filename_id}_patient_dynamics.pdf"
            )
    
            shutil.copy2(
                summary["figure_png"],
                exemplar_png,
            )
    
            shutil.copy2(
                summary["figure_pdf"],
                exemplar_pdf,
            )
    
            print("[OK] Copied identical exemplar Figure 4:")
            print(f"     {exemplar_png}")
            print(f"     {exemplar_pdf}")

    summary_table = pd.DataFrame(summary_rows)
    numeric_columns = summary_table.select_dtypes(include="number").columns
    summary_table[numeric_columns] = summary_table[numeric_columns].round(6)
    summary_path = args.out_dir / "Supplementary_Table_patient_PPC_summary.csv"
    summary_table.to_csv(summary_path, index=False)

    print("[OK] Saved all-patient PPC summary:")
    print(f"     {summary_path}")
    print("[OK] Saved all-patient figures under:")
    print(f"     {all_patient_dir}")


if __name__ == "__main__":
    main()
