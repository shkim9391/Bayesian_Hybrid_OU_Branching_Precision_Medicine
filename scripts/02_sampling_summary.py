from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_BASE = Path(
    "/scripts/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Table S1 and Figure S1 sampling summaries."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Revision directory containing the input files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional explicit input file. Preferred: longitudinal_metadata_debug.csv; "
            "fallback: kmt2a_longitudinal_clean.xlsx."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <base>/sampling_characteristics.",
    )
    return parser.parse_args()


def normalize_name(name: object) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_column(columns: list[object], candidates: list[str]) -> object | None:
    lookup = {normalize_name(col): col for col in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in lookup:
            return lookup[key]
    return None


def load_excel_with_matching_sheet(path: Path) -> pd.DataFrame:
    """Read the first Excel sheet containing patient and time information."""
    workbook = pd.ExcelFile(path)
    errors: list[str] = []

    for sheet in workbook.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception as exc:  # pragma: no cover - defensive logging
            errors.append(f"{sheet}: {exc}")
            continue

        patient_col = find_column(
            list(df.columns), ["Patient_ID", "Patient ID", "patient", "patient_id"]
        )
        time_col = find_column(
            list(df.columns), ["t_years", "t", "Day", "days", "time", "time_years"]
        )
        if patient_col is not None and time_col is not None:
            print(f"Using Excel sheet: {sheet}")
            return df

    detail = "; ".join(errors) if errors else "no matching columns found"
    raise ValueError(
        f"No sheet in {path.name} contained recognizable patient and time columns ({detail})."
    )


def choose_input(base: Path, explicit_input: Path | None) -> Path:
    if explicit_input is not None:
        path = explicit_input.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    preferred = base / "longitudinal_metadata_debug.csv"
    fallback = base / "kmt2a_longitudinal_clean.xlsx"
    model_input = base / "series_auto.csv"

    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    if model_input.exists():
        warnings.warn(
            "Using series_auto.csv. It contains separate x and n rows; "
            "the script will collapse duplicate Patient_ID-time combinations."
        )
        return model_input

    raise FileNotFoundError(
        "No suitable input was found. Expected longitudinal_metadata_debug.csv, "
        "kmt2a_longitudinal_clean.xlsx, or series_auto.csv in:\n"
        f"  {base}"
    )


def load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return load_excel_with_matching_sheet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def standardize_input(df: pd.DataFrame) -> pd.DataFrame:
    patient_col = find_column(
        list(df.columns), ["Patient_ID", "Patient ID", "patient", "patient_id"]
    )
    group_col = find_column(
        list(df.columns), ["Group", "Clinical group", "clinical_group"]
    )
    years_col = find_column(
        list(df.columns), ["t_years", "time_years", "t (years)", "t"]
    )
    day_col = find_column(list(df.columns), ["Day", "days", "time_day"])

    if patient_col is None:
        raise ValueError(
            "Could not identify a patient column. Expected Patient_ID or a close variant."
        )
    if years_col is None and day_col is None:
        raise ValueError(
            "Could not identify a time column. Expected t_years, t, or Day."
        )

    out = pd.DataFrame()
    out["Patient_ID"] = df[patient_col].astype(str).str.strip()
    out["Clinical_group"] = (
        df[group_col].astype(str).str.strip() if group_col is not None else "Not specified"
    )

    if years_col is not None:
        out["t_years"] = pd.to_numeric(df[years_col], errors="coerce")
    else:
        out["t_years"] = pd.to_numeric(df[day_col], errors="coerce") / 365.25

    if day_col is not None:
        out["Day"] = pd.to_numeric(df[day_col], errors="coerce")
    else:
        out["Day"] = out["t_years"] * 365.25

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["Patient_ID", "t_years"])
    out = out[out["Patient_ID"].ne("") & out["Patient_ID"].ne("nan")]

    if out.empty:
        raise ValueError("No valid patient-time observations remained after cleaning.")

    return out


def safe_cv(values: np.ndarray) -> float:
    """Sample CV of positive intervals; undefined for fewer than two intervals."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        return np.nan
    mean_value = values.mean()
    if mean_value <= 0:
        return np.nan
    return float(values.std(ddof=1) / mean_value)


def compute_sampling_statistics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    patient_rows: list[dict[str, object]] = []
    all_intervals: list[float] = []

    for patient_id, patient_df in data.groupby("Patient_ID", sort=True):
        raw_times = np.sort(patient_df["t_years"].to_numpy(dtype=float))
        unique_times = np.unique(raw_times)
        intervals = np.diff(unique_times)
        positive_intervals = intervals[intervals > 0]
        all_intervals.extend(positive_intervals.tolist())

        groups = sorted(
            g for g in patient_df["Clinical_group"].dropna().astype(str).unique() if g
        )
        group_label = "; ".join(groups) if groups else "Not specified"

        follow_up = float(unique_times[-1] - unique_times[0]) if unique_times.size else np.nan
        patient_rows.append(
            {
                "Patient_ID": patient_id,
                "Clinical_group": group_label,
                "n_observations": int(raw_times.size),
                "n_unique_timepoints": int(unique_times.size),
                "same_time_replicates": int(raw_times.size - unique_times.size),
                "follow_up_years": follow_up,
                "median_interval_years": (
                    float(np.median(positive_intervals))
                    if positive_intervals.size
                    else np.nan
                ),
                "min_interval_years": (
                    float(np.min(positive_intervals))
                    if positive_intervals.size
                    else np.nan
                ),
                "max_interval_years": (
                    float(np.max(positive_intervals))
                    if positive_intervals.size
                    else np.nan
                ),
                "interval_cv": safe_cv(positive_intervals),
            }
        )

    patient_table = pd.DataFrame(patient_rows)
    intervals_array = np.asarray(all_intervals, dtype=float)

    n_obs = patient_table["n_observations"].to_numpy(dtype=float)
    n_time = patient_table["n_unique_timepoints"].to_numpy(dtype=float)
    follow = patient_table["follow_up_years"].to_numpy(dtype=float)
    cv_values = patient_table["interval_cv"].dropna().to_numpy(dtype=float)

    def fmt_int_range(values: np.ndarray) -> str:
        return f"{int(np.nanmin(values))}–{int(np.nanmax(values))}"

    def fmt_range(values: np.ndarray, decimals: int = 3) -> str:
        return f"{np.nanmin(values):.{decimals}f}–{np.nanmax(values):.{decimals}f}"

    summary_rows = [
        ("Number of patients", int(patient_table.shape[0])),
        ("Total longitudinal observations", int(n_obs.sum())),
        ("Total unique patient-time sampling points", int(n_time.sum())),
        ("Mean observations per patient", round(float(np.mean(n_obs)), 2)),
        ("Median observations per patient", round(float(np.median(n_obs)), 2)),
        ("Range of observations per patient", fmt_int_range(n_obs)),
        ("Mean unique timepoints per patient", round(float(np.mean(n_time)), 2)),
        ("Median unique timepoints per patient", round(float(np.median(n_time)), 2)),
        ("Range of unique timepoints per patient", fmt_int_range(n_time)),
        ("Median follow-up duration (years)", round(float(np.nanmedian(follow)), 3)),
        ("Range of follow-up duration (years)", fmt_range(follow)),
        (
            "Median positive sampling interval (years)",
            round(float(np.nanmedian(intervals_array)), 3),
        ),
        ("Range of positive sampling intervals (years)", fmt_range(intervals_array)),
        (
            "Median sampling irregularity index (interval CV)",
            round(float(np.nanmedian(cv_values)), 3) if cv_values.size else "NA",
        ),
        (
            "IQR of sampling irregularity index (interval CV)",
            (
                f"{np.nanpercentile(cv_values, 25):.3f}–"
                f"{np.nanpercentile(cv_values, 75):.3f}"
                if cv_values.size
                else "NA"
            ),
        ),
        (
            "Patients contributing an interval CV",
            f"{cv_values.size} of {patient_table.shape[0]}",
        ),
        (
            "Patients with at least 5 unique timepoints",
            f"{int((n_time >= 5).sum())} ({100 * (n_time >= 5).mean():.1f}%)",
        ),
        (
            "Patients with at least 10 unique timepoints",
            f"{int((n_time >= 10).sum())} ({100 * (n_time >= 10).mean():.1f}%)",
        ),
    ]

    summary_table = pd.DataFrame(summary_rows, columns=["Characteristic", "Value"])
    return summary_table, patient_table, intervals_array


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def make_figure(
    patient_table: pd.DataFrame,
    all_intervals: np.ndarray,
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    obs = patient_table["n_unique_timepoints"].dropna().to_numpy(dtype=float)
    follow = patient_table["follow_up_years"].dropna().to_numpy(dtype=float)
    intervals = all_intervals[np.isfinite(all_intervals) & (all_intervals > 0)]
    cvs = patient_table["interval_cv"].dropna().to_numpy(dtype=float)

    # A. Unique timepoints per patient
    ax = axes[0, 0]
    bins = np.arange(obs.min() - 0.5, obs.max() + 1.5, 1)
    ax.hist(obs, bins=bins, edgecolor="black", linewidth=0.8)
    ax.axvline(np.median(obs), linestyle="--", linewidth=1.2, label=f"Median = {np.median(obs):.1f}")
    ax.set_xlabel("Unique longitudinal timepoints per patient")
    ax.set_ylabel("Number of patients")
    ax.legend(frameon=False)
    add_panel_label(ax, "A")

    # B. Follow-up duration
    ax = axes[0, 1]
    ax.hist(follow, bins="auto", edgecolor="black", linewidth=0.8)
    ax.axvline(np.median(follow), linestyle="--", linewidth=1.2, label=f"Median = {np.median(follow):.2f} y")
    ax.set_xlabel("Follow-up duration (years)")
    ax.set_ylabel("Number of patients")
    ax.legend(frameon=False)
    add_panel_label(ax, "B")

    # C. Positive consecutive sampling intervals
    ax = axes[1, 0]
    ax.hist(intervals, bins="auto", edgecolor="black", linewidth=0.8)
    ax.axvline(np.median(intervals), linestyle="--", linewidth=1.2, label=f"Median = {np.median(intervals):.2f} y")
    ax.set_xlabel("Positive consecutive sampling interval (years)")
    ax.set_ylabel("Number of intervals")
    ax.legend(frameon=False)
    add_panel_label(ax, "C")

    # D. Patient-level interval CV
    ax = axes[1, 1]
    if cvs.size:
        ax.hist(cvs, bins="auto", edgecolor="black", linewidth=0.8)
        ax.axvline(np.median(cvs), linestyle="--", linewidth=1.2, label=f"Median = {np.median(cvs):.2f}")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "Interval CV unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Sampling irregularity index (CV of intervals)")
    ax.set_ylabel("Number of patients")
    add_panel_label(ax, "D")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Longitudinal sampling characteristics of the KMT2A-rearranged leukemia cohort", fontsize=14)
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    base = args.base.expanduser().resolve()
    outdir = (args.outdir or (base / "sampling_characteristics")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    input_path = choose_input(base, args.input)
    print(f"Input: {input_path}")

    raw = load_input(input_path)
    data = standardize_input(raw)
    summary_table, patient_table, all_intervals = compute_sampling_statistics(data)

    summary_path = outdir / "Table_S1_sampling_summary.csv"
    patient_path = outdir / "Table_S1_patient_sampling_characteristics.csv"
    figure_png = outdir / "Figure_S1_sampling_characteristics.png"
    figure_pdf = outdir / "Figure_S1_sampling_characteristics.pdf"

    summary_table.to_csv(summary_path, index=False)
    patient_table.to_csv(patient_path, index=False, float_format="%.6f")
    make_figure(patient_table, all_intervals, figure_png, figure_pdf)

    print("\nCohort summary")
    print(summary_table.to_string(index=False))
    print("\nCreated:")
    for path in (summary_path, patient_path, figure_png, figure_pdf):
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
