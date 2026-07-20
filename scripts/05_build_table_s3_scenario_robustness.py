from __future__ import annotations

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Default project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = (
    Path.home()
    /scripts/ "05_build_table_s3_scenario_robustness.py"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "Figure_2"
    / "out_simulation_benchmark"
)

DEFAULT_ESTIMATES = (
    INPUT_DIR
    / "Table_S2_simulation_estimates.csv"
)

DEFAULT_SCENARIOS = (
    INPUT_DIR
    / "Table_S2_simulation_scenarios.csv"
)

DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "Figure_2"
    / "out_table_s3"
)


PARAMETERS = ("mu", "theta", "sigma")

TRUE_COLUMNS = {
    "mu": "mu_true",
    "theta": "theta_true",
    "sigma": "sigma_true",
}

LOWER_COLUMNS = {
    "mu": "mu_lower",
    "theta": "theta_lower",
    "sigma": "sigma_lower",
}

UPPER_COLUMNS = {
    "mu": "mu_upper",
    "theta": "theta_upper",
    "sigma": "sigma_upper",
}

COVERED_COLUMNS = {
    "mu": "mu_covered",
    "theta": "theta_covered",
    "sigma": "sigma_covered",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def safe_percent(numerator: int | float, denominator: int | float) -> float:
    """Return percentage or NaN when denominator is zero."""
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return np.nan
    return 100.0 * float(numerator) / float(denominator)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation with protection against insufficient variation."""
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    x_use = x.loc[mask]
    y_use = y.loc[mask]
    if x_use.nunique() < 2 or y_use.nunique() < 2:
        return np.nan
    return float(x_use.corr(y_use, method="spearman"))


def criterion_note(method: str) -> str:
    """Describe the method-specific reliability definition."""
    method_lower = str(method).strip().lower()

    if method_lower == "bayesian":
        return (
            "Finite posterior estimate; sampling completed; no boundary contact; "
            "and prespecified MCMC diagnostic criteria passed."
        )
    if method_lower == "mle":
        return (
            "Finite estimate; optimizer reported success; and no parameter "
            "boundary contact."
        )
    if method_lower == "mom":
        return (
            "Direct moment estimate was identifiable, finite, and did not "
            "contact a prespecified boundary."
        )
    if "imputed" in method_lower:
        return (
            "Moment estimate after prespecified model-assisted imputation was "
            "finite and did not contact a prespecified boundary."
        )

    return (
        "Finite estimate and method-specific success/reliability fields supplied "
        "by the simulation benchmark."
    )


def validate_estimates(df: pd.DataFrame) -> None:
    """Validate required columns before summarization."""
    required = {
        "simulation_id",
        "scenario",
        "method",
        "finite_estimate",
        "success",
        "reliable_estimate",
        *PARAMETERS,
        *TRUE_COLUMNS.values(),
    }

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "The estimates file is missing required columns:\n  "
            + "\n  ".join(missing)
        )

    if df.empty:
        raise ValueError("The estimates file contains no rows.")

    duplicate_mask = df.duplicated(
        subset=["simulation_id", "method"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, ["simulation_id", "method"]]
            .drop_duplicates()
            .head(10)
        )
        raise ValueError(
            "Duplicate simulation_id × method rows were detected. Examples:\n"
            + examples.to_string(index=False)
        )


def normalize_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce benchmark indicator columns to clean booleans."""
    out = df.copy()

    boolean_columns = [
        "finite_estimate",
        "success",
        "reliable_estimate",
        "boundary_contact",
    ]

    truth_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    for col in boolean_columns:
        if col not in out.columns:
            continue

        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].fillna(False)
        else:
            out[col] = (
                out[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(truth_map)
                .fillna(False)
                .astype(bool)
            )

    return out


def load_inputs(
    estimates_path: Path,
    scenarios_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load and validate the benchmark files."""
    if not estimates_path.exists():
        raise FileNotFoundError(
            f"Simulation estimates file not found:\n{estimates_path}"
        )

    estimates = pd.read_csv(estimates_path)
    estimates = normalize_boolean_columns(estimates)
    validate_estimates(estimates)

    scenarios = None
    if scenarios_path is not None and scenarios_path.exists():
        scenarios = pd.read_csv(scenarios_path)

        required_scenario_columns = {
            "scenario",
            *TRUE_COLUMNS.values(),
        }
        missing = sorted(required_scenario_columns.difference(scenarios.columns))
        if missing:
            raise ValueError(
                "The scenarios file is missing required columns:\n  "
                + "\n  ".join(missing)
            )

        scenarios = scenarios.drop_duplicates(subset=["scenario"]).copy()

    return estimates, scenarios


# ---------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------
def summarize_group(
    group: pd.DataFrame,
    parameter: str,
    all_simulation_n: int,
) -> dict:
    """Summarize one scenario × method × parameter group."""
    true_col = TRUE_COLUMNS[parameter]
    covered_col = COVERED_COLUMNS[parameter]

    estimate = pd.to_numeric(group[parameter], errors="coerce")
    truth = pd.to_numeric(group[true_col], errors="coerce")

    finite_mask = (
        group["finite_estimate"].astype(bool)
        & estimate.notna()
        & np.isfinite(estimate)
    )
    success_mask = group["success"].astype(bool)
    reliable_mask = group["reliable_estimate"].astype(bool) & finite_mask

    valid = finite_mask & truth.notna() & np.isfinite(truth)
    error = estimate.loc[valid] - truth.loc[valid]
    absolute_error = error.abs()

    n_rows = int(len(group))
    finite_n = int(finite_mask.sum())
    success_n = int(success_mask.sum())
    reliable_n = int(reliable_mask.sum())

    if valid.any():
        bias = float(error.mean())
        mae = float(absolute_error.mean())
        median_ae = float(absolute_error.median())
        rmse = float(np.sqrt(np.mean(np.square(error))))
        spearman = safe_spearman(
            truth.loc[valid],
            estimate.loc[valid],
        )
    else:
        bias = np.nan
        mae = np.nan
        median_ae = np.nan
        rmse = np.nan
        spearman = np.nan

    coverage_n = 0
    coverage_95_percent = np.nan

    if covered_col in group.columns:
        covered = pd.to_numeric(group[covered_col], errors="coerce")
        coverage_mask = valid & covered.notna()
        coverage_n = int(coverage_mask.sum())

        if coverage_n > 0:
            coverage_95_percent = float(
                100.0 * covered.loc[coverage_mask].mean()
            )

    true_values = truth.dropna().unique()
    true_value = (
        float(true_values[0])
        if len(true_values) == 1
        else np.nan
    )

    return {
        "parameter": parameter,
        "true_value": true_value,
        "all_simulations_n": int(all_simulation_n),
        "method_rows_n": n_rows,
        "finite_n": finite_n,
        "finite_percent_of_all_simulations": safe_percent(
            finite_n,
            all_simulation_n,
        ),
        "success_n": success_n,
        "success_percent_of_all_simulations": safe_percent(
            success_n,
            all_simulation_n,
        ),
        "reliable_n": reliable_n,
        "reliable_percent_of_all_simulations": safe_percent(
            reliable_n,
            all_simulation_n,
        ),
        "reliable_percent_among_finite": safe_percent(
            reliable_n,
            finite_n,
        ),
        "bias": bias,
        "mae": mae,
        "median_absolute_error": median_ae,
        "rmse": rmse,
        "spearman_true_vs_estimated": spearman,
        "coverage_evaluable_n": coverage_n,
        "coverage_95_percent": coverage_95_percent,
    }


def build_scenario_table(
    estimates: pd.DataFrame,
    scenarios: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build scenario-stratified Table S3."""
    rows: list[dict] = []

    scenario_order = (
        scenarios["scenario"].astype(str).tolist()
        if scenarios is not None
        else sorted(estimates["scenario"].astype(str).unique())
    )

    method_order = list(
        dict.fromkeys(estimates["method"].astype(str).tolist())
    )

    for scenario in scenario_order:
        scenario_df = estimates.loc[
            estimates["scenario"].astype(str) == str(scenario)
        ].copy()

        if scenario_df.empty:
            warnings.warn(
                f"Scenario {scenario!r} appears in the scenario table "
                "but has no estimate rows."
            )
            continue

        all_simulation_n = int(
            scenario_df["simulation_id"].nunique()
        )

        for method in method_order:
            method_df = scenario_df.loc[
                scenario_df["method"].astype(str) == str(method)
            ].copy()

            if method_df.empty:
                continue

            for parameter in PARAMETERS:
                row = summarize_group(
                    method_df,
                    parameter,
                    all_simulation_n=all_simulation_n,
                )
                row.update(
                    {
                        "scenario": scenario,
                        "method": method,
                        "reliability_definition": criterion_note(method),
                    }
                )
                rows.append(row)

    table = pd.DataFrame(rows)

    preferred_columns = [
        "scenario",
        "method",
        "parameter",
        "true_value",
        "all_simulations_n",
        "method_rows_n",
        "finite_n",
        "finite_percent_of_all_simulations",
        "success_n",
        "success_percent_of_all_simulations",
        "reliable_n",
        "reliable_percent_of_all_simulations",
        "reliable_percent_among_finite",
        "bias",
        "mae",
        "median_absolute_error",
        "rmse",
        "spearman_true_vs_estimated",
        "coverage_evaluable_n",
        "coverage_95_percent",
        "reliability_definition",
    ]

    table = table[preferred_columns]

    return table


def build_overall_table(
    estimates: pd.DataFrame,
) -> pd.DataFrame:
    """Build all-scenario method summary."""
    rows: list[dict] = []

    total_simulations = int(
        estimates["simulation_id"].nunique()
    )

    method_order = list(
        dict.fromkeys(estimates["method"].astype(str).tolist())
    )

    for method in method_order:
        method_df = estimates.loc[
            estimates["method"].astype(str) == str(method)
        ].copy()

        for parameter in PARAMETERS:
            row = summarize_group(
                method_df,
                parameter,
                all_simulation_n=total_simulations,
            )
            row.update(
                {
                    "method": method,
                    "reliability_definition": criterion_note(method),
                }
            )
            rows.append(row)

    table = pd.DataFrame(rows)

    preferred_columns = [
        "method",
        "parameter",
        "all_simulations_n",
        "method_rows_n",
        "finite_n",
        "finite_percent_of_all_simulations",
        "success_n",
        "success_percent_of_all_simulations",
        "reliable_n",
        "reliable_percent_of_all_simulations",
        "reliable_percent_among_finite",
        "bias",
        "mae",
        "median_absolute_error",
        "rmse",
        "spearman_true_vs_estimated",
        "coverage_evaluable_n",
        "coverage_95_percent",
        "reliability_definition",
    ]

    table = table[preferred_columns]

    return table


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
def round_numeric_columns(
    df: pd.DataFrame,
    digits: int = 4,
) -> pd.DataFrame:
    """Round floating-point columns while preserving integer counts."""
    out = df.copy()
    float_columns = out.select_dtypes(
        include=["float16", "float32", "float64"]
    ).columns
    out[float_columns] = out[float_columns].round(digits)
    return out


def write_workbook(
    scenario_table: pd.DataFrame,
    overall_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write an Excel workbook with explanatory notes."""
    notes = pd.DataFrame(
        {
            "Item": [
                "Purpose",
                "Accuracy metrics",
                "Reliability",
                "Reliability denominator",
                "Coverage",
                "MoM and imputed MoM",
                "Interpretation",
            ],
            "Description": [
                (
                    "Scenario-stratified recovery and computational robustness "
                    "for the completed OU simulation benchmark."
                ),
                (
                    "Bias, MAE, median absolute error, and RMSE are evaluated "
                    "against known simulation-generating parameters."
                ),
                (
                    "Reliability is method-specific and should not be interpreted "
                    "as an identical diagnostic test across methods."
                ),
                (
                    "The table reports reliability both as a percentage of all "
                    "simulations and among finite estimates."
                ),
                (
                    "95% interval coverage is reported only when interval bounds "
                    "or a precomputed coverage indicator are available, ordinarily "
                    "for Bayesian posterior intervals."
                ),
                (
                    "Direct MoM and model-assisted imputed MoM are retained as "
                    "separate methods because they differ in information source "
                    "and availability."
                ),
                (
                    "No composite ranking is generated. Accuracy, availability, "
                    "and computational reliability are reported separately."
                ),
            ],
        }
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        scenario_table.to_excel(
            writer,
            sheet_name="Scenario_stratified",
            index=False,
        )
        overall_table.to_excel(
            writer,
            sheet_name="Overall_summary",
            index=False,
        )
        notes.to_excel(
            writer,
            sheet_name="Notes",
            index=False,
        )

        for sheet_name in [
            "Scenario_stratified",
            "Overall_summary",
            "Notes",
        ]:
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter

                for cell in column_cells:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))

                worksheet.column_dimensions[column_letter].width = min(
                    max(max_length + 2, 10),
                    60,
                )


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build scenario-stratified Table S3 from the completed "
            "OU simulation benchmark."
        )
    )

    parser.add_argument(
        "--estimates",
        type=Path,
        default=DEFAULT_ESTIMATES,
        help="Path to Table_S2_simulation_estimates.csv.",
    )

    parser.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_SCENARIOS,
        help="Path to Table_S2_simulation_scenarios.csv.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for Table S3 files.",
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Number of decimal places for reported metrics.",
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    estimates, scenarios = load_inputs(
        estimates_path=args.estimates,
        scenarios_path=args.scenarios,
    )

    scenario_table = build_scenario_table(
        estimates=estimates,
        scenarios=scenarios,
    )

    overall_table = build_overall_table(
        estimates=estimates,
    )

    scenario_table = round_numeric_columns(
        scenario_table,
        digits=args.digits,
    )

    overall_table = round_numeric_columns(
        overall_table,
        digits=args.digits,
    )

    scenario_csv = (
        args.out
        / "Table_S3_scenario_stratified_method_robustness.csv"
    )

    overall_csv = (
        args.out
        / "Table_S3_method_summary.csv"
    )

    workbook_path = (
        args.out
        / "Table_S3_OU_method_robustness.xlsx"
    )

    scenario_table.to_csv(
        scenario_csv,
        index=False,
    )

    overall_table.to_csv(
        overall_csv,
        index=False,
    )

    write_workbook(
        scenario_table=scenario_table,
        overall_table=overall_table,
        output_path=workbook_path,
    )

    print("[OK] Saved scenario-stratified Table S3:")
    print(f"     {scenario_csv}")

    print("[OK] Saved overall method summary:")
    print(f"     {overall_csv}")

    print("[OK] Saved Excel workbook:")
    print(f"     {workbook_path}")

    print("\n[SUMMARY] Overall reliability by method:")
    reliability_view = (
        overall_table[
            [
                "method",
                "all_simulations_n",
                "method_rows_n",
                "finite_n",
                "reliable_n",
                "reliable_percent_of_all_simulations",
                "reliable_percent_among_finite",
            ]
        ]
        .drop_duplicates(subset=["method"])
        .reset_index(drop=True)
    )

    print(
        reliability_view.to_string(
            index=False,
        )
    )


if __name__ == "__main__":
    main()
