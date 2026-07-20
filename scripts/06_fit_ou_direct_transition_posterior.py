from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 13


# ---------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------
def read_series(path: Path) -> pd.DataFrame:
    """Read and clean the longitudinal series table."""
    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]

    required = {
        "Patient_ID",
        "series",
        "t",
        "value",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"[series_csv] Missing columns: {sorted(missing)}. "
            f"Found: {df.columns.tolist()}"
        )

    df["Patient_ID"] = (
        df["Patient_ID"]
        .astype("string")
        .str.strip()
    )

    df["series"] = (
        df["series"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["t"] = pd.to_numeric(
        df["t"],
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = (
        df.loc[
            df["series"].isin(
                [
                    "x",
                    "n",
                ]
            )
        ]
        .dropna(
            subset=[
                "Patient_ID",
                "series",
                "t",
                "value",
            ]
        )
        .copy()
    )

    empty_patient_ids = (
        df["Patient_ID"]
        .astype(str)
        .str.len()
        .eq(0)
    )

    if empty_patient_ids.any():
        df = df.loc[~empty_patient_ids].copy()

    # Collapse repeated observations at the same patient-series-time
    # combination. For continuous traits, use the arithmetic mean.
    duplicate_counts = (
        df.groupby(
            [
                "Patient_ID",
                "series",
                "t",
            ]
        )
        .size()
    )

    duplicate_groups_n = int(
        (duplicate_counts > 1).sum()
    )

    if duplicate_groups_n > 0:
        print(
            "[INFO] Collapsing "
            f"{duplicate_groups_n} duplicate "
            "Patient_ID-series-time combinations by mean."
        )

    df = (
        df.groupby(
            [
                "Patient_ID",
                "series",
                "t",
            ],
            as_index=False,
        )
        .agg(
            value=(
                "value",
                "mean",
            )
        )
    )

    df = (
        df.sort_values(
            [
                "Patient_ID",
                "series",
                "t",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return df


def read_patient_groups(path: Path) -> pd.DataFrame:
    """Read patient-group assignments and reject conflicts."""
    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]

    lower_to_original = {
        str(column).lower(): column
        for column in df.columns
    }

    rename_map = {}

    if (
        "patient_id" in lower_to_original
        and "Patient_ID" not in df.columns
    ):
        rename_map[
            lower_to_original["patient_id"]
        ] = "Patient_ID"

    elif (
        "patient" in lower_to_original
        and "Patient_ID" not in df.columns
    ):
        rename_map[
            lower_to_original["patient"]
        ] = "Patient_ID"

    if (
        "group" in lower_to_original
        and "Group" not in df.columns
    ):
        rename_map[
            lower_to_original["group"]
        ] = "Group"

    if rename_map:
        df = df.rename(
            columns=rename_map
        )

    required = {
        "Patient_ID",
        "Group",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"[patient_group_csv] Missing columns: {sorted(missing)}. "
            f"Found: {df.columns.tolist()}"
        )

    df["Patient_ID"] = (
        df["Patient_ID"]
        .astype("string")
        .str.strip()
    )

    df["Group"] = (
        df["Group"]
        .astype("string")
        .str.strip()
    )

    df = (
        df.dropna(
            subset=[
                "Patient_ID",
                "Group",
            ]
        )
        .copy()
    )

    df = df.loc[
        df["Patient_ID"].astype(str).str.len() > 0
    ].copy()

    df = df.loc[
        df["Group"].astype(str).str.len() > 0
    ].copy()

    conflicting_groups = (
        df.groupby(
            "Patient_ID"
        )["Group"]
        .nunique()
    )

    conflicting_groups = conflicting_groups.loc[
        conflicting_groups > 1
    ]

    if not conflicting_groups.empty:
        conflicting_ids = (
            conflicting_groups
            .index
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Conflicting Group assignments were found for "
            f"{len(conflicting_ids)} patients. "
            f"Examples: {conflicting_ids[:10]}"
        )

    df = (
        df.drop_duplicates(
            subset=[
                "Patient_ID",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ---------------------------------------------------------------------
# Patient-series and naming helpers
# ---------------------------------------------------------------------
def safe_variable_suffix(
    patient_id: str,
) -> str:
    """
    Convert a patient ID into a stable PyMC/xarray-compatible suffix.
    """
    suffix = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        str(patient_id).strip(),
    )

    suffix = suffix.strip("_")

    if not suffix:
        raise ValueError(
            f"Patient ID {patient_id!r} cannot be converted "
            "to a valid posterior-variable suffix."
        )

    if suffix[0].isdigit():
        suffix = f"P_{suffix}"

    return suffix


def get_patient_x_series(
    series_df: pd.DataFrame,
    patient_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted x-series times and values for one patient."""
    patient_df = (
        series_df.loc[
            (
                series_df["Patient_ID"].astype(str)
                == str(patient_id)
            )
            & (
                series_df["series"] == "x"
            )
        ]
        .sort_values(
            "t",
            kind="mergesort",
        )
        .copy()
    )

    times = patient_df["t"].to_numpy(
        dtype=float
    )

    values = patient_df["value"].to_numpy(
        dtype=float
    )

    return times, values


# ---------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------
def fit_and_write(
    series_df: pd.DataFrame,
    patient_group_df: pd.DataFrame,
    out_dir: Path,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
) -> None:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_df = (
        series_df.loc[
            series_df["series"] == "x"
        ]
        .sort_values(
            [
                "Patient_ID",
                "t",
            ],
            kind="mergesort",
        )
        .copy()
    )

    if x_df.empty:
        raise ValueError(
            "No rows with series == 'x' were found."
        )

    x_values = x_df["value"].to_numpy(
        dtype=float
    )

    x_sd = (
        float(
            np.std(
                x_values,
                ddof=1,
            )
        )
        if len(x_values) > 1
        else np.nan
    )

    print(
        "[INFO] x-series scale: "
        f"n={len(x_values)}, "
        f"mean={np.mean(x_values):.6f}, "
        f"sd={x_sd:.6f}, "
        f"min={np.min(x_values):.6f}, "
        f"max={np.max(x_values):.6f}"
    )

    patients = []

    for patient_id, patient_df in x_df.groupby(
        "Patient_ID",
        sort=False,
    ):
        times = patient_df["t"].to_numpy(
            dtype=float
        )

        if (
            len(times) >= 2
            and np.all(
                np.isfinite(times)
            )
            and np.any(
                np.diff(times) > 0
            )
        ):
            patients.append(
                str(patient_id)
            )

    all_x_patients = int(
        x_df["Patient_ID"]
        .astype(str)
        .nunique()
    )

    print(
        f"[INFO] Patients with an x series: {all_x_patients}; "
        "patients contributing at least one positive transition: "
        f"{len(patients)}"
    )

    if not patients:
        raise ValueError(
            "No patients contain at least one positive consecutive "
            "time interval."
        )

    patient_variable_suffix = {
        patient_id: safe_variable_suffix(
            patient_id
        )
        for patient_id in patients
    }

    suffixes = list(
        patient_variable_suffix.values()
    )

    if len(set(suffixes)) != len(suffixes):
        suffix_to_ids = {}

        for patient_id, suffix in patient_variable_suffix.items():
            suffix_to_ids.setdefault(
                suffix,
                [],
            ).append(
                patient_id
            )

        collisions = {
            suffix: ids
            for suffix, ids in suffix_to_ids.items()
            if len(ids) > 1
        }

        raise ValueError(
            "Two or more patient IDs produce the same sanitized "
            f"posterior-variable suffix: {collisions}"
        )

    patient_group_indexed = (
        patient_group_df
        .set_index(
            "Patient_ID"
        )
        .reindex(
            patients
        )
    )

    if patient_group_indexed["Group"].isna().any():
        missing = (
            patient_group_indexed.index[
                patient_group_indexed["Group"].isna()
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"Missing Group labels for {len(missing)} patients in "
            f"patient_group_csv. Examples: {missing[:10]}"
        )

    group_names = sorted(
        patient_group_indexed["Group"]
        .astype(str)
        .unique()
        .tolist()
    )

    if not group_names:
        raise ValueError(
            "No valid patient groups were found."
        )

    group_to_index = {
        group_name: group_index
        for group_index, group_name in enumerate(
            group_names
        )
    }

    patient_group_index = {
        patient_id: group_to_index[
            str(
                patient_group_indexed.loc[
                    patient_id,
                    "Group",
                ]
            )
        ]
        for patient_id in patients
    }

    n_groups = len(
        group_names
    )

    epsilon = 1e-8
    total_valid_transitions = 0

    with pm.Model() as model:
        theta_raw = pm.Normal(
            "theta_raw",
            mu=0.0,
            sigma=1.0,
            shape=n_groups,
        )

        sigma_raw = pm.Normal(
            "sigma_raw",
            mu=0.0,
            sigma=1.0,
            shape=n_groups,
        )

        theta = pm.Deterministic(
            "theta",
            0.05
            + pm.math.log1pexp(
                theta_raw
            ),
        )

        sigma = pm.Deterministic(
            "sigma",
            0.02
            + 0.50
            * pm.math.log1pexp(
                sigma_raw
            ),
        )

        tau_x = pm.HalfNormal(
            "tau_x",
            sigma=0.30,
        )

        mu_mu = pm.Normal(
            "mu_mu",
            mu=0.0,
            sigma=1.0,
        )

        mu_sigma = pm.HalfNormal(
            "mu_sigma",
            sigma=1.0,
        )

        for patient_id in patients:
            group_index = patient_group_index[
                patient_id
            ]

            suffix = patient_variable_suffix[
                patient_id
            ]

            times, values = get_patient_x_series(
                series_df,
                patient_id,
            )

            z_mu = pm.Normal(
                f"z_mu_{suffix}",
                mu=0.0,
                sigma=1.0,
            )

            patient_mu = pm.Deterministic(
                f"mu_{suffix}",
                mu_mu
                + mu_sigma
                * z_mu,
            )

            time_differences = np.diff(
                times
            )

            valid_transition = (
                np.isfinite(
                    time_differences
                )
                & (
                    time_differences > 0
                )
            )

            delta_t = (
                time_differences[
                    valid_transition
                ]
                .astype(float)
            )

            if len(delta_t) == 0:
                continue

            previous_values = (
                values[:-1][
                    valid_transition
                ]
                .astype(float)
            )

            next_values = (
                values[1:][
                    valid_transition
                ]
                .astype(float)
            )

            if not (
                np.all(
                    np.isfinite(
                        previous_values
                    )
                )
                and np.all(
                    np.isfinite(
                        next_values
                    )
                )
            ):
                raise ValueError(
                    f"Nonfinite x values detected for patient "
                    f"{patient_id!r}."
                )

            total_valid_transitions += len(
                delta_t
            )

            delta_t_tensor = pt.as_tensor_variable(
                delta_t
            )

            previous_values_tensor = pt.as_tensor_variable(
                previous_values
            )

            decay = pt.exp(
                -theta[group_index]
                * delta_t_tensor
            )

            transition_mean = (
                patient_mu
                + (
                    previous_values_tensor
                    - patient_mu
                )
                * decay
            )

            process_variance = (
                (
                    sigma[group_index] ** 2
                )
                / (
                    2.0
                    * theta[group_index]
                )
                * (
                    1.0
                    - pt.exp(
                        -2.0
                        * theta[group_index]
                        * delta_t_tensor
                    )
                )
            )

            total_sd = pt.sqrt(
                process_variance
                + tau_x**2
                + epsilon
            )

            pm.Normal(
                f"x_transition_obs_{suffix}",
                mu=transition_mean,
                sigma=total_sd,
                observed=next_values,
            )

    if total_valid_transitions == 0:
        raise ValueError(
            "No valid positive-time OU transitions were created."
        )

    print(
        "[INFO] Valid OU transitions:",
        total_valid_transitions,
    )

    print(
        "[INFO] Number of groups:",
        n_groups,
    )

    print(
        "[INFO] Group order:",
        group_names,
    )

    print(
        "[INFO] Number of free random variables:",
        len(
            model.free_RVs
        ),
    )

    print(
        "[INFO] Number of observed variables:",
        len(
            model.observed_RVs
        ),
    )

    print(
        "[INFO] Free variables:",
        [
            variable.name
            for variable in model.free_RVs
        ],
    )

    idata = pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=min(
            chains,
            4,
        ),
        random_seed=SEED,
        target_accept=target_accept,
        init="jitter+adapt_diag",
        return_inferencedata=True,
        compute_convergence_checks=True,
        discard_tuned_samples=True,
        progressbar=True,
        model=model,
    )

    # -----------------------------------------------------------------
    # Posterior trace
    # -----------------------------------------------------------------
    trace_path = (
        out_dir
        / "ou_direct_transition_trace.nc"
    )

    az.to_netcdf(
        idata,
        trace_path,
    )

    # -----------------------------------------------------------------
    # Posterior summary
    # -----------------------------------------------------------------
    core_variables = [
        "theta_raw",
        "sigma_raw",
        "theta",
        "sigma",
        "tau_x",
        "mu_mu",
        "mu_sigma",
    ]

    z_mu_variables = sorted(
        variable_name
        for variable_name in idata.posterior.data_vars
        if variable_name.startswith(
            "z_mu_"
        )
    )

    patient_mu_variables = sorted(
        variable_name
        for variable_name in idata.posterior.data_vars
        if (
            variable_name.startswith(
                "mu_"
            )
            and variable_name
            not in {
                "mu_mu",
                "mu_sigma",
            }
        )
    )

    summary_variables = (
        core_variables
        + z_mu_variables
        + patient_mu_variables
    )

    posterior_summary = az.summary(
        idata,
        var_names=summary_variables,
        round_to=6,
    )

    posterior_summary_path = (
        out_dir
        / "ou_direct_transition_posterior_summary.csv"
    )

    posterior_summary.to_csv(
        posterior_summary_path
    )

    # -----------------------------------------------------------------
    # Group order
    # -----------------------------------------------------------------
    group_order_path = (
        out_dir
        / "ou_direct_transition_group_order.csv"
    )

    pd.DataFrame(
        {
            "group_index": np.arange(
                n_groups,
                dtype=int,
            ),
            "Group": group_names,
        }
    ).to_csv(
        group_order_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Patient-variable mapping
    # -----------------------------------------------------------------
    patient_variable_map_path = (
        out_dir
        / "ou_direct_transition_patient_variable_map.csv"
    )

    patient_variable_map = pd.DataFrame(
        {
            "Patient_ID": patients,
            "Group": [
                str(
                    patient_group_indexed.loc[
                        patient_id,
                        "Group",
                    ]
                )
                for patient_id in patients
            ],
            "group_index": [
                patient_group_index[
                    patient_id
                ]
                for patient_id in patients
            ],
            "variable_suffix": [
                patient_variable_suffix[
                    patient_id
                ]
                for patient_id in patients
            ],
            "z_mu_variable": [
                "z_mu_"
                + patient_variable_suffix[
                    patient_id
                ]
                for patient_id in patients
            ],
            "mu_variable": [
                "mu_"
                + patient_variable_suffix[
                    patient_id
                ]
                for patient_id in patients
            ],
        }
    )

    patient_variable_map.to_csv(
        patient_variable_map_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Convergence summary
    # -----------------------------------------------------------------
    convergence_rows = [
        {
            "Metric": "Maximum R-hat",
            "Value": float(
                posterior_summary[
                    "r_hat"
                ].max()
            ),
        },
        {
            "Metric": "Minimum bulk ESS",
            "Value": float(
                posterior_summary[
                    "ess_bulk"
                ].min()
            ),
        },
        {
            "Metric": "Minimum tail ESS",
            "Value": float(
                posterior_summary[
                    "ess_tail"
                ].min()
            ),
        },
    ]

    if (
        "sample_stats"
        in idata.groups()
        and "diverging"
        in idata.sample_stats
    ):
        divergence_count = int(
            idata.sample_stats[
                "diverging"
            ].sum()
        )

        convergence_rows.append(
            {
                "Metric": "Divergent transitions",
                "Value": divergence_count,
            }
        )

    if (
        "sample_stats"
        in idata.groups()
        and "reached_max_treedepth"
        in idata.sample_stats
    ):
        maximum_tree_depth_events = int(
            idata.sample_stats[
                "reached_max_treedepth"
            ].sum()
        )

        convergence_rows.append(
            {
                "Metric": "Maximum tree-depth events",
                "Value": maximum_tree_depth_events,
            }
        )

    convergence_summary = pd.DataFrame(
        convergence_rows
    )

    convergence_summary_path = (
        out_dir
        / "ou_direct_transition_convergence_summary.csv"
    )

    convergence_summary.to_csv(
        convergence_summary_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Run metadata
    # -----------------------------------------------------------------
    run_metadata_path = (
        out_dir
        / "ou_direct_transition_run_metadata.csv"
    )

    run_metadata = pd.DataFrame(
        {
            "setting": [
                "script",
                "random_seed",
                "draws",
                "tune",
                "chains",
                "target_accept",
                "x_observations",
                "x_patients_total",
                "patients_modeled",
                "groups",
                "valid_transitions",
                "x_mean",
                "x_sd",
                "x_min",
                "x_max",
            ],
            "value": [
                "06_fit_ou_direct_transition_posterior.py",
                SEED,
                draws,
                tune,
                chains,
                target_accept,
                len(
                    x_values
                ),
                all_x_patients,
                len(
                    patients
                ),
                n_groups,
                total_valid_transitions,
                float(
                    np.mean(
                        x_values
                    )
                ),
                x_sd,
                float(
                    np.min(
                        x_values
                    )
                ),
                float(
                    np.max(
                        x_values
                    )
                ),
            ],
        }
    )

    run_metadata.to_csv(
        run_metadata_path,
        index=False,
    )

    print(
        f"[OK] wrote {trace_path}"
    )

    print(
        f"[OK] wrote {posterior_summary_path}"
    )

    print(
        f"[OK] wrote {group_order_path}"
    )

    print(
        f"[OK] wrote {convergence_summary_path}"
    )

    print(
        f"[OK] wrote {patient_variable_map_path}"
    )

    print(
        f"[OK] wrote {run_metadata_path}"
    )

    print(
        f"[OK] groups (order): {group_names}"
    )

    print(
        "\nConvergence summary"
    )

    print(
        convergence_summary.to_string(
            index=False
        )
    )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the hierarchical OU direct-transition posterior "
            "used to regenerate Figure 3."
        )
    )

    parser.add_argument(
        "--series_csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--patient_group_csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--draws",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--tune",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--chains",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--target_accept",
        type=float,
        default=0.95,
    )

    args = parser.parse_args()

    if args.draws < 1:
        raise ValueError(
            "--draws must be at least 1."
        )

    if args.tune < 0:
        raise ValueError(
            "--tune cannot be negative."
        )

    if args.chains < 1:
        raise ValueError(
            "--chains must be at least 1."
        )

    if not (
        0.0
        < args.target_accept
        < 1.0
    ):
        raise ValueError(
            "--target_accept must be between 0 and 1."
        )

    if not args.series_csv.exists():
        raise FileNotFoundError(
            args.series_csv
        )

    if not args.patient_group_csv.exists():
        raise FileNotFoundError(
            args.patient_group_csv
        )

    series_df = read_series(
        args.series_csv
    )

    patient_group_df = read_patient_groups(
        args.patient_group_csv
    )

    fit_and_write(
        series_df=series_df,
        patient_group_df=patient_group_df,
        out_dir=args.out_dir,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
    )


if __name__ == "__main__":
    main()
