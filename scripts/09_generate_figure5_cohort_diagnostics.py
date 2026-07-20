from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SEED = 2026

PROJECT_ROOT = (
    Path.home()
    /scripts/ "09_generate_figure5_cohort_diagnostics.py"
)

POSTERIOR_DIR = (
    PROJECT_ROOT
    / "Figure_3"
    / "out_ou_direct_transition"
)

DEFAULT_SERIES = PROJECT_ROOT / "series_auto.csv"

DEFAULT_TRACE = (
    POSTERIOR_DIR
    / "ou_direct_transition_trace.nc"
)

DEFAULT_GROUP_ORDER = (
    POSTERIOR_DIR
    / "ou_direct_transition_group_order.csv"
)

DEFAULT_PATIENT_MAP = (
    POSTERIOR_DIR
    / "ou_direct_transition_patient_variable_map.csv"
)

DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "Figure_5"
    / "out_ou_direct_transition_diagnostics"
)


def read_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"Patient_ID", "series", "t", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Series file is missing columns: {sorted(missing)}"
        )

    df["Patient_ID"] = df["Patient_ID"].astype("string").str.strip()
    df["series"] = df["series"].astype("string").str.strip().str.lower()
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = (
        df.loc[df["series"].isin(["x", "n"])]
        .dropna(subset=["Patient_ID", "series", "t", "value"])
        .copy()
    )

    duplicate_counts = (
        df.groupby(["Patient_ID", "series", "t"]).size()
    )
    duplicate_groups = int((duplicate_counts > 1).sum())

    if duplicate_groups:
        print(
            "[INFO] Collapsing "
            f"{duplicate_groups} duplicate "
            "Patient_ID-series-time combinations by mean."
        )

    df = (
        df.groupby(
            ["Patient_ID", "series", "t"],
            as_index=False,
        )
        .agg(value=("value", "mean"))
        .sort_values(
            ["Patient_ID", "series", "t"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return df


def load_group_order(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"group_index", "Group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Group-order file is missing columns: {sorted(missing)}"
        )

    df = df.sort_values("group_index").reset_index(drop=True)
    expected = np.arange(len(df))
    observed = df["group_index"].to_numpy(dtype=int)

    if not np.array_equal(observed, expected):
        raise ValueError(
            "group_index must be consecutive and start at zero."
        )

    return df


def load_patient_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "Patient_ID",
        "Group",
        "group_index",
        "mu_variable",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Patient map is missing columns: {sorted(missing)}"
        )

    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()
    return df


def posterior_vector(
    idata: az.InferenceData,
    variable: str,
) -> np.ndarray:
    if variable not in idata.posterior:
        raise KeyError(
            f"Posterior variable {variable!r} was not found."
        )

    return np.asarray(
        idata.posterior[variable].values,
        dtype=float,
    ).reshape(-1)


def posterior_group_matrix(
    idata: az.InferenceData,
    variable: str,
    n_groups: int,
) -> np.ndarray:
    if variable not in idata.posterior:
        raise KeyError(
            f"Posterior variable {variable!r} was not found."
        )

    values = np.asarray(
        idata.posterior[variable].values,
        dtype=float,
    )

    if values.shape[-1] != n_groups:
        raise ValueError(
            f"{variable} has {values.shape[-1]} groups; "
            f"expected {n_groups}."
        )

    return values.reshape(-1, n_groups)


def deterministic_mean_paths(
    times: np.ndarray,
    initial_value: float,
    mu_samples: np.ndarray,
    theta_samples: np.ndarray,
) -> np.ndarray:
    posterior_n = min(
        len(mu_samples),
        len(theta_samples),
    )

    paths = np.empty(
        (posterior_n, len(times)),
        dtype=float,
    )
    paths[:, 0] = float(initial_value)

    for draw in range(posterior_n):
        current = float(initial_value)
        mu = float(mu_samples[draw])
        theta = float(theta_samples[draw])

        for i in range(1, len(times)):
            dt = float(times[i] - times[i - 1])
            current = (
                mu
                + (current - mu)
                * np.exp(-theta * dt)
            )
            paths[draw, i] = current

    return paths


def simulate_ppc_paths(
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

    posterior_n = min(
        len(mu_samples),
        len(theta_samples),
        len(sigma_samples),
        len(tau_samples),
    )

    indices = rng.integers(
        0,
        posterior_n,
        size=n_paths,
    )

    paths = np.empty(
        (n_paths, len(times)),
        dtype=float,
    )
    paths[:, 0] = float(initial_value)

    for path_index, posterior_index in enumerate(indices):
        mu = float(mu_samples[posterior_index])
        theta = float(theta_samples[posterior_index])
        sigma = float(sigma_samples[posterior_index])
        tau_x = float(tau_samples[posterior_index])

        current = float(initial_value)

        for i in range(1, len(times)):
            dt = float(times[i] - times[i - 1])
            decay = np.exp(-theta * dt)

            transition_mean = (
                mu
                + (current - mu) * decay
            )

            process_variance = (
                sigma**2
                / (2.0 * theta)
                * (
                    1.0
                    - np.exp(-2.0 * theta * dt)
                )
            )

            total_variance = (
                process_variance
                + tau_x**2
            )

            current = rng.normal(
                transition_mean,
                np.sqrt(max(total_variance, 1e-12)),
            )

            paths[path_index, i] = current

    return paths


def mean_transition_loglikelihood(
    times: np.ndarray,
    values: np.ndarray,
    mu_samples: np.ndarray,
    theta_samples: np.ndarray,
    sigma_samples: np.ndarray,
    tau_samples: np.ndarray,
) -> float:
    """
    Mean posterior expected log-likelihood per observed transition.
    """
    posterior_n = min(
        len(mu_samples),
        len(theta_samples),
        len(sigma_samples),
        len(tau_samples),
    )

    loglik_draws = np.empty(
        posterior_n,
        dtype=float,
    )

    for draw in range(posterior_n):
        mu = float(mu_samples[draw])
        theta = float(theta_samples[draw])
        sigma = float(sigma_samples[draw])
        tau_x = float(tau_samples[draw])

        contributions = []

        for i in range(1, len(times)):
            dt = float(times[i] - times[i - 1])
            x_prev = float(values[i - 1])
            x_next = float(values[i])

            decay = np.exp(-theta * dt)

            transition_mean = (
                mu
                + (x_prev - mu) * decay
            )

            process_variance = (
                sigma**2
                / (2.0 * theta)
                * (
                    1.0
                    - np.exp(-2.0 * theta * dt)
                )
            )

            total_variance = (
                process_variance
                + tau_x**2
                + 1e-12
            )

            contribution = (
                -0.5 * np.log(2.0 * np.pi * total_variance)
                -0.5
                * (x_next - transition_mean) ** 2
                / total_variance
            )

            contributions.append(contribution)

        loglik_draws[draw] = float(
            np.mean(contributions)
        )

    return float(
        np.mean(loglik_draws)
    )


def patient_diagnostic_row(
    patient_id: str,
    series_df: pd.DataFrame,
    patient_record: pd.Series,
    idata: az.InferenceData,
    theta_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    tau_samples: np.ndarray,
    n_ppc: int,
    credible_mass: float,
    seed: int,
) -> dict:
    x_df = (
        series_df.loc[
            (
                series_df["Patient_ID"].astype(str)
                == patient_id
            )
            & (
                series_df["series"] == "x"
            )
        ]
        .sort_values("t")
    )

    times = x_df["t"].to_numpy(dtype=float)
    values = x_df["value"].to_numpy(dtype=float)

    if len(times) < 2:
        raise ValueError(
            f"Patient {patient_id} has fewer than two x observations."
        )

    if np.any(np.diff(times) <= 0):
        raise ValueError(
            f"Patient {patient_id} has nonpositive time differences."
        )

    group_index = int(
        patient_record["group_index"]
    )

    group_name = str(
        patient_record["Group"]
    )

    mu_variable = str(
        patient_record["mu_variable"]
    )

    mu_samples = posterior_vector(
        idata,
        mu_variable,
    )

    theta_samples = theta_matrix[
        :,
        group_index,
    ]

    sigma_samples = sigma_matrix[
        :,
        group_index,
    ]

    mean_paths = deterministic_mean_paths(
        times=times,
        initial_value=float(values[0]),
        mu_samples=mu_samples,
        theta_samples=theta_samples,
    )

    posterior_mean_path = mean_paths.mean(axis=0)

    residuals = (
        values
        - posterior_mean_path
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residuals**2
            )
        )
    )

    sse = float(
        np.sum(
            residuals**2
        )
    )

    sst = float(
        np.sum(
            (
                values
                - np.mean(values)
            ) ** 2
        )
    )

    predictive_r2 = (
        1.0 - sse / sst
        if sst > 0.0
        else np.nan
    )

    ppc_paths = simulate_ppc_paths(
        times=times,
        initial_value=float(values[0]),
        mu_samples=mu_samples,
        theta_samples=theta_samples,
        sigma_samples=sigma_samples,
        tau_samples=tau_samples,
        n_paths=n_ppc,
        seed=seed,
    )

    alpha = (
        1.0 - credible_mass
    ) / 2.0

    lower = np.percentile(
        ppc_paths,
        100.0 * alpha,
        axis=0,
    )

    upper = np.percentile(
        ppc_paths,
        100.0 * (1.0 - alpha),
        axis=0,
    )

    coverage = float(
        np.mean(
            (
                values >= lower
            )
            & (
                values <= upper
            )
        )
    )

    interval_width = float(
        np.mean(
            upper - lower
        )
    )

    mean_loglik = mean_transition_loglikelihood(
        times=times,
        values=values,
        mu_samples=mu_samples,
        theta_samples=theta_samples,
        sigma_samples=sigma_samples,
        tau_samples=tau_samples,
    )

    return {
        "Patient_ID": patient_id,
        "Group": group_name,
        "group_index": group_index,
        "n_observations": int(len(times)),
        "n_transitions": int(len(times) - 1),
        "time_span_years": float(
            times[-1] - times[0]
        ),
        "rmse": rmse,
        "ppc_95": coverage,
        "ppc_95_percent": 100.0 * coverage,
        "mean_ppc_interval_width": interval_width,
        "predictive_r2": predictive_r2,
        "mean_loglik_per_transition": mean_loglik,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)

    means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        means[i] = np.mean(sample)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def group_summary_table(
    patient_table: pd.DataFrame,
    group_order: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    metrics = [
        "rmse",
        "ppc_95",
        "ppc_95_percent",
        "mean_ppc_interval_width",
        "predictive_r2",
        "mean_loglik_per_transition",
    ]

    rows = []

    for group_index, group_name in enumerate(group_order):
        group_df = patient_table.loc[
            patient_table["Group"] == group_name
        ]

        for metric_index, metric in enumerate(metrics):
            values = pd.to_numeric(
                group_df[metric],
                errors="coerce",
            )
            finite = values[
                np.isfinite(values)
            ].to_numpy(dtype=float)

            lower, upper = bootstrap_mean_ci(
                finite,
                n_bootstrap=n_bootstrap,
                seed=seed
                + 100 * group_index
                + metric_index,
            )

            rows.append(
                {
                    "group_index": group_index,
                    "Group": group_name,
                    "metric": metric,
                    "n_patients_total": int(len(group_df)),
                    "n_patients_evaluable": int(len(finite)),
                    "mean": (
                        float(np.mean(finite))
                        if len(finite)
                        else np.nan
                    ),
                    "sd": (
                        float(np.std(finite, ddof=1))
                        if len(finite) > 1
                        else np.nan
                    ),
                    "median": (
                        float(np.median(finite))
                        if len(finite)
                        else np.nan
                    ),
                    "minimum": (
                        float(np.min(finite))
                        if len(finite)
                        else np.nan
                    ),
                    "maximum": (
                        float(np.max(finite))
                        if len(finite)
                        else np.nan
                    ),
                    "bootstrap_mean_ci_lower": lower,
                    "bootstrap_mean_ci_upper": upper,
                }
            )

    return pd.DataFrame(rows)


def spearman_table(
    patient_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate pairwise Spearman correlations among patient-level diagnostics.

    Diagonal comparisons are assigned rho=1 and p=0 explicitly.
    This avoids duplicate-column selection when a metric is compared
    with itself.
    """
    metrics = [
        "rmse",
        "ppc_95",
        "predictive_r2",
        "mean_loglik_per_transition",
    ]

    correlation = pd.DataFrame(
        np.nan,
        index=metrics,
        columns=metrics,
        dtype=float,
    )

    pvalue = pd.DataFrame(
        np.nan,
        index=metrics,
        columns=metrics,
        dtype=float,
    )

    n_paired = pd.DataFrame(
        0,
        index=metrics,
        columns=metrics,
        dtype=int,
    )

    for row_metric in metrics:
        for column_metric in metrics:

            x = (
                pd.to_numeric(
                    patient_table[row_metric],
                    errors="coerce",
                )
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
            )

            y = (
                pd.to_numeric(
                    patient_table[column_metric],
                    errors="coerce",
                )
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
            )

            valid = (
                x.notna()
                & y.notna()
            )

            x_use = x.loc[
                valid
            ].to_numpy(
                dtype=float
            )

            y_use = y.loc[
                valid
            ].to_numpy(
                dtype=float
            )

            paired_count = len(
                x_use
            )

            n_paired.loc[
                row_metric,
                column_metric,
            ] = paired_count

            if row_metric == column_metric:
                if paired_count >= 2:
                    rho = 1.0
                    p = 0.0
                else:
                    rho = np.nan
                    p = np.nan

            elif paired_count < 3:
                rho = np.nan
                p = np.nan

            elif (
                np.unique(
                    x_use
                ).size < 2
                or np.unique(
                    y_use
                ).size < 2
            ):
                rho = np.nan
                p = np.nan

            else:
                result = spearmanr(
                    x_use,
                    y_use,
                )

                rho = float(
                    result.statistic
                )

                p = float(
                    result.pvalue
                )

            correlation.loc[
                row_metric,
                column_metric,
            ] = (
                float(rho)
                if np.isfinite(rho)
                else np.nan
            )

            pvalue.loc[
                row_metric,
                column_metric,
            ] = (
                float(p)
                if np.isfinite(p)
                else np.nan
            )

    long_rows = []

    for row_metric in metrics:
        for column_metric in metrics:
            long_rows.append(
                {
                    "metric_1": row_metric,
                    "metric_2": column_metric,
                    "n_paired": int(
                        n_paired.loc[
                            row_metric,
                            column_metric,
                        ]
                    ),
                    "spearman_rho": correlation.loc[
                        row_metric,
                        column_metric,
                    ],
                    "p_value": pvalue.loc[
                        row_metric,
                        column_metric,
                    ],
                }
            )

    correlation_table = pd.DataFrame(
        long_rows
    )

    return (
        correlation,
        correlation_table,
    )


def draw_group_violin(
    axis: plt.Axes,
    patient_table: pd.DataFrame,
    group_order: list[str],
    metric: str,
    ylabel: str,
    title: str,
    horizontal_reference: float | None = None,
) -> None:
    data = []

    for group_name in group_order:
        values = pd.to_numeric(
            patient_table.loc[
                patient_table["Group"] == group_name,
                metric,
            ],
            errors="coerce",
        )
        values = values[
            np.isfinite(values)
        ].to_numpy(dtype=float)

        # matplotlib violinplot requires at least two values.
        if len(values) == 1:
            plotting_values = np.array(
                [
                    values[0] - 1e-8,
                    values[0] + 1e-8,
                ]
            )
        elif len(values) == 0:
            plotting_values = np.array(
                [
                    np.nan,
                    np.nan,
                ]
            )
        else:
            plotting_values = values

        data.append(
            plotting_values
        )

    positions = np.arange(
        1,
        len(group_order) + 1,
    )

    valid_indices = [
        i
        for i, values in enumerate(data)
        if np.all(np.isfinite(values))
    ]

    if valid_indices:
        valid_data = [
            data[i]
            for i in valid_indices
        ]
        valid_positions = [
            positions[i]
            for i in valid_indices
        ]

        violin = axis.violinplot(
            valid_data,
            positions=valid_positions,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in violin["bodies"]:
            body.set_facecolor("0.82")
            body.set_edgecolor("0.35")
            body.set_alpha(0.9)

    for position, group_name in zip(
        positions,
        group_order,
    ):
        values = pd.to_numeric(
            patient_table.loc[
                patient_table["Group"] == group_name,
                metric,
            ],
            errors="coerce",
        )
        values = values[
            np.isfinite(values)
        ].to_numpy(dtype=float)

        if len(values) == 0:
            continue

        jitter = np.linspace(
            -0.06,
            0.06,
            len(values),
        )

        axis.scatter(
            position + jitter,
            values,
            s=14,
            color="black",
            alpha=0.65,
            zorder=4,
        )

        if len(values) >= 2:
            quartiles = np.percentile(
                values,
                [
                    25,
                    50,
                    75,
                ],
            )

            axis.plot(
                [
                    position - 0.12,
                    position + 0.12,
                ],
                [
                    quartiles[1],
                    quartiles[1],
                ],
                color="black",
                linewidth=1.4,
                zorder=5,
            )

            axis.plot(
                [
                    position,
                    position,
                ],
                [
                    quartiles[0],
                    quartiles[2],
                ],
                color="black",
                linewidth=2.5,
                zorder=5,
            )

    if horizontal_reference is not None:
        axis.axhline(
            horizontal_reference,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        group_order,
        rotation=38,
        ha="right",
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_xlabel(
        "Clinical group"
    )

    axis.set_title(
        title,
        loc="left",
        fontweight="bold",
    )


def generate_figure(
    patient_table: pd.DataFrame,
    group_order: list[str],
    correlation_matrix: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
) -> None:
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

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(8.2, 7.0),
        constrained_layout=True,
    )

    draw_group_violin(
        axis=axes[0, 0],
        patient_table=patient_table,
        group_order=group_order,
        metric="rmse",
        ylabel="RMSE",
        title="A  RMSE by clinical group",
    )

    draw_group_violin(
        axis=axes[0, 1],
        patient_table=patient_table,
        group_order=group_order,
        metric="ppc_95",
        ylabel="Coverage of 95% predictive band",
        title="B  Posterior predictive coverage",
        horizontal_reference=0.95,
    )

    predictive_r2 = pd.to_numeric(
        patient_table["predictive_r2"],
        errors="coerce",
    )

    predictive_r2 = predictive_r2[
        np.isfinite(predictive_r2)
    ].to_numpy(dtype=float)

    axes[1, 0].hist(
        predictive_r2,
        bins=min(
            15,
            max(
                5,
                int(
                    np.sqrt(
                        len(predictive_r2)
                    )
                )
                + 2,
            ),
        ),
        facecolor="0.78",
        edgecolor="0.25",
        linewidth=0.8,
    )

    axes[1, 0].axvline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=1.2,
    )

    axes[1, 0].set_xlabel(
        r"Predictive $R^2$"
    )

    axes[1, 0].set_ylabel(
        "Number of patients"
    )

    axes[1, 0].set_title(
        r"C  Patient-level predictive $R^2$",
        loc="left",
        fontweight="bold",
    )

    matrix = correlation_matrix.to_numpy(
        dtype=float
    )

    image = axes[1, 1].imshow(
        matrix,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm_r",
        aspect="auto",
    )

    metric_labels = {
        "rmse": "RMSE",
        "ppc_95": "PPC coverage",
        "predictive_r2": r"Predictive $R^2$",
        "mean_loglik_per_transition": "Mean log likelihood",
    }

    labels = [
        metric_labels.get(
            value,
            value,
        )
        for value in correlation_matrix.columns
    ]

    axes[1, 1].set_xticks(
        np.arange(
            len(labels)
        )
    )

    axes[1, 1].set_yticks(
        np.arange(
            len(labels)
        )
    )

    axes[1, 1].set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    axes[1, 1].set_yticklabels(
        labels,
    )

    axes[1, 1].set_title(
        "D  Spearman correlation among diagnostics",
        loc="left",
        fontweight="bold",
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            if np.isfinite(
                value
            ):
                text_color = (
                    "white"
                    if abs(
                        value
                    )
                    >= 0.55
                    else "black"
                )

                axes[1, 1].text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )

    colorbar = figure.colorbar(
        image,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        r"Spearman $\rho$"
    )

    figure.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight",
    )

    figure.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def write_workbook(
    patient_table: pd.DataFrame,
    group_table: pd.DataFrame,
    correlation_table: pd.DataFrame,
    output_path: Path,
) -> None:
    notes = pd.DataFrame(
        {
            "Item": [
                "RMSE",
                "Posterior predictive coverage",
                "Predictive R-squared",
                "Mean log likelihood",
                "Group confidence intervals",
                "Interpretation of negative predictive R-squared",
            ],
            "Definition": [
                (
                    "Root mean squared error between observed x values and "
                    "the posterior-mean deterministic OU trajectory."
                ),
                (
                    "Fraction of observed x values within the patient-level "
                    "95% posterior predictive band."
                ),
                (
                    "1 - SSE/SST, using the posterior-mean deterministic "
                    "trajectory. This is a prediction diagnostic, not a "
                    "posterior convergence statistic."
                ),
                (
                    "Posterior expected Gaussian transition log-likelihood, "
                    "averaged across observed transitions."
                ),
                (
                    "Table S11 reports nonparametric bootstrap 95% confidence "
                    "intervals for group mean diagnostic values."
                ),
                (
                    "Negative values indicate that the posterior-mean trajectory "
                    "predicts that patient's observations worse than the observed "
                    "patient-specific mean. They do not invalidate parameter "
                    "inference or MCMC convergence."
                ),
            ],
        }
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        patient_table.to_excel(
            writer,
            sheet_name="Table_S10_patient",
            index=False,
        )

        group_table.to_excel(
            writer,
            sheet_name="Table_S11_group",
            index=False,
        )

        correlation_table.to_excel(
            writer,
            sheet_name="S12_correlations",
            index=False,
        )

        notes.to_excel(
            writer,
            sheet_name="Notes",
            index=False,
        )

        for sheet_name in writer.book.sheetnames:
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                maximum = max(
                    len(
                        str(
                            cell.value
                            if cell.value is not None
                            else ""
                        )
                    )
                    for cell in column_cells
                )

                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = min(
                    max(
                        maximum + 2,
                        10,
                    ),
                    55,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate revised Figure 5 and Supplementary Tables S10-12 "
            "from the hierarchical OU direct-transition posterior."
        )
    )

    parser.add_argument(
        "--series_csv",
        type=Path,
        default=DEFAULT_SERIES,
    )

    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_TRACE,
    )

    parser.add_argument(
        "--group_order",
        type=Path,
        default=DEFAULT_GROUP_ORDER,
    )

    parser.add_argument(
        "--patient_map",
        type=Path,
        default=DEFAULT_PATIENT_MAP,
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    parser.add_argument(
        "--n_ppc",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--credible_mass",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    for path in (
        args.series_csv,
        args.trace,
        args.group_order,
        args.patient_map,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.n_ppc < 100:
        raise ValueError(
            "--n_ppc must be at least 100."
        )

    if args.n_bootstrap < 100:
        raise ValueError(
            "--n_bootstrap must be at least 100."
        )

    if not 0.0 < args.credible_mass < 1.0:
        raise ValueError(
            "--credible_mass must be between 0 and 1."
        )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    series_df = read_series(
        args.series_csv
    )

    group_df = load_group_order(
        args.group_order
    )

    patient_map = load_patient_map(
        args.patient_map
    )

    idata = az.from_netcdf(
        args.trace
    )

    group_order = (
        group_df["Group"]
        .astype(str)
        .tolist()
    )

    n_groups = len(group_order)

    theta_matrix = posterior_group_matrix(
        idata,
        "theta",
        n_groups,
    )

    sigma_matrix = posterior_group_matrix(
        idata,
        "sigma",
        n_groups,
    )

    tau_samples = posterior_vector(
        idata,
        "tau_x",
    )

    mapped_patients = set(
        patient_map["Patient_ID"].astype(str)
    )

    series_patients = set(
        series_df.loc[
            series_df["series"] == "x",
            "Patient_ID",
        ].astype(str)
    )

    eligible_patients = sorted(
        mapped_patients & series_patients
    )

    rows = []

    print(
        f"[INFO] Calculating diagnostics for "
        f"{len(eligible_patients)} modeled patients."
    )

    for patient_index, patient_id in enumerate(
        eligible_patients
    ):
        patient_record = (
            patient_map.loc[
                patient_map["Patient_ID"].astype(str)
                == patient_id
            ]
            .iloc[0]
        )

        row = patient_diagnostic_row(
            patient_id=patient_id,
            series_df=series_df,
            patient_record=patient_record,
            idata=idata,
            theta_matrix=theta_matrix,
            sigma_matrix=sigma_matrix,
            tau_samples=tau_samples,
            n_ppc=args.n_ppc,
            credible_mass=args.credible_mass,
            seed=args.seed + patient_index,
        )

        rows.append(row)

        print(
            f"[OK] {patient_id}: "
            f"RMSE={row['rmse']:.4f}; "
            f"PPC={row['ppc_95_percent']:.1f}%; "
            f"predictive R2={row['predictive_r2']:.4f}"
        )

    patient_table = pd.DataFrame(rows)

    patient_table["Group"] = pd.Categorical(
        patient_table["Group"],
        categories=group_order,
        ordered=True,
    )

    patient_table = (
        patient_table.sort_values(
            ["Group", "Patient_ID"]
        )
        .reset_index(drop=True)
    )

    group_table = group_summary_table(
        patient_table=patient_table,
        group_order=group_order,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 10000,
    )

    correlation_matrix, correlation_table = spearman_table(
        patient_table
    )

    for table in (
        patient_table,
        group_table,
        correlation_table,
    ):
        numeric_columns = table.select_dtypes(
            include=[
                "float16",
                "float32",
                "float64",
            ]
        ).columns

        table[numeric_columns] = table[
            numeric_columns
        ].round(args.digits)

    patient_path = (
        args.out_dir
        / "Table_S10_patient_level_model_diagnostics.csv"
    )

    group_path = (
        args.out_dir
        / "Table_S11_group_level_model_diagnostics.csv"
    )

    correlation_path = (
        args.out_dir
        / "Table_S12_diagnostic_spearman_correlations.csv"
    )

    workbook_path = (
        args.out_dir
        / "Table_S10-S12_model_diagnostics.xlsx"
    )

    patient_table.to_csv(
        patient_path,
        index=False,
    )

    group_table.to_csv(
        group_path,
        index=False,
    )

    correlation_table.to_csv(
        correlation_path,
        index=False,
    )

    write_workbook(
        patient_table=patient_table,
        group_table=group_table,
        correlation_table=correlation_table,
        output_path=workbook_path,
    )

    output_png = (
        args.out_dir
        / "Figure5_model_diagnostics.png"
    )

    output_pdf = (
        args.out_dir
        / "Figure5_model_diagnostics.pdf"
    )

    generate_figure(
        patient_table=patient_table,
        group_order=group_order,
        correlation_matrix=correlation_matrix,
        output_png=output_png,
        output_pdf=output_pdf,
    )

    print("[OK] Saved Figure 5:")
    print(f"     {output_png}")
    print(f"     {output_pdf}")

    print("[OK] Saved Table S10:")
    print(f"     {patient_path}")

    print("[OK] Saved Table S11:")
    print(f"     {group_path}")
    print(f"     {correlation_path}")

    print("[OK] Saved combined workbook:")
    print(f"     {workbook_path}")


if __name__ == "__main__":
    main()
