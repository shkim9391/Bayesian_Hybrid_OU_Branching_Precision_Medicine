from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.optimize import minimize

SEED = 13
DAYS_PER_YEAR = 365.25

PROJECT_ROOT = Path(
    "/scripts/ "03_simulate_ou_calibration_benchmark.py"
)
EXCEL_PATH = PROJECT_ROOT / "kmt2a_longitudinal_data.xlsx"
EXCEL_SHEET = "Samples"
OUT_DIR = PROJECT_ROOT / "Figure_2" / "out_simulation_benchmark"

TABLE_SCENARIOS = OUT_DIR / "Table_S2_simulation_scenarios.csv"
TABLE_ESTIMATES = OUT_DIR / "Table_S2_simulation_estimates.csv"
TABLE_PERFORMANCE = OUT_DIR / "Table_S2_simulation_performance.csv"
TABLE_RELIABILITY = OUT_DIR / "Table_S2_simulation_reliability.csv"
CHECKPOINT = OUT_DIR / "simulation_checkpoint.csv"

METHOD_ORDER = ["Bayesian", "MLE", "MoM", "MoM (imputed)"]
PARAMETERS = ["mu", "theta", "sigma"]

DEFAULT_SCENARIOS = [
    {"scenario": "S1", "mu_true": 2.0, "theta_true": 0.25, "sigma_true": 0.25},
    {"scenario": "S2", "mu_true": 2.0, "theta_true": 1.00, "sigma_true": 0.50},
    {"scenario": "S3", "mu_true": 3.5, "theta_true": 0.50, "sigma_true": 0.75},
    {"scenario": "S4", "mu_true": 3.5, "theta_true": 2.00, "sigma_true": 0.50},
    {"scenario": "S5", "mu_true": 5.0, "theta_true": 1.00, "sigma_true": 0.25},
    {"scenario": "S6", "mu_true": 5.0, "theta_true": 3.00, "sigma_true": 1.00},
]


def finite_parameters(mu, theta, sigma) -> bool:
    return bool(np.isfinite(mu) and np.isfinite(theta) and np.isfinite(sigma))


def empty_diagnostics() -> dict:
    return {
        "failure_reason": "",
        "boundary_mu_lower": False,
        "boundary_mu_upper": False,
        "boundary_theta_lower": False,
        "boundary_theta_upper": False,
        "boundary_sigma_lower": False,
        "boundary_sigma_upper": False,
        "max_rhat": np.nan,
        "min_ess_bulk": np.nan,
        "min_ess_tail": np.nan,
        "divergences": np.nan,
        "max_treedepth_events": np.nan,
        "mu_lower": np.nan,
        "mu_upper": np.nan,
        "theta_lower": np.nan,
        "theta_upper": np.nan,
        "sigma_lower": np.nan,
        "sigma_upper": np.nan,
        "imputation_source": "",
    }


def make_result_row(
    simulation_id,
    scenario,
    replicate,
    schedule_patient_id,
    method,
    mu_true,
    theta_true,
    sigma_true,
    mu,
    theta,
    sigma,
    success,
    n_timepoints,
    **diagnostics,
):
    row = {
        "simulation_id": str(simulation_id),
        "scenario": str(scenario),
        "replicate": int(replicate),
        "schedule_patient_id": str(schedule_patient_id),
        "method": str(method),
        "mu_true": float(mu_true),
        "theta_true": float(theta_true),
        "sigma_true": float(sigma_true),
        "mu": float(mu) if np.isfinite(mu) else np.nan,
        "theta": float(theta) if np.isfinite(theta) else np.nan,
        "sigma": float(sigma) if np.isfinite(sigma) else np.nan,
        "finite_estimate": finite_parameters(mu, theta, sigma),
        "success": bool(success),
        "n_timepoints": int(n_timepoints),
        "n_transitions": int(max(n_timepoints - 1, 0)),
    }
    row.update(empty_diagnostics())
    row.update(diagnostics)
    return row


def load_empirical_schedules(xlsx_path: Path) -> dict[str, np.ndarray]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Metadata workbook not found:\n{xlsx_path}")

    df = pd.read_excel(xlsx_path, sheet_name=EXCEL_SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    lookup = {c.lower(): c for c in df.columns}
    missing = [c for c in ["patient", "day"] if c not in lookup]
    if missing:
        raise ValueError(
            f"Samples sheet must contain Patient and Day. "
            f"Missing: {missing}. Found: {df.columns.tolist()}"
        )

    d = df.rename(
        columns={lookup["patient"]: "Patient_ID", lookup["day"]: "Day"}
    ).copy()
    d["Patient_ID"] = d["Patient_ID"].astype("string").str.strip()
    d["Day"] = pd.to_numeric(d["Day"], errors="coerce")
    d = d.dropna(subset=["Patient_ID", "Day"]).copy()
    d["t"] = d["Day"].astype(float) / DAYS_PER_YEAR

    schedules = {}
    for patient_id, group in d.groupby("Patient_ID", sort=True):
        times = np.sort(group["t"].dropna().unique().astype(float))
        if len(times) >= 2:
            schedules[str(patient_id)] = times

    if not schedules:
        raise ValueError("No schedules with at least two unique timepoints were found.")
    return schedules


def ou_transition_moments(x0, mu, theta, sigma, dt):
    decay = np.exp(-theta * dt)
    mean = mu + (x0 - mu) * decay
    variance = (sigma**2) / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * dt))
    return mean, variance


def simulate_ou_trajectory(times, mu, theta, sigma, rng):
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing and contain at least two values.")
    if theta <= 0 or sigma <= 0:
        raise ValueError("theta and sigma must be positive.")

    values = np.empty(len(times), dtype=float)
    values[0] = rng.normal(mu, sigma / np.sqrt(2.0 * theta))
    for i in range(1, len(times)):
        dt = float(times[i] - times[i - 1])
        mean, variance = ou_transition_moments(values[i - 1], mu, theta, sigma, dt)
        values[i] = rng.normal(mean, np.sqrt(max(variance, 1e-12)))
    return values


def ou_mle_fit(t, x):
    dt = np.diff(t)
    x_prev = x[:-1]
    x_next = x[1:]

    def nll(params):
        mu, theta, sigma = params
        if theta <= 0 or sigma <= 0:
            return np.inf
        mean, variance = ou_transition_moments(x_prev, mu, theta, sigma, dt)
        if (
            np.any(~np.isfinite(mean))
            or np.any(~np.isfinite(variance))
            or np.any(variance <= 0)
        ):
            return np.inf
        return float(
            0.5
            * np.sum(
                np.log(2.0 * np.pi * variance)
                + ((x_next - mean) ** 2) / variance
            )
        )

    return minimize(
        nll,
        x0=[float(np.mean(x)), 0.5, max(float(np.std(x)), 0.1)],
        bounds=[(-10.0, 10.0), (1e-6, 10.0), (1e-6, 10.0)],
        method="L-BFGS-B",
    )


def ou_mom_estimator(t, x):
    """Simple diagnostic baseline, not a full irregular-time estimator."""
    if len(t) < 3:
        raise ValueError("MoM requires at least three timepoints.")
    dx = np.diff(x)
    variance_x = np.var(x)
    if variance_x <= 1e-12:
        raise ValueError("MoM is not identifiable because variance is negligible.")
    theta = -np.cov(x[:-1], dx)[0, 1] / variance_x
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("MoM produced nonpositive or nonfinite theta.")
    mu = float(np.mean(x))
    residual = dx - (-theta) * (x[:-1] - mu)
    sigma = float(np.std(residual, ddof=1) / np.sqrt(2.0))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("MoM produced nonpositive or nonfinite sigma.")
    return mu, float(theta), sigma


def impute_points_from_mle(t_obs, x_obs, mu, theta, sigma, t_new):
    values = []
    for new_time in t_new:
        previous_index = np.where(t_obs <= new_time)[0].max()
        x0 = float(x_obs[previous_index])
        t0 = float(t_obs[previous_index])
        mean, _ = ou_transition_moments(x0, mu, theta, sigma, float(new_time - t0))
        values.append(float(mean))
    return np.asarray(values, dtype=float)


def run_mom_on_augmented(t_obs, x_obs, t_new, x_new):
    t_all = np.concatenate([t_obs, t_new])
    x_all = np.concatenate([x_obs, x_new])
    order = np.argsort(t_all)
    return ou_mom_estimator(t_all[order], x_all[order])


def fit_simulated_series(
    simulation_id,
    scenario,
    replicate,
    schedule_patient_id,
    times,
    values,
    mu_true,
    theta_true,
    sigma_true,
    draws,
    tune,
    chains,
    cores,
    target_accept,
    min_ess,
    seed,
):
    t = np.asarray(times, dtype=float)
    x = np.asarray(values, dtype=float)
    n_timepoints = len(t)
    rows = []

    # MLE
    mle_result = ou_mle_fit(t, x)
    if mle_result.success and np.all(np.isfinite(mle_result.x)):
        mu_mle, theta_mle, sigma_mle = mle_result.x
    else:
        mu_mle = theta_mle = sigma_mle = np.nan

    mle_success = bool(
        mle_result.success and finite_parameters(mu_mle, theta_mle, sigma_mle)
    )
    boundaries = {
        "boundary_mu_lower": bool(np.isfinite(mu_mle) and np.isclose(mu_mle, -10.0, atol=1e-6)),
        "boundary_mu_upper": bool(np.isfinite(mu_mle) and np.isclose(mu_mle, 10.0, atol=1e-6)),
        "boundary_theta_lower": bool(np.isfinite(theta_mle) and np.isclose(theta_mle, 1e-6, atol=1e-8)),
        "boundary_theta_upper": bool(np.isfinite(theta_mle) and np.isclose(theta_mle, 10.0, atol=1e-6)),
        "boundary_sigma_lower": bool(np.isfinite(sigma_mle) and np.isclose(sigma_mle, 1e-6, atol=1e-8)),
        "boundary_sigma_upper": bool(np.isfinite(sigma_mle) and np.isclose(sigma_mle, 10.0, atol=1e-6)),
    }
    rows.append(
        make_result_row(
            simulation_id, scenario, replicate, schedule_patient_id, "MLE",
            mu_true, theta_true, sigma_true,
            mu_mle, theta_mle, sigma_mle, mle_success, n_timepoints,
            failure_reason=(
                "" if mle_success
                else str(mle_result.message) if not mle_result.success
                else "Optimizer returned nonfinite estimates."
            ),
            **boundaries,
        )
    )

    # Bayesian NUTS
    try:
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=3.5, sigma=2.5)
            log_theta = pm.Normal("log_theta", mu=np.log(1.0), sigma=1.25)
            theta = pm.Deterministic("theta", pm.math.exp(log_theta))
            log_sigma = pm.Normal("log_sigma", mu=np.log(0.5), sigma=1.25)
            sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

            dt = np.diff(t)
            for i in range(1, len(t)):
                current_dt = float(dt[i - 1])
                previous_x = float(x[i - 1])
                decay = pm.math.exp(-theta * current_dt)
                transition_mean = mu + (previous_x - mu) * decay
                transition_variance = (
                    sigma**2 / (2.0 * theta)
                    * (1.0 - pm.math.exp(-2.0 * theta * current_dt))
                )
                pm.Normal(
                    f"x_transition_{i}",
                    mu=transition_mean,
                    sigma=pm.math.sqrt(pm.math.maximum(transition_variance, 1e-12)),
                    observed=float(x[i]),
                )

            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                init="jitter+adapt_diag",
                random_seed=seed,
                return_inferencedata=True,
                compute_convergence_checks=True,
                progressbar=False,
            )

        summary = az.summary(
            idata,
            var_names=["mu", "theta", "sigma"],
            hdi_prob=0.95,
            round_to=10,
        )
        mu_bayes = float(summary.loc["mu", "mean"])
        theta_bayes = float(summary.loc["theta", "mean"])
        sigma_bayes = float(summary.loc["sigma", "mean"])
        max_rhat = float(summary["r_hat"].max())
        min_ess_bulk = float(summary["ess_bulk"].min())
        min_ess_tail = float(summary["ess_tail"].min())
        divergences = int(idata.sample_stats["diverging"].sum().item())
        max_treedepth_events = (
            int(idata.sample_stats["reached_max_treedepth"].sum().item())
            if "reached_max_treedepth" in idata.sample_stats else 0
        )
        bayesian_success = bool(
            np.isfinite(max_rhat)
            and max_rhat <= 1.01
            and divergences == 0
            and max_treedepth_events == 0
            and min_ess_bulk >= min_ess
            and min_ess_tail >= min_ess
        )

        failures = []
        if max_rhat > 1.01:
            failures.append(f"max_rhat={max_rhat:.4f}")
        if divergences > 0:
            failures.append(f"divergences={divergences}")
        if max_treedepth_events > 0:
            failures.append(f"max_treedepth={max_treedepth_events}")
        if min_ess_bulk < min_ess:
            failures.append(f"min_ess_bulk={min_ess_bulk:.1f}")
        if min_ess_tail < min_ess:
            failures.append(f"min_ess_tail={min_ess_tail:.1f}")

        low_col = "hdi_2.5%"
        high_col = "hdi_97.5%"
        rows.append(
            make_result_row(
                simulation_id, scenario, replicate, schedule_patient_id, "Bayesian",
                mu_true, theta_true, sigma_true,
                mu_bayes, theta_bayes, sigma_bayes,
                bayesian_success, n_timepoints,
                failure_reason="; ".join(failures),
                max_rhat=max_rhat,
                min_ess_bulk=min_ess_bulk,
                min_ess_tail=min_ess_tail,
                divergences=divergences,
                max_treedepth_events=max_treedepth_events,
                mu_lower=float(summary.loc["mu", low_col]),
                mu_upper=float(summary.loc["mu", high_col]),
                theta_lower=float(summary.loc["theta", low_col]),
                theta_upper=float(summary.loc["theta", high_col]),
                sigma_lower=float(summary.loc["sigma", low_col]),
                sigma_upper=float(summary.loc["sigma", high_col]),
            )
        )
    except Exception as exc:
        warnings.warn(f"Bayesian fit failed for {simulation_id}: {exc}")
        rows.append(
            make_result_row(
                simulation_id, scenario, replicate, schedule_patient_id, "Bayesian",
                mu_true, theta_true, sigma_true,
                np.nan, np.nan, np.nan, False, n_timepoints,
                failure_reason=str(exc),
            )
        )

    # MoM or MLE-imputed MoM. MLE imputation avoids using Bayesian estimates.
    if n_timepoints >= 3:
        try:
            mu_mom, theta_mom, sigma_mom = ou_mom_estimator(t, x)
            rows.append(
                make_result_row(
                    simulation_id, scenario, replicate, schedule_patient_id, "MoM",
                    mu_true, theta_true, sigma_true,
                    mu_mom, theta_mom, sigma_mom, True, n_timepoints,
                )
            )
        except Exception as exc:
            rows.append(
                make_result_row(
                    simulation_id, scenario, replicate, schedule_patient_id, "MoM",
                    mu_true, theta_true, sigma_true,
                    np.nan, np.nan, np.nan, False, n_timepoints,
                    failure_reason=str(exc),
                )
            )
    else:
        try:
            if not finite_parameters(mu_mle, theta_mle, sigma_mle):
                raise ValueError("Finite MLE estimates were unavailable for imputation.")
            new_times = np.linspace(float(t.min()), float(t.max()), 3)[1:-1]
            new_values = impute_points_from_mle(
                t, x, mu_mle, theta_mle, sigma_mle, new_times
            )
            mu_mom, theta_mom, sigma_mom = run_mom_on_augmented(
                t, x, new_times, new_values
            )
            rows.append(
                make_result_row(
                    simulation_id, scenario, replicate, schedule_patient_id,
                    "MoM (imputed)", mu_true, theta_true, sigma_true,
                    mu_mom, theta_mom, sigma_mom, True, n_timepoints,
                    imputation_source="MLE",
                )
            )
        except Exception as exc:
            rows.append(
                make_result_row(
                    simulation_id, scenario, replicate, schedule_patient_id,
                    "MoM (imputed)", mu_true, theta_true, sigma_true,
                    np.nan, np.nan, np.nan, False, n_timepoints,
                    failure_reason=str(exc), imputation_source="MLE",
                )
            )

    return rows


def add_reliability_and_errors(df):
    out = df.copy()
    bool_columns = [
        "finite_estimate", "success",
        "boundary_mu_lower", "boundary_mu_upper",
        "boundary_theta_lower", "boundary_theta_upper",
        "boundary_sigma_lower", "boundary_sigma_upper",
    ]
    for column in bool_columns:
        out[column] = out[column].fillna(False).astype(bool)

    boundary_columns = [
        "boundary_mu_lower", "boundary_mu_upper",
        "boundary_theta_lower", "boundary_theta_upper",
        "boundary_sigma_lower", "boundary_sigma_upper",
    ]
    out["boundary_contact"] = out[boundary_columns].any(axis=1)
    out["reliable_estimate"] = False

    out.loc[
        (out["method"] == "Bayesian") & out["finite_estimate"] & out["success"],
        "reliable_estimate",
    ] = True
    out.loc[
        (out["method"] == "MLE") & out["finite_estimate"] & out["success"]
        & ~out["boundary_contact"],
        "reliable_estimate",
    ] = True
    out.loc[
        out["method"].isin(["MoM", "MoM (imputed)"])
        & out["finite_estimate"] & out["success"]
        & (out["theta"] > 0) & (out["sigma"] > 0),
        "reliable_estimate",
    ] = True

    for parameter in PARAMETERS:
        true_col = f"{parameter}_true"
        out[f"{parameter}_error"] = out[parameter] - out[true_col]
        out[f"{parameter}_abs_error"] = np.abs(out[f"{parameter}_error"])
        out[f"{parameter}_relative_error"] = out[f"{parameter}_error"] / out[true_col]
        lower_col = f"{parameter}_lower"
        upper_col = f"{parameter}_upper"
        out[f"{parameter}_covered"] = np.where(
            out[lower_col].notna() & out[upper_col].notna(),
            (out[lower_col] <= out[true_col]) & (out[true_col] <= out[upper_col]),
            np.nan,
        )
    return out


def build_performance_table(df):
    rows = []
    for method in METHOD_ORDER:
        method_df = df.loc[df["method"] == method].copy()
        for parameter in PARAMETERS:
            valid = method_df.loc[
                method_df[parameter].notna() & np.isfinite(method_df[parameter])
            ].copy()
            errors = valid[f"{parameter}_error"].to_numpy(dtype=float)
            abs_errors = np.abs(errors)
            coverage_values = valid[f"{parameter}_covered"].dropna()
            rows.append(
                {
                    "method": method,
                    "parameter": parameter,
                    "n_total": int(len(method_df)),
                    "n_finite": int(len(valid)),
                    "n_reliable": int(valid["reliable_estimate"].sum()),
                    "bias": float(np.mean(errors)) if len(errors) else np.nan,
                    "mae": float(np.mean(abs_errors)) if len(abs_errors) else np.nan,
                    "median_ae": float(np.median(abs_errors)) if len(abs_errors) else np.nan,
                    "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
                    "coverage_95_percent": (
                        100.0 * coverage_values.astype(bool).mean()
                        if len(coverage_values) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_reliability_table(df):
    total_simulations = int(df["simulation_id"].nunique())
    rows = []
    for method in METHOD_ORDER:
        method_df = df.loc[df["method"] == method].copy()
        finite_n = int(method_df["finite_estimate"].sum())
        success_n = int(method_df["success"].sum())
        reliable_n = int(method_df["reliable_estimate"].sum())
        rows.append(
            {
                "method": method,
                "total_simulations": total_simulations,
                "method_rows": int(len(method_df)),
                "finite_n": finite_n,
                "finite_percent": 100.0 * finite_n / total_simulations,
                "success_n": success_n,
                "success_percent": 100.0 * success_n / total_simulations,
                "reliable_n": reliable_n,
                "reliable_percent": 100.0 * reliable_n / total_simulations,
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=Path, default=EXCEL_PATH)
    parser.add_argument("--replicates-per-scenario", type=int, default=10)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--min-ess", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scenarios-json", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = DEFAULT_SCENARIOS
    if args.scenarios_json is not None:
        with open(args.scenarios_json, "r", encoding="utf-8") as handle:
            scenarios = json.load(handle)

    scenario_df = pd.DataFrame(scenarios)
    required = {"scenario", "mu_true", "theta_true", "sigma_true"}
    missing = required - set(scenario_df.columns)
    if missing:
        raise ValueError(f"Scenario definition is missing columns: {sorted(missing)}")
    scenario_df.to_csv(TABLE_SCENARIOS, index=False)

    schedules = load_empirical_schedules(args.excel)
    schedule_ids = sorted(schedules)
    rng = np.random.default_rng(args.seed)

    completed_ids = set()
    result_rows = []
    if args.resume and CHECKPOINT.exists():
        checkpoint_df = pd.read_csv(CHECKPOINT)
        result_rows = checkpoint_df.to_dict("records")
        completed_ids = set(checkpoint_df["simulation_id"].astype(str).unique())
        print(f"[INFO] Resuming with {len(completed_ids)} completed simulations.")

    total = len(scenarios) * args.replicates_per_scenario
    current = 0
    for scenario_index, scenario in enumerate(scenarios):
        for replicate in range(1, args.replicates_per_scenario + 1):
            current += 1
            simulation_id = f"{scenario['scenario']}_R{replicate:03d}"
            if simulation_id in completed_ids:
                continue

            schedule_patient_id = str(rng.choice(schedule_ids))
            times = schedules[schedule_patient_id]
            sim_seed = args.seed + scenario_index * 100_000 + replicate * 100
            values = simulate_ou_trajectory(
                times,
                float(scenario["mu_true"]),
                float(scenario["theta_true"]),
                float(scenario["sigma_true"]),
                np.random.default_rng(sim_seed),
            )

            print(
                f"[INFO] {current}/{total} {simulation_id}: "
                f"schedule={schedule_patient_id}, n={len(times)}"
            )

            result_rows.extend(
                fit_simulated_series(
                    simulation_id=simulation_id,
                    scenario=str(scenario["scenario"]),
                    replicate=replicate,
                    schedule_patient_id=schedule_patient_id,
                    times=times,
                    values=values,
                    mu_true=float(scenario["mu_true"]),
                    theta_true=float(scenario["theta_true"]),
                    sigma_true=float(scenario["sigma_true"]),
                    draws=args.draws,
                    tune=args.tune,
                    chains=args.chains,
                    cores=args.cores,
                    target_accept=args.target_accept,
                    min_ess=args.min_ess,
                    seed=sim_seed + 1,
                )
            )
            pd.DataFrame(result_rows).to_csv(CHECKPOINT, index=False)

    estimates_df = add_reliability_and_errors(pd.DataFrame(result_rows))
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    estimates_df["_method_order"] = estimates_df["method"].map(method_rank)
    estimates_df = (
        estimates_df.sort_values(
            ["scenario", "replicate", "_method_order"], kind="mergesort"
        )
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )

    performance_df = build_performance_table(estimates_df)
    reliability_df = build_reliability_table(estimates_df)

    estimates_df.to_csv(TABLE_ESTIMATES, index=False)
    performance_df.to_csv(TABLE_PERFORMANCE, index=False)
    reliability_df.to_csv(TABLE_RELIABILITY, index=False)

    print("\n[OK] Saved simulation scenarios:")
    print(f"     {TABLE_SCENARIOS}")
    print("[OK] Saved simulation estimates:")
    print(f"     {TABLE_ESTIMATES}")
    print("[OK] Saved simulation performance:")
    print(f"     {TABLE_PERFORMANCE}")
    print("[OK] Saved simulation reliability:")
    print(f"     {TABLE_RELIABILITY}")


if __name__ == "__main__":
    main()
