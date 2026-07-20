from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 13

PROJECT_ROOT = Path(
    "/scripts/ "04_plot_simulation_calibration_figure.py"
)
OUT_DIR = PROJECT_ROOT / "Figure_2" / "out_simulation_benchmark"

TABLE_ESTIMATES = OUT_DIR / "Table_S2_simulation_estimates.csv"
TABLE_PERFORMANCE = OUT_DIR / "Table_S2_simulation_performance.csv"
TABLE_RELIABILITY = OUT_DIR / "Table_S2_simulation_reliability.csv"

FIGURE_PNG = OUT_DIR / "Figure2_simulation_calibration_benchmark.png"
FIGURE_PDF = OUT_DIR / "Figure2_simulation_calibration_benchmark.pdf"

METHOD_ORDER = ["Bayesian", "MLE", "MoM", "MoM (imputed)"]
PARAMETER_INFO = {
    "mu": {
        "panel": "A",
        "title": r"Equilibrium state, $\mu$",
        "x_label": r"True $\mu$",
        "y_label": r"Estimated $\mu$",
        "log_scale": False,
    },
    "theta": {
        "panel": "B",
        "title": r"Mean-reversion rate, $\theta$",
        "x_label": r"True $\theta$ (year$^{-1}$)",
        "y_label": r"Estimated $\theta$ (year$^{-1}$)",
        "log_scale": True,
    },
    "sigma": {
        "panel": "C",
        "title": r"Diffusion scale, $\sigma$",
        "x_label": r"True $\sigma$ (year$^{-1/2}$)",
        "y_label": r"Estimated $\sigma$ (year$^{-1/2}$)",
        "log_scale": True,
    },
}


def coerce_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if series.dtype == object:
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )
    return series.fillna(False).astype(bool)


def prepare_estimates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["finite_estimate", "success", "reliable_estimate"]:
        if column not in out.columns:
            raise ValueError(f"Required estimates column missing: {column}")
        out[column] = coerce_boolean(out[column])
    return out


def plot_recovery_panel(ax, estimates_df, parameter, rng):
    info = PARAMETER_INFO[parameter]
    true_col = f"{parameter}_true"

    finite = estimates_df.loc[
        estimates_df["finite_estimate"]
        & estimates_df[parameter].notna()
        & estimates_df[true_col].notna()
        & np.isfinite(estimates_df[parameter])
        & np.isfinite(estimates_df[true_col])
    ].copy()

    if info["log_scale"]:
        finite = finite.loc[
            (finite[parameter] > 0) & (finite[true_col] > 0)
        ].copy()

    if finite.empty:
        ax.text(0.5, 0.5, "No finite estimates", ha="center", va="center")
        ax.set_title(info["title"], fontweight="bold")
        return

    all_values = np.concatenate(
        [
            finite[true_col].to_numpy(dtype=float),
            finite[parameter].to_numpy(dtype=float),
        ]
    )
    lower = float(np.nanmin(all_values))
    upper = float(np.nanmax(all_values))

    if info["log_scale"]:
        lower = max(lower * 0.75, 1e-4)
        upper *= 1.25
    else:
        padding = max((upper - lower) * 0.08, 0.1)
        lower -= padding
        upper += padding

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.0,
        label="Perfect recovery",
        zorder=1,
    )

    for method in METHOD_ORDER:
        method_df = finite.loc[finite["method"] == method].copy()
        if method_df.empty:
            continue

        if info["log_scale"]:
            x_values = method_df[true_col].to_numpy(dtype=float) * np.exp(
                rng.uniform(-0.025, 0.025, size=len(method_df))
            )
        else:
            scale = max(upper - lower, 1.0)
            x_values = (
                method_df[true_col].to_numpy(dtype=float)
                + rng.uniform(-0.0075, 0.0075, size=len(method_df)) * scale
            )

        reliable = method_df["reliable_estimate"].to_numpy(dtype=bool)
        y_values = method_df[parameter].to_numpy(dtype=float)

        if np.any(~reliable):
            ax.scatter(
                x_values[~reliable],
                y_values[~reliable],
                marker="x",
                s=24,
                linewidths=0.9,
                alpha=0.55,
                label=(
                    f"{method}: finite but unreliable"
                    if parameter == "mu" else None
                ),
                zorder=2,
            )

        if np.any(reliable):
            ax.scatter(
                x_values[reliable],
                y_values[reliable],
                marker="o",
                s=23,
                edgecolors="black",
                linewidths=0.3,
                alpha=0.72,
                label=(f"{method}: reliable" if parameter == "mu" else None),
                zorder=3,
            )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    if info["log_scale"]:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(info["x_label"])
    ax.set_ylabel(info["y_label"])
    ax.set_title(info["title"], fontweight="bold")
    ax.text(
        -0.15,
        1.05,
        info["panel"],
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )


def plot_reliability_panel(ax, reliability_df):
    ordered = reliability_df.set_index("method").reindex(METHOD_ORDER)
    if ordered[["finite_percent", "success_percent", "reliable_percent"]].isna().all().all():
        raise ValueError("Reliability table does not contain usable percentages.")

    x = np.arange(len(METHOD_ORDER))
    width = 0.24

    ax.bar(x - width, ordered["finite_percent"], width=width, label="Finite")
    ax.bar(
        x,
        ordered["success_percent"],
        width=width,
        label="Algorithm/diagnostics passed",
    )
    ax.bar(
        x + width,
        ordered["reliable_percent"],
        width=width,
        label="Reliable",
    )

    for method_index, method in enumerate(METHOD_ORDER):
        if method not in ordered.index or pd.isna(ordered.loc[method, "reliable_n"]):
            continue
        reliable_n = int(ordered.loc[method, "reliable_n"])
        reliable_percent = float(ordered.loc[method, "reliable_percent"])
        ax.text(
            method_index + width,
            reliable_percent + 1.5,
            str(reliable_n),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=28, ha="right")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Simulated datasets (%)")
    ax.set_title("Method reliability", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.text(
        -0.15,
        1.05,
        "D",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        0.98,
        0.03,
        "Numbers above reliable bars indicate n",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimates", type=Path, default=TABLE_ESTIMATES)
    parser.add_argument("--performance", type=Path, default=TABLE_PERFORMANCE)
    parser.add_argument("--reliability", type=Path, default=TABLE_RELIABILITY)
    parser.add_argument("--output-png", type=Path, default=FIGURE_PNG)
    parser.add_argument("--output-pdf", type=Path, default=FIGURE_PDF)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in [args.estimates, args.performance, args.reliability]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required benchmark output not found:\n{path}\n"
                "Run 03_simulate_ou_calibration_benchmark.py first."
            )

    estimates_df = prepare_estimates(pd.read_csv(args.estimates))
    performance_df = pd.read_csv(args.performance)
    reliability_df = pd.read_csv(args.reliability)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)

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

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.0, 6.7),
        dpi=600,
        constrained_layout=False,
    )
    
    rng = np.random.default_rng(SEED)
    
    plot_recovery_panel(axes[0, 0], estimates_df, "mu", rng)
    plot_recovery_panel(axes[0, 1], estimates_df, "theta", rng)
    plot_recovery_panel(axes[1, 0], estimates_df, "sigma", rng)
    plot_reliability_panel(axes[1, 1], reliability_df)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    if handles:
        unique = dict(zip(labels, handles))
    
        fig.legend(
            handles=list(unique.values()),
            labels=list(unique.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.035),
            ncol=4,
            frameon=False,
            fontsize=8,
            columnspacing=1.5,
            handletextpad=0.6,
        )
    
    fig.subplots_adjust(
        top=0.90,
        bottom=0.08,
        left=0.08,
        right=0.98,
        hspace=0.32,
        wspace=0.18,
    )
    
    fig.savefig(
        args.output_png,
        bbox_inches="tight",
    )
    
    fig.savefig(
        args.output_pdf,
        bbox_inches="tight",
    )
    
    plt.close(fig)

    print("[OK] Saved simulation benchmark Figure_2:")
    print(f"     {args.output_png}")
    print(f"     {args.output_pdf}")


if __name__ == "__main__":
    main()
