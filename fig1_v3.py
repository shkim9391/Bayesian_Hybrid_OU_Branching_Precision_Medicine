"""
Figure 1 schematic for:
"Stage-specific evolutionary dynamics in pediatric leukemia inferred by Bayesian stochastic modeling"

Outputs:
    figure1_schematic_v3.png
    figure1_schematic_v3.pdf
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D


# ----------------------------
# Global style
# ----------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "figure.titlesize": 16,
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ----------------------------
# Colors
# ----------------------------
BLUE = "#4C78A8"
BLUE_LIGHT = "#DCEAF7"

GREEN = "#59A14F"
GREEN_LIGHT = "#E4F2E0"

ORANGE = "#F28E2B"
ORANGE_LIGHT = "#FCE8D6"

PURPLE = "#9C6ADE"
PURPLE_LIGHT = "#EEE4FB"

RED = "#E15759"

GRAY = "#6E6E6E"
DARK = "#1F1F1F"


# ----------------------------
# Helpers
# ----------------------------
def add_panel_box(ax, edgecolor: str, facecolor: str, title: str, letter: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    outer = FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.01,rounding_size=0.028",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor="white",
        zorder=0
    )
    ax.add_patch(outer)

    header = FancyBboxPatch(
        (0.01, 0.855), 0.98, 0.135,
        boxstyle="round,pad=0.01,rounding_size=0.028",
        linewidth=0,
        facecolor=facecolor,
        zorder=0
    )
    ax.add_patch(header)
    ax.add_patch(Rectangle((0.01, 0.855), 0.98, 0.08, color=facecolor, ec="none", zorder=0))

    circ = Circle((0.065, 0.922), 0.030, facecolor=edgecolor, edgecolor="none")
    ax.add_patch(circ)
    ax.text(0.065, 0.922, letter, ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")

    ax.text(0.12, 0.922, title, ha="left", va="center",
            color=DARK, fontsize=14, fontweight="bold")


def draw_patient_icon(ax, x: float, y: float, scale: float = 1.0, color: str = BLUE) -> None:
    head = Circle((x, y + 0.065 * scale), 0.020 * scale, color=color, ec="none")
    ax.add_patch(head)
    ax.add_line(Line2D([x, x], [y - 0.015 * scale, y + 0.040 * scale], lw=3.0, color=color))
    ax.add_line(Line2D([x - 0.028 * scale, x + 0.028 * scale], [y + 0.01 * scale, y + 0.01 * scale], lw=3.0, color=color))
    ax.add_line(Line2D([x, x - 0.026 * scale], [y - 0.015 * scale, y - 0.065 * scale], lw=3.0, color=color))
    ax.add_line(Line2D([x, x + 0.026 * scale], [y - 0.015 * scale, y - 0.065 * scale], lw=3.0, color=color))


def make_ou_paths(
    n_paths: int = 4,
    n_steps: int = 160,
    mu: float = 0.0,
    theta: float = 1.25,
    sigma: float = 0.22,
    x0: float = 0.2,
    dt: float = 0.03,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * dt
    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = x0 + rng.normal(0, 0.12, size=n_paths)

    for i in range(n_paths):
        for k in range(1, n_steps):
            prev = paths[i, k - 1]
            drift = theta * (mu - prev) * dt
            diffusion = sigma * np.sqrt(dt) * rng.normal()
            paths[i, k] = prev + drift + diffusion

    return t, paths


def draw_vertical_axis(ax, x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=DARK))
    ax.text(x - 0.012, y1 + 0.004, "High", fontsize=8.5, ha="center", va="bottom")
    ax.text(x - 0.012, y0 - 0.008, "Low", fontsize=8.5, ha="center", va="top")


def draw_boxplot_schematic(ax, x, y, width, low, q1, med, q3, high, color):
    ax.add_line(Line2D([x, x], [y + low, y + high], color=color, lw=1.2))
    ax.add_line(Line2D([x - width * 0.25, x + width * 0.25], [y + low, y + low], color=color, lw=1.2))
    ax.add_line(Line2D([x - width * 0.25, x + width * 0.25], [y + high, y + high], color=color, lw=1.2))
    rect = Rectangle((x - width / 2, y + q1), width, q3 - q1,
                     facecolor=color + "30", edgecolor=color, lw=1.2)
    ax.add_patch(rect)
    ax.add_line(Line2D([x - width / 2, x + width / 2], [y + med, y + med], color=color, lw=1.2))


# ----------------------------
# Panel A
# ----------------------------
def draw_panel_a(ax) -> None:
    add_panel_box(ax, BLUE, BLUE_LIGHT, "Longitudinal Pediatric Leukemia Data", "A")

    ax.text(
        0.05, 0.81,
        "Sparse and noisy clinical measurements capture tumor burden\nacross disease stages.",
        ha="left", va="top", fontsize=11.5, color=DARK
    )

    draw_patient_icon(ax, 0.09, 0.40, scale=1.0, color=BLUE)
    ax.text(0.09, 0.52, "Patient i", ha="center", va="bottom",
            color=BLUE, fontweight="bold", fontsize=11)

    x0, y0, w, h = 0.37, 0.20, 0.55, 0.42
    ax.add_line(Line2D([x0, x0], [y0, y0 + h], color=DARK, lw=1.5))
    ax.add_line(Line2D([x0, x0 + w], [y0, y0], color=DARK, lw=1.5))
    ax.text(x0 + w / 2, y0 - 0.05, "Clinical Time", ha="center", va="top", fontsize=12.5)
    ax.text(x0 - 0.22, y0 + h / 2, "Tumor Burden\n(e.g., VAF)", ha="left", va="center", fontsize=12.5)

    xs = np.array([0.08, 0.22, 0.36, 0.50, 0.67, 0.83, 0.97])
    ys = np.array([0.82, 0.56, 0.39, 0.34, 0.17, 0.36, 0.70])
    X = x0 + w * xs
    Y = y0 + h * ys

    ax.plot(X, Y, color=BLUE, lw=1.9)
    ax.scatter(X, Y, s=52, color="#2A549A", zorder=3)

    err = np.array([0.08, 0.06, 0.03, 0.03, 0.05, 0.05, 0.08]) * h
    for xx, yy, ee in zip(X, Y, err):
        ax.add_line(Line2D([xx, xx], [yy - ee, yy + ee], color="0.75", lw=1.0, zorder=1))

    ax.text(x0 + 0.12 * w, y0 + h + 0.02, "Diagnosis", color=RED, fontsize=10.0, ha="center")
    ax.text(x0 + 0.53 * w, y0 + h + 0.02, "Remission", color=GREEN, fontsize=10.0, ha="center")
    ax.text(x0 + 0.89 * w, y0 + h - 0.02, "Relapse/\nRefractory", color=ORANGE, fontsize=10.0, ha="center")

    ax.annotate(
        "Measurement\nnoise",
        xy=(X[1], Y[1]),
        xytext=(0.50, 0.53),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0),
        color=BLUE, fontsize=8, ha="center"
    )
    ax.annotate(
        "Irregular\nsampling",
        xy=(X[5], Y[5]),
        xytext=(0.80, 0.50),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0),
        color=BLUE, fontsize=8, ha="center"
    )


# ----------------------------
# Panel B
# ----------------------------
def draw_panel_b(ax) -> None:
    add_panel_box(ax, GREEN, GREEN_LIGHT, "Ornstein–Uhlenbeck (OU) Dynamics", "B")

    ax.text(
        0.05, 0.81,
        "Tumor evolution fluctuates around a long-term equilibrium\n"
        "under stabilizing forces and stochastic variation.",
        ha="left", va="top", fontsize=11.5, color=DARK
    )

    eq_box = FancyBboxPatch(
        (0.05, 0.20), 0.38, 0.37,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=0.8, edgecolor="none", facecolor="#EEF5EA"
    )
    ax.add_patch(eq_box)

    ax.text(0.24, 0.55, r"$dX(t)=\theta(\mu-X(t))dt+\sigma\,dW(t)$",
            ha="center", va="center", fontsize=11)

    ax.text(0.08, 0.46, r"$\mu$", color=GREEN, fontsize=14, fontweight="bold")
    ax.text(0.14, 0.46, "Long-term equilibrium\n(stable state)", fontsize=9, va="center")

    ax.text(0.08, 0.35, r"$\theta$", color=PURPLE, fontsize=14, fontweight="bold")
    ax.text(0.14, 0.35, "Mean reversion\n(restoring force)", fontsize=9, va="center")

    ax.text(0.08, 0.24, r"$\sigma$", color=ORANGE, fontsize=14, fontweight="bold")
    ax.text(0.14, 0.24, "Stochastic variability\n(random fluctuations)", fontsize=9, va="center")

    x0, y0, w, h = 0.56, 0.22, 0.37, 0.38
    ax.add_line(Line2D([x0, x0], [y0, y0 + h], color=DARK, lw=1.5))
    ax.add_line(Line2D([x0, x0 + w], [y0, y0], color=DARK, lw=1.5))
    ax.text(x0 + w / 2, y0 - 0.04, "Time $t$", ha="center", va="top", fontsize=12.5)
    ax.text(x0 + 0.01, y0 + h - 0.01, "Tumor state $X(t)$", ha="left", va="bottom", fontsize=11.0)

    mu_y = y0 + 0.48 * h
    ax.add_line(Line2D([x0, x0 + w], [mu_y, mu_y], color=GREEN, lw=1.3, ls="--"))
    ax.text(x0 - 0.03, mu_y, r"$\mu$", color=GREEN, fontsize=16, va="center", ha="right")

    t, paths = make_ou_paths()
    t_scaled = x0 + (t - t.min()) / (t.max() - t.min()) * w
    for i in range(paths.shape[0]):
        p = (paths[i] - paths.min()) / (paths.max() - paths.min())
        y_scaled = y0 + 0.16 * h + p * 0.58 * h
        ax.plot(t_scaled, y_scaled, color=PURPLE, lw=1.6, alpha=0.9)

    ax.annotate(
        "Possible tumor\ntrajectories",
        xy=(x0 + 0.73 * w, y0 + 0.53 * h),
        xytext=(0.86, 0.49),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0),
        color=PURPLE, fontsize=8, ha="center"
    )


# ----------------------------
# Panel C
# ----------------------------
def draw_panel_c(ax) -> None:
    add_panel_box(ax, ORANGE, ORANGE_LIGHT, "Bayesian Calibration with Latent States", "C")

    ax.text(
        0.05, 0.81,
        "A latent-state model infers hidden tumor states\nand quantifies parameter uncertainty.",
        ha="left", va="top", fontsize=11, color=DARK
    )

    # Left graphical model box
    gm = FancyBboxPatch(
        (0.05, 0.18), 0.50, 0.44,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0, edgecolor="#E9D0AF", facecolor="#FBF3EA"
    )
    ax.add_patch(gm)

    ax.text(0.08, 0.58, "Priors", color=ORANGE, fontsize=11.0, fontweight="bold")

    # Priors
    prior_x = [0.17, 0.27, 0.37, 0.48]
    prior_labels = [r"$\mu$", r"$\theta$", r"$\sigma$", r"$\sigma_{obs}$"]
    for xx, lab in zip(prior_x, prior_labels):
        circ = Circle((xx, 0.53), 0.035, facecolor="#FBE5CC", edgecolor=ORANGE, lw=1.0)
        ax.add_patch(circ)
        ax.text(xx, 0.53, lab, ha="center", va="center", fontsize=13)

    # Latent nodes
    latent_x = [0.17, 0.34, 0.51]
    latent_labels = [r"$X_1$", r"$X_2$", r"$X_n$"]
    for xx, lab in zip(latent_x, latent_labels):
        circ = Circle((xx, 0.36), 0.043, facecolor="#E9E0FA", edgecolor=PURPLE, lw=1.0)
        ax.add_patch(circ)
        ax.text(xx, 0.36, lab, ha="center", va="center", fontsize=14)

    # Observed nodes
    obs_x = [0.17, 0.34, 0.51]
    obs_labels = [r"$Y_1$", r"$Y_2$", r"$Y_n$"]
    for xx, lab in zip(obs_x, obs_labels):
        circ = Circle((xx, 0.22), 0.043, facecolor="#DFEBFA", edgecolor=BLUE, lw=1.0)
        ax.add_patch(circ)
        ax.text(xx, 0.22, lab, ha="center", va="center", fontsize=14)

    # Arrows: priors to latent / obs
    ax.add_patch(FancyArrowPatch((0.17, 0.495), (0.17, 0.405), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=GRAY))
    ax.add_patch(FancyArrowPatch((0.27, 0.495), (0.33, 0.40), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=GRAY))
    ax.add_patch(FancyArrowPatch((0.37, 0.495), (0.50, 0.40), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=GRAY))
    ax.add_patch(FancyArrowPatch((0.48, 0.495), (0.51, 0.265), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=GRAY))

    # Arrows: latent chain
    ax.add_patch(FancyArrowPatch((0.215, 0.36), (0.295, 0.36), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=GRAY))
    ax.add_patch(FancyArrowPatch((0.385, 0.36), (0.465, 0.36), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=GRAY))
    ax.text(0.425, 0.36, r"$\cdots$", color=GRAY, va="center", ha="center", fontsize=13)

    # Arrows: latent to observed
    for xx in obs_x:
        ax.add_patch(FancyArrowPatch((xx, 0.315), (xx, 0.26), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=GRAY))

    # Side labels
    ax.text(0.085, 0.36, "Latent", color=PURPLE, fontsize=8.0, ha="center", va="center")
    ax.text(0.076, 0.22, "Observed", color=BLUE, fontsize=8.0, ha="center", va="center")

    # Right posterior box
    box = FancyBboxPatch(
        (0.68, 0.28), 0.22, 0.24,
        boxstyle="round,pad=0.06,rounding_size=0.03",
        linewidth=1.0, edgecolor="#E0E0E0", facecolor="#FAFAFA"
    )
    ax.add_patch(box)
    ax.text(0.79, 0.53, "Posterior uncertainty", ha="center", va="center",
            fontsize=11.5, color="#2F3E8F")

    xx = np.linspace(0.2, 1.27, 200)
    curves = [
        (0.48, GREEN, r"$\mu$"),
        (0.68, PURPLE, r"$\theta$"),
        (0.86, ORANGE, r"$\sigma$"),
        (0.106, BLUE, r"$\sigma_{obs}$"),
    ]
    for mean, color, _ in curves:
        yy = np.exp(-0.5 * ((xx - mean) / 0.08) ** 2)
        yy = yy / yy.max()
        X = 0.67 + xx * 0.16
        Y = 0.30 + yy * 0.11
        ax.plot(X, Y, color=color, lw=1.6)

    ax.text(0.89, 0.42, r"$\mu$", color=GREEN, fontsize=10.5)
    ax.text(0.89, 0.37, r"$\theta$", color=PURPLE, fontsize=10.5)
    ax.text(0.89, 0.32, r"$\sigma$", color=ORANGE, fontsize=10.5)
    ax.text(0.89, 0.27, r"$\sigma_{obs}$", color=BLUE, fontsize=10.5)

    ax.text(0.78, 0.27, "Parameter value", ha="center", va="top", fontsize=9.5)
    ax.text(0.68, 0.37, "Density", ha="center", va="center", rotation=90, fontsize=9.5)

    # Shorter formula strip
    formula = FancyBboxPatch(
        (0.05, 0.06), 0.50, 0.065,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=0, facecolor="#EEE7FA"
    )
    ax.add_patch(formula)
    ax.text(
        0.30, 0.092,
        r"$p(\mu,\theta,\sigma,\sigma_{obs},X \mid Y,t)$",
        ha="center", va="center", fontsize=13
    )

# ----------------------------
# Panel D
# ----------------------------
def draw_panel_d(ax) -> None:
    add_panel_box(ax, PURPLE, PURPLE_LIGHT, "Stage-Specific Evolutionary Dynamics", "D")

    ax.text(
        0.05, 0.81,
        "Bayesian-calibrated parameters reveal biologically\nmeaningful differences across disease stages.",
        ha="left", va="top", fontsize=11, color=DARK
    )

    # Left and right comparison boxes
    remission_box = FancyBboxPatch(
        (0.06, 0.19), 0.40, 0.44,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0, edgecolor="#D7CAE9", facecolor="#FAF8FD"
    )
    relapse_box = FancyBboxPatch(
        (0.54, 0.19), 0.40, 0.44,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0, edgecolor="#E8D6C8", facecolor="#FFF9F4"
    )
    ax.add_patch(remission_box)
    ax.add_patch(relapse_box)

    # Headers
    rem_hdr = FancyBboxPatch(
        (0.19, 0.56), 0.15, 0.05,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0, facecolor="#CFE8C9"
    )
    rel_hdr = FancyBboxPatch(
        (0.59, 0.56), 0.30, 0.05,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0, facecolor="#F8D7B8"
    )
    ax.add_patch(rem_hdr)
    ax.add_patch(rel_hdr)

    ax.text(0.265, 0.585, "Remission", ha="center", va="center", fontsize=12.5, fontweight="bold")
    ax.text(0.74, 0.585, "Relapse / Refractory", ha="center", va="center", fontsize=12.5, fontweight="bold")

    # Shorter descriptors
    ax.text(0.265, 0.52, "Higher stabilization", ha="center", va="top", fontsize=10.5)
    ax.text(0.74, 0.52, "Higher variability", ha="center", va="top", fontsize=10.5)

    # Variable labels only
    ax.text(0.18, 0.42, r"$\theta$", ha="center", va="center", fontsize=13)
    ax.text(0.34, 0.36, r"$\sigma$", ha="center", va="center", fontsize=13)
    ax.text(0.67, 0.36, r"$\theta$", ha="center", va="center", fontsize=13)
    ax.text(0.83, 0.43, r"$\sigma$", ha="center", va="center", fontsize=13)

    # Axes
    draw_vertical_axis(ax, 0.13, 0.24, 0.35)
    draw_vertical_axis(ax, 0.29, 0.24, 0.35)
    draw_vertical_axis(ax, 0.62, 0.24, 0.35)
    draw_vertical_axis(ax, 0.78, 0.24, 0.35)

    # Boxplots
    draw_boxplot_schematic(ax, 0.18, 0.20, 0.075,
                           low=0.02, q1=0.07, med=0.13, q3=0.17, high=0.19,
                           color=GREEN)
    draw_boxplot_schematic(ax, 0.34, 0.20, 0.075,
                           low=0.02, q1=0.04, med=0.08, q3=0.11, high=0.13,
                           color=GREEN)

    draw_boxplot_schematic(ax, 0.67, 0.20, 0.075,
                           low=0.02, q1=0.04, med=0.08, q3=0.11, high=0.13,
                           color=RED)
    draw_boxplot_schematic(ax, 0.83, 0.20, 0.075,
                           low=0.02, q1=0.07, med=0.13, q3=0.18, high=0.20,
                           color=RED)

    # Bottom strip
    strip = FancyBboxPatch(
        (0.15, 0.065), 0.70, 0.045,
        boxstyle="round,pad=0.050,rounding_size=0.02",
        linewidth=0, facecolor="#D9E3F7"
    )
    ax.add_patch(strip)
    ax.text(
        0.50, 0.087,
        "Outcome: stage-specific dynamics with quantified uncertainty",
        ha="center", va="center", fontsize=9.8, fontweight="bold", color=DARK
    )

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    fig = plt.figure(figsize=(12.5, 7.2), facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        left=0.03, right=0.99,
        top=0.88, bottom=0.14,
        wspace=0.018, hspace=0.03
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    draw_panel_a(axA)
    draw_panel_b(axB)
    draw_panel_c(axC)
    draw_panel_d(axD)

    fig.savefig("figure1_schematic_v3.png", dpi=300, bbox_inches="tight")
    fig.savefig("figure1_schematic_v3.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
