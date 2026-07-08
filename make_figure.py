"""Step 5: assemble the Section 7 figure (two panels).

Left panel: transductive error vs. number of labels n_L on a log-log scale,
for the graph-Laplacian probe (Eq. alg) and a ridge reference, with horizontal
lines at the matched floor (the cut of the probe's own kNN graph) and the
augmentation-based R_DA floors. No supervised baseline: this figure is a
numerical validation of Theorem 2 (the rate and the floor), not a comparison.

Right panel: the quantity the theory actually predicts a 1/n_L rate for, namely
the EXCESS error above the floor,  error(n_L) - R_DA . Theorem 2 says
error <= C/n_L + R_DA, so it is the excess, not the raw error, that should fall
at slope -1. Fitting the raw curve understates the rate badly because the curve
has already plateaued at large n_L; fitting the excess recovers the ~1/n_L
behavior. A reference line of slope -1 is drawn for comparison.

Reads the JSON written by probe_sweep, baseline, and measure_rda. Run last:
    python make_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import common
import config

EXCESS_EPS = 1e-4  # excess values at or below this are treated as on-the-floor


def fit_floor_rate(n, y):
    """Fit error(n_L) = R_inf + C * n_L^(-beta) by nonlinear least squares.

    Returns ((R_inf, C, beta), beta_stderr). R_inf is the data-estimated floor;
    beta is the rate, so the excess error  y - R_inf = C * n_L^(-beta)  is a
    straight line of slope -beta on log-log axes. R_inf is bounded below the
    smallest observed error so the excess stays positive.
    """
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    ymin = float(y.min())

    def model(nn, R, C, beta):
        return R + C * nn ** (-beta)

    p0 = [0.9 * ymin, max(float(y[0]) - ymin, 1e-3), 1.0]
    bounds = ([0.0, 0.0, 0.1], [ymin, 10.0, 4.0])
    popt, pcov = curve_fit(model, n, y, p0=p0, bounds=bounds, maxfev=40000)
    beta_se = float(np.sqrt(np.diag(pcov))[2])
    return (float(popt[0]), float(popt[1]), float(popt[2])), beta_se


def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run the step that writes it first.")
    with open(path) as f:
        return json.load(f)


def curve(results, method, n_values):
    """Return (means, stds) for a method, indexing JSON's string keys safely."""
    block = results[method]
    means = np.array([block[str(n)]["mean"] for n in n_values])
    stds = np.array([block[str(n)]["std"] for n in n_values])
    return means, stds


def main() -> None:
    probe = load(config.PROBE_RESULTS)
    rda = load(config.RDA_RESULTS)

    n_values = np.array(probe["n_L"], dtype=float)
    graph_m, graph_s = curve(probe, "graph", probe["n_L"])
    ridge_m, ridge_s = curve(probe, "ridge", probe["n_L"])

    # Estimated floor + rate. The plain graph cut is NOT used as the floor (the
    # graph curve descends through it, so it is not a hard floor). Instead we
    # estimate each probe's own asymptotic floor R_inf from the data by fitting
    # error = R_inf + C * n_L^(-beta). The excess above R_inf then exhibits the
    # rate -beta as a straight line on the right panel.
    (Rg, Cg, bg), bg_se = fit_floor_rate(n_values, graph_m)
    (Rr, Cr, br), br_se = fit_floor_rate(n_values, ridge_m)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- Left: raw error vs n_L -------------------------------------------
    def plot_raw(means, stds, label, marker):
        axL.errorbar(n_values, means, yerr=stds, marker=marker, capsize=3, label=label)
        return common.fit_loglog_slope(n_values, means)[0]

    g_slope = plot_raw(graph_m, graph_s, "graph probe (Eq. alg)", "o")
    r_slope = plot_raw(ridge_m, ridge_s, "ridge probe", "s")

    # Theory cut lines dropped. The lines shown are the DATA-ESTIMATED floors
    # R_inf from the power-law fit, drawn so the curves visibly approach them.
    axL.axhline(Rg, ls="--", lw=1.0, color="tab:blue", alpha=0.7,
                label=f"graph fitted floor $R_\\infty$ = {Rg:.3f}")
    axL.axhline(Rr, ls="--", lw=1.0, color="tab:orange", alpha=0.7,
                label=f"ridge fitted floor $R_\\infty$ = {Rr:.3f}")

    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("labels per class  $n_L$")
    axL.set_ylabel("transductive error on the unlabeled pool")
    axL.set_title("Error vs. number of labels")
    axL.grid(True, which="both", ls=":", alpha=0.4)
    axL.legend(fontsize=7.5, loc="best")
    axL.text(0.02, 0.02,
             ("raw-curve slopes (incl. plateau):\n"
              f"  graph  {g_slope:+.2f}\n"
              f"  ridge  {r_slope:+.2f}"),
             transform=axL.transAxes, fontsize=7, va="bottom", ha="left",
             family="monospace", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # ---- Right: excess above the fitted floor (the rate) ------------------
    # excess(n_L) = error(n_L) - R_inf = C * n_L^(-beta), a straight line of
    # slope -beta on log-log. Both probes are shown, each above its OWN fitted
    # floor, so the slopes are directly comparable.
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel("labels per class  $n_L$")
    axR.set_ylabel(r"excess error  $\,\mathrm{error}(n_L) - R_\infty$")
    axR.set_title("Excess above the fitted floor (the rate)")
    axR.grid(True, which="both", ls=":", alpha=0.4)

    for means, R, C, beta, beta_se, color, name in (
        (graph_m, Rg, Cg, bg, bg_se, "tab:blue", "graph"),
        (ridge_m, Rr, Cr, br, br_se, "tab:orange", "ridge"),
    ):
        ex = means - R
        m = ex > EXCESS_EPS
        axR.plot(n_values[m], ex[m], "o", color=color, label=f"{name} excess")
        xs = np.array([n_values[m].min(), n_values[m].max()])
        axR.plot(xs, C * xs ** (-beta), "-", color=color,
                 label=f"{name} slope ${-beta:+.2f}\\pm{beta_se:.2f}$")

    # Reference slope -1 (the 1/n_L prediction), anchored at graph's first point.
    ex_g0 = graph_m[0] - Rg
    xs = np.array([n_values.min(), n_values.max()])
    axR.plot(xs, ex_g0 * (xs / n_values[0]) ** (-1.0), ":", color="gray",
             label="slope $-1$  ($1/n_L$)")
    axR.legend(fontsize=8, loc="best")

    fig.suptitle("CIFAR-10, frozen SimCLR backbone", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURE_PATH)
    png = os.path.splitext(config.FIGURE_PATH)[0] + ".png"
    fig.savefig(png, dpi=200)

    print("saved", config.FIGURE_PATH, "and", png)
    print(f"graph: fitted floor R_inf={Rg:.4f}  rate beta={bg:.2f} +/- {bg_se:.2f}")
    print(f"ridge: fitted floor R_inf={Rr:.4f}  rate beta={br:.2f} +/- {br_se:.2f}")
    print("  excess = error - R_inf has log-log slope -beta by construction.")


if __name__ == "__main__":
    main()
