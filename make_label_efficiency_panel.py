"""Descriptive real-data panel: SSL label efficiency on CIFAR-10.

Reads the already-computed transductive probe sweep (cache/probe_results.json)
and draws ONE panel: transductive accuracy on the unlabeled pool versus the
number of labels, for the graph-Laplacian probe (Eq. alg, the estimator the
theory analyzes) and a ridge linear-probe reference. The point of the figure is
only that the phenomenon is real on real data: a simple probe on frozen SimCLR
features reaches the backbone's ~90% accuracy ceiling using a few percent of the
labels. It makes no floor or cut claim (those were dropped); the mechanism is
verified in the synthetic experiments.

Deliberately torch-free (json + numpy + matplotlib), like synthetic_verify.py,
so it runs without the torch env. Writes label_efficiency.pdf and .png to cache.

Run:
    python make_label_efficiency_panel.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(CACHE, "probe_results.json")

N_CLASSES = 10
TRAIN_SIZE = 50000           # CIFAR-10 train split
BACKBONE_TEST_ACC = 90.12    # edadaltocg/resnet50_simclr_cifar10 reported test acc


def load():
    with open(RESULTS) as f:
        r = json.load(f)
    n_L = [int(n) for n in r["n_L"]]

    def curve(key):
        mean = np.array([r[key][str(n)]["mean"] for n in n_L])
        std = np.array([r[key][str(n)]["std"] for n in n_L])
        return 100.0 * (1.0 - mean), 100.0 * std   # accuracy %, std in points

    return n_L, curve("graph"), curve("ridge")


def main():
    n_L, (acc_g, sd_g), (acc_r, sd_r) = load()
    labels = np.array(n_L) * N_CLASSES          # total labeled images

    fig, ax = plt.subplots(figsize=(5.4, 3.8))

    # Ridge reference first (lighter), graph-Laplacian probe on top.
    ax.plot(labels, acc_r, "s--", color="0.55", mfc="white", ms=5, lw=1.4,
            label="ridge linear probe (reference)", zorder=2)
    ax.fill_between(labels, acc_r - sd_r, acc_r + sd_r, color="0.55", alpha=0.15, lw=0)

    ax.plot(labels, acc_g, "o-", color="#1f77b4", ms=5, lw=1.8,
            label="graph-Laplacian probe (Eq. alg)", zorder=3)
    ax.fill_between(labels, acc_g - sd_g, acc_g + sd_g, color="#1f77b4", alpha=0.18, lw=0)

    # Backbone accuracy ceiling.
    ax.axhline(BACKBONE_TEST_ACC, ls=":", color="0.3", lw=1.2)
    ax.text(labels[0], BACKBONE_TEST_ACC + 0.25,
            f"SimCLR backbone test accuracy ({BACKBONE_TEST_ACC:.1f}%)",
            fontsize=8, color="0.3", va="bottom")

    # Annotate the ~4%-of-labels operating point (n_L = 200 -> 2000 labels).
    if 200 in n_L:
        i = n_L.index(200)
        x, y = labels[i], acc_g[i]
        pct = 100.0 * x / TRAIN_SIZE
        ax.annotate(f"~{y:.0f}% accuracy\nat {pct:.0f}% of labels ({x:,} labels)",
                    xy=(x, y), xytext=(x * 1.15, y - 6.5), fontsize=8.5,
                    ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0))

    ax.set_xscale("log")
    ax.set_xlabel("number of labeled images (log scale)")
    ax.set_ylabel("transductive accuracy on unlabeled pool (%)")
    ax.set_title("CIFAR-10: frozen SimCLR features are label-efficient", fontsize=10.5)

    # Top axis: the same ticks read as a fraction of the training labels.
    sec = ax.secondary_xaxis("top",
                             functions=(lambda v: 100.0 * v / TRAIN_SIZE,
                                        lambda p: p / 100.0 * TRAIN_SIZE))
    sec.set_xlabel("% of training labels", fontsize=9)

    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(CACHE, f"label_efficiency.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
