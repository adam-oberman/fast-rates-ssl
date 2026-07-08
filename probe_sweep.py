"""Step 2: probe sweep. Transductive error versus the number of labels n_L.

Two estimators on the frozen features, both evaluated on the FIXED unlabeled
pool (the points not in the labeled subset), to match the transductive theory.

  graph : the graph-Laplacian-regularized classifier of Eq. (alg). On the
          symmetric-normalized Laplacian L of the kNN augmentation-graph proxy,
          solve, with J the diagonal selector of labeled rows and Y one-hot,
              (J + lambda * L) F = J Y,
          one conjugate-gradient solve per class, then predict argmax over
          classes. Theorem 2 predicts this curve falls at ~1/n_L and plateaus
          at the data-augmentation alignment error R_DA(y).

  ridge : a plain ridge linear probe fit on the labeled features only. A
          simpler reference curve; not expected to share the graph floor.

Each n_L is averaged over N_SUBSETS class-balanced labeled subsets, drawn from
the master seed, and we report mean and standard deviation of the error.

Run:
    !python probe_sweep.py            # full sweep
    !python probe_sweep.py --smoke    # 2 values of n_L, 3 subsets
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import torch

import common
import config


def graph_probe_error(L, labeled_idx, labels, unlabeled_mask, lam, n_classes):
    """Transductive error of the Eq. (alg) estimator on the unlabeled pool."""
    m = L.shape[0]
    labeled_np = labeled_idx.numpy()
    J = np.zeros(m)
    J[labeled_np] = 1.0
    A = (sp.diags(J) + lam * L).tocsr()  # SPD when each component has a label

    y = labels.numpy()
    scores = np.zeros((m, n_classes))
    for c in range(n_classes):
        b = np.zeros(m)
        b[labeled_np] = (y[labeled_np] == c).astype(float)
        scores[:, c] = common.cg_solve(A, b)
    pred = scores.argmax(axis=1)
    return float((pred[unlabeled_mask] != y[unlabeled_mask]).mean())


def ridge_probe_error(feats_np, labeled_idx, labels, unlabeled_mask, alpha, n_classes):
    """Transductive error of a ridge linear probe trained on labeled features.

    feats_np is the feature matrix as a numpy array, passed in once so it is not
    reconverted on every subset. Predictions are computed for all points with a
    single (m x d)(d x C) matmul, which avoids copying the large unlabeled block
    out of the array on each call (the memory-friendly choice on 8 GB).
    """
    XL = feats_np[labeled_idx.numpy()]
    yL = labels[labeled_idx].numpy()
    YL = np.eye(n_classes)[yL]
    d = XL.shape[1]
    Wd = np.linalg.solve(XL.T @ XL + alpha * np.eye(d), XL.T @ YL)  # (d, C)
    pred_all = (feats_np @ Wd).argmax(axis=1)
    y = labels.numpy()
    return float((pred_all[unlabeled_mask] != y[unlabeled_mask]).mean())


def main(smoke: bool = False) -> None:
    if smoke:
        config.apply_smoke()
    common.set_seed(config.MASTER_SEED)

    feats, labels = common.load_features()
    feats_np = feats.numpy()  # converted once; reused by the ridge probe
    m = feats.shape[0]
    print(f"pool: {m} points, {config.FEATURE_DIM}-d features")

    t0 = time.time()
    L = common.load_or_build_graph(feats, config.KNN_K)
    print(f"graph ready ({time.time() - t0:.1f}s), nnz={L.nnz}")

    rng = np.random.default_rng(config.MASTER_SEED)
    results = {"n_L": config.N_L_VALUES, "graph": {}, "ridge": {},
               "meta": {"knn_k": config.KNN_K, "lambda": config.LAPLACIAN_REG,
                        "ridge_alpha": config.RIDGE_ALPHA, "n_subsets": config.N_SUBSETS}}

    for n_L in config.N_L_VALUES:
        graph_errs, ridge_errs = [], []
        for s in range(config.N_SUBSETS):
            labeled_idx = common.sample_labeled_indices(
                labels, n_L, config.N_CLASSES, rng
            )
            unlabeled_mask = np.ones(m, dtype=bool)
            unlabeled_mask[labeled_idx.numpy()] = False

            graph_errs.append(graph_probe_error(
                L, labeled_idx, labels, unlabeled_mask,
                config.LAPLACIAN_REG, config.N_CLASSES))
            ridge_errs.append(ridge_probe_error(
                feats_np, labeled_idx, labels, unlabeled_mask,
                config.RIDGE_ALPHA, config.N_CLASSES))

        results["graph"][n_L] = {"mean": float(np.mean(graph_errs)),
                                 "std": float(np.std(graph_errs))}
        results["ridge"][n_L] = {"mean": float(np.mean(ridge_errs)),
                                 "std": float(np.std(ridge_errs))}
        print(f"n_L={n_L:4d}  graph={np.mean(graph_errs):.4f}  "
              f"ridge={np.mean(ridge_errs):.4f}")

    if not smoke:
        with open(config.PROBE_RESULTS, "w") as f:
            json.dump(results, f, indent=2)
        print("saved", config.PROBE_RESULTS)

    # Report fitted slopes over the descending part of each curve.
    for method in ("graph", "ridge"):
        xs = config.N_L_VALUES
        ys = [results[method][n]["mean"] for n in xs]
        slope, _ = common.fit_loglog_slope(xs, ys)
        print(f"{method} log-log slope (all points): {slope:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(**vars(ap.parse_args()))
