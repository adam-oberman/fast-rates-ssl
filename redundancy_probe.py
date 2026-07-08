"""Experiment C: redundancy = identification.

The paper's identification (Theorem 3) is that a frozen SSL backbone has already
performed the graph-Laplacian smoothing our theory describes. Experiment C turns
that into a positive result: graph-Laplacian regularization should help a lot on
features that never saw the augmentation graph (raw pixels, a random-init
backbone) and be redundant on SSL features, and the size of the help should
scale with the augmentation-graph cut in that feature space.

A spectrum of feature spaces, ordered by how much they could have absorbed the
augmentation graph:

  pixel    : flattened normalized CIFAR images (3*32*32 = 3072-d), L2-normalized.
  randinit : the SAME CIFAR ResNet-50 architecture as the SSL backbone (3x3
             stride-1 conv1, no maxpool), random frozen weights (seeded), eval,
             features L2-normalized. The architecture-matched control.
  ssl      : the cached frozen SimCLR features (features.pt), reused, not
             re-extracted.

For each space, holding the protocol identical (class-balanced subsets,
L2-normalized features, k=15, N_SUBSETS, transductive error on the unlabeled
pool), we build the kNN graph and its symmetric normalized Laplacian, measure
the symmetric cut, and run the graph and ridge probes over the n_L grid. The
gain is

    Delta = err_ridge - err_graph        (best lambda, best alpha).

Credibility hinges on tuning: to claim "redundant" we give graph regularization
its best lambda on SSL features and still find no gain. So lambda (graph) and
alpha (ridge) are each swept and the best chosen per space at a representative
mid n_L.

Per-space cache files (cache/expC_<space>_*) keep the real features.pt and
knn_graph.npz untouched. Smoke runs use *_smoke caches and the 2000-image pool.

Run:
    python redundancy_probe.py --smoke    # build 3 spaces, show the per-space
                                          # table at one n_L and lambda=1.0
    python redundancy_probe.py            # tuned-lambda full grid + Delta + JSON
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

import common
import config
import extract_features
import probe_sweep

# --- Experiment C knobs (local; shared protocol knobs come from config) ------
SPACES = ["pixel", "randinit", "ssl"]
RANDINIT_SEED = 0                      # reproducible random-init backbone
LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]   # graph probe, around the old 1.0
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]    # ridge probe
DEFAULT_LAMBDA = 1.0                   # the single lambda used for the smoke table
DEFAULT_ALPHA = 1.0
RESULTS_PATH = os.path.join(config.CACHE_DIR, "redundancy_results.json")


# ---------------------------------------------------------------------------
# Per-space cache paths (never the real features.pt / knn_graph.npz)
# ---------------------------------------------------------------------------
def _sfx(smoke):
    return "_smoke" if smoke else ""


def feat_cache(space, smoke):
    return os.path.join(config.CACHE_DIR, f"expC_{space}_features{_sfx(smoke)}.pt")


def graph_cache(space, smoke):
    return os.path.join(config.CACHE_DIR, f"expC_{space}_knn{_sfx(smoke)}.npz")


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------
def build_randinit_backbone(seed):
    """SSL backbone architecture with random frozen weights (the control).

    Identical to common.build_backbone's architecture (CIFAR stem: 3x3 stride-1
    conv1, no maxpool, identity head) but never trained. Seeding before
    construction fixes the random init so the features are reproducible.
    """
    import torchvision.models as tvm

    common.set_seed(seed)
    net = tvm.resnet50(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval().to(config.DEVICE)
    return net


def _cifar_loader(smoke):
    ds = extract_features.ensure_cifar()
    if smoke:
        ds = torch.utils.data.Subset(ds, range(2000))
    return torch.utils.data.DataLoader(
        ds, batch_size=config.EXTRACT_BATCH, shuffle=False, num_workers=2
    )


def extract_space(space, smoke):
    """Return L2-normalized features (torch float [N, d]) for a non-ssl space."""
    loader = _cifar_loader(smoke)
    net = build_randinit_backbone(RANDINIT_SEED) if space == "randinit" else None
    feats = []
    for i, (imgs, _lab) in enumerate(loader):
        if space == "pixel":
            z = imgs.flatten(1).to(config.DEVICE)
            z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
            feats.append(z.cpu())
        else:  # randinit
            feats.append(common.embed(net, imgs).cpu())
        if i % 20 == 0:
            print(f"  [{space}] batch {i}/{len(loader)}")
    return torch.cat(feats).float()


def get_features(space, smoke):
    """(features, labels) for a space; features cached per-space, labels shared.

    Labels come from the smoke-aware config.LABELS_PATH; image order is
    deterministic (shuffle=False) so the same row index is the same image in
    every space, which is what makes the spaces apples-to-apples.
    """
    labels = torch.load(config.LABELS_PATH).long()
    if space == "ssl":
        feats, ssl_labels = common.load_features()
        assert torch.equal(ssl_labels.long(), labels), "ssl label order mismatch"
        return feats.float(), labels

    path = feat_cache(space, smoke)
    if os.path.exists(path):
        feats = torch.load(path)
        if feats.shape[0] == labels.shape[0]:
            return feats.float(), labels
    feats = extract_space(space, smoke)
    torch.save(feats, path)
    return feats, labels


def get_graph(space, feats, smoke):
    """Symmetric kNN affinity W (CSR) for a space, cached per-space."""
    path = graph_cache(space, smoke)
    if os.path.exists(path):
        W = sp.load_npz(path)
        if W.shape[0] == feats.shape[0]:
            return W
    W = common.build_knn_graph(feats, config.KNN_K)
    sp.save_npz(path, W)
    return W


def build_space(space, smoke):
    """Assemble everything a space needs: features, labels, affinity, Laplacian, cut."""
    t0 = time.time()
    feats, labels = get_features(space, smoke)
    W = get_graph(space, feats, smoke)
    L = common.normalized_laplacian(W)
    cut = common.graph_cut_rda(W, labels.numpy())["symmetric"]
    print(f"[{space:8s}] feats {tuple(feats.shape)}  nnz={W.nnz}  "
          f"cut_sym={cut:.4f}  ({time.time() - t0:.1f}s)")
    return {"feats": feats, "labels": labels, "W": W, "L": L, "cut": cut}


# ---------------------------------------------------------------------------
# Probe evaluation (reuses probe_sweep estimators unchanged)
# ---------------------------------------------------------------------------
def probe_errors(space_data, n_L, lam, alpha, n_subsets, seed):
    """Mean (graph_err, ridge_err) over class-balanced subsets at fixed n_L.

    The same seed gives identical labeled subsets across spaces and lambdas, so
    every comparison is on the same draws.
    """
    feats_np = space_data["feats"].numpy()
    labels = space_data["labels"]
    L = space_data["L"]
    m = feats_np.shape[0]
    rng = np.random.default_rng(seed)
    g_errs, r_errs = [], []
    for _ in range(n_subsets):
        labeled_idx = common.sample_labeled_indices(
            labels, n_L, config.N_CLASSES, rng)
        unlabeled_mask = np.ones(m, dtype=bool)
        unlabeled_mask[labeled_idx.numpy()] = False
        g_errs.append(probe_sweep.graph_probe_error(
            L, labeled_idx, labels, unlabeled_mask, lam, config.N_CLASSES))
        r_errs.append(probe_sweep.ridge_probe_error(
            feats_np, labeled_idx, labels, unlabeled_mask, alpha, config.N_CLASSES))
    return float(np.mean(g_errs)), float(np.mean(r_errs))


def per_space_table(spaces, n_L, lam, alpha, n_subsets, seed):
    """The milestone table: space x {cut, graph err, ridge err, Delta}."""
    print(f"\nPer-space table at n_L={n_L}, lambda={lam}, alpha={alpha}, "
          f"n_subsets={n_subsets}")
    print(f"  {'space':8s} {'cut_sym':>8} {'graph_err':>10} {'ridge_err':>10} "
          f"{'Delta':>9}")
    rows = []
    for space, data in spaces.items():
        g, r = probe_errors(data, n_L, lam, alpha, n_subsets, seed)
        delta = r - g
        rows.append({"space": space, "cut_sym": data["cut"],
                     "graph_err": g, "ridge_err": r, "delta": delta})
        print(f"  {space:8s} {data['cut']:8.4f} {g:10.4f} {r:10.4f} "
              f"{delta:+9.4f}")
    print("  (Delta = ridge - graph; positive means graph regularization helps)")
    return rows


# ---------------------------------------------------------------------------
# Tuned-lambda full grid (run only after the smoke table is approved)
# ---------------------------------------------------------------------------
def tune(space_data, mid_n_L, n_subsets, seed):
    """Best lambda (graph) and best alpha (ridge) by mean error at mid n_L."""
    best_lam, best_lam_err = None, np.inf
    best_alpha, best_alpha_err = None, np.inf
    for lam in LAMBDA_GRID:
        g, _ = probe_errors(space_data, mid_n_L, lam, DEFAULT_ALPHA, n_subsets, seed)
        if g < best_lam_err:
            best_lam, best_lam_err = lam, g
    for alpha in ALPHA_GRID:
        _, r = probe_errors(space_data, mid_n_L, DEFAULT_LAMBDA, alpha, n_subsets, seed)
        if r < best_alpha_err:
            best_alpha, best_alpha_err = alpha, r
    return best_lam, best_alpha, best_lam_err, best_alpha_err


def full_run(spaces, smoke):
    n_subsets = config.N_SUBSETS
    n_L_grid = config.N_L_VALUES
    mid_n_L = n_L_grid[len(n_L_grid) // 2]
    print(f"\nTuning at mid n_L={mid_n_L} (lambda grid {LAMBDA_GRID}, "
          f"alpha grid {ALPHA_GRID})")
    results = {"meta": {"knn_k": config.KNN_K, "n_subsets": n_subsets,
                        "n_L": n_L_grid, "mid_n_L": mid_n_L,
                        "lambda_grid": LAMBDA_GRID, "alpha_grid": ALPHA_GRID,
                        "seed": config.MASTER_SEED}, "spaces": {}}
    for space, data in spaces.items():
        best_lam, best_alpha, le, ae = tune(data, mid_n_L, n_subsets, config.MASTER_SEED)
        print(f"[{space:8s}] best lambda={best_lam} (err {le:.4f}), "
              f"best alpha={best_alpha} (err {ae:.4f})")
        curve = {"cut_sym": data["cut"], "best_lambda": best_lam,
                 "best_alpha": best_alpha, "graph": {}, "ridge": {}, "delta": {}}
        for n_L in n_L_grid:
            g, r = probe_errors(data, n_L, best_lam, best_alpha,
                                n_subsets, config.MASTER_SEED + 1000 + n_L)
            curve["graph"][n_L] = g
            curve["ridge"][n_L] = r
            curve["delta"][n_L] = r - g
            print(f"    n_L={n_L:4d}  graph={g:.4f}  ridge={r:.4f}  "
                  f"Delta={r - g:+.4f}")
        results["spaces"][space] = curve
    if not smoke:
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print("saved", RESULTS_PATH)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(smoke=False, full=False):
    if smoke:
        config.apply_smoke()
        print("[smoke] 2000-image pool, *_smoke caches, reduced sweeps")
    common.set_seed(config.MASTER_SEED)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    print(f"building {len(SPACES)} feature spaces: {SPACES}")
    spaces = {s: build_space(s, smoke) for s in SPACES}

    if full and not smoke:
        full_run(spaces, smoke)
    else:
        # Milestone: per-space table at one n_L and one lambda, before the grid.
        n_L = config.N_L_VALUES[-1] if smoke else 100
        per_space_table(spaces, n_L, DEFAULT_LAMBDA, DEFAULT_ALPHA,
                        config.N_SUBSETS, config.MASTER_SEED)
        if not smoke:
            print("\n(table only; pass --full to run the tuned-lambda grid)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="run the tuned-lambda grid (default is the table only)")
    main(**vars(ap.parse_args()))
