"""Shared utilities for the Section 7 experiment scripts.

Nothing here runs on import. The heavy objects (backbone, kNN graph) are built
by explicit calls so each script controls when work happens.
"""
from __future__ import annotations

import inspect
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

import config


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
def build_backbone() -> nn.Module:
    """Frozen CIFAR ResNet-50 SimCLR backbone with the fc head removed.

    Reproduces the architecture the checkpoint was trained with: a 3x3
    stride-1 conv1 and no maxpool (the CIFAR stem), the ResNet-50 body, and an
    identity head. Returns an eval-mode module on config.DEVICE with gradients
    disabled. The assertions are the load-time sanity check from the notebook:
    only fc is allowed to be missing, and nothing should be unexpected.
    """
    import torchvision.models as tvm
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(repo_id=config.BACKBONE_REPO, filename=config.BACKBONE_FILE)
    state = torch.load(ckpt, map_location="cpu")

    net = tvm.resnet50(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()

    missing, unexpected = net.load_state_dict(state, strict=False)
    assert unexpected == [], f"unexpected checkpoint keys: {unexpected}"
    assert set(missing) <= {"fc.weight", "fc.bias"}, f"unexpected missing keys: {missing}"

    net.fc = nn.Identity()
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval().to(config.DEVICE)
    return net


@torch.no_grad()
def embed(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Embed a batch of normalized images and L2-normalize the features."""
    z = backbone(images.to(config.DEVICE))
    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return z


# ---------------------------------------------------------------------------
# Cached features
# ---------------------------------------------------------------------------
def load_features() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the cached (features, labels) produced by extract_features.py."""
    if not (os.path.exists(config.FEATURES_PATH) and os.path.exists(config.LABELS_PATH)):
        raise FileNotFoundError(
            "Cached features not found. Run extract_features.py first."
        )
    feats = torch.load(config.FEATURES_PATH).float()
    labels = torch.load(config.LABELS_PATH).long()
    # Defensive re-normalization; extract_features already L2-normalizes.
    feats = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return feats, labels


# ---------------------------------------------------------------------------
# Class-balanced labeled subsets
# ---------------------------------------------------------------------------
def sample_labeled_indices(
    labels: torch.Tensor, n_per_class: int, n_classes: int, rng: np.random.Generator
) -> torch.Tensor:
    """Exactly n_per_class indices drawn from each class, without replacement.

    rng is a numpy Generator so the draw is reproducible from the master seed.
    Returns a 1-D LongTensor of labeled indices into the pool.
    """
    labels_np = labels.numpy()
    chosen = []
    for c in range(n_classes):
        idx_c = np.where(labels_np == c)[0]
        chosen.append(rng.choice(idx_c, size=n_per_class, replace=False))
    return torch.from_numpy(np.concatenate(chosen)).long()


# ---------------------------------------------------------------------------
# kNN augmentation-graph proxy and its normalized Laplacian
# ---------------------------------------------------------------------------
def build_knn_graph(feats: torch.Tensor, k: int, batch: int = None):
    """Symmetric kNN affinity over all pool points, as a scipy CSR matrix.

    feats is (m, d) and L2-normalized, so cosine similarity equals the dot
    product. For each node we keep its k nearest neighbors (self excluded),
    clip negative cosines to zero, then symmetrize via W = max(W, W^T). This is
    the practical stand-in for the augmentation graph of Section 2: two images
    are joined when their features are close, which is exactly where the
    backbone maps augmentation-related views (Theorem 3).

    Rows are processed in blocks of `batch` (config.GRAPH_BATCH) so the full
    m x m similarity is never materialized, which keeps peak memory low.
    """
    import scipy.sparse as sp

    if batch is None:
        batch = config.GRAPH_BATCH
    m = feats.shape[0]
    F = feats.to(config.DEVICE)
    rows, cols, vals = [], [], []
    for start in range(0, m, batch):
        end = min(start + batch, m)
        sims = F[start:end] @ F.T  # (b, m)
        # Exclude self-similarity before taking the top-k.
        ar = torch.arange(start, end, device=F.device)
        sims[torch.arange(end - start, device=F.device), ar] = -2.0
        topv, topi = sims.topk(k, dim=1)
        r_idx = ar.repeat_interleave(k)
        rows.append(r_idx.cpu())
        cols.append(topi.reshape(-1).cpu())
        vals.append(topv.reshape(-1).cpu())
    rows = torch.cat(rows).numpy()
    cols = torch.cat(cols).numpy()
    vals = torch.cat(vals).numpy().clip(min=0.0)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(m, m))
    W = W.maximum(W.T)  # undirected graph
    return W


def normalized_laplacian(W):
    """Symmetric normalized Laplacian L = I - D^{-1/2} W D^{-1/2} (CSR).

    This is the L_S normalization that Theorem 2 is stated for.
    """
    import scipy.sparse as sp

    deg = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    Dinv = sp.diags(dinv)
    n = W.shape[0]
    return (sp.identity(n, format="csr") - Dinv @ W @ Dinv).tocsr()


def load_or_build_affinity(feats: torch.Tensor, k: int):
    """Return the symmetric kNN affinity W (CSR), caching it to GRAPH_PATH."""
    import scipy.sparse as sp

    if os.path.exists(config.GRAPH_PATH):
        W = sp.load_npz(config.GRAPH_PATH)
        if W.shape[0] == feats.shape[0]:
            return W
    W = build_knn_graph(feats, k)
    os.makedirs(os.path.dirname(config.GRAPH_PATH), exist_ok=True)
    sp.save_npz(config.GRAPH_PATH, W)
    return W


def load_or_build_graph(feats: torch.Tensor, k: int):
    """Return the symmetric normalized Laplacian of the cached affinity."""
    return normalized_laplacian(load_or_build_affinity(feats, k))


def graph_cut_rda(W, labels) -> dict:
    """R_DA as the normalized cut of the probe's own kNN graph.

    This is the matched floor: the fraction of the graph's edge mass that runs
    between differently labeled nodes, on the exact graph the probe regularizes
    on. Two normalizations are returned. `plain` is the raw edge-mass fraction;
    `symmetric` weights each edge by 1/sqrt(deg_i deg_j), i.e. it is the
    off-diagonal mass of the symmetric normalized affinity D^{-1/2} W D^{-1/2},
    matching the L_S normalization in Theorem 2. Theorem 2 predicts the graph
    probe plateaus at `symmetric`.
    """
    coo = W.tocoo()
    i, j, w = coo.row, coo.col, coo.data
    y = np.asarray(labels)
    deg = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    w_sym = w * dinv[i] * dinv[j]
    cross = y[i] != y[j]
    plain = float(w[cross].sum() / max(w.sum(), 1e-12))
    symmetric = float(w_sym[cross].sum() / max(w_sym.sum(), 1e-12))
    return {"plain": plain, "symmetric": symmetric}


# ---------------------------------------------------------------------------
# Sparse CG that works across scipy versions (tol vs rtol)
# ---------------------------------------------------------------------------
def cg_solve(A, b):
    """Conjugate-gradient solve of A x = b for symmetric positive-definite A."""
    from scipy.sparse.linalg import cg

    kwargs = {"maxiter": config.CG_MAXITER}
    params = inspect.signature(cg).parameters
    if "rtol" in params:  # scipy >= 1.12
        kwargs["rtol"] = config.CG_TOL
    else:
        kwargs["tol"] = config.CG_TOL
    x, _info = cg(A, b, **kwargs)
    return x


# ---------------------------------------------------------------------------
# Log-log slope fitting
# ---------------------------------------------------------------------------
def fit_loglog_slope(x, y) -> Tuple[float, float]:
    """Least-squares slope and intercept of log(y) against log(x)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(intercept)
