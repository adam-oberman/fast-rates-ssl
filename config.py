"""Central configuration for the Section 7 experiment.

Target environment: Google Colab with Google Drive mounted. Every artifact is
cached under DRIVE_DIR so that reruns and Colab reconnects are instant. All
five scripts import from here, so this is the single place to change paths,
the n_L sweep, seeds, and hyperparameters.

The experiment checks the two predictions of Theorem 2:
  (i)  transductive error falls at the fast rate ~1/n_L in the labeled count,
  (ii) it plateaus at a floor equal to the data-augmentation alignment error
       R_DA(y).
"""
from __future__ import annotations

import os
import torch

# --- Paths (local) ----------------------------------------------------------
# Everything caches next to the code, under ./cache and ./data, so reruns are
# instant and nothing depends on an external mount.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CIFAR_ROOT = os.path.join(BASE_DIR, "data")  # CIFAR-10 downloads/extracts here

FEATURES_PATH = os.path.join(CACHE_DIR, "features.pt")
LABELS_PATH = os.path.join(CACHE_DIR, "labels.pt")
GRAPH_PATH = os.path.join(CACHE_DIR, "knn_graph.npz")
PROBE_RESULTS = os.path.join(CACHE_DIR, "probe_results.json")
BASELINE_RESULTS = os.path.join(CACHE_DIR, "baseline_results.json")
RDA_RESULTS = os.path.join(CACHE_DIR, "rda_results.json")
FIGURE_PATH = os.path.join(CACHE_DIR, "accuracy_vs_labels.pdf")

# --- Backbone ---------------------------------------------------------------
BACKBONE_REPO = "edadaltocg/resnet50_simclr_cifar10"
BACKBONE_FILE = "pytorch_model.bin"
FEATURE_DIM = 2048

# CIFAR-10 channel statistics, used by every transform that feeds the backbone.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

# --- Experiment sweep -------------------------------------------------------
# Extended to large n_L so the rate and the plateau are unambiguous. CIFAR-10
# has 5000 labels/class, so even n_L=2000 (20k labeled) leaves a 30k unlabeled
# pool for the transductive evaluation.
N_L_VALUES = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]  # labels PER CLASS
N_CLASSES = 10
N_SUBSETS = 20      # random class-balanced labeled subsets averaged per n_L
MASTER_SEED = 0

# --- Graph-Laplacian probe (Eq. alg) ----------------------------------------
KNN_K = 15           # neighbors in the augmentation-graph proxy
LAPLACIAN_REG = 1.0  # lambda in  min ||f - y||^2_S + lambda * f^T L f
CG_TOL = 1e-5
CG_MAXITER = 2000

# --- Ridge reference probe --------------------------------------------------
RIDGE_ALPHA = 1.0

# --- Supervised-from-scratch baseline ---------------------------------------
BASELINE_EPOCHS = 60
BASELINE_LR = 0.05          # SGD + momentum + cosine schedule (CIFAR recipe)
BASELINE_MOMENTUM = 0.9
BASELINE_WD = 5e-4
BASELINE_LABEL_SMOOTH = 0.1
BASELINE_BATCH = 128
BASELINE_SEEDS = 3
BASELINE_EVAL_CAP = 10000  # cap unlabeled-pool eval size for speed (0 = all)

# --- R_DA estimator ---------------------------------------------------------
RDA_BANK_PER_CLASS = 100   # labeled bank size per class
RDA_N_ANCHORS = 2000       # labeled images used as augmentation seeds
RDA_VIEWS_PER_ANCHOR = 8   # augmentation draws per anchor
RDA_KNN = 15               # neighbors among the labeled bank
AUG_STRENGTHS = ["weak", "aggressive"]

# --- Device + threads -------------------------------------------------------
# This Intel iMac has no GPU PyTorch can use (CUDA is NVIDIA-only; MPS is Apple
# Silicon only), so DEVICE resolves to cpu here. The cuda -> mps -> cpu order
# keeps the code portable to other machines.
def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick_device()

# On CPU, use every core for the matmuls in extraction and the graph build.
if DEVICE.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 1)

# --- Memory / batch knobs (sized for 8 GB RAM) ------------------------------
EXTRACT_BATCH = 128   # images per forward pass in feature extraction
GRAPH_BATCH = 1024    # rows per similarity block when building the kNN graph

# --- Smoke test -------------------------------------------------------------
# Every script accepts --smoke, which shrinks the work so the whole pipeline
# runs end to end in well under a minute before you commit to the full sweep.
SMOKE = {
    "N_L_VALUES": [5, 20],
    "N_SUBSETS": 3,
    "BASELINE_EPOCHS": 2,
    "BASELINE_SEEDS": 1,
    "BASELINE_EVAL_CAP": 2000,
    "RDA_N_ANCHORS": 200,
    "RDA_VIEWS_PER_ANCHOR": 4,
    "RDA_BANK_PER_CLASS": 50,
}


def apply_smoke() -> None:
    """Overwrite module globals with their smoke-test values, in place.

    Smoke runs also use separate cache files so a quick sanity pass never
    clobbers the full 50,000-image features or the real kNN graph.
    """
    g = globals()
    for key, value in SMOKE.items():
        g[key] = value
    g["FEATURES_PATH"] = os.path.join(CACHE_DIR, "features_smoke.pt")
    g["LABELS_PATH"] = os.path.join(CACHE_DIR, "labels_smoke.pt")
    g["GRAPH_PATH"] = os.path.join(CACHE_DIR, "knn_graph_smoke.npz")
