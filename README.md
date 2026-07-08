# Section 7 experiment code

Code behind Section 7 of *Fast Rates for Semi-Supervised Learning via Data-Augmentation Graph Regularization*. The finished Section 7 is the synthetic Experiment A (three panels: additive floor vs cut, leave-one-out stability at slope $-1.00$, the downstream envelope) plus a descriptive CIFAR label-efficiency panel. The rate is shown via the stability lemma (slope $-1.00$), not the shallow excess-risk slope, and there is no plateau-at-$R_{\mathrm{DA}}$ claim.

Target environment is a local Python install on an Intel iMac, CPU only (PyTorch cannot use the AMD GPU on macOS, and MPS is Apple Silicon only). Everything caches under `./cache`, so reruns are instant.

## Scripts

Current Section 7 (all torch-free):

- `synthetic_verify.py` : Experiment A tests and numbers.
- `fig_pipeline.py` : renders the three synthetic panels (`synthetic_cut/stability/rate.pdf`, $m=6000$) and copies them into `../PAPER/figures/`.
- `make_label_efficiency_panel.py` : the CIFAR label-efficiency panel, built from `cache/probe_results.json`.
- `rate_diagnostic.py` : the record behind the stability / ${tr}\mathbf K$ rate wording. Keep it.

The CIFAR feature/probe pipeline (it produced `cache/probe_results.json`, the input to the label-efficiency panel) and two shared modules:

- `config.py` : all paths, the $n_L$ sweep, seeds, and hyperparameters. One place to edit.
- `common.py` : backbone build, feature loading, class-balanced sampling, the kNN graph and its normalized Laplacian, the version-safe CG solve, log-log slope fitting.
- `extract_features.py` : frozen SimCLR features for all 50,000 training images, L2-normalized, cached.
- `probe_sweep.py` : the graph-Laplacian probe (Eq. alg) and a ridge reference probe, transductive error vs $n_L$.
- `measure_rda.py` : the $R_{\mathrm{DA}}(y)$ nearest-neighbor estimator with the input-space circularity check. Kept for reference; no longer overlaid as a plateau.
- `redundancy_probe.py` : the set-aside Experiment C.

Kept in the repo but not run or plotted: `baseline.py` (the from-scratch CNN; the slow comparator is a $1/\sqrt{n_L}$ reference line, not a trained net) and `make_figure.py` (the old single-figure plateau script, superseded by `fig_pipeline.py` and `make_label_efficiency_panel.py`).

## Environment

The torch `.venv` was rebuilt on 18 June 2026 and now exists (Python 3.11 via Homebrew, `torch==2.2.2`, `numpy==1.26.x`). The pins matter on Intel macOS: `torch==2.2.2` is the last release with Intel-Mac wheels, and it predates NumPy 2, so `numpy<2` is pinned to avoid an import-time crash ("Numpy is not available"). The system Python (3.14) cannot run torch 2.2.2.

A second torch-free venv `.venv-numpy` (numpy, scipy, matplotlib) is enough for the synthetic panels (`fig_pipeline.py`, `synthetic_verify.py`) and the CIFAR panel (`make_label_efficiency_panel.py`).

If the venvs are lost (they are not synced via iCloud), rebuild torch from inside this `Code` folder:

```
brew install python@3.11
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Confirm the stack is healthy:

```
python -c "import torch, numpy as np; print(torch.__version__, np.__version__); print(torch.from_numpy(np.ones(3)).sum().item())"
```

CIFAR-10 downloads itself into `./data` on the first run, so there is no manual data step. Re-activate the venv with `source .venv/bin/activate` (or `.venv-numpy`) each Terminal session before running anything.

## Run order

Smoke-test first. Every CIFAR script takes `--smoke`, which shrinks the sweep and uses a separate `*_smoke` cache, so it finishes in about a minute and never touches the real artifacts:

```
./run.sh smoke
```

The synthetic figures are torch-free and quick (this is all Section 7 needs to rebuild):

```
python fig_pipeline.py --part stab && python fig_pipeline.py --part cutenv && python fig_pipeline.py --part render
```

This prints the stability slope near $-1.00$ and the cut correlation near $0.99$, and refreshes the three PDFs in `../PAPER/figures/`.

The CIFAR feature/probe pipeline is only needed to regenerate `cache/probe_results.json`. Extraction is the one slow step on CPU and is done once; the rest read its cache:

```
python extract_features.py            # ~15-25 min on this CPU, then cached forever
python probe_sweep.py                 # builds the kNN graph once, caches it
python measure_rda.py                 # optional; R_DA reference numbers
python make_label_efficiency_panel.py # the CIFAR panel from the cached probe results
```

## What lands in ./cache

`features.pt`, `labels.pt`, `knn_graph.npz`, `probe_results.json`, `rda_results.json`, and the figure PDFs and PNGs. The `.gitignore` keeps the big binaries (`*.pt`, `*.npz`) and the dataset out of git but tracks the JSON results and the figures. Delete any JSON or the graph to force that step to recompute; delete `features.pt` only if you change the backbone. Smoke runs write `*_smoke` copies you can ignore or delete.

## Method notes

The graph estimator solves the transductive system $(J + \lambda L)F = JY$ on the symmetric normalized Laplacian $L$ of the kNN graph, one conjugate-gradient solve per class, then predicts the argmax. Error is always measured on the fixed unlabeled pool, never a held-out test set, to match the transductive statement. Features are L2-normalized before both the probe and the kNN, so cosine similarity is just the dot product.

