"""Experiment A: controlled synthetic verification of Theorem 2.

A balanced planted-partition (stochastic block) graph is a model where every
object the theorem references is known in closed form: the labels, the weights,
the S-normalized Laplacian L_S, and therefore cut(L_S, y). The loss is squared
loss (a = 1) and lambda is set by us, so the prefactor lambda/a is known. This
lets us probe the two things CIFAR could not: how the asymptotic error depends
on the cut (the additive-vs-multiplicative question), and the n_L rate against
an EXACT, computed (not fitted) floor.

ESTIMATOR (Eq. alg, exactly as in PAPER/sections/02-setup.tex). The objective is
the *averaged* labeled loss plus the kernel regularizer,

    f = argmin_g  (1/n_L) sum_{i in L} (g_i - y_i)^2  +  lambda * g^T K^{-1} g ,
        with  K^{-1} = alpha S^{-1} + L_S .

The averaging by n_L is load-bearing: the stationarity condition is

    ( J + lambda * n_L * K^{-1} ) f_c = J Y_c ,                       (estimator)

so the regularizer scales with the labeled count n_L. The full-label oracle in
the theorem averages over all m nodes, giving

    ( I + lambda * m * K^{-1} ) f_c = Y_c .                           (oracle)

One CG solve per class; predict by argmax; read error on the unlabeled nodes.
At the label indicator the regularizer energy is lambda*(alpha*s + cut), the
proof's (1/a)*lambda*(alpha*s + cut) term, so the additive floor prediction is
(lambda/a)*cut with the label-independent alpha*s folding into the rate term.

TWO METRICS, read from the same score matrix F, because the theorem bounds the
squared loss and the figures elsewhere report 0/1 error, and they behave
differently:
  - 0/1 error : fraction of argmax(F) != y. The classification error.
  - sq risk   : mean_i sum_c (F[i,c] - 1[y_i=c])^2. The squared (admissible)
                loss the estimator trains on and the theorem bounds.

THREE TESTS (see EXPERIMENT_PLAN.md, Experiment A):
  1. RATE.  Fix epsilon. Sweep n_L; plot excess squared risk above the exact
     oracle floor on log-log, and also report the 0/1 error decay.
  2. CUT.   Sweep epsilon; plot the full-label oracle floor against the known
     cut(L_S, y), in BOTH metrics. A line through the origin in the squared-loss
     risk is the ADDITIVE form; a 0/1 floor that stays ~0 is MULTIPLICATIVE.
  3. STABILITY (optional).  Spot-check |f(x_i) - f^{\\i}(x_i)|, the quantity the
     stability lemma bounds by sigma*K_ii/(2 lambda n_L). Pass --stability.

Pure numpy/scipy, runs in seconds. The three numerical helpers (normalized
Laplacian, version-safe CG solve, log-log slope fit) are inlined rather than
imported from common.py, which imports torch at module load for the CIFAR
pipeline; this script is deliberately torch-free so it runs standalone. The
inlined helpers are the same algorithms as their common.py counterparts.

Run:
    python synthetic_verify.py --smoke    # tiny graph, both tests, seconds
    python synthetic_verify.py            # full defaults
    python synthetic_verify.py --stability
"""
from __future__ import annotations

import argparse
import inspect
import os
import random

import numpy as np
import scipy.sparse as sp

# Figures land next to the other cached artifacts.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


# ---------------------------------------------------------------------------
# Numerical helpers (mirrors of the pure-numpy parts of common.py)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalized_laplacian(W):
    """Symmetric normalized Laplacian L_S = I - D^{-1/2} W D^{-1/2} (CSR)."""
    deg = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    Dinv = sp.diags(dinv)
    n = W.shape[0]
    return (sp.identity(n, format="csr") - Dinv @ W @ Dinv).tocsr()


def cg_solve(A, b, tol=1e-8, maxiter=5000):
    """Conjugate-gradient solve of A x = b for symmetric positive-definite A.

    The rtol/tol keyword shim keeps it working across scipy versions, exactly
    as common.cg_solve does.
    """
    from scipy.sparse.linalg import cg

    kwargs = {"maxiter": maxiter}
    params = inspect.signature(cg).parameters
    if "rtol" in params:        # scipy >= 1.12
        kwargs["rtol"] = tol
    else:
        kwargs["tol"] = tol
    x, _info = cg(A, b, **kwargs)
    return x


def fit_loglog_slope(x, y):
    """Least-squares slope and intercept of log(y) against log(x)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# Parameters (self-contained; only CACHE_DIR is shared with the CIFAR config).
# n_L below always means labels PER CLASS; the total labeled count is K*n_L,
# which is the n_L of the paper's averaged loss (it averages over |S^L|).
# ---------------------------------------------------------------------------
PARAMS = {
    "M": 3000,            # nodes (balanced over K classes)
    "K": 10,              # classes, to mirror CIFAR-10
    "P_IN": 0.08,         # within-class edge probability (mean in-degree ~24)
    "SEED": 0,            # master seed; graphs and label draws derive from it

    # Estimator (Eq. alg). a = 1 for squared loss, so lambda/a = LAMBDA. The
    # regularizer enters as lambda*N (estimator) and lambda*m (oracle), N=K*n_L.
    "LAMBDA": 1e-3,       # the averaged-loss regularization strength
    "ALPHA": 1e-3,        # S^{-1} coefficient in K^{-1}; folds into the rate term

    # Test 1 (rate): one graph at a fixed cross-class strength, n_L swept.
    "EPS_RATE": 0.03,     # p_out = EPS_RATE * P_IN
    "N_PER_CLASS": [2, 5, 10, 20, 50, 100, 150, 200],
    "N_SUBSETS": 15,      # class-balanced labeled subsets averaged per n_L

    # Test 2 (cut-dependence): a sweep of cross-class strengths.
    "EPS_GRID": [0.005, 0.01, 0.02, 0.04, 0.07],
}

SMOKE = {
    "M": 600,
    "P_IN": 0.15,
    "N_PER_CLASS": [2, 5, 20],
    "N_SUBSETS": 3,
    "EPS_GRID": [0.01, 0.04],
}


# ---------------------------------------------------------------------------
# Planted-partition (stochastic block) graph
# ---------------------------------------------------------------------------
def build_sbm(m, K, p_in, p_out, rng):
    """Balanced SBM: m nodes, K equal blocks, within-prob p_in, cross-prob p_out.

    Returns (W, y) with W a symmetric unweighted CSR adjacency (no self-loops)
    and y the block labels. Balance (Assumption 2) holds by construction:
    exactly m/K nodes per class. The realized cut is computed from W directly,
    so it is known exactly regardless of the random draw.
    """
    assert m % K == 0, "m must be divisible by K for a balanced graph"
    per = m // K
    y = np.repeat(np.arange(K), per)

    same = y[:, None] == y[None, :]            # (m, m) block-membership mask
    R = rng.random((m, m))
    A = np.where(same, R < p_in, R < p_out)    # Bernoulli edges per block pair
    A = np.triu(A, 1)                          # keep upper triangle, no diagonal
    A = A | A.T                                # symmetrize
    W = sp.csr_matrix(A.astype(float))
    return W, y


def cut_S(LS, y, K):
    """Exact cut(L_S, y) = sum_k g_k^T L_S g_k, the S-normalized Laplacian energy
    at the one-hot label indicators. This is precisely eq:cut in the paper.
    """
    G = np.eye(K)[y]                # (m, K) one-hot
    return float(np.sum(G * (LS @ G)))


def plain_cut_fraction(W, y):
    """Raw cross-class edge-mass fraction, reported only for context."""
    coo = W.tocoo()
    cross = y[coo.row] != y[coo.col]
    return float(coo.data[cross].sum() / max(coo.data.sum(), 1e-12))


# ---------------------------------------------------------------------------
# Eq. (alg) estimator
# ---------------------------------------------------------------------------
def kernel_inv(LS, Sinv, alpha):
    """K^{-1} = alpha * S^{-1} + L_S as a CSR matrix, built once per graph."""
    return (alpha * sp.diags(Sinv) + LS).tocsr()


def solve_scores(Kinv, labeled_idx, y, K, reg):
    """Solve ( J + reg * K^{-1} ) f_c = J Y_c per class; return (m, K) scores.

    reg = lambda * n_L for the estimator (n_L = total labeled), lambda * m for
    the full-label oracle.
    """
    m = Kinv.shape[0]
    J = np.zeros(m)
    J[labeled_idx] = 1.0
    A = (sp.diags(J) + reg * Kinv).tocsr()
    F = np.zeros((m, K))
    for c in range(K):
        b = np.zeros(m)
        b[labeled_idx] = (y[labeled_idx] == c).astype(float)
        F[:, c] = cg_solve(A, b)
    return F


def metrics(F, y, K, mask):
    """(0/1 error, squared-loss risk) of scores F over the nodes in `mask`."""
    G = np.eye(K)[y]
    err01 = float((F.argmax(axis=1)[mask] != y[mask]).mean())
    risk = float(np.mean(np.sum((F[mask] - G[mask]) ** 2, axis=1)))
    return err01, risk


def balanced_labeled(y, n_per_class, K, rng):
    """n_per_class indices from each class, without replacement (class-balanced)."""
    chosen = [rng.choice(np.where(y == c)[0], size=n_per_class, replace=False)
              for c in range(K)]
    return np.concatenate(chosen)


def build_graph(P, eps, rng):
    """Build the SBM at cross-strength eps; return (W, y, LS, Kinv, cut)."""
    W, y = build_sbm(P["M"], P["K"], P["P_IN"], eps * P["P_IN"], rng)
    LS = normalized_laplacian(W)
    Sinv = 1.0 / np.clip(np.asarray(W.sum(axis=1)).ravel(), 1e-12, None)
    Kinv = kernel_inv(LS, Sinv, P["ALPHA"])
    return W, y, LS, Kinv, cut_S(LS, y, P["K"])


def oracle_floors(Kinv, y, K, lam):
    """Exact full-label oracle floors: label every node, reg = lambda*m.

    Returns (err01_floor, risk_floor). risk_floor is the estimator's
    irreducible regularization bias in squared loss, the n_L -> infinity
    asymptote the additive (lambda/a)*cut term is meant to predict. Computed,
    not fitted.
    """
    m = Kinv.shape[0]
    F = solve_scores(Kinv, np.arange(m), y, K, lam * m)
    return metrics(F, y, K, np.ones(m, dtype=bool))


# ---------------------------------------------------------------------------
# Test 1: n_L rate against the exact (computed) oracle floor
# ---------------------------------------------------------------------------
def test_rate(P, rng):
    print("\n=== Test 1: n_L rate against the exact full-label oracle floor ===")
    W, y, LS, Kinv, cut = build_graph(P, P["EPS_RATE"], rng)
    m, lam = P["M"], P["LAMBDA"]
    err_floor, risk_floor = oracle_floors(Kinv, y, P["K"], lam)
    print(f"epsilon={P['EPS_RATE']}  p_out={P['EPS_RATE'] * P['P_IN']:.4f}  "
          f"cut(L_S,y)={cut:.2f}  cut/m={cut / m:.4f}  "
          f"plain-cut-frac={plain_cut_fraction(W, y):.4f}")
    print(f"oracle floors:  0/1 error={err_floor:.4f}   "
          f"squared-loss risk={risk_floor:.4f}")
    print(f"  {'n_L':>4} {'N=K*n_L':>8}  {'err01':>7} {'risk':>8} {'excess_risk':>12}")

    rows = []
    for npc in P["N_PER_CLASS"]:
        N = P["K"] * npc
        e01, rk = [], []
        for _ in range(P["N_SUBSETS"]):
            labeled = balanced_labeled(y, npc, P["K"], rng)
            unlabeled = np.ones(m, dtype=bool)
            unlabeled[labeled] = False
            F = solve_scores(Kinv, labeled, y, P["K"], lam * N)
            a, b = metrics(F, y, P["K"], unlabeled)
            e01.append(a); rk.append(b)
        mean_e01, mean_rk = float(np.mean(e01)), float(np.mean(rk))
        excess = mean_rk - risk_floor
        rows.append((npc, N, mean_e01, mean_rk, excess))
        print(f"  {npc:4d} {N:8d}  {mean_e01:7.4f} {mean_rk:8.4f} {excess:+12.4f}")

    # Rate of the excess SQUARED-LOSS RISK above the exact oracle floor, vs N.
    pos = [(N, ex) for _, N, _, _, ex in rows if ex > 0]
    if len(pos) >= 2:
        xs, ys = zip(*pos)
        slope, _ = fit_loglog_slope(xs, ys)
        print(f"excess-risk log-log slope vs N = {slope:.2f}")
    else:
        slope = float("nan")
        print("  not enough positive-excess points to fit a slope")
    # The 0/1 error decay, for comparison (it has no floor here; see Test 2).
    e01s = [(P["K"] * npc, e) for npc, _, e, _, _ in rows if e > 0]
    if len(e01s) >= 2:
        xs, ys = zip(*e01s)
        s01, _ = fit_loglog_slope(xs, ys)
        print(f"0/1-error    log-log slope vs N = {s01:.2f}")

    return {"rows": rows, "err_floor": err_floor, "risk_floor": risk_floor,
            "cut": cut, "slope": slope}


# ---------------------------------------------------------------------------
# Test 2: cut-dependence, additive vs multiplicative -- read in BOTH metrics
# ---------------------------------------------------------------------------
def test_cut(P, rng):
    print("\n=== Test 2: full-label oracle floor vs the known cut ===")
    print(f"  {'eps':>6} {'cut':>8} {'cut/m':>7} {'err01_floor':>12} "
          f"{'risk_floor':>11}")
    pts = []
    for eps in P["EPS_GRID"]:
        W, y, LS, Kinv, cut = build_graph(P, eps, rng)
        err_floor, risk_floor = oracle_floors(Kinv, y, P["K"], P["LAMBDA"])
        pts.append((eps, cut, err_floor, risk_floor))
        print(f"  {eps:6.3f} {cut:8.2f} {cut / P['M']:7.4f} "
              f"{err_floor:12.4f} {risk_floor:11.4f}")

    cuts = np.array([c for _, c, _, _ in pts])
    err_floors = np.array([e for _, _, e, _ in pts])
    risk_floors = np.array([r for _, _, _, r in pts])

    # Additive prediction lives in the SQUARED-LOSS risk: floor = slope * cut.
    slope0 = float(cuts @ risk_floors / max(cuts @ cuts, 1e-12))
    A = np.vstack([cuts, np.ones_like(cuts)]).T
    (slope1, intercept1), *_ = np.linalg.lstsq(A, risk_floors, rcond=None)
    corr_risk = (float(np.corrcoef(cuts, risk_floors)[0, 1])
                 if len(cuts) > 1 else float("nan"))

    print(f"\n  squared-loss risk floor vs cut:")
    print(f"    through-origin slope = {slope0:.2e}   "
          f"free-line slope={slope1:.2e}, intercept={intercept1:+.4f}")
    print(f"    corr(cut, risk_floor) = {corr_risk:.3f}, "
          f"risk range [{risk_floors.min():.4f}, {risk_floors.max():.4f}]")
    print(f"    (additive upper bound is (lambda/a)*cut; "
          f"realized slope <= lambda={P['LAMBDA']:.1e})")
    print(f"  0/1 error floor across the sweep: "
          f"max = {err_floors.max():.4f}")

    risk_additive = (corr_risk > 0.95 and risk_floors.max() > 5e-3
                     and abs(intercept1) < 0.2 * risk_floors.max())
    err_multiplicative = err_floors.max() < 5e-3
    print("\n  VERDICT (depends on which loss the theorem's 'err' denotes):")
    print("    - 0/1 classification error: "
          + ("MULTIPLICATIVE (floor ~ 0 for every cut; error -> 0)."
             if err_multiplicative else "shows a nonzero 0/1 floor; inspect."))
    print("    - squared-loss (admissible) risk: "
          + ("ADDITIVE (floor grows ~linearly with cut, through the origin)."
             if risk_additive else "cut-dependence unclear; inspect the plot."))

    return {"pts": pts, "slope0": slope0, "slope1": slope1,
            "intercept1": intercept1, "corr_risk": corr_risk,
            "err_multiplicative": err_multiplicative,
            "risk_additive": risk_additive}


# ---------------------------------------------------------------------------
# Test 3 (optional): stability lemma spot-check
# ---------------------------------------------------------------------------
def test_stability(P, rng):
    print("\n=== Test 3 (optional): stability |f - f^\\i| (lemma bound "
          "sigma*K_ii/(2 lambda n_L)) ===")
    W, y, LS, Kinv, cut = build_graph(P, P["EPS_RATE"], rng)
    m, lam = P["M"], P["LAMBDA"]
    G = np.eye(P["K"])[y]

    rows = []
    for npc in P["N_PER_CLASS"]:
        N = P["K"] * npc
        labeled = balanced_labeled(y, npc, P["K"], rng)
        F = solve_scores(Kinv, labeled, y, P["K"], lam * N)
        drop = rng.choice(labeled, size=min(20, len(labeled)), replace=False)
        diffs = []
        for i in drop:
            F_loo = solve_scores(Kinv, labeled[labeled != i], y, P["K"], lam * N)
            diffs.append(np.abs(F[i] - F_loo[i]).max())
        rows.append((N, float(np.mean(diffs))))
        print(f"  n_L(per class)={npc:4d}  N={N:5d}  "
              f"mean |f - f^\\i| = {np.mean(diffs):.5f}")
    xs = [N for N, _ in rows]
    ys = [d for _, d in rows]
    slope, _ = fit_loglog_slope(xs, ys)
    print(f"stability log-log slope vs N = {slope:.2f}")
    return {"rows": rows, "slope": slope}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(rate, cut, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: excess squared-loss risk vs N (log-log), with a 1/N guide and
    # the 0/1 error overlaid. The guide is a reference, NOT a fit.
    fig, ax = plt.subplots(figsize=(5.4, 4))
    Ns = np.array([N for _, N, _, _, ex in rate["rows"] if ex > 0])
    ex = np.array([ex for _, N, _, _, ex in rate["rows"] if ex > 0])
    ax.loglog(Ns, ex, "o-", color="C0",
              label=f"excess risk (slope {rate['slope']:.2f})")
    ax.loglog(Ns, ex[0] * (Ns / Ns[0]) ** -1.0, "k--", alpha=0.6,
              label="1/N reference")
    e01 = [(N, e) for _, N, e, _, _ in rate["rows"] if e > 0]
    if e01:
        xs, ys = zip(*e01)
        ax.loglog(xs, ys, "s-", color="C3", alpha=0.8, label="0/1 error")
    ax.set_xlabel("total labels  N = K * n_L")
    ax.set_ylabel("excess risk / 0-1 error")
    ax.set_title(f"Test 1: rate (risk floor = {rate['risk_floor']:.3f})")
    ax.legend(fontsize=8); fig.tight_layout()
    p1 = os.path.join(out_dir, "synthetic_rate.png")
    fig.savefig(p1, dpi=140); plt.close(fig)

    # Figure 2: oracle floors vs cut, in BOTH metrics. Squared-loss risk floor
    # is the additive signal (line through the origin); 0/1 floor ~ 0.
    cuts = np.array([c for _, c, _, _ in cut["pts"]])
    err_floors = np.array([e for _, _, e, _ in cut["pts"]])
    risk_floors = np.array([r for _, _, _, r in cut["pts"]])

    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.plot(cuts, risk_floors, "o", ms=7, color="C0",
            label="squared-loss risk floor (additive)")
    xs = np.linspace(0, cuts.max() * 1.05, 50)
    ax.plot(xs, cut["slope0"] * xs, "-", color="C0", alpha=0.8,
            label="through-origin fit")
    ax.plot(cuts, err_floors, "s", ms=7, color="C3",
            label="0/1 error floor (~0, multiplicative)")
    ax.set_xlabel("cut(L_S, y)")
    ax.set_ylabel("n_L -> inf asymptote (oracle floor)")
    ax.set_title("Test 2: additive in risk, multiplicative in 0/1")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8); fig.tight_layout()
    p2 = os.path.join(out_dir, "synthetic_cut.png")
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"\nsaved {p1}\nsaved {p2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(smoke=False, stability=False, no_plots=False):
    P = dict(PARAMS)
    if smoke:
        P.update(SMOKE)
        print("[smoke] tiny graph, reduced sweeps")
    set_seed(P["SEED"])
    rng = np.random.default_rng(P["SEED"])

    print(f"planted-partition SBM: m={P['M']}, K={P['K']}, "
          f"{P['M'] // P['K']}/class, p_in={P['P_IN']}, "
          f"lambda/a={P['LAMBDA']:.1e}, alpha={P['ALPHA']:.1e}")

    rate = test_rate(P, rng)
    cut = test_cut(P, rng)
    if stability:
        test_stability(P, rng)

    if not no_plots:
        make_plots(rate, cut, CACHE_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stability", action="store_true",
                    help="also run the optional stability spot-check (test 3)")
    ap.add_argument("--no-plots", action="store_true")
    main(**vars(ap.parse_args()))
