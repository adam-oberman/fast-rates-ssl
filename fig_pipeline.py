"""Larger-m figure pipeline for Experiment A. Three parts, cached to npz so each
fits the shell time budget, then rendered to PDF/PNG.

  --part cutenv    : cut-dependence (lambda=1e-3) + downstream rate envelope
                     (excess squared risk + 0/1 error, lambda=1e-3)
  --part stab      : leave-one-out stability |f-f^\\i| (lambda=1e-1, where the
                     stability constant is small enough that the asymptotic
                     1/N tail is visible; slope ~ -1.00), the rate at its source
  --part render    : build synthetic_cut.pdf, synthetic_rate.pdf,
                     synthetic_stability.pdf from the cached npz, into cache/
                     and (if present) ../PAPER/figures/

m=6000, K=10. Torch-free. Reproduces the Section 7 / RATE_FINDINGS.md figures.
Run order: `python fig_pipeline.py --part stab && python fig_pipeline.py --part
cutenv && python fig_pipeline.py --part render`.
"""
from __future__ import annotations
import argparse, inspect, os, shutil
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "cache")            # npz + png/pdf land in Code/cache
PAPER_FIG = os.path.join(BASE, os.pardir, "PAPER", "figures")
os.makedirs(OUT, exist_ok=True)
M, K, P_IN, ALPHA, EPS = 6000, 10, 0.03, 1e-3, 0.03
LAM_FLOOR = 1e-3      # cut + envelope panels (Experiment A config)
LAM_STAB = 1e-1       # stability panel: smaller constant so the 1/N tail is visible
EPS_GRID = [0.005, 0.01, 0.02, 0.04, 0.07]
NPCS = [5, 10, 20, 50, 100, 200, 300]
NPCS_STAB = [10, 20, 50, 100, 200, 400, 500]


def normalized_laplacian(W):
    deg = np.asarray(W.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    D = sp.diags(dinv)
    return (sp.identity(W.shape[0], format="csr") - D @ W @ D).tocsr()


def cg_solve(A, b, tol=1e-8, maxiter=6000):
    kw = {"maxiter": maxiter}
    kw["rtol" if "rtol" in inspect.signature(cg).parameters else "tol"] = tol
    return cg(A, b, **kw)[0]


def fit_slope(x, y):
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.linalg.lstsq(np.vstack([lx, np.ones_like(lx)]).T, ly, rcond=None)[0][0])


def build_sbm(m, K, p_in, p_out, rng):
    y = np.repeat(np.arange(K), m // K)
    same = y[:, None] == y[None, :]
    R = rng.random((m, m))
    A = np.where(same, R < p_in, R < p_out)
    A = np.triu(A, 1); A = A | A.T
    return sp.csr_matrix(A.astype(float)), y


def kernel_inv(LS, W, alpha):
    Sinv = 1.0 / np.clip(np.asarray(W.sum(1)).ravel(), 1e-12, None)
    return (alpha * sp.diags(Sinv) + LS).tocsr()


def solve_scores(Kinv, lab, y, K, reg):
    m = Kinv.shape[0]
    J = np.zeros(m); J[lab] = 1.0
    A = (sp.diags(J) + reg * Kinv).tocsr()
    F = np.zeros((m, K))
    for c in range(K):
        b = np.zeros(m); b[lab] = (y[lab] == c).astype(float)
        F[:, c] = cg_solve(A, b)
    return F


def cut_S(LS, y, K):
    G = np.eye(K)[y]
    return float(np.sum(G * (LS @ G)))


def risk_on(F, y, K, mask):
    G = np.eye(K)[y]
    return float(np.mean(np.sum((F[mask] - G[mask]) ** 2, 1)))


def err_on(F, y, mask):
    return float((F.argmax(1)[mask] != y[mask]).mean())


def bal(pool, y, npc, K, rng):
    return np.concatenate([rng.choice(pool[y[pool] == c], npc, replace=False) for c in range(K)])


def part_cutenv():
    rng = np.random.default_rng(0)
    # cut-dependence
    cuts, floors = [], []
    for eps in EPS_GRID:
        W, y = build_sbm(M, K, P_IN, eps * P_IN, rng)
        LS = normalized_laplacian(W); Kinv = kernel_inv(LS, W, ALPHA)
        Fo = solve_scores(Kinv, np.arange(M), y, K, LAM_FLOOR * M)
        cuts.append(cut_S(LS, y, K)); floors.append(risk_on(Fo, y, K, np.ones(M, bool)))
    cuts = np.array(cuts); floors = np.array(floors)
    slope0 = float(cuts @ floors / (cuts @ cuts))
    corr = float(np.corrcoef(cuts, floors)[0, 1])
    # envelope at fixed eps
    W, y = build_sbm(M, K, P_IN, EPS * P_IN, rng)
    LS = normalized_laplacian(W); Kinv = kernel_inv(LS, W, ALPHA)
    ev = bal(np.arange(M), y, 150, K, rng); evm = np.zeros(M, bool); evm[ev] = True
    pool = np.where(~evm)[0]
    floor = risk_on(solve_scores(Kinv, np.arange(M), y, K, LAM_FLOOR * M), y, K, evm)
    Ns, exc, e01 = [], [], []
    for npc in NPCS:
        N = K * npc; rs, es = [], []
        for _ in range(6):
            lab = bal(pool, y, npc, K, rng)
            F = solve_scores(Kinv, lab, y, K, LAM_FLOOR * N)
            rs.append(risk_on(F, y, K, evm)); es.append(err_on(F, y, evm))
        Ns.append(N); exc.append(np.mean(rs) - floor); e01.append(np.mean(es))
    np.savez(os.path.join(OUT, "fig_cutenv.npz"), cuts=cuts, floors=floors,
             slope0=slope0, corr=corr, Ns=Ns, exc=exc, e01=e01, floor=floor)
    print(f"cut: corr={corr:.3f} through-origin slope={slope0:.2e} "
          f"(lambda/a={LAM_FLOOR:.0e}); floors {floors.min():.4f}-{floors.max():.4f}")
    pos = np.array(exc) > 0
    print(f"envelope: excess slope={fit_slope(np.array(Ns)[pos], np.array(exc)[pos]):+.2f}  "
          f"0/1 slope={fit_slope(Ns, e01):+.2f}")


def part_stab():
    rng = np.random.default_rng(1)
    W, y = build_sbm(M, K, P_IN, EPS * P_IN, rng)
    LS = normalized_laplacian(W); Kinv = kernel_inv(LS, W, ALPHA)
    Ns, stab = [], []
    for npc in NPCS_STAB:
        N = K * npc
        lab = bal(np.arange(M), y, npc, K, rng)
        F = solve_scores(Kinv, lab, y, K, LAM_STAB * N)
        drop = rng.choice(lab, size=min(8, len(lab)), replace=False)
        d = [np.abs(F[i] - solve_scores(Kinv, lab[lab != i], y, K, LAM_STAB * N)[i]).max()
             for i in drop]
        Ns.append(N); stab.append(float(np.mean(d)))
    Ns = np.array(Ns); stab = np.array(stab)
    tail = Ns >= Ns.max() / 8
    np.savez(os.path.join(OUT, "fig_stab.npz"), Ns=Ns, stab=stab,
             slope_all=fit_slope(Ns, stab), slope_tail=fit_slope(Ns[tail], stab[tail]))
    print(f"stability (lambda={LAM_STAB:.0e}): slope(all)={fit_slope(Ns, stab):+.2f}  "
          f"slope(tail)={fit_slope(Ns[tail], stab[tail]):+.2f}")
    print("   N:    " + " ".join(f"{n:5d}" for n in Ns))
    print("   stab: " + " ".join(f"{v:.4f}" for v in stab))


def part_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ce = np.load(os.path.join(OUT, "fig_cutenv.npz"))
    st = np.load(os.path.join(OUT, "fig_stab.npz"))

    # cut panel
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cuts, floors = ce["cuts"], ce["floors"]
    xs = np.linspace(0, cuts.max() * 1.05, 50)
    ax.plot(xs, ce["slope0"] * xs, "-", color="C0", alpha=0.8, label="through-origin fit")
    ax.plot(cuts, floors, "o", ms=7, color="C0", label="squared-loss floor")
    ax.plot(cuts, np.zeros_like(cuts), "s", ms=6, color="C3", label="0/1 floor ($\\approx$0)")
    ax.set_xlabel(r"$\mathrm{cut}(\mathcal{L}_S, y)$"); ax.set_ylabel(r"$n_L\to\infty$ floor")
    ax.set_xlim(left=0); ax.set_ylim(bottom=-0.002)
    ax.set_title(f"additive in surrogate (corr {ce['corr']:.3f})", fontsize=9)
    ax.legend(fontsize=7.5); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "synthetic_cut.pdf")); fig.savefig(os.path.join(OUT, "synthetic_cut.png"), dpi=140); plt.close(fig)

    # stability panel
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    Ns, stab = st["Ns"], st["stab"]
    ax.loglog(Ns, stab, "o-", color="C2", label=f"stability $|f-f^{{\\backslash i}}|$ (slope {float(st['slope_tail']):.2f})")
    ax.loglog(Ns, stab[0] * (Ns / Ns[0]) ** -1.0, "k--", alpha=0.6, label=r"$1/N$ reference")
    ax.set_xlabel(r"total labels $N=K\,n_L$"); ax.set_ylabel(r"mean $|f-f^{\backslash i}|$")
    ax.set_title(r"rate at its source: lemma's $1/n_L$", fontsize=9)
    ax.legend(fontsize=7.5); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "synthetic_stability.pdf")); fig.savefig(os.path.join(OUT, "synthetic_stability.png"), dpi=140); plt.close(fig)

    # downstream envelope panel
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    Ns = ce["Ns"]; exc = ce["exc"]; e01 = ce["e01"]
    pe = np.array(exc) > 0
    ax.loglog(np.array(Ns)[pe], np.array(exc)[pe], "o-", color="C0", label="excess risk (envelope)")
    p1 = np.array(e01) > 0
    ax.loglog(np.array(Ns)[p1], np.array(e01)[p1], "s-", color="C3", alpha=0.85, label="0/1 error")
    Na = np.array(Ns, float)
    ax.loglog(Na, exc[0] * (Na / Na[0]) ** -1.0, "k--", alpha=0.55, label=r"$1/N$")
    ax.loglog(Na, exc[0] * (Na / Na[0]) ** -0.5, ":", color="gray", alpha=0.8, label=r"$1/\sqrt{N}$")
    ax.set_xlabel(r"total labels $N=K\,n_L$"); ax.set_ylabel("excess risk / 0-1 error")
    ax.set_title("downstream: envelope loose (large tr K)", fontsize=9)
    ax.legend(fontsize=7.5); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "synthetic_rate.pdf")); fig.savefig(os.path.join(OUT, "synthetic_rate.png"), dpi=140); plt.close(fig)
    print("rendered synthetic_cut/.pdf, synthetic_stability.pdf, synthetic_rate.pdf to", OUT)
    if os.path.isdir(PAPER_FIG):
        for name in ("synthetic_cut.pdf", "synthetic_stability.pdf", "synthetic_rate.pdf"):
            shutil.copy(os.path.join(OUT, name), os.path.join(PAPER_FIG, name))
        print("copied the three PDFs into", os.path.normpath(PAPER_FIG))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["cutenv", "stab", "render"])
    a = ap.parse_args()
    {"cutenv": part_cutenv, "stab": part_stab, "render": part_render}[a.part]()
