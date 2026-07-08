"""Step 4: measure the data-augmentation alignment error R_DA(y).

R_DA is the fraction of augmentation-graph edge mass that runs between
differently labeled nodes: the normalized cut of Eq. (cut). It is a property
of the augmentation distribution and the labels together, defined before any
classifier exists, and Theorem 2 predicts it is the floor of the graph-probe
error curve.

Estimator (nearest-neighbor, forward chain). Take a labeled anchor x0 with
label y0, apply the augmentation to get a view v = M(x0), embed it, and find
v's nearest neighbors among a fixed labeled bank. The neighbor weight that
lands on points with a label other than y0 is edge mass crossing a boundary.
An augmented view never gets a new label: it keeps y0, and we only ask whether
augmentation pushed it into another class's territory.

We report three normalizations of the same affinity:
  forward    : row-normalized (each view distributes weight 1 over neighbors),
  backward   : column-normalized (per landing point in the bank),
  symmetric  : geometric mean of the two, the L_S normalization in Theorem 2.
Forward and backward should bracket the symmetric value.

Two augmentation strengths are measured so the floor can move:
  weak       : mild, label-respecting crops and jitter -> small R_DA,
  aggressive : full SimCLR augmentation -> larger R_DA.

Circularity check. R_DA is also computed in input space (flattened normalized
pixels) instead of backbone features, so the floor is estimated independently
of the model whose error it predicts. The two should track each other.

Run:
    !python measure_rda.py            # full, both strengths, both spaces
    !python measure_rda.py --smoke    # 200 anchors, smaller bank
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

import common
import config


def augmentation(strength: str) -> T.Compose:
    norm = T.Normalize(config.CIFAR_MEAN, config.CIFAR_STD)
    if strength == "weak":
        return T.Compose([
            T.RandomResizedCrop(32, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.0)], p=0.5),
            T.ToTensor(), norm,
        ])
    if strength == "aggressive":
        return T.Compose([
            T.RandomResizedCrop(32, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(), norm,
        ])
    raise ValueError(f"unknown strength {strength!r}")


def cut_from_affinity(view_feats, view_labels, bank_feats, bank_labels, k):
    """Forward, backward, and symmetric normalized cut from a kNN affinity.

    view_feats : (V, d) embedded augmented views, L2-normalized.
    bank_feats : (B, d) embedded labeled bank, L2-normalized.
    Returns (forward, backward, symmetric) crossing fractions in [0, 1].
    """
    dev = config.DEVICE
    V = view_feats.to(dev)
    Bf = bank_feats.to(dev)
    vlab = view_labels.to(dev)
    blab = bank_labels.to(dev)

    sims = (V @ Bf.T).clamp(min=0.0)  # (V, B) cosine, negatives dropped
    topv, topi = sims.topk(k, dim=1)  # (V, k)
    neigh_lab = blab[topi]            # (V, k)
    differ = (neigh_lab != vlab[:, None]).float()

    # Forward: row-normalize each view's neighbor weights.
    p = topv / topv.sum(dim=1, keepdim=True).clamp_min(1e-12)
    forward = (p * differ).sum(dim=1).mean().item()

    # Rebuild a sparse (V, B) affinity holding only the top-k entries, then
    # column-normalize for the backward chain.
    A = torch.zeros(V.shape[0], Bf.shape[0], device=dev)
    A.scatter_(1, topi, topv)
    col_sum = A.sum(dim=0, keepdim=True)                     # (1, B)
    active = col_sum.squeeze(0) > 0
    q = A / col_sum.clamp_min(1e-12)                          # column-normalized
    differ_full = (blab[None, :] != vlab[:, None]).float()   # (V, B)
    backward = (q * differ_full).sum().item() / max(int(active.sum().item()), 1)

    symmetric = float(np.sqrt(max(forward, 0.0) * max(backward, 0.0)))
    return forward, backward, symmetric


def embed_views_model(backbone, view_tensor):
    return common.embed(backbone, view_tensor).cpu()


def embed_views_input(view_tensor):
    """Input-space 'embedding': flattened, L2-normalized pixels."""
    x = view_tensor.flatten(1)
    return (x / x.norm(dim=1, keepdim=True).clamp_min(1e-12))


def main(smoke: bool = False) -> None:
    if smoke:
        config.apply_smoke()
    common.set_seed(config.MASTER_SEED)

    feats, labels = common.load_features()
    rng = np.random.default_rng(config.MASTER_SEED)

    # Labeled bank: class-balanced, fixed across strengths.
    bank_idx = common.sample_labeled_indices(
        labels, config.RDA_BANK_PER_CLASS, config.N_CLASSES, rng)
    bank_labels = labels[bank_idx]
    bank_feats_model = feats[bank_idx]

    backbone = common.build_backbone()
    raw_ds = torchvision.datasets.CIFAR10(
        root=config.CIFAR_ROOT, train=True, download=True, transform=None)
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(config.CIFAR_MEAN, config.CIFAR_STD)])

    # Input-space bank: clean normalized pixels for the bank images.
    bank_imgs = torch.stack([eval_tf(raw_ds[i][0]) for i in bank_idx.tolist()])
    bank_feats_input = embed_views_input(bank_imgs)

    # Anchors are labeled images sampled from the bank.
    anchor_pos = rng.choice(len(bank_idx),
                            size=min(config.RDA_N_ANCHORS, len(bank_idx)),
                            replace=False)
    anchor_idx = bank_idx[anchor_pos]
    anchor_labels = labels[anchor_idx]
    anchor_pils = [raw_ds[i][0] for i in anchor_idx.tolist()]

    results = {"meta": {"bank_per_class": config.RDA_BANK_PER_CLASS,
                        "n_anchors": int(len(anchor_idx)),
                        "views_per_anchor": config.RDA_VIEWS_PER_ANCHOR,
                        "knn": config.RDA_KNN}}

    # Matched floor: the normalized cut of the exact kNN graph the probe uses
    # (same KNN_K), computed directly from the graph weights and labels. This is
    # the floor Theorem 2 predicts the graph probe plateaus at, on the same
    # graph, so it is the apples-to-apples comparison for the figure.
    W = common.load_or_build_affinity(feats, config.KNN_K)
    results["graph_cut"] = common.graph_cut_rda(W, labels.numpy())
    results["graph_cut"]["knn_k"] = config.KNN_K
    print(f"[graph cut ] matched floor sym={results['graph_cut']['symmetric']:.4f} "
          f"(plain={results['graph_cut']['plain']:.4f}) on the probe's kNN graph")

    for strength in config.AUG_STRENGTHS:
        aug = augmentation(strength)
        vpa = config.RDA_VIEWS_PER_ANCHOR

        view_tensors, view_labels = [], []
        for pil, lab in zip(anchor_pils, anchor_labels.tolist()):
            for _ in range(vpa):
                view_tensors.append(aug(pil))
                view_labels.append(lab)
        view_tensor = torch.stack(view_tensors)
        view_labels = torch.tensor(view_labels)

        # Model space.
        vfeat_model = []
        for s in range(0, view_tensor.shape[0], 512):
            vfeat_model.append(embed_views_model(backbone, view_tensor[s:s + 512]))
        vfeat_model = torch.cat(vfeat_model)
        fwd_m, bwd_m, sym_m = cut_from_affinity(
            vfeat_model, view_labels, bank_feats_model, bank_labels, config.RDA_KNN)

        # Input space (circularity check).
        vfeat_input = embed_views_input(view_tensor)
        fwd_i, bwd_i, sym_i = cut_from_affinity(
            vfeat_input, view_labels, bank_feats_input, bank_labels, config.RDA_KNN)

        results[strength] = {
            "model": {"forward": fwd_m, "backward": bwd_m, "symmetric": sym_m},
            "input": {"forward": fwd_i, "backward": bwd_i, "symmetric": sym_i},
        }
        print(f"[{strength:10s}] model R_DA sym={sym_m:.4f} "
              f"(fwd={fwd_m:.4f}, bwd={bwd_m:.4f}) | "
              f"input sym={sym_i:.4f}")

    if not smoke:
        with open(config.RDA_RESULTS, "w") as f:
            json.dump(results, f, indent=2)
        print("saved", config.RDA_RESULTS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(**vars(ap.parse_args()))
