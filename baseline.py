"""Step 3: supervised-from-scratch baseline.

A small CNN trained from random initialization on the raw images, for the same
n_L sweep, to exhibit the slower ~1/sqrt(n_L) slope that the frozen-feature
probe is meant to beat. Error is measured on the same fixed unlabeled training
pool used by probe_sweep (capped for speed), so the curves are comparable.

The labeled subsets reuse common.sample_labeled_indices with the master seed,
so for a given n_L the baseline trains on the same images the probe labels.
Indices line up with the cached features because the dataset order is fixed.

Run:
    !python baseline.py            # full sweep, BASELINE_SEEDS seeds
    !python baseline.py --smoke    # tiny: 2 n_L, 1 seed, 2 epochs
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import common
import config


class SmallCNN(nn.Module):
    """Compact 3-block CNN, deliberately modest so n_L drives the accuracy."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(128 * 4 * 4, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def make_datasets():
    """Train split with light augmentation, and a clean copy for evaluation."""
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(config.CIFAR_MEAN, config.CIFAR_STD),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(config.CIFAR_MEAN, config.CIFAR_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=config.CIFAR_ROOT, train=True, download=True, transform=train_tf)
    eval_ds = torchvision.datasets.CIFAR10(
        root=config.CIFAR_ROOT, train=True, download=True, transform=eval_tf)
    return train_ds, eval_ds


def train_once(train_ds, eval_ds, labeled_idx, unlabeled_idx, seed):
    common.set_seed(seed)
    device = config.DEVICE
    net = SmallCNN(config.N_CLASSES).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=config.BASELINE_LR,
                          momentum=config.BASELINE_MOMENTUM,
                          weight_decay=config.BASELINE_WD, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.BASELINE_EPOCHS)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.BASELINE_LABEL_SMOOTH)

    # num_workers=0: this function is called many times in a loop, and spawning
    # DataLoader worker subprocesses each time exhausts macOS's low default
    # file-descriptor limit ("Too many open files"). The data is tiny and
    # CPU-bound, so in-process loading is fine and avoids the leak.
    labeled_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_ds, labeled_idx.tolist()),
        batch_size=min(config.BASELINE_BATCH, len(labeled_idx)),
        shuffle=True, num_workers=0, drop_last=False)

    net.train()
    for _ in range(config.BASELINE_EPOCHS):
        for imgs, lab in labeled_loader:
            opt.zero_grad()
            loss = loss_fn(net(imgs.to(device)), lab.to(device))
            loss.backward()
            opt.step()
        sched.step()

    eval_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(eval_ds, unlabeled_idx.tolist()),
        batch_size=512, shuffle=False, num_workers=0)
    net.eval()
    wrong = total = 0
    with torch.no_grad():
        for imgs, lab in eval_loader:
            pred = net(imgs.to(device)).argmax(1).cpu()
            wrong += (pred != lab).sum().item()
            total += lab.numel()
    return wrong / total


def main(smoke: bool = False) -> None:
    if smoke:
        config.apply_smoke()
    common.set_seed(config.MASTER_SEED)

    _, labels = common.load_features()  # labels align with dataset order
    m = labels.shape[0]
    train_ds, eval_ds = make_datasets()
    rng = np.random.default_rng(config.MASTER_SEED)

    results = {"n_L": config.N_L_VALUES, "baseline": {},
               "meta": {"epochs": config.BASELINE_EPOCHS,
                        "seeds": config.BASELINE_SEEDS,
                        "eval_cap": config.BASELINE_EVAL_CAP}}

    for n_L in config.N_L_VALUES:
        errs = []
        for seed in range(config.BASELINE_SEEDS):
            labeled_idx = common.sample_labeled_indices(
                labels, n_L, config.N_CLASSES, rng)
            unlabeled_mask = np.ones(m, dtype=bool)
            unlabeled_mask[labeled_idx.numpy()] = False
            unlabeled_idx = np.where(unlabeled_mask)[0]
            if config.BASELINE_EVAL_CAP and len(unlabeled_idx) > config.BASELINE_EVAL_CAP:
                unlabeled_idx = rng.choice(
                    unlabeled_idx, size=config.BASELINE_EVAL_CAP, replace=False)
            errs.append(train_once(
                train_ds, eval_ds, labeled_idx,
                torch.from_numpy(unlabeled_idx), config.MASTER_SEED + seed))
        results["baseline"][n_L] = {"mean": float(np.mean(errs)),
                                    "std": float(np.std(errs))}
        print(f"n_L={n_L:4d}  baseline={np.mean(errs):.4f} (+/- {np.std(errs):.4f})")

    if not smoke:
        with open(config.BASELINE_RESULTS, "w") as f:
            json.dump(results, f, indent=2)
        print("saved", config.BASELINE_RESULTS)

    xs = config.N_L_VALUES
    ys = [results["baseline"][n]["mean"] for n in xs]
    slope, _ = common.fit_loglog_slope(xs, ys)
    print(f"baseline log-log slope: {slope:.2f}  (expected near -0.5)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(**vars(ap.parse_args()))
