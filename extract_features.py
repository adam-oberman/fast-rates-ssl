"""Step 1: extract frozen SimCLR features for the CIFAR-10 training set.

Pushes every training image once through the frozen backbone, L2-normalizes,
and caches (features, labels) to Drive. This is the only GPU-touching step;
everything downstream is linear algebra on the cached vectors.

Run in Colab after mounting Drive (see README):
    !python extract_features.py            # full 50,000 images
    !python extract_features.py --smoke    # first 2,000, just a sanity check

Image order is deterministic (shuffle=False), so the row index of a feature
equals its dataset index. Every later script relies on that alignment.
"""
from __future__ import annotations

import argparse
import os

import torch
import torchvision
import torchvision.transforms as T

import common
import config


def ensure_cifar() -> torchvision.datasets.CIFAR10:
    """CIFAR-10 train split, normalized only (no augmentation here).

    download=True is cheap when the data is already present at CIFAR_ROOT: the
    integrity check passes and nothing is fetched. The README setup cell places
    the data there to avoid the slow mirror.
    """
    os.makedirs(config.CIFAR_ROOT, exist_ok=True)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(config.CIFAR_MEAN, config.CIFAR_STD),
    ])
    return torchvision.datasets.CIFAR10(
        root=config.CIFAR_ROOT, train=True, download=True, transform=transform
    )


def main(smoke: bool = False) -> None:
    if smoke:
        config.apply_smoke()  # repoints to the *_smoke cache files
    common.set_seed(config.MASTER_SEED)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    if os.path.exists(config.FEATURES_PATH) and os.path.exists(config.LABELS_PATH):
        feats = torch.load(config.FEATURES_PATH)
        print(f"Features already cached {tuple(feats.shape)} at {config.FEATURES_PATH}; nothing to do.")
        return

    backbone = common.build_backbone()
    dataset = ensure_cifar()
    if smoke:
        dataset = torch.utils.data.Subset(dataset, range(2000))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.EXTRACT_BATCH, shuffle=False, num_workers=2
    )

    feats, labels = [], []
    for i, (imgs, lab) in enumerate(loader):
        feats.append(common.embed(backbone, imgs).cpu())
        labels.append(lab)
        if i % 10 == 0:
            print(f"batch {i}/{len(loader)}")
    feats = torch.cat(feats)
    labels = torch.cat(labels)

    torch.save(feats, config.FEATURES_PATH)
    torch.save(labels, config.LABELS_PATH)
    tag = "smoke " if smoke else ""
    print(f"saved {tag}features {tuple(feats.shape)} and labels to {config.CACHE_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2,000-image sanity run")
    main(**vars(ap.parse_args()))
