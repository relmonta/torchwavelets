"""Application 3: CIFAR-10 classification with DTCWT scattering features.

The classic Mallat recipe: a fixed (untrained) scattering front-end extracts
locally shift-invariant, deformation-stable features, and a linear classifier
is trained on top. Here ``DTCWTScattering(J=2, max_order=2)`` is applied to
every 32x32 colour image, the invariants are globally average-pooled (147
dimensions), and a multinomial logistic regression is fitted on the 50k
training images. A raw-pixel logistic baseline is reported for comparison.

Expected result (fixed seed): scattering ~50-60% vs raw pixels ~30-40% on the
10k test images, with 5x fewer dimensions.

Usage::

    python examples/application_cifar10.py [--out docs/figures/cifar10_scattering.png]
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from _data import CIFAR_CLASSES, load_cifar10
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import torchwavelets as tw


def scattering_features(images, batch=2048):
    """2x2-average-pooled DTCWT scattering invariants, shape (N, 588)."""
    net = tw.DTCWTScattering(J=2, max_order=2)
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            xb = images[i : i + batch]
            s0, invs = net(xb)
            parts = [F.avg_pool2d(s0, 4)]
            parts += [F.avg_pool2d(v, 4) for _, v in invs]
            feats.append(torch.cat(parts, dim=1).flatten(1))
    return torch.cat(feats, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/cifar10_scattering.png")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    x_tr, y_tr, x_te, y_te = load_cifar10()
    print(f"CIFAR-10: {x_tr.shape[0]} train / {x_te.shape[0]} test images")

    # per-channel standardisation (train stats)
    mean = x_tr.mean(axis=(0, 1, 2), keepdims=True) / 255.0
    std = x_tr.std(axis=(0, 1, 2), keepdims=True) / 255.0
    x_tr_n = (x_tr / 255.0 - mean) / std
    x_te_n = (x_te / 255.0 - mean) / std

    # scattering features
    t0 = time.time()
    F_tr = scattering_features(torch.tensor(x_tr_n.transpose(0, 3, 1, 2)))
    F_te = scattering_features(torch.tensor(x_te_n.transpose(0, 3, 1, 2)))
    print(f"scattering features {tuple(F_tr.shape)} in {time.time() - t0:.0f}s")

    clf = LogisticRegression(max_iter=5000, C=1.0, tol=1e-4)
    sc = StandardScaler().fit(F_tr.numpy())
    clf.fit(sc.transform(F_tr.numpy()), y_tr)
    acc_scat = accuracy_score(y_te, clf.predict(sc.transform(F_te.numpy())))
    print(f"scattering  accuracy: {acc_scat * 100:.1f}%")

    # raw-pixel baseline (subsample train for speed: 20k)
    n_base = min(20000, len(y_tr))
    idx = rng.choice(len(y_tr), n_base, replace=False)
    sc_pix = StandardScaler().fit(x_tr_n[idx].reshape(len(idx), -1))
    clf_pix = LogisticRegression(max_iter=5000, C=1.0, tol=1e-4)
    clf_pix.fit(sc_pix.transform(x_tr_n[idx].reshape(len(idx), -1)), y_tr[idx])
    acc_pix = accuracy_score(y_te, clf_pix.predict(sc_pix.transform(x_te_n.reshape(len(y_te), -1))))
    print(f"raw pixels  accuracy: {acc_pix * 100:.1f}%")

    # --- figure --------------------------------------------------------------
    cm = confusion_matrix(y_te, clf.predict(sc.transform(F_te.numpy())), normalize="true")
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.2, 1, 1])

    preds = clf.predict(sc.transform(F_te.numpy()))
    for i in range(6):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.imshow(x_te[i].astype(np.float64) / 255.0)
        ok = preds[i] == y_te[i]
        ax.set_title(
            f"{CIFAR_CLASSES[y_te[i]]} → {CIFAR_CLASSES[preds[i]]}{'' if ok else '  (WRONG)'}",
            fontsize=9,
            color=("black" if ok else "red"),
        )
        ax.axis("off")

    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CIFAR_CLASSES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(10), CIFAR_CLASSES, fontsize=7)
    ax.set_title("confusion matrix (scattering)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 1:])
    ax.bar(
        ["raw pixels\n(3072 dims)", "DTCWT scattering\n(588 dims)"],
        [acc_pix * 100, acc_scat * 100],
        color=["#9aa5b1", "#2f6f4f"],
        width=0.55,
    )
    for i, v in enumerate([acc_pix * 100, acc_scat * 100]):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("logistic regression on CIFAR-10", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "CIFAR-10 with DTCWT scattering features: a fixed 588-d front-end beats 3072-d raw pixels",
        fontsize=13,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
