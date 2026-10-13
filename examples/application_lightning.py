"""Application 4: training with PyTorch Lightning + precision management.

A small CIFAR-10 classifier is trained with PyTorch Lightning: a **frozen**
``DTCWTScattering`` front-end (no learned parameters) feeds a learnable MLP
head. The same script doubles as a *precision audit*: it prints the dtype of
every stage of the model inside Lightning's mixed-precision autocast context,
demonstrating the documented contract:

* DWT / SWT: output dtype == autocast dtype (e.g. bfloat16 under
  ``bf16-mixed``) — autocast-native.
* DTCWT / scattering: computed in float32 (no complex16/bf16 kernels) and
  returned as float32 — the documented exception, which never crashes AMP.
* The head runs in the autocast dtype; Lightning handles the loss/backward.

Usage::

    pip install lightning
    python examples/application_lightning.py [--epochs 2] [--train 10000]

The script picks ``precision="16-mixed"`` on CUDA and ``"bf16-mixed"`` on CPU.
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import lightning
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from _data import load_cifar10

import torchwavelets as tw


class ScatteringClassifier(lightning.LightningModule):
    """Frozen torchwavelets layers (DWT + DTCWT scattering) + learned MLP head.

    The model contains both a critically sampled DWT layer (autocast-native:
    its output follows the Trainer precision) and a DTCWT scattering front-end
    (computed in float32: no complex bf16 kernels), so the precision audit
    below shows the full contract.
    """

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.dwt = tw.DWT2D(J=2, wave="db2", mode="symmetric")
        self.front = tw.DTCWTScattering(J=2, max_order=2)
        self.head = nn.Sequential(
            nn.Linear(588 + 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10),
        )
        self.lr = lr
        self.losses: list[torch.Tensor] = []
        self.audit: dict[str, str] = {}

    def forward(self, x):
        # precision audit: record the dtype of every stage inside the
        # Trainer's autocast context
        yl, _ = self.dwt(x)
        s0, invs = self.front(x)
        parts = [F.avg_pool2d(s0, 4)]
        parts += [F.avg_pool2d(v, 4) for _, v in invs]
        feats = torch.cat(parts, dim=1).flatten(1)
        feats = torch.cat([feats, yl.mean(dim=(-1, -2))], dim=1)
        logits = self.head(feats)
        self.audit = {
            "input": str(x.dtype),
            "DWT2D lowpass (autocast-native)": str(yl.dtype),
            "scattering order-0 (DTCWT)": str(s0.dtype),
            "scattering invariants (DTCWT)": str(invs[0][1].dtype),
            "features (pooled)": str(feats.dtype),
            "logits (head)": str(logits.dtype),
        }
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.losses.append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return F.cross_entropy(self(x), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.head.parameters(), lr=self.lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/lightning_training.png")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    torch.manual_seed(0)

    x_tr, y_tr, x_te, y_te = load_cifar10()
    idx = np.random.default_rng(0).choice(len(y_tr), args.train, replace=False)
    te = np.random.default_rng(1).choice(len(y_te), args.test, replace=False)

    mean = x_tr.mean(axis=(0, 1, 2), keepdims=True) / 255.0
    std = x_tr.std(axis=(0, 1, 2), keepdims=True) / 255.0

    def norm(a):
        return torch.tensor((a / 255.0 - mean) / std).float().permute(0, 3, 1, 2)

    ds_tr = torch.utils.data.TensorDataset(norm(x_tr[idx]), torch.tensor(y_tr[idx]))
    ds_te = torch.utils.data.TensorDataset(norm(x_te[te]), torch.tensor(y_te[te]))
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=args.batch)

    model = ScatteringClassifier()
    precision = "16-mixed" if torch.cuda.is_available() else "bf16-mixed"
    print(f"accelerator: {'gpu' if torch.cuda.is_available() else 'cpu'}, precision: {precision}")
    trainer = lightning.Trainer(
        max_epochs=args.epochs,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="auto",
    )
    t0 = time.time()
    trainer.fit(model, dl_tr, dl_te)
    print(f"training: {time.time() - t0:.0f}s, final loss {model.losses[-1].item():.3f}")

    # snapshot the audit from inside the Trainer's autocast context (the
    # evaluation loop below runs outside it and would overwrite the dtypes)
    audit = dict(model.audit)

    # --- precision audit -----------------------------------------------------
    print("\nprecision audit (inside the Lightning autocast context):")
    for k, v in audit.items():
        print(f"  {k:26s}: {v}")

    # accuracy on the test split
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl_te:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    print(f"test accuracy: {100 * correct / total:.1f}%")

    # --- figure --------------------------------------------------------------
    losses = torch.stack(model.losses).numpy()
    steps = np.arange(len(losses)) * args.batch
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, width_ratios=[1.3, 1])
    axes[0].plot(steps, losses, lw=1.5, color="#2f6f4f")
    axes[0].set_xlabel("training samples seen")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title(f"Lightning training ({args.epochs} epochs, {precision})", fontsize=10)
    axes[0].grid(alpha=0.3)

    rows = [[k, v] for k, v in audit.items()]
    tbl = axes[1].table(
        cellText=rows,
        colLabels=["stage", "dtype"],
        loc="center",
        cellLoc="left",
        colWidths=[0.68, 0.32],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#2f6f4f")
            cell.set_text_props(color="white")
    axes[1].axis("off")
    axes[1].set_title(
        f"precision audit under {precision}\n(test accuracy {100 * correct / total:.1f}%)",
        fontsize=10,
    )
    fig.suptitle("torchwavelets under PyTorch Lightning mixed precision", fontsize=13)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
