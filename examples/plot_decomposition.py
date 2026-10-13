"""Plot a 2-level 2-D DWT decomposition and its reconstruction.

Saves ``docs/figures/decomposition.png``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import torchwavelets as tw


def synthetic_image(n: int = 256, seed: int = 0) -> np.ndarray:
    """A structured test image: gratings, blobs, edges and noise."""
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = (
        0.5 * np.sin(2 * np.pi * 8 * xx)
        + 0.3 * np.sin(2 * np.pi * 4 * yy + 1.0)
        + 0.4 * np.exp(-((xx - 0.3) ** 2 + (yy - 0.7) ** 2) / 0.02)
        + 0.5 * ((xx + yy) % 1 < 0.02)
    )
    img += 0.05 * g.standard_normal((n, n))
    img -= img.min()
    return img / img.max()


def main() -> None:
    img = synthetic_image()
    x = torch.tensor(img, dtype=torch.float64)[None, None]

    yl, yh = tw.dwt2d(x, wave="db4", mode="symmetric", level=2)
    xr = tw.idwt2d((yl, yh), wave="db4", mode="symmetric")

    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    vmax = np.abs(img).max()

    def imshow(ax, arr, title, cmap="gray"):
        im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        return im

    imshow(axes[0, 0], img, "original")
    imshow(axes[0, 1], xr[0, 0].numpy(), "reconstruction (db4, J=2)")
    imshow(
        axes[0, 2],
        xr[0, 0].numpy() - img,
        "reconstruction error\n(rel. err = "
        f"{np.linalg.norm(xr[0, 0].numpy() - img) / np.linalg.norm(img):.2e})",
    )
    imshow(axes[0, 3], yl[0, 0].numpy(), "LL2 lowpass")

    titles = ["LH2", "HL2", "HH2", "LH1 (finest)"]
    bands = [yh[1][0, 0, 0], yh[1][0, 0, 1], yh[1][0, 0, 2], yh[0][0, 0, 0]]
    for ax, b, t in zip(axes[1], bands, titles, strict=True):
        arr = b.numpy()
        v = np.abs(arr).max()
        ax.imshow(arr, cmap="gray", vmin=-v, vmax=v)
        ax.set_title(t)
        ax.axis("off")

    fig.suptitle("2-D DWT with torchwavelets: 2-level db4 decomposition", fontsize=14)
    fig.tight_layout()
    fig.savefig("docs/figures/decomposition.png", dpi=130)
    print("saved docs/figures/decomposition.png")


if __name__ == "__main__":
    main()
