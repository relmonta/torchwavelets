"""Plot a 3-level 2-D DTCWT: the 6 complex orientations per scale and the
reconstruction.

Saves ``docs/figures/dtcwt.png``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from plot_decomposition import synthetic_image  # noqa: E402

import torchwavelets as tw

ORIENTATIONS = ["15°", "45°", "75°", "105°", "135°", "165°"]


def main() -> None:
    img = synthetic_image()
    x = torch.tensor(img, dtype=torch.float64)[None, None]

    yl, yh = tw.dtcwt2d(x, biort="near_sym_a", qshift="qshift_a", J=3)
    xr = tw.idtcwt2d((yl, yh))
    rel_err = np.linalg.norm(xr[0, 0].numpy() - img) / np.linalg.norm(img)

    fig, axes = plt.subplots(2, 6, figsize=(17, 6.2))

    def imshow(ax, arr, title, cmap="gray"):
        v = np.abs(arr).max()
        ax.imshow(arr, cmap=cmap, vmin=-v, vmax=v)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Top row: magnitude of the 6 orientations at the finest scale.
    mags = [np.abs(yh[0][0, 0, k].numpy()) for k in range(6)]
    for k in range(6):
        imshow(axes[0, k], mags[k], f"|{ORIENTATIONS[k]}| (scale 1)")

    # Bottom row: magnitude of the 6 orientations at scale 2, plus
    # the reconstruction error as the last panel.
    mags2 = [np.abs(yh[1][0, 0, k].numpy()) for k in range(6)]
    for k in range(5):
        imshow(axes[1, k], mags2[k], f"|{ORIENTATIONS[k]}| (scale 2)")
    imshow(axes[1, 5], xr[0, 0].numpy() - img, f"recon. err\n({rel_err:.2e})")

    fig.suptitle(
        "2-D DTCWT with torchwavelets: 6 complex orientations per scale "
        "(near_sym_a / qshift_a, J=3)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig("docs/figures/dtcwt.png", dpi=130)
    print("saved docs/figures/dtcwt.png")


if __name__ == "__main__":
    main()
