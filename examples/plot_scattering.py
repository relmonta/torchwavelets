"""Plot the DTCWT scattering feature map: first- and second-order invariants
of a synthetic image, plus a shift-invariance demonstration.

Saves ``docs/figures/scattering.png``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from plot_decomposition import synthetic_image  # noqa: E402

import torchwavelets as tw

ORIENTATIONS = ["15°", "45°", "75°", "105°", "135°", "165°"]


def main() -> None:
    img = synthetic_image(256)
    x = torch.tensor(img, dtype=torch.float64)[None, None]

    layer = tw.ScatLayerj2(magbias=1e-2)
    z = layer(x)
    print("scattering output:", tuple(z.shape))

    fig, axes = plt.subplots(2, 7, figsize=(18, 6.4))

    def imshow(ax, arr, title):
        v = np.abs(arr).max()
        ax.imshow(arr, cmap="gray", vmin=-v, vmax=v)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    imshow(axes[0, 0], img, "input")
    for k in range(6):
        imshow(axes[0, k + 1], z[0, k + 1].numpy(), f"|W| scale1 {ORIENTATIONS[k]}")

    imshow(axes[1, 0], z[0, 0].numpy(), "smoothed lowpass")
    for k in range(6):
        imshow(axes[1, k + 1], z[0, 7 + k].numpy(), f"|W| scale2 {ORIENTATIONS[k]}")

    # shift invariance: translate the input by 4 px and compare
    shift = 4
    pad = F.pad(x, (0, shift, 0, shift), mode="reflect")
    xs = pad[..., shift : shift + 256, shift : shift + 256]
    zs = layer(xs)
    aligned = torch.roll(zs, shifts=(1, 1), dims=(-1, -2))
    m = 8
    err = (aligned[..., m:-m, m:-m] - z[..., m:-m, m:-m]).abs().max().item()
    scale = z.abs().max().item()
    print(f"shift invariance: rel err after alignment = {err / scale:.3e}")

    fig.suptitle(
        "DTCWT scattering (ScatLayerj2): first- and second-order invariants "
        f"— shift-invariant to {err / scale:.1e} rel.",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig("docs/figures/scattering.png", dpi=130)
    print("saved docs/figures/scattering.png")


if __name__ == "__main__":
    main()
