"""Boundary modes compared on a 1-D chirp signal.

Saves ``docs/figures/modes.png``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import torchwavelets as tw

MODES = ["zero", "symmetric", "reflect", "periodization", "periodic", "constant"]


def main() -> None:
    n = 256
    t = np.linspace(0, 1, n)
    signal = np.sin(2 * np.pi * (10 + 60 * t) * t) + 0.3 * np.cos(2 * np.pi * 25 * t)
    x = torch.tensor(signal, dtype=torch.float64)[None, None]

    fig, axes = plt.subplots(len(MODES), 2, figsize=(12, 12), sharex=True)

    for i, mode in enumerate(MODES):
        yl, yh = tw.dwt1d(x, wave="db4", mode=mode, level=4)
        xr = tw.idwt1d((yl, yh), wave="db4", mode=mode)
        err = np.linalg.norm(xr[0, 0].numpy() - signal) / np.linalg.norm(signal)

        axes[i, 0].plot(signal, lw=1, label="original")
        axes[i, 0].plot(xr[0, 0].numpy(), lw=1, ls="--", label="reconstruction")
        axes[i, 0].set_ylabel(mode, rotation=0, ha="right", va="center")
        axes[i, 0].set_title(f"rel. err = {err:.2e}", fontsize=9)
        axes[i, 0].legend(loc="lower right", fontsize=7)
        axes[i, 1].plot(yl[0, 0].numpy(), lw=1)
        axes[i, 1].set_title(f"lowpass after J=4 ({yl.shape[-1]} samples)", fontsize=9)
        axes[i, 1].axis("off")

    axes[0, 0].set_title("1-D DWT (db4, J=4): reconstruction", fontsize=10)
    fig.suptitle("Boundary modes in torchwavelets", fontsize=14)
    fig.tight_layout()
    fig.savefig("docs/figures/modes.png", dpi=130)
    print("saved docs/figures/modes.png")


if __name__ == "__main__":
    main()
