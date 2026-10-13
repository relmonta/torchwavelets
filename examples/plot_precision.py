"""Perfect-reconstruction error as a function of precision and of the number
of decomposition levels.

Saves ``docs/figures/precision.png``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

import torchwavelets as tw

DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
LEVELS = [1, 2, 3, 4, 5]


def _roundtrip_err(x: torch.Tensor, dtype: torch.dtype, level: int) -> float:
    yl, yh = tw.dwt2d(x.to(dtype), wave="db4", mode="symmetric", level=level)
    xr = tw.idwt2d((yl, yh), wave="db4", mode="symmetric").double()
    return (x - xr).norm().item() / x.norm().item()


def main() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 1, 128, 128, generator=g)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: relative error vs dtype (J=4, symmetric, db4)
    errs = [_roundtrip_err(x, dt, 4) for dt in DTYPES]
    ax1.bar([str(d).replace("torch.", "") for d in DTYPES], errs, color="#4C72B0")
    ax1.set_yscale("log")
    ax1.set_ylabel("relative reconstruction error")
    ax1.set_xlabel("precision (input dtype)")
    ax1.set_title("Round-trip error by dtype (db4, J=4, symmetric)")
    for i, e in enumerate(errs):
        ax1.text(i, e * 1.3, f"{e:.1e}", ha="center", fontsize=9)

    # Panel 2: error vs levels (float32 vs float64)
    err32 = [_roundtrip_err(x, torch.float32, j) for j in LEVELS]
    err64 = [_roundtrip_err(x, torch.float64, j) for j in LEVELS]
    ax2.plot(LEVELS, err32, "o-", label="float32")
    ax2.plot(LEVELS, err64, "s--", label="float64")
    ax2.set_yscale("log")
    ax2.set_xlabel("number of levels J")
    ax2.set_ylabel("relative reconstruction error")
    ax2.set_title("Round-trip error vs levels (db4, symmetric)")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("docs/figures/precision.png", dpi=130)
    print("saved docs/figures/precision.png")


if __name__ == "__main__":
    main()
