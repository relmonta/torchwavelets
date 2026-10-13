"""Showcase: every transform in ``torchwavelets`` applied to a real photograph.

Renders a 2x3 gallery figure: original image, DWT 2D, DTCWT 2D, SWT 2D,
wavelet packets 2D and DTCWT scattering invariants.

Usage::

    python examples/plot_transform_gallery.py [--out docs/figures/transform_gallery.png]

The photo (``peppers``, USC-SIPI) is downloaded to ``~/data/torchwavelets``
on first use.
"""

from __future__ import annotations

import argparse
import os
import urllib.request

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import torch

import torchwavelets as tw

DATA_DIR = os.path.expanduser("~/data/torchwavelets")
PEPPERS_URL = "https://sipi.usc.edu/database/misc/4.2.07.tiff"


def load_photo(size: int = 512) -> torch.Tensor:
    """Return a centred ``size x size`` grayscale crop of the peppers photo."""
    from PIL import Image

    path = os.path.join(DATA_DIR, "peppers.tiff")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading peppers photo -> {path}")
        urllib.request.urlretrieve(PEPPERS_URL, path)
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    h, w = arr.shape
    r = (h - size) // 2
    c = (w - size) // 2
    arr = arr[r : r + size, c : c + size] / 255.0
    return torch.tensor(arr)[None, None]


def imshow(ax, img, title, cmap="gray", vmin=None, vmax=None):
    img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0, 0] if img.shape[1] == 1 else img[0]
    if vmin is None and cmap != "gray":
        vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def norm_img(a, symmetric=True, plo=1, phi=99):
    """Per-block percentile normalisation to [-1, 1] (symmetric) or [0, 1]."""
    lo, hi = np.percentile(np.abs(a) if symmetric else a, [plo, phi])
    if symmetric:
        m = max(abs(lo), abs(hi), 1e-12)
        return np.clip(a / m, -1, 1)
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def dwt_composite(ll, details):
    """Classic 2-level DWT pyramid: LL2 + 3 level-2 details in the top-left
    quadrant, level-1 details filling the other three quadrants."""
    big = norm_img(ll[0, 0].numpy())
    h, w = big.shape
    level1 = [norm_img(details[0][0, 0, k].numpy()) for k in range(3)]
    level2 = [norm_img(details[1][0, 0, k].numpy()) for k in range(3)]
    comp = np.zeros((4 * h, 4 * w))
    comp[:h, :w] = big
    comp[:h, w : 2 * w] = level2[0]
    comp[h : 2 * h, :w] = level2[1]
    comp[h : 2 * h, w : 2 * w] = level2[2]
    for d, (r, c) in zip(level1, [(0, 2 * w), (2 * h, 0), (2 * h, 2 * w)], strict=False):
        scale = max(1, (2 * h) // d.shape[0])
        comp[r : r + 2 * h, c : c + 2 * w] = np.kron(d, np.ones((scale, scale)))
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/transform_gallery.png")
    args = ap.parse_args()

    torch.manual_seed(0)
    x = load_photo(512)
    print("photo:", tuple(x.shape))

    # --- transforms ----------------------------------------------------------
    dwt = tw.DWT2D(J=2, wave="db4", mode="periodization")
    ll, details = dwt(x)  # details[j]: (N, C, 3, H, W) = (cH, cV, cD)

    dtcwt = tw.DTCWT2D(J=2)
    _, yh = dtcwt(x)  # yh[j]: (N, C, 6, H, W) complex orientation bandpasses

    swt = tw.swt2d(x, "db2", level=2)  # list[(cA, [cH, cV, cD])], coarsest first

    wp = tw.WaveletPacket2D(x, "db2", mode="symmetric", maxlevel=2)
    nodes = wp.get_level(2)

    scat = tw.ScatLayerj2()(x)  # (N, 49, H/4, W/4)
    print("scattering:", tuple(scat.shape))

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # row 1 -------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    imshow(ax, x, "input photograph (512 x 512)")

    ax = fig.add_subplot(gs[0, 1])
    comp = dwt_composite(ll, details)
    ax.imshow(comp, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title("DWT 2D (db4, periodization, 2 levels)\nLL2 + 6 detail subbands", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    mags = yh[1].abs()[0, 0].numpy()  # (6, H, W): level-2 orientation magnitudes
    mosaic = np.concatenate(
        [
            np.concatenate([mags[i] for i in (0, 1, 2)], axis=1),
            np.concatenate([mags[i] for i in (3, 4, 5)], axis=1),
        ],
        axis=0,
    )
    ax.imshow(np.log1p(mosaic), cmap="magma", interpolation="nearest")
    ax.set_title("DTCWT 2D (level 2)\n6 complex orientations (log magnitude)", fontsize=9)
    ax.axis("off")

    # row 2 -------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    cA, cD = swt[0]
    grid = np.concatenate(
        [norm_img(cA[0, 0].numpy(), symmetric=False), norm_img(cD[0][0, 0].numpy())], axis=1
    )
    ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title("SWT 2D (db2, 2 levels, à-trous)\napproximation + detail", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    lv = min(nodes[0].data.shape[-2:])
    mosaic = np.zeros((lv * 4 + 3, lv * 4 + 3))
    for i, nd in enumerate(nodes):
        d = norm_img(nd.data[0, 0].numpy())
        scale = max(1, lv // d.shape[0])
        d = np.kron(d, np.ones((scale, scale)))[:lv, :lv]
        r, c = (i // 4) * (lv + 1), (i % 4) * (lv + 1)
        mosaic[r : r + lv, c : c + lv] = d
    ax.imshow(mosaic, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(
        "Wavelet packets 2D (db2, symmetric, 2 levels)\n16 nodes (a/h/v/d path ordering)",
        fontsize=9,
    )
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    chs = scat[0].numpy()  # (49, H/4, W/4)
    size = chs.shape[-1]
    sq = int(np.ceil(np.sqrt(chs.shape[0])))
    mosaic = np.zeros((sq * size + sq - 1, sq * size + sq - 1))
    for i in range(chs.shape[0]):
        r, c = i // sq, i % sq
        mosaic[r * (size + 1) : r * (size + 1) + size, c * (size + 1) : c * (size + 1) + size] = (
            norm_img(chs[i], symmetric=False)
        )
    ax.imshow(mosaic, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("DTCWT scattering (ScatLayerj2)\n49 shift-invariant channels", fontsize=9)
    ax.axis("off")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
