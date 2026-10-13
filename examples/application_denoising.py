"""Application 1: image denoising with the dual-tree complex wavelet transform.

A real photograph is corrupted with white Gaussian noise (sigma = 25/255) and
denoised by per-level universal soft thresholding of the wavelet coefficients,
once with the real DWT and once with the redundant, nearly shift-invariant
DTCWT (complex soft thresholding of the six orientation bandpasses). The
redundancy of the DTCWT lets the complex threshold keep the signal while
suppressing noise, which typically yields 1-2 dB better PSNR.

Usage::

    python examples/application_denoising.py [--out docs/figures/denoising.png]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from _data import load_photo

import torchwavelets as tw


def psnr(x, y):
    mse = ((x - y) ** 2).mean()
    return 10 * np.log10(1.0 / mse)


def soft_threshold(x, lam):
    return torch.sign(x) * torch.clamp(x.abs() - lam, min=0.0)


def complex_soft_threshold(z, lam):
    """Kingsbury complex shrinkage: ``z * max(1 - lam/|z|, 0)``."""
    mag = z.abs()
    scale = torch.clamp(1.0 - lam / (mag + 1e-12), min=0.0)
    return z * scale


def dwt_denoise(x, sigma, wave="db4", J=4, mode="periodization", k=3.0):
    """Per-level ``k * sigma_j`` soft thresholding with the DWT.

    ``sigma_j`` is the measured standard deviation of the level-j detail
    coefficients of a pure-noise forward pass (the DWT is orthonormal, so
    ``sigma_j == sigma`` for all levels).
    """
    noise = torch.randn_like(x) * sigma
    _, nhps = tw.dwt2d(noise, wave=wave, mode=mode, level=J)
    lo, hps = tw.dwt2d(x, wave=wave, mode=mode, level=J)
    thresh = []
    for h, nh in zip(hps, nhps, strict=True):
        lam = k * nh.std()
        thresh.append(soft_threshold(h, lam))
    # reconstruct level by level
    cur = lo
    for j in range(J - 1, -1, -1):
        cur = tw.idwt2d((cur, [thresh[j]]), wave=wave, mode=mode)
    return cur


def dtcwt_denoise(x, sigma, J=4, k=3.0):
    """Per-level ``k * sigma_j`` complex soft thresholding with the DTCWT.

    ``sigma_j`` is the measured standard deviation of the modulus of the
    level-j orientation bandpasses of a pure-noise forward pass (the DTCWT's
    2x redundancy lowers the coefficient noise floor well below ``sigma``).
    """
    fwd = tw.DTCWT2D(J=J)
    inv = tw.IDTCWT2D()
    _, nyh = fwd(torch.randn_like(x) * sigma)
    yl, yh = fwd(x)
    yh_th = []
    for h, nh in zip(yh, nyh, strict=True):
        lam = k * nh.abs().std()
        yh_th.append(complex_soft_threshold(h, lam))
    return inv((yl, yh_th))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/denoising.png")
    ap.add_argument("--sigma", type=float, default=25.0 / 255.0)
    args = ap.parse_args()
    torch.manual_seed(0)

    x = torch.tensor(load_photo(512))[None, None]
    noise = torch.randn_like(x) * args.sigma
    y = torch.clamp(x + noise, 0.0, 1.0)
    print(f"noisy PSNR: {psnr(x, y):.2f} dB")

    xd = dwt_denoise(y, args.sigma)
    xc = dtcwt_denoise(y, args.sigma)
    xd = torch.clamp(xd, 0.0, 1.0)
    xc = torch.clamp(xc, 0.0, 1.0)
    p_d, p_c = psnr(x, xd), psnr(x, xc)
    print(f"DWT   denoised PSNR: {p_d:.2f} dB")
    print(f"DTCWT denoised PSNR: {p_c:.2f} dB  (gain {p_c - p_d:+.2f} dB)")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    for ax, img, title in zip(
        axes.ravel(),
        [x, y, xd, xc],
        [
            "original (noiseless)",
            f"noisy  (sigma = {args.sigma * 255:.0f}/255, PSNR {psnr(x, y):.1f} dB)",
            f"DWT  db4 soft threshold  (PSNR {p_d:.1f} dB)",
            f"DTCWT complex threshold (PSNR {p_c:.1f} dB)",
        ],
        strict=False,
    ):
        ax.imshow(img[0, 0].numpy(), cmap="gray")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("Wavelet image denoising: DWT vs DTCWT (torchwavelets)", fontsize=14)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
