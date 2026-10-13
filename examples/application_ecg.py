"""Application 2: ECG denoising with the stationary (undecimated) wavelet transform.

Real 360 Hz ECG (MIT-BIH Arrhythmia record 100, PhysioNet) is corrupted with
white Gaussian noise and denoised by soft thresholding of the wavelet
coefficients. The SWT (à-trous) version is compared with the critically
sampled DWT: because the SWT keeps every translation of the wavelet at every
scale, a thresholded ``(cA, cD)`` pyramid is guaranteed to be shift-consistent
and gives lower reconstruction RMSE at the same threshold rule.

Usage::

    python examples/application_ecg.py [--out docs/figures/ecg_denoising.png]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from _data import load_mitdb_ecg

import torchwavelets as tw


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def soft_threshold(x, lam):
    return torch.sign(x) * torch.clamp(x.abs() - lam, min=0.0)


def swt_denoise(x, sigma, wave="db4", level=5, k=3.0):
    """Per-level ``k * sigma_j`` soft thresholding with the SWT.

    ``sigma_j`` is measured from a pure-noise forward pass of the same
    transform (the SWT keeps the noise level in every detail level).
    """
    nhps = tw.swt1d(torch.randn_like(x) * sigma, wave, level=level, norm=False)
    coeffs = tw.swt1d(x, wave, level=level, norm=False)
    th = []
    for (cA, cD), (_, ncD) in zip(coeffs, nhps, strict=True):
        th.append((cA, soft_threshold(cD, k * ncD.std())))
    return tw.iswt1d(th, wave, norm=False)


def dwt_denoise(x, sigma, wave="db4", level=5, k=3.0):
    """Per-level ``k * sigma_j`` soft thresholding with the DWT."""
    _, nhps = tw.dwt1d(torch.randn_like(x) * sigma, wave, mode="symmetric", level=level)
    lo, hps = tw.dwt1d(x, wave, mode="symmetric", level=level)
    th = []
    for h, nh in zip(hps, nhps, strict=True):
        th.append(soft_threshold(h, k * nh.std()))
    cur = lo
    for j in range(level - 1, -1, -1):
        cur = tw.idwt1d((cur, [th[j]]), wave=wave, mode="symmetric")
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/ecg_denoising.png")
    ap.add_argument("--sigma", type=float, default=0.05)  # mV
    args = ap.parse_args()
    torch.manual_seed(0)

    x = torch.tensor(load_mitdb_ecg(channel=0, nsamples=8192))[None, None]
    noise = torch.randn_like(x) * args.sigma
    y = x + noise
    snr0 = 20 * np.log10(x.std() / noise.std())

    xs = torch.clamp(swt_denoise(y, args.sigma), -2.0, 2.0)
    xd = torch.clamp(dwt_denoise(y, args.sigma), -2.0, 2.0)
    r_s, r_d = rmse(x.numpy(), xs.numpy()), rmse(x.numpy(), xd.numpy())
    snr_s = 20 * np.log10(x.std() / (xs - x).std())
    snr_d = 20 * np.log10(x.std() / (xd - x).std())
    print(f"noisy SNR: {snr0:.1f} dB")
    print(f"SWT  denoised: RMSE {r_s * 1000:.1f} uV, SNR {snr_s:.1f} dB")
    print(f"DWT  denoised: RMSE {r_d * 1000:.1f} uV, SNR {snr_d:.1f} dB")

    fs = 360.0
    t = np.arange(x.numel()) / fs
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    for ax, sig, title in zip(
        axes,
        [x.numpy()[0, 0], y.numpy()[0, 0], xs.numpy()[0, 0], xd.numpy()[0, 0]],
        [
            "clean ECG (MIT-BIH 100, ch 0)",
            f"noisy (sigma = {args.sigma * 1000:.0f} uV, SNR {snr0:.1f} dB)",
            f"SWT denoised  (RMSE {r_s * 1000:.1f} uV, SNR {snr_s:.1f} dB)",
            f"DWT denoised  (RMSE {r_d * 1000:.1f} uV, SNR {snr_d:.1f} dB)",
        ],
        strict=False,
    ):
        ax.plot(t, sig, lw=0.8, color="black")
        ax.set_ylabel("mV")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, t[-1])
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        "ECG denoising: SWT (à-trous) vs DWT, per-level 3-sigma soft thresholding", fontsize=14
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
