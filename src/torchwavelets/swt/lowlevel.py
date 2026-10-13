"""Low-level stationary (undecimated / a trous) wavelet filter banks.

The SWT is the undecimated periodic wavelet transform: at level ``j`` the
analysis filters are dilated (zero-stuffed) by ``2**(j-1)`` and applied with
no decimation, so every subband has the same length as the input. The
phase conventions match PyWavelets exactly (verified bit-for-bit by the
parity tests).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["swt_axis", "dilate_filter"]


def dilate_filter(h: torch.Tensor, fstep: int) -> torch.Tensor:
    """Insert ``fstep - 1`` zeros between filter taps (a trous dilation)."""
    L = h.numel()
    e = torch.zeros(1, 1, L * fstep, 1, dtype=h.dtype, device=h.device)
    e[0, 0, ::fstep, 0] = h.reshape(-1)
    return e


def swt_axis(x: torch.Tensor, h: torch.Tensor, dim: int, fstep: int) -> torch.Tensor:
    """One stationary analysis filter along ``dim`` (no decimation).

    Args:
        x: ``(N, C, H, W)`` input.
        h: reversed analysis filter, ``(1, 1, L, 1)`` (column orientation;
            transposed internally for ``dim == 3``).
        dim: spatial dimension (2 or 3).
        fstep: dilation factor ``2**(level - 1)``.

    Returns:
        ``(N, C, ...)`` with the same spatial size as ``x``.
    """
    n = x.shape[dim]
    L = h.numel()
    Ld = L * fstep  # dilated filter length
    e = dilate_filter(h, fstep)
    if dim == 3:
        e = e.transpose(2, 3).contiguous()
    # pywt: out[i] = sum_m dec[m] x[(i + (L/2 - m)*fstep) mod n]; with the
    # reversed-filter convention this is out[i] = sum_k h_rev[k]
    # x[(i + (k - L/2 + 1)*fstep) mod n].
    shift = fstep * (L // 2 - 1)
    idx = (torch.arange(n + Ld - 1, device=x.device) - shift) % n
    xp = x.index_select(dim, idx)
    stride = (1, 1)
    C = x.shape[1]
    return F.conv2d(xp, e.repeat(C, 1, 1, 1), stride=stride, groups=C)
