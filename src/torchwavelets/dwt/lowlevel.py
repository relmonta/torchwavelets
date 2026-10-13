"""Low-level 1-D analysis/synthesis filter banks operating on 4-D tensors.

All functions operate on ``(N, C, H, W)`` tensors and filter along one spatial
axis (``dim=2`` for columns, ``dim=3`` for rows). Filters must be 4-D tensors
of shape ``(1, 1, L, 1)`` (column) or ``(1, 1, 1, L)`` (row); analysis filters
must already be reversed (see :mod:`torchwavelets.core.filters`).

Everything is built from plain torch ops (``conv2d`` / ``conv_transpose2d`` /
index-based padding), so gradients flow through all dtypes, autocast works
natively, and no custom ``autograd.Function`` is required.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.modes import pad_indices, padding_sizes

__all__ = ["afb1d", "sfb1d"]


def _dim_slice(dim: int, sl) -> tuple[slice, ...]:
    s = [slice(None)] * 4
    s[dim] = sl
    return tuple(s)


def _cat_last(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Append a copy of the last sample along ``dim`` (used for odd lengths)."""
    idx = torch.tensor([x.shape[dim] - 1], device=x.device)
    return torch.cat((x, x.index_select(dim, idx)), dim=dim)


def afb1d(
    x: torch.Tensor,
    h0: torch.Tensor,
    h1: torch.Tensor,
    mode: str,
    dim: int,
) -> torch.Tensor:
    """1-D analysis filter bank along ``dim`` with decimation by 2.

    Args:
        x: 4-D input ``(N, C, H, W)``.
        h0, h1: 4-D analysis filters (reversed), shape ``(1, 1, L, 1)`` if
            ``dim == 2`` else ``(1, 1, 1, L)``.
        mode: boundary mode (see :mod:`torchwavelets.core.modes`).
        dim: spatial dimension to filter along (2 or 3).

    Returns:
        ``(N, 2C, ...)`` tensor with the lowpass/highpass outputs interleaved
        along the channel dimension as ``[lo_0, hi_0, lo_1, hi_1, ...]``.
    """
    C = x.shape[1]
    L = h0.numel()
    stride = (2, 1) if dim == 2 else (1, 2)
    h = torch.cat([h0, h1] * C, dim=0)

    if mode == "periodization":
        # Circular transform with the *reversed* analysis filter (conv2d is a
        # correlation): c[k] = sum_m h[m] * x[(2k + L/2 - m) mod n], which is
        # PyWavelets' periodization convention (bit-exact vs pywt.dwt).
        n = x.shape[dim]
        if n % 2 == 1:
            x = _cat_last(x, dim)
            n += 1
        shift = L // 2 - 1
        idx = (torch.arange(n + L - 2, device=x.device) - shift) % n
        xp = x.index_select(dim, idx)
        return F.conv2d(xp, h, stride=stride, groups=C)

    n = x.shape[dim]
    before, after = padding_sizes(n, L, mode)
    if mode == "zero":
        shape = list(x.shape)
        shape[dim] = n + before + after
        padded = torch.zeros(shape, dtype=x.dtype, device=x.device)
        padded[_dim_slice(dim, slice(before, before + n))] = x
    elif mode == "smooth":
        # PyWavelets 'smooth': linear extrapolation from the edges,
        # left pad (k = before..1): x0 + k*(x0 - x1), right pad (k = 1..after):
        # x[-1] + k*(x[-1] - x[-2]).
        x0 = x[_dim_slice(dim, slice(0, 1))]
        x1 = x[_dim_slice(dim, slice(1, 2))] if n >= 2 else torch.zeros_like(x0)
        xm1 = x[_dim_slice(dim, slice(-1, None))]
        xm2 = x[_dim_slice(dim, slice(-2, -1))] if n >= 2 else torch.zeros_like(xm1)
        kshape = [1, 1, 1, 1]
        kshape[dim] = -1
        k_left = torch.arange(before, 0, -1, dtype=x.dtype, device=x.device).view(kshape)
        k_right = torch.arange(1, after + 1, dtype=x.dtype, device=x.device).view(kshape)
        left = x0 + k_left * (x0 - x1)
        right = xm1 + k_right * (xm1 - xm2)
        padded = torch.cat([left, x, right], dim=dim)
    else:
        idx = pad_indices(n, before, after, mode, x.device)
        padded = x.index_select(dim, idx)
    return F.conv2d(padded, h, stride=stride, groups=C)


def sfb1d(
    lo: torch.Tensor,
    hi: torch.Tensor,
    g0: torch.Tensor,
    g1: torch.Tensor,
    mode: str,
    dim: int,
) -> torch.Tensor:
    """1-D synthesis filter bank along ``dim`` with upsampling by 2.

    Args:
        lo, hi: 4-D lowpass/highpass coefficients ``(N, C, ...)``.
        g0, g1: 4-D synthesis filters (natural order), shape ``(1, 1, L, 1)``
            if ``dim == 2`` else ``(1, 1, 1, L)``.
        mode: boundary mode.
        dim: spatial dimension (2 or 3).

    Returns:
        Reconstructed ``(N, C, ...)`` tensor.
    """
    C = lo.shape[1]
    L = g0.numel()
    stride = (2, 1) if dim == 2 else (1, 2)
    g0 = torch.cat([g0] * C, dim=0)
    g1 = torch.cat([g1] * C, dim=0)

    if mode == "periodization":
        # Circular synthesis: y[m] = sum_j g[j] * u[(m + j - L/2) mod 2n] with
        # u the upsampled (zero-inserted) coefficient signal. g is flipped
        # because conv2d is a correlation (PyWavelets' periodization, exact).
        n_in = lo.shape[dim]
        n_out = 2 * n_in
        u_lo = torch.zeros(
            *lo.shape[:dim], n_out, *lo.shape[dim + 1 :], dtype=lo.dtype, device=lo.device
        )
        u_hi = torch.zeros_like(u_lo)
        u_lo[_dim_slice(dim, slice(None, None, 2))] = lo
        u_hi[_dim_slice(dim, slice(None, None, 2))] = hi
        idx = (torch.arange(n_out + L - 1, device=lo.device) - L // 2) % n_out
        # Flip along the filter-tap dimension: conv2d is a correlation.
        # g0/g1 are already tiled to C channels above.
        flip = -2 if dim == 2 else -1
        g0 = g0.flip(flip)
        g1 = g1.flip(flip)
        y = F.conv2d(u_lo.index_select(dim, idx), g0, groups=C) + F.conv2d(
            u_hi.index_select(dim, idx), g1, groups=C
        )
        return y

    pad = (L - 2, 0) if dim == 2 else (0, L - 2)
    y = F.conv_transpose2d(lo, g0, stride=stride, padding=pad, groups=C)
    y = y + F.conv_transpose2d(hi, g1, stride=stride, padding=pad, groups=C)
    return y
