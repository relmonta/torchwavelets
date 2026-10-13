"""Single-scale DTCWT stages (plain functions, native complex tensors)."""

from __future__ import annotations

import torch

from .lowlevel import (
    coldfilt,
    colfilter,
    colifilt,
    highs_to_orientations,
    orientations_to_highs,
    rowdfilt,
    rowfilter,
    rowifilt,
)

__all__ = ["get_dimensions5", "fwd_j1", "inv_j1", "fwd_j2plus", "inv_j2plus"]


def get_dimensions5(o_dim: int) -> tuple[int, int, int]:
    """Map ``o_dim`` to ``(o_dim, h_dim, w_dim)`` for 5-D complex tensors
    ``(N, C, O, H, W)``."""
    o_dim = o_dim % 5
    if o_dim == 4:
        h_dim, w_dim = 2, 3
    elif o_dim == 3:
        h_dim, w_dim = 2, 4
    else:
        h_dim, w_dim = 3, 4
    return o_dim, h_dim, w_dim


def fwd_j1(x: torch.Tensor, h0o, h1o, skip_hps: bool, o_dim: int):
    """Level-1 forward DTCWT (biorthogonal filters)."""
    if not skip_hps:
        lo = rowfilter(x, h0o)
        hi = rowfilter(x, h1o)
        ll = colfilter(lo, h0o)
        lh = colfilter(lo, h1o)
        hl = colfilter(hi, h0o)
        hh = colfilter(hi, h1o)
        highs = highs_to_orientations(lh, hl, hh, o_dim)
    else:
        ll = colfilter(rowfilter(x, h0o), h0o)
        highs = None
    return ll, highs


def inv_j1(ll: torch.Tensor, highs, g0o, g1o, o_dim: int, h_dim: int, w_dim: int) -> torch.Tensor:
    """Level-1 inverse DTCWT."""
    if highs is None:
        return rowfilter(colfilter(ll, g0o), g0o)
    lh, hl, hh = orientations_to_highs(highs, o_dim)

    # Possibly crop the lowpass to match the bandpass sizes.
    r, c = ll.shape[2:]
    r1, c1 = highs.shape[h_dim], highs.shape[w_dim]
    if r != r1 * 2:
        ll = ll[:, :, 1:-1]
    if c != c1 * 2:
        ll = ll[:, :, :, 1:-1]

    hi = colfilter(hh, g1o) + colfilter(hl, g0o)
    lo = colfilter(lh, g1o) + colfilter(ll, g0o)
    return rowfilter(hi, g1o) + rowfilter(lo, g0o)


def fwd_j2plus(x: torch.Tensor, h0a, h1a, h0b, h1b, skip_hps: bool, o_dim: int):
    """Level >= 2 forward DTCWT (quarter-sample-shift filters)."""
    if not skip_hps:
        lo = rowdfilt(x, h0b, h0a, False)
        hi = rowdfilt(x, h1b, h1a, True)
        ll = coldfilt(lo, h0b, h0a, False)
        lh = coldfilt(lo, h1b, h1a, True)
        hl = coldfilt(hi, h0b, h0a, False)
        hh = coldfilt(hi, h1b, h1a, True)
        highs = highs_to_orientations(lh, hl, hh, o_dim)
    else:
        ll = coldfilt(rowdfilt(x, h0b, h0a, False), h0b, h0a, False)
        highs = None
    return ll, highs


def inv_j2plus(ll: torch.Tensor, highs, g0a, g1a, g0b, g1b, o_dim: int, h_dim: int, w_dim: int):
    """Level >= 2 inverse DTCWT."""
    if highs is None:
        return rowifilt(colifilt(ll, g0b, g0a, False), g0b, g0a, False)
    lh, hl, hh = orientations_to_highs(highs, o_dim)

    hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a, False)
    lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a, False)
    return rowifilt(hi, g1b, g1a, True) + rowifilt(lo, g0b, g0a, False)
