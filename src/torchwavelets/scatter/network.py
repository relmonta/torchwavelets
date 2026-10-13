"""General (arbitrary-order) DTCWT scattering network.

A scattering network computes locally shift-invariant feature maps by
iterating wavelet bandpasses with the smooth modulus ``sqrt(|Wx|^2 + b^2) - b``
and smoothing the intermediate invariants with the wavelet lowpass. With the
DTCWT's near shift-invariance this gives a small, cheap, highly invariant
front-end for classification and segmentation.

Paths are sequences of increasing scales ``(j1 < j2 < ... < jm)`` with
``m <= max_order``. For each path the network returns one invariant:
``avg_pool2d(lowpass, 2)`` when the path can still be extended (the smoothed
modulus feeding the next order) and the raw ``|bandpass|`` at the terminal
paths. With ``(J=2, max_order=2)`` the output matches
:class:`~torchwavelets.scatter.lowlevel.ScatLayerj2` exactly.

Shift invariance: the raw level-1 modulus ``|W_1 x|`` is only *weakly* shift
invariant (the single-tree level-1 design), so individual raw bandpass
invariants drift with sub-scale shifts; the smoothed invariants (pooled
lowpasses) and the order-0 lowpass are exactly invariant at dyadic integer
shifts, and the invariance improves with scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dtcwt._autocast import autocast_off
from ..dtcwt.coeffs import biort as _biort
from ..dtcwt.coeffs import qshift as _qshift
from ..dtcwt.transform2d import (
    _compute_dtype,
    _dtcwt2d_impl,
    _prep_filt,
    _validate_names,
)

__all__ = ["DTCWTScattering", "dtcwt_scattering"]


def _smooth_mag(highs: torch.Tensor, magbias: float) -> torch.Tensor:
    """Smooth modulus ``sqrt(|z|^2 + b^2) - b`` of complex bandpasses."""
    return torch.sqrt(highs.abs().pow(2) + magbias**2) - magbias


def _resolve_filters(biort, qshift, dtype, device):
    _validate_names(biort, qshift)
    h0o, _, h1o, _ = _biort(biort)
    h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
    return {
        "h0o": _prep_filt(h0o, dtype, device),
        "h1o": _prep_filt(h1o, dtype, device),
        "h0a": _prep_filt(h0a, dtype, device),
        "h0b": _prep_filt(h0b, dtype, device),
        "h1a": _prep_filt(h1a, dtype, device),
        "h1b": _prep_filt(h1b, dtype, device),
    }


def _dtcwt_levels(u: torch.Tensor, f, levels: int):
    """``levels``-level DTCWT of ``u``; returns ``(lowpass, bandpass)``."""
    yl, yh = _dtcwt2d_impl(u, f, levels, [False] * levels, [False] * levels, 2)
    return yl, yh[-1]


def _scattering_impl(x, f, J, max_order, magbias):
    """BFS path expansion; returns ``(s0, [(path, invariant), ...])``.

    Invariant semantics (matching ScatLayerj2): the invariant of an
    extendable path is the smoothed (avg-pooled) lowpass of the level-1
    transform of its modulus; terminal paths return the raw modulus.
    """
    if J < 1:
        raise ValueError("J must be >= 1 for a scattering network")
    # order-0 invariant: the smoothed final lowpass
    yl, _ = _dtcwt_levels(x, f, J)
    s0 = F.avg_pool2d(yl, 2)

    results = []
    # frontier entries: (u, current_scale, path, pending_path)
    # pending_path = the path whose smoothed invariant comes from the level-1
    # transform of u (computed at this node).
    frontier = [(x, 0, (), None)]
    for _ in range(max_order):
        next_frontier = []
        for u, scale, path, pending in frontier:
            # level-1 transform of u: needed for the pending invariant and as
            # the first step of the j-loop below.
            ll1, hh1 = _dtcwt_levels(u, f, 1)
            u1 = _flatten_orientations(_smooth_mag(hh1, magbias))
            if pending is not None:
                results.append((pending, F.avg_pool2d(ll1, 2)))
            for j in range(scale + 1, J + 1):
                if j == scale + 1:
                    ll, hh, u_next = ll1, hh1, u1
                else:
                    ll, hh = _dtcwt_levels(u, f, j - scale)
                    u_next = _flatten_orientations(_smooth_mag(hh, magbias))
                new_path = path + (j,)
                if j < J and len(new_path) < max_order:
                    next_frontier.append((u_next, j, new_path, new_path))
                else:
                    results.append((new_path, _upstream_order(u_next, len(new_path))))
        frontier = next_frontier
    results.sort(key=lambda pv: (len(pv[0]), pv[0]))
    return s0, results


def _flatten_orientations(u: torch.Tensor) -> torch.Tensor:
    """``(N, C, 6, H, W)`` modulus -> ``(N, 6*C, H, W)``."""
    b, ch, _, h, w = u.shape
    return u.reshape(b, ch * 6, h, w)


def _upstream_order(u: torch.Tensor, m: int) -> torch.Tensor:
    """Rotate the orientation blocks to the reference channel order.

    Internally the moduli propagate channel-major as
    ``(C, o1st, o2nd, ..., om)``; the reference (``ScatLayerj2``) stacks the
    first orientation block last: ``(C, o2nd, ..., om, o1st)``. For a
    first-order modulus (``m == 1``) this is the identity — the reference
    keeps the moduli channel-major ``(C, o1st)``.
    """
    b, c6, h, w = u.shape
    c = c6 // 6**m
    blocks = u.reshape(b, c, *([6] * m), h, w)
    perm = (0, 1) + tuple(range(3, m + 2)) + (2, m + 2, m + 3)
    return blocks.permute(perm).reshape(b, c6, h, w)


def dtcwt_scattering(
    x: torch.Tensor,
    J: int = 3,
    max_order: int = 2,
    biort="near_sym_a",
    qshift="qshift_a",
    magbias: float = 1e-2,
) -> tuple[torch.Tensor, list[tuple[tuple[int, ...], torch.Tensor]]]:
    """Forward DTCWT scattering network.

    Args:
        x (torch.Tensor): ``(N, C, H, W)`` input.
        J (int): number of wavelet scales.
        max_order (int): maximum scattering order.
        biort (str): filter bank name.
        qshift (str): filter bank name.
        magbias (float): smoothing bias of the modulus.

    Returns:
        tuple[torch.Tensor, list[tuple[tuple[int, ...], torch.Tensor]]]:
        ``(s0, invariants)``: the smoothed order-0 lowpass
        ``(N, C, H/2^J, W/2^J)`` and a list of ``(path, tensor)`` pairs, one
        per path (orders 1..max_order, ordered by order then scale). The
        invariant size depends only on the path's final scale ``jm``:
        ``H/2^jm``.
    """
    if x.ndim != 4:
        raise ValueError(f"dtcwt_scattering expects (N, C, H, W); got {tuple(x.shape)}")
    if isinstance(J, bool) or not isinstance(J, int) or J < 1:
        raise ValueError(f"J must be a positive int; got {J!r}")
    if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 1:
        raise ValueError(f"max_order must be a positive int; got {max_order!r}")
    dtype = _compute_dtype(x)
    f = _resolve_filters(biort, qshift, dtype, x.device)
    with autocast_off(x.device):
        return _scattering_impl(x.to(dtype), f, J, max_order, float(magbias))


class DTCWTScattering(nn.Module):
    """DTCWT scattering network module (see :func:`dtcwt_scattering`)."""

    def __init__(self, J=3, max_order=2, biort="near_sym_a", qshift="qshift_a", magbias=1e-2):
        super().__init__()
        if isinstance(J, bool) or not isinstance(J, int) or J < 1:
            raise ValueError(f"J must be a positive int; got {J!r}")
        if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 1:
            raise ValueError(f"max_order must be a positive int; got {max_order!r}")
        self.J = J
        self.max_order = max_order
        self.biort = biort
        self.qshift = qshift
        self.magbias = float(magbias)
        filts = _resolve_filters(biort, qshift, torch.float64, torch.device("cpu"))
        for name, t in filts.items():
            self.register_buffer(name, t)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"DTCWTScattering expects (N, C, H, W); got {tuple(x.shape)}")
        dtype = _compute_dtype(x)
        f = {
            n: getattr(self, n).to(dtype=dtype, device=x.device)
            for n in ("h0o", "h1o", "h0a", "h0b", "h1a", "h1b")
        }
        with autocast_off(x.device):
            return _scattering_impl(x.to(dtype), f, self.J, self.max_order, self.magbias)

    def extra_repr(self) -> str:
        return (
            f"J={self.J}, max_order={self.max_order}, "
            f"biort={self.biort!r}, qshift={self.qshift!r}, "
            f"magbias={self.magbias}"
        )
