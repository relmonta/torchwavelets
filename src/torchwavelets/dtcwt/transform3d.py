"""3-D dual-tree complex wavelet transform (functional and module API).

The 3-D DTCWT (Kingsbury's design, as in the ``dtcwt`` numpy package) runs
four separable trees with the level-1 biorthogonal and level-2+
quarter-sample-shift filters on a doubled work grid and combines the octant
outputs into 28 complex orientation subbands per level (7 ``(H/L)`` face
combinations x 4 subbands each).

Precision follows the input (float32 -> complex64, float64 -> complex128);
half inputs are computed in float32. Boundary handling is Kingsbury's
symmetric extension (``ext_mode=4``: each input dimension must be even;
higher levels are edge-padded to a multiple of 4).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..dwt._validation import check_spatial
from ._autocast import autocast_off
from .coeffs import biort as _biort
from .coeffs import qshift as _qshift
from .lowlevel import coldfilt, colfilter, colifilt
from .transform2d import _compute_dtype, _prep_filt, _validate_names

__all__ = ["dtcwt3d", "idtcwt3d", "DTCWT3D", "IDTCWT3D"]


def _axis_fn(x: torch.Tensor, fn, axis: int):
    """Apply ``fn`` (filters the dim-2 of a 4-D view) along ``axis`` of 5-D."""
    order = [0, 1, axis] + [i for i in (2, 3, 4) if i != axis]
    xp = x.permute(order)
    shp = xp.shape
    y = fn(xp.reshape(shp[0], shp[1], shp[2], shp[3] * shp[4]))
    out_shp = list(shp)
    out_shp[2] = y.shape[2]  # the filtered dim may change size
    y = y.reshape(out_shp)
    inv = [0, 1] + [order.index(i) for i in (2, 3, 4)]
    return y.permute(inv)


def _colfilter_axis(x, h, axis):
    return _axis_fn(x, lambda t: colfilter(t, h), axis)


def _coldfilt_axis(x, ha, hb, highpass, axis):
    return _axis_fn(x, lambda t: coldfilt(t, ha, hb, highpass), axis)


def _colifilt_axis(x, ha, hb, highpass, axis):
    return _axis_fn(x, lambda t: colifilt(t, ha, hb, highpass), axis)


def _pad_to_multiple(x, mult: int) -> torch.Tensor:
    for axis in (2, 3, 4):
        n = x.shape[axis]
        if n % mult != 0:
            before = (
                x[..., :1, :, :]
                if axis == 2
                else x[..., :, :1, :]
                if axis == 3
                else x[..., :, :, :1]
            )
            after = (
                x[..., -1:, :, :]
                if axis == 2
                else x[..., :, -1:, :]
                if axis == 3
                else x[..., :, :, -1:]
            )
            x = torch.cat((before, x, after), dim=axis)
    return x


def _cube2c(y: torch.Tensor) -> torch.Tensor:
    """Octant (quad-tree) -> 4 complex subbands (numpy dtcwt cube2c)."""
    A = y[..., 0::2, 0::2, 0::2]
    B = y[..., 0::2, 1::2, 0::2]
    C = y[..., 1::2, 0::2, 0::2]
    D = y[..., 1::2, 1::2, 0::2]
    E = y[..., 0::2, 0::2, 1::2]
    F = y[..., 0::2, 1::2, 1::2]
    G = y[..., 1::2, 0::2, 1::2]
    H = y[..., 1::2, 1::2, 1::2]
    p = 0.5 * (A - G - D - F) + 0.5j * (B - H + C + E)
    q = 0.5 * (A - G + D + F) + 0.5j * (-B + H + C + E)
    r = 0.5 * (A + G + D - F) + 0.5j * (B + H - C + E)
    s = 0.5 * (A + G - D + F) + 0.5j * (-B - H - C + E)
    return torch.stack((p, q, r, s), dim=-1)


def _c2cube(z: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_cube2c` (numpy dtcwt c2cube)."""
    p, q, r, s = torch.unbind(z, dim=-1)
    pr, pi = p.real, p.imag
    qr, qi = q.real, q.imag
    rr, ri = r.real, r.imag
    sr, si = s.real, s.imag
    shp = z.shape[:-1]
    y = torch.zeros(shp[:2] + tuple(2 * s for s in shp[2:]), dtype=pr.dtype, device=z.device)
    y[..., 0::2, 0::2, 0::2] = pr + qr + rr + sr
    y[..., 1::2, 0::2, 1::2] = -pr - qr + rr + sr
    y[..., 1::2, 1::2, 0::2] = -pr + qr + rr - sr
    y[..., 0::2, 1::2, 1::2] = -pr + qr - rr + sr
    y[..., 0::2, 1::2, 0::2] = pi - qi + ri - si
    y[..., 1::2, 1::2, 1::2] = -pi + qi + ri - si
    y[..., 1::2, 0::2, 0::2] = pi + qi - ri - si
    y[..., 0::2, 0::2, 1::2] = pi + qi + ri + si
    return y * 0.5


def _resolve(biort, qshift, dtype, device):
    _validate_names(biort, qshift)
    h0o, g0o, h1o, g1o = _biort(biort)
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
    return {
        "h0o": _prep_filt(h0o, dtype, device),
        "g0o": _prep_filt(g0o, dtype, device),
        "h1o": _prep_filt(h1o, dtype, device),
        "g1o": _prep_filt(g1o, dtype, device),
        "h0a": _prep_filt(h0a, dtype, device),
        "h0b": _prep_filt(h0b, dtype, device),
        "h1a": _prep_filt(h1a, dtype, device),
        "h1b": _prep_filt(h1b, dtype, device),
        "g0a": _prep_filt(g0a, dtype, device),
        "g0b": _prep_filt(g0b, dtype, device),
        "g1a": _prep_filt(g1a, dtype, device),
        "g1b": _prep_filt(g1b, dtype, device),
    }


def _level1_xfm(x, h0o, h1o):
    D, H, W = x.shape[2:]
    L = h0o.numel()
    d2, h2, w2 = 2 * D, 2 * H, 2 * W
    if L % 2 == 0:
        d2, h2, w2 = d2 + 2, h2 + 2, w2 + 2
    work = x.new_zeros((x.shape[0], x.shape[1], d2, h2, w2))
    if L % 2 == 0:
        work[..., :D, :H, :W] = x
        work[..., D, :H, :W] = x[..., -1:, :, :]
        work[..., :D, H, :W] = x[..., :, -1:, :]
        work[..., :D, :H, W] = x[..., :, :, -1:]
        work[..., D, H, W] = x[..., -1:, -1:, -1:]
    else:
        work[..., :D, :H, :W] = x

    # 1: filter along W (first-D x first-H block; fills both W halves)
    blk = work[..., :D, :H, :W]
    work[..., :D, :H, W:] = _colfilter_axis(blk, h1o, 4)
    work[..., :D, :H, :W] = _colfilter_axis(blk, h0o, 4)

    # 2: filter along H (into 2H), then along D
    blk2 = work[..., :D, :H, :]
    y2 = torch.cat((_colfilter_axis(blk2, h0o, 3), _colfilter_axis(blk2, h1o, 3)), dim=3)
    work[..., :D, :, :] = _colfilter_axis(y2, h0o, 2)
    work[..., D:, :, :] = _colfilter_axis(y2, h1o, 2)

    return (
        work[..., :D, :H, :W],
        torch.cat(
            (
                _cube2c(work[..., :D, H : 2 * H, :W]),
                _cube2c(work[..., D : 2 * D, :H, :W]),
                _cube2c(work[..., D : 2 * D, H : 2 * H, :W]),
                _cube2c(work[..., :D, :H, W : 2 * W]),
                _cube2c(work[..., :D, H : 2 * H, W : 2 * W]),
                _cube2c(work[..., D : 2 * D, :H, W : 2 * W]),
                _cube2c(work[..., D : 2 * D, H : 2 * H, W : 2 * W]),
            ),
            dim=-1,
        ),
    )


def _level2_xfm(x, h0a, h0b, h1a, h1b):
    x = _pad_to_multiple(x, 4)
    D, H, W = x.shape[2:]

    # 1: filter along W (full D, H); both outputs from the original input
    hi_w = _coldfilt_axis(x, h1b, h1a, True, 4)
    lo_w = _coldfilt_axis(x, h0b, h0a, False, 4)
    work = torch.cat((lo_w, hi_w), dim=4)

    # 2: filter along H, concatenate, then along D
    a = _coldfilt_axis(work, h0b, h0a, False, 3)
    b = _coldfilt_axis(work, h1b, h1a, True, 3)
    y2 = torch.cat((a, b), dim=3)
    work = torch.cat(
        (
            _coldfilt_axis(y2, h0b, h0a, False, 2),
            _coldfilt_axis(y2, h1b, h1a, True, 2),
        ),
        dim=2,
    )

    D2, H2, W2 = work.shape[2:]
    s0a = slice(0, D2 >> 1)
    s0b = slice(D2 >> 1, None)
    s1a = slice(0, H2 >> 1)
    s1b = slice(H2 >> 1, None)
    s2a = slice(0, W2 >> 1)
    s2b = slice(W2 >> 1, None)
    return (
        work[..., s0a, s1a, s2a],
        torch.cat(
            (
                _cube2c(work[..., s0a, s1b, s2a]),
                _cube2c(work[..., s0b, s1a, s2a]),
                _cube2c(work[..., s0b, s1b, s2a]),
                _cube2c(work[..., s0a, s1a, s2b]),
                _cube2c(work[..., s0a, s1b, s2b]),
                _cube2c(work[..., s0b, s1a, s2b]),
                _cube2c(work[..., s0b, s1b, s2b]),
            ),
            dim=-1,
        ),
    )


def _level1_ifm(yl, yh, g0o, g1o):
    D, H, W = yl.shape[2:]
    L = g0o.numel()
    d2, h2, w2 = 2 * D, 2 * H, 2 * W
    if L % 2 == 0:
        d2, h2, w2 = d2 + 2, h2 + 2, w2 + 2
    work = yl.new_zeros((yl.shape[0], yl.shape[1], d2, h2, w2))
    work[..., :D, :H, :W] = yl
    work[..., :D, H : 2 * H, :W] = _c2cube(yh[..., 0:4])
    work[..., D : 2 * D, :H, :W] = _c2cube(yh[..., 4:8])
    work[..., D : 2 * D, H : 2 * H, :W] = _c2cube(yh[..., 8:12])
    work[..., :D, :H, W : 2 * W] = _c2cube(yh[..., 12:16])
    work[..., :D, H : 2 * H, W : 2 * W] = _c2cube(yh[..., 16:20])
    work[..., D : 2 * D, :H, W : 2 * W] = _c2cube(yh[..., 20:24])
    work[..., D : 2 * D, H : 2 * H, W : 2 * W] = _c2cube(yh[..., 24:28])

    # rows (H): filter both H halves along H, then combine along D
    y = _colfilter_axis(work[..., :, :H, :], g0o, 3) + _colfilter_axis(work[..., :, H:, :], g1o, 3)
    work[..., :D, :H, :] = _colfilter_axis(y[..., :D, :, :], g0o, 2) + _colfilter_axis(
        y[..., D:, :, :], g1o, 2
    )

    # depth (D): filter both D halves along W, combine into the first W half
    z = _colfilter_axis(work[..., :D, :H, :W], g0o, 4) + _colfilter_axis(
        work[..., :D, :H, W:], g1o, 4
    )
    work[..., :D, :H, :W] = z

    if L % 2 == 0:
        return work[..., 1 : d2 >> 1, 1 : h2 >> 1, 1 : w2 >> 1]
    return work[..., :D, :H, :W]


def _level2_ifm(yl, yh, g0a, g0b, g1a, g1b, prev_level_size):
    D, H, W = yl.shape[2:]
    work = yl.new_zeros((yl.shape[0], yl.shape[1], 2 * D, 2 * H, 2 * W))
    work[..., :D, :H, :W] = yl
    work[..., :D, H:, :W] = _c2cube(yh[..., 0:4])
    work[..., D:, :H, :W] = _c2cube(yh[..., 4:8])
    work[..., D:, H:, :W] = _c2cube(yh[..., 8:12])
    work[..., :D, :H, W:] = _c2cube(yh[..., 12:16])
    work[..., :D, H:, W:] = _c2cube(yh[..., 16:20])
    work[..., D:, :H, W:] = _c2cube(yh[..., 20:24])
    work[..., D:, H:, W:] = _c2cube(yh[..., 24:28])

    # rows (H): combine the H halves, then the D halves
    y = _colifilt_axis(work[..., :, :H, :], g0b, g0a, False, 3) + _colifilt_axis(
        work[..., :, H:, :], g1b, g1a, True, 3
    )
    work = _colifilt_axis(y[..., :D, :, :], g0b, g0a, False, 2) + _colifilt_axis(
        y[..., D:, :, :], g1b, g1a, True, 2
    )

    # depth (D): combine the W halves
    work = _colifilt_axis(work[..., :, :, :W], g0b, g0a, False, 4) + _colifilt_axis(
        work[..., :, :, W:], g1b, g1a, True, 4
    )

    if 2 * yh.shape[2] != prev_level_size[0]:
        work = work[..., 1:-1, :, :]
    if 2 * yh.shape[3] != prev_level_size[1]:
        work = work[..., :, 1:-1, :]
    if 2 * yh.shape[4] != prev_level_size[2]:
        work = work[..., :, :, 1:-1]
    return work


def _dtcwt3d_impl(x, f, J):
    yh = []
    yl = x
    for level in range(J):
        if level == 0:
            yl, h = _level1_xfm(yl, f["h0o"], f["h1o"])
        else:
            yl, h = _level2_xfm(yl, f["h0a"], f["h0b"], f["h1a"], f["h1b"])
        yh.append(h)
    return yl, yh


def _validate_even3d(dims):
    """The 3-D work grid requires even spatial dims at every level."""
    check_spatial(dims, ("D", "H", "W"), "dtcwt3d")
    bad = [d for d in dims if d % 2 != 0]
    if bad:
        raise ValueError(
            "dtcwt3d requires even spatial dimensions (the doubled work grid "
            f"needs even sizes); got {tuple(dims)}"
        )


def _idtcwt3d_impl(yl, yh, f):
    J = len(yh)
    for level in range(J - 1, 0, -1):
        prev = yh[level - 1].shape[2:5]
        yl = _level2_ifm(yl, yh[level], f["g0a"], f["g0b"], f["g1a"], f["g1b"], prev)
    yl = _level1_ifm(yl, yh[0], f["g0o"], f["g1o"])
    return yl


def dtcwt3d(
    x: torch.Tensor, biort="near_sym_a", qshift="qshift_a", J=3
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Forward 3-D DTCWT.

    Args:
        x (torch.Tensor): ``(N, C, D, H, W)`` input with even spatial dims.
        biort (str): filter bank name.
        qshift (str): filter bank name.
        J (int): number of levels.

    Returns:
        tuple[torch.Tensor, list[torch.Tensor]]: ``(yl, yh)``: real lowpass
        ``(N, C, D', H', W')`` and a list of ``J`` complex bandpasses
        ``(N, C, D'', H'', W'', 28)`` (finest first).
    """
    if x.ndim != 5:
        raise ValueError(f"dtcwt3d expects (N, C, D, H, W); got {tuple(x.shape)}")
    if isinstance(J, bool) or not isinstance(J, int) or J < 0:
        raise ValueError(f"J must be a non-negative int; got {J!r}")
    if J == 0:
        return x, None
    _validate_even3d(x.shape[2:])
    dtype = _compute_dtype(x)
    f = _resolve(biort, qshift, dtype, x.device)
    with autocast_off(x.device):
        return _dtcwt3d_impl(x.to(dtype), f, J)


def idtcwt3d(coeffs, biort="near_sym_a", qshift="qshift_a") -> torch.Tensor:
    """Inverse 3-D DTCWT (exact inverse of :func:`dtcwt3d`)."""
    yl, yh = coeffs
    if yl is None:
        raise ValueError("The lowpass coefficient tensor yl cannot be None")
    dtype = _compute_dtype(yl)
    f = _resolve(biort, qshift, dtype, yl.device)
    with autocast_off(yl.device):
        return _idtcwt3d_impl(yl.to(dtype), yh, f)


class DTCWT3D(nn.Module):
    """3-D DTCWT module (see :func:`dtcwt3d`)."""

    def __init__(self, biort="near_sym_a", qshift="qshift_a", J=3):
        super().__init__()
        if isinstance(J, bool) or not isinstance(J, int) or J < 0:
            raise ValueError(f"J must be a non-negative int; got {J!r}")
        self.biort = biort
        self.qshift = qshift
        self.J = J
        filts = _resolve(biort, qshift, torch.float64, torch.device("cpu"))
        for name in ("h0o", "h1o", "h0a", "h1a", "h0b", "h1b"):
            self.register_buffer(name, filts[name])

    def forward(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"DTCWT3D expects (N, C, D, H, W); got {tuple(x.shape)}")
        if self.J == 0:
            return x, None
        _validate_even3d(x.shape[2:])
        dtype = _compute_dtype(x)
        f = {
            n: getattr(self, n).to(dtype=dtype, device=x.device)
            for n in ("h0o", "h1o", "h0a", "h1a", "h0b", "h1b")
        }
        with autocast_off(x.device):
            return _dtcwt3d_impl(x.to(dtype), f, self.J)

    def extra_repr(self) -> str:
        return f"biort={self.biort!r}, qshift={self.qshift!r}, J={self.J}"


class IDTCWT3D(nn.Module):
    """3-D inverse DTCWT module (see :func:`idtcwt3d`)."""

    def __init__(self, biort="near_sym_a", qshift="qshift_a"):
        super().__init__()
        self.biort = biort
        self.qshift = qshift
        filts = _resolve(biort, qshift, torch.float64, torch.device("cpu"))
        for name in ("g0o", "g1o", "g0a", "g1a", "g0b", "g1b"):
            self.register_buffer(name, filts[name])

    def forward(self, coeffs):
        yl, yh = coeffs
        if yl is None:
            raise ValueError("The lowpass coefficient tensor yl cannot be None")
        if yl.ndim != 5:
            raise ValueError(f"IDTCWT3D expects 5-D coefficients; got shape {tuple(yl.shape)}")
        dtype = _compute_dtype(yl)
        f = {
            n: getattr(self, n).to(dtype=dtype, device=yl.device)
            for n in ("g0o", "g1o", "g0a", "g1a", "g0b", "g1b")
        }
        with autocast_off(yl.device):
            return _idtcwt3d_impl(yl.to(dtype), yh, f)

    def extra_repr(self) -> str:
        return f"biort={self.biort!r}, qshift={self.qshift!r}"
