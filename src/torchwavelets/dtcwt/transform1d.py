"""1-D dual-tree complex wavelet transform (functional and module API).

The 1-D DTCWT produces complex bandpass coefficients from the two
decimation phases of the highpass filtering (real = even phase, imaginary =
odd phase), so a single 1-D DTCWT level yields a complex band that is
approximately shift-invariant and analytic. Levels 2+ use the
quarter-sample-shift filters, exactly as in the 2-D transform.

Precision follows the input (float32 -> complex64, float64 -> complex128);
half inputs are computed in float32 (no usable complex16 kernels).
Boundary handling is Kingsbury's symmetric extension.
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

__all__ = ["dtcwt1d", "idtcwt1d", "DTCWT1D", "IDTCWT1D"]


def _to_col(x: torch.Tensor) -> torch.Tensor:
    """``(N, C, L)`` -> ``(N, C, L, 1)`` column layout (like numpy dtcwt)."""
    return x.unsqueeze(-1)


def _from_col(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(-1)


def _complex_from_phases(hi: torch.Tensor) -> torch.Tensor:
    """Highpass phases -> complex bandpass: ``hi[..., ::2] + 1j hi[..., 1::2]``."""
    return torch.complex(hi[..., ::2, :], hi[..., 1::2, :])


def _phases_from_complex(z: torch.Tensor) -> torch.Tensor:
    """Complex bandpass -> interleaved real/imag phases (inverse of above)."""
    b, ch, r, c = z.shape
    out = torch.empty((b, ch, 2 * r, c), dtype=z.real.dtype, device=z.device)
    out[..., ::2, :] = z.real
    out[..., 1::2, :] = z.imag
    return out


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


def _dtcwt1d_impl(x, f, J, include_scale):
    """Forward on the column layout ``(N, C, L, 1)``."""
    scales = [x.new_zeros([])] * J
    hi = colfilter(x, f["h1o"])
    lo = colfilter(x, f["h0o"])
    yh = [_complex_from_phases(hi)]
    if include_scale:
        scales[0] = lo
    for j in range(1, J):
        if lo.shape[2] % 4 != 0:
            lo = torch.cat((lo[:, :, 0:1], lo, lo[:, :, -1:]), dim=2)
        hi = coldfilt(lo, f["h1b"], f["h1a"], True)
        lo = coldfilt(lo, f["h0b"], f["h0a"], False)
        yh.append(_complex_from_phases(hi))
        if include_scale:
            scales[j] = lo
    if include_scale:
        return scales, yh
    return lo, yh


def _idtcwt1d_impl(lo, yh, f):
    """Inverse on the column layout."""
    J = len(yh)
    if J == 0:
        return lo
    for j in range(J - 1, 0, -1):
        hj = yh[j]
        if hj is None:  # skipped bandpass: treat as complex zeros
            cdtype = torch.complex128 if lo.dtype == torch.float64 else torch.complex64
            hj = torch.zeros(
                (lo.shape[0], lo.shape[1], lo.shape[2] // 2, lo.shape[3]),
                dtype=cdtype,
                device=lo.device,
            )
        hi = _phases_from_complex(hj)
        lo = colifilt(lo, f["g0b"], f["g0a"], False) + colifilt(hi, f["g1b"], f["g1a"], True)
        nxt = yh[j - 1]
        if nxt is not None and lo.shape[2] != 2 * nxt.shape[2]:
            lo = lo[:, :, 1:-1]
    hj = yh[0]
    if hj is None:
        cdtype = torch.complex128 if lo.dtype == torch.float64 else torch.complex64
        hj = torch.zeros(
            (lo.shape[0], lo.shape[1], lo.shape[2] // 2, lo.shape[3]),
            dtype=cdtype,
            device=lo.device,
        )
    hi = _phases_from_complex(hj)
    z = colfilter(lo, f["g0o"]) + colfilter(hi, f["g1o"])
    return z


def dtcwt1d(
    x: torch.Tensor, biort="near_sym_a", qshift="qshift_a", J=3, include_scale=False
) -> tuple[torch.Tensor | list[torch.Tensor], list[torch.Tensor] | None]:
    """Forward 1-D dual-tree complex wavelet transform.

    Args:
        x (torch.Tensor): ``(N, C, L)`` input.
        biort (str): filter bank name.
        qshift (str): filter bank name.
        J (int): number of levels (``0`` returns ``(x, None)``).
        include_scale (bool): also return the intermediate lowpasses.

    Odd-length inputs are padded by one sample (the last sample is repeated)
    and the inverse returns the padded length ``L + 1``, matching the 2-D
    transform's convention.

    Returns:
        tuple[torch.Tensor, list[torch.Tensor]]: ``(yl, yh)``: lowpass
        ``(N, C, L')`` and a list of ``J`` complex bandpasses ``(N, C, L'')``
        (finest first).
    """
    if x.ndim != 3:
        raise ValueError(f"dtcwt1d expects (N, C, L); got {tuple(x.shape)}")
    check_spatial(x.shape[-1:], ("L",), "dtcwt1d")
    if isinstance(J, bool) or not isinstance(J, int) or J < 0:
        raise ValueError(f"J must be a non-negative int; got {J!r}")
    dtype = _compute_dtype(x)
    f = _resolve(biort, qshift, dtype, x.device)
    x = x.to(dtype)
    if x.shape[-1] % 2 != 0:
        x = torch.cat((x, x[..., -1:]), dim=-1)
    x4 = _to_col(x)
    if J == 0:
        return x, None
    with autocast_off(x.device):
        out = _dtcwt1d_impl(x4, f, J, bool(include_scale))
    if include_scale:
        scales, yh = out
        return [s.squeeze(-1) for s in scales], [h.squeeze(-1) for h in yh]
    lo, yh = out
    return lo.squeeze(-1), [h.squeeze(-1) for h in yh]


def idtcwt1d(coeffs, biort="near_sym_a", qshift="qshift_a") -> torch.Tensor:
    """Inverse 1-D DTCWT (exact inverse of :func:`dtcwt1d`)."""
    yl, yh = coeffs
    if yl is None:
        raise ValueError("The lowpass coefficient tensor yl cannot be None")
    dtype = _compute_dtype(yl)
    f = _resolve(biort, qshift, dtype, yl.device)
    lo = _to_col(yl.to(dtype))
    if yh is None:  # J == 0 forward
        return yl
    highs = [None if h is None else _to_col(h) for h in yh]
    with autocast_off(yl.device):
        z = _idtcwt1d_impl(lo, highs, f)
    return _from_col(z)


class DTCWT1D(nn.Module):
    """1-D DTCWT module (see :func:`dtcwt1d`)."""

    def __init__(self, biort="near_sym_a", qshift="qshift_a", J=3, include_scale=False):
        super().__init__()
        if isinstance(J, bool) or not isinstance(J, int) or J < 0:
            raise ValueError(f"J must be a non-negative int; got {J!r}")
        self.biort = biort
        self.qshift = qshift
        self.J = J
        self.include_scale = bool(include_scale)
        filts = _resolve(biort, qshift, torch.float64, torch.device("cpu"))
        for name in ("h0o", "h1o", "h0a", "h1a", "h0b", "h1b"):
            self.register_buffer(name, filts[name])

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"DTCWT1D expects (N, C, L); got {tuple(x.shape)}")
        check_spatial(x.shape[-1:], ("L",), "DTCWT1D")
        if self.J == 0:
            return x, None
        dtype = _compute_dtype(x)
        x = x.to(dtype)
        if x.shape[-1] % 2 != 0:
            x = torch.cat((x, x[..., -1:]), dim=-1)
        f = {
            n: getattr(self, n).to(dtype=dtype, device=x.device)
            for n in ("h0o", "h1o", "h0a", "h1a", "h0b", "h1b")
        }
        with autocast_off(x.device):
            out = _dtcwt1d_impl(_to_col(x), f, self.J, self.include_scale)
        if self.include_scale:
            scales, yh = out
            return [s.squeeze(-1) for s in scales], [h.squeeze(-1) for h in yh]
        lo, yh = out
        return lo.squeeze(-1), [h.squeeze(-1) for h in yh]

    def extra_repr(self) -> str:
        return f"biort={self.biort!r}, qshift={self.qshift!r}, J={self.J}"


class IDTCWT1D(nn.Module):
    """1-D inverse DTCWT module (see :func:`idtcwt1d`)."""

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
        dtype = _compute_dtype(yl)
        f = {
            n: getattr(self, n).to(dtype=dtype, device=yl.device)
            for n in ("g0o", "g1o", "g0a", "g1a", "g0b", "g1b")
        }
        lo = _to_col(yl.to(dtype))
        if yh is None:  # J == 0 forward
            return yl
        highs = [None if h is None else _to_col(h) for h in yh]
        with autocast_off(yl.device):
            return _from_col(_idtcwt1d_impl(lo, highs, f))

    def extra_repr(self) -> str:
        return f"biort={self.biort!r}, qshift={self.qshift!r}"
