"""2-D dual-tree complex wavelet transform (functional and module API).

The bandpass coefficients are native complex tensors (``torch.complex64`` /
``torch.complex128``): ``yh[j]`` has shape ``(N, C, 6, H, W)`` with the 6
orientations (15, 45, 75, 105, 135, 165 degrees) along dim 2.

Precision follows the input: float32 -> complex64, float64 -> complex128.
Half inputs (float16/bfloat16) are computed in float32 and return complex64
(torch has no usable complex16 kernels). Boundary handling follows
Kingsbury's symmetric extension (the DTCWT is designed for it); other modes
are not supported.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..dwt._validation import check_spatial
from ._autocast import autocast_off
from .coeffs import biort as _biort
from .coeffs import qshift as _qshift
from .transform_funcs import fwd_j1, fwd_j2plus, get_dimensions5, inv_j1, inv_j2plus

__all__ = ["dtcwt2d", "idtcwt2d", "DTCWT2D", "IDTCWT2D"]

_SUPPORTED_BIORTS = ("antonini", "legall", "near_sym_a", "near_sym_b")
_SUPPORTED_QSHIFTS = ("qshift_06", "qshift_32", "qshift_a", "qshift_b", "qshift_c", "qshift_d")


def _validate_names(biort: str, qshift: str) -> None:
    if biort not in _SUPPORTED_BIORTS:
        if biort in ("farras", "near_sym_a2"):
            raise ValueError(
                f"biort {biort!r} is not usable as a level-1 bank: the shipped "
                "data only contains its two-tree (level >= 2) filters. Use "
                "one of the supported level-1 banks: "
                f"{_SUPPORTED_BIORTS}."
            )
        raise ValueError(
            f"Unknown biort {biort!r}; supported: {_SUPPORTED_BIORTS} "
            "(near_sym_b_bp needs the rotationally-symmetric variant, not "
            "implemented yet)"
        )
    if qshift not in _SUPPORTED_QSHIFTS:
        raise ValueError(
            f"Unknown qshift {qshift!r}; supported: {_SUPPORTED_QSHIFTS} "
            "(qshift_b_bp not implemented yet)"
        )


def _prep_filt(h, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Prepare a filter for conv2d: reversed, shaped ``(1, 1, L, 1)``
    (column orientation; ``rowfilter`` transposes internally)."""
    h = np.asarray(h[::-1], dtype=np.float64).ravel()
    h = h[:, None][None, None, :]
    return torch.tensor(np.ascontiguousarray(h), dtype=dtype, device=device)


def resolve_dtcwt_filters(biort: str, qshift: str, dtype: torch.dtype, device: torch.device):
    """All analysis/synthesis filters for the DTCWT, shaped for conv2d.

    Returns a dict with keys ``h0o, h1o, h0a, h1a, h0b, h1b, g0o, g1o,
    g0a, g1a, g0b, g1b`` (column orientation, ``(1, 1, L, 1)``; ``rowfilter``
    transposes internally).
    """
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


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """float32 for half inputs (no usable complex16 kernels), else input dtype."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"DTCWT expects a float32 or float64 input; got {x.dtype}")
    return x.dtype


def _norm_flags(flag, J: int) -> list[bool]:
    if isinstance(flag, (list, tuple, np.ndarray)):
        return list(flag)
    return [flag] * J


def _dtcwt2d_impl(x, f, J, skip_hps, include_scale, o_dim):
    o_dim, _, _ = get_dimensions5(o_dim)
    if J == 0:
        return x, None
    scales = [x.new_zeros([])] * J
    r, c = x.shape[2:]
    if r % 2 != 0:
        x = torch.cat((x, x[:, :, -1:]), dim=2)
    if c % 2 != 0:
        x = torch.cat((x, x[:, :, :, -1:]), dim=3)

    low, highs = fwd_j1(x, f["h0o"], f["h1o"], skip_hps[0], o_dim)
    yh = [highs]
    if include_scale[0]:
        scales[0] = low
    for j in range(1, J):
        r, c = low.shape[2:]
        if r % 4 != 0:
            low = torch.cat((low[:, :, 0:1], low, low[:, :, -1:]), dim=2)
        if c % 4 != 0:
            low = torch.cat((low[:, :, :, 0:1], low, low[:, :, :, -1:]), dim=3)
        low, highs = fwd_j2plus(low, f["h0a"], f["h1a"], f["h0b"], f["h1b"], skip_hps[j], o_dim)
        yh.append(highs)
        if include_scale[j]:
            scales[j] = low
    if True in include_scale:
        return scales, yh
    return low, yh


def _idtcwt2d_impl(yl, yh, f, o_dim):
    J = len(yh)
    o_dim, h_dim, w_dim = get_dimensions5(o_dim)
    low = yl
    for j in range(J - 1, 0, -1):
        s = yh[j]
        if s is not None:
            r, c = low.shape[2:]
            r1, c1 = s.shape[h_dim], s.shape[w_dim]
            if r != r1 * 2:
                low = low[:, :, 1:-1]
            if c != c1 * 2:
                low = low[:, :, :, 1:-1]
        low = inv_j2plus(low, s, f["g0a"], f["g1a"], f["g0b"], f["g1b"], o_dim, h_dim, w_dim)
    s = yh[0]
    if s is not None:
        r, c = low.shape[2:]
        r1, c1 = s.shape[h_dim], s.shape[w_dim]
        if r != r1 * 2:
            low = low[:, :, 1:-1]
        if c != c1 * 2:
            low = low[:, :, :, 1:-1]
    return inv_j1(low, s, f["g0o"], f["g1o"], o_dim, h_dim, w_dim)


def dtcwt2d(
    x: torch.Tensor,
    biort: str = "near_sym_a",
    qshift: str = "qshift_a",
    J: int = 3,
    skip_hps=False,
    include_scale=False,
    o_dim: int = 2,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Forward 2-D dual-tree complex wavelet transform.

    Args:
        x (torch.Tensor): ``(N, C, H, W)`` input.
        biort (str): filter bank name (see
            :mod:`torchwavelets.dtcwt.coeffs`).
        qshift (str): filter bank name.
        J (int): number of levels (``0`` returns ``(x, None)``).
        skip_hps (bool or list[bool]): one per level, to skip bandpasses.
        include_scale (bool or list[bool]): one per level, to also return the
            intermediate lowpasses.
        o_dim (int): dimension for the 6 orientations (default 2).

    Returns:
        tuple[torch.Tensor, list[torch.Tensor]]: ``(yl, yh)``: lowpass
        ``(N, C, H', W')`` and a list of ``J`` complex bandpasses
        ``(N, C, 6, H'', W'')`` (finest first).
    """
    if x.ndim != 4:
        raise ValueError(f"dtcwt2d expects a 4-D input (N, C, H, W); got shape {tuple(x.shape)}")
    check_spatial(x.shape[2:], ("H", "W"), "dtcwt2d")
    if isinstance(J, bool) or not isinstance(J, int) or J < 0:
        raise ValueError(f"J must be a non-negative int; got {J!r}")
    dtype = _compute_dtype(x)
    f = resolve_dtcwt_filters(biort, qshift, dtype, x.device)
    x = x.to(dtype)
    with autocast_off(x.device):
        return _dtcwt2d_impl(
            x, f, J, _norm_flags(skip_hps, J), _norm_flags(include_scale, J), o_dim
        )


def idtcwt2d(
    coeffs, biort: str = "near_sym_a", qshift: str = "qshift_a", o_dim: int = 2
) -> torch.Tensor:
    """Inverse 2-D DTCWT.

    Args:
        coeffs (tuple): ``(yl, yh)`` as returned by :func:`dtcwt2d` (any
            bandpass may be ``None``, treated as zeros).
        biort (str): as in :func:`dtcwt2d`.
        qshift (str): as in :func:`dtcwt2d`.
        o_dim (int): as in :func:`dtcwt2d`.

    Returns:
        torch.Tensor: reconstructed ``(N, C, H, W)`` tensor.
    """
    yl, yh = coeffs
    if yl is None:
        raise ValueError("The lowpass coefficient tensor yl cannot be None")
    dtype = _compute_dtype(yl)
    f = resolve_dtcwt_filters(biort, qshift, dtype, yl.device)
    with autocast_off(yl.device):
        return _idtcwt2d_impl(yl.to(dtype), yh, f, o_dim)


class DTCWT2D(nn.Module):
    """2-D DTCWT module (see :func:`dtcwt2d`).

    Filters are resolved eagerly at construction; the forward pass is pure
    torch ops, so the module is ``torch.compile``-compatible.
    """

    def __init__(
        self,
        biort="near_sym_a",
        qshift="qshift_a",
        J=3,
        skip_hps=False,
        include_scale=False,
        o_dim=2,
    ):
        super().__init__()
        if isinstance(J, bool) or not isinstance(J, int) or J < 0:
            raise ValueError(f"J must be a non-negative int; got {J!r}")
        self.biort = biort
        self.qshift = qshift
        self.J = J
        self.o_dim = o_dim
        self.skip_hps = _norm_flags(skip_hps, J)
        self.include_scale = _norm_flags(include_scale, J)

        filts = resolve_dtcwt_filters(biort, qshift, torch.float64, torch.device("cpu"))
        for name in ("h0o", "h1o", "h0a", "h1a", "h0b", "h1b"):
            self.register_buffer(name, filts[name])

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"DTCWT2D expects (N, C, H, W); got {tuple(x.shape)}")
        check_spatial(x.shape[2:], ("H", "W"), "DTCWT2D")
        dtype = _compute_dtype(x)
        f = {
            name: getattr(self, name).to(dtype=dtype, device=x.device)
            for name in ("h0o", "h1o", "h0a", "h1a", "h0b", "h1b")
        }
        with autocast_off(x.device):
            return _dtcwt2d_impl(
                x.to(dtype), f, self.J, self.skip_hps, self.include_scale, self.o_dim
            )

    def extra_repr(self) -> str:
        return f"biort={self.biort!r}, qshift={self.qshift!r}, J={self.J}"


class IDTCWT2D(nn.Module):
    """2-D inverse DTCWT module (see :func:`idtcwt2d`)."""

    def __init__(self, biort="near_sym_a", qshift="qshift_a", o_dim=2):
        super().__init__()
        self.biort = biort
        self.qshift = qshift
        self.o_dim = o_dim
        filts = resolve_dtcwt_filters(biort, qshift, torch.float64, torch.device("cpu"))
        for name in ("g0o", "g1o", "g0a", "g1a", "g0b", "g1b"):
            self.register_buffer(name, filts[name])

    def forward(self, coeffs):
        yl, _ = coeffs
        if yl is None:
            raise ValueError("The lowpass coefficient tensor yl cannot be None")
        if yl.ndim != 4:
            raise ValueError(f"IDTCWT2D expects 4-D coefficients; got shape {tuple(yl.shape)}")
        dtype = _compute_dtype(yl)
        f = {
            name: getattr(self, name).to(dtype=dtype, device=yl.device)
            for name in ("g0o", "g1o", "g0a", "g1a", "g0b", "g1b")
        }
        with autocast_off(yl.device):
            return _idtcwt2d_impl(yl.to(dtype), coeffs[1], f, self.o_dim)

    def extra_repr(self) -> str:
        return f"biort={self.biort!r}, qshift={self.qshift!r}"
