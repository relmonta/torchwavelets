"""2-D discrete wavelet transform (functional and module API).

The module API resolves its filters into registered buffers at construction
time (eager) and only performs torch ops (``.to(dtype, device)`` + convs) in
``forward``, so it is fully traceable by ``torch.compile``.
"""

from __future__ import annotations

import torch

from ..core.filters import filter_bank, resolve_2d, wave_name
from ..core.modes import validate_mode
from . import lowlevel
from ._validation import check_reflect_length, check_spatial, validate_level, warn_if_too_high

__all__ = ["dwt2d", "idwt2d", "DWT2D", "IDWT2D"]

_IS_COMPILING = getattr(torch.compiler, "is_compiling", lambda: False)

_DWT2D_NAMES = ("h0_col", "h1_col", "h0_row", "h1_row")
_IDWT2D_NAMES = ("g0_col", "g1_col", "g0_row", "g1_row")


def _dwt2d_impl(x: torch.Tensor, h0c, h1c, h0r, h1r, mode: str, level: int):
    yl = x
    highs = []
    for _ in range(level):
        lohi = lowlevel.afb1d(yl, h0r, h1r, mode, 3)  # rows
        y = lowlevel.afb1d(lohi, h0c, h1c, mode, 2)  # columns
        s = y.shape
        y = y.reshape(s[0], s[1] // 4, 4, s[-2], s[-1])
        yl = y[:, :, 0].contiguous()
        highs.append(y[:, :, 1:].contiguous())
    return yl, highs


def _idwt2d_impl(yl: torch.Tensor, highs, g0c, g1c, g0r, g1r, mode: str) -> torch.Tensor:
    y = yl
    for h in reversed(highs):
        if h is None:
            h = torch.zeros(
                y.shape[0],
                y.shape[1],
                3,
                y.shape[-2],
                y.shape[-1],
                dtype=y.dtype,
                device=y.device,
            )
        else:
            # PyWavelets convention: drop extra samples introduced by
            # odd-length previous levels before synthesizing.
            if y.shape[-2] == h.shape[-2] + 1:
                y = y[..., :-1, :]
            if y.shape[-1] == h.shape[-1] + 1:
                y = y[..., :-1]
        if y.shape[-2] != h.shape[-2] or y.shape[-1] != h.shape[-1]:
            raise ValueError(
                "coefficient shape mismatch: approximation "
                f"{tuple(y.shape)} vs detail {tuple(h.shape)} at the same level"
            )
        lh, hl, hh = torch.unbind(h, dim=2)
        lo = lowlevel.sfb1d(y, lh, g0c, g1c, mode, 2)
        hi = lowlevel.sfb1d(hl, hh, g0c, g1c, mode, 2)
        y = lowlevel.sfb1d(lo, hi, g0r, g1r, mode, 3)
    return y


def dwt2d(
    x: torch.Tensor, wave="db1", mode: str = "symmetric", level: int = 1
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Forward 2-D DWT (separable, row then column filtering).

    Args:
        x (torch.Tensor): ``(N, C, H, W)`` input.
        wave (str or pywt.Wavelet or tuple): wavelet name, ``pywt.Wavelet`` or
            custom filter bank.
        mode (str): boundary mode.
        level (int): number of decomposition levels (``0`` returns ``(x, [])``).

    Returns:
        ``(yl, yh)`` where ``yl`` is the final lowpass ``(N, C, H', W')`` and
        ``yh`` is a list of ``level`` tensors of shape
        ``(N, C, 3, H'', W'')`` holding the ``(LH, HL, HH)`` subbands
        (finest first). ``LH``/``HL``/``HH`` match PyWavelets'
        ``cH``/``cV``/``cD``. The output dtype matches the input dtype.
    """
    if x.ndim != 4:
        raise ValueError(f"dwt2d expects a 4-D input (N, C, H, W); got shape {tuple(x.shape)}")
    check_spatial(x.shape[2:], ("H", "W"), "dwt2d")
    mode = validate_mode(mode)
    check_reflect_length(x.shape[2:], ("H", "W"), mode, "dwt2d")
    validate_level(level)
    mode = validate_mode(mode)

    warn_if_too_high(min(x.shape[-2:]), filter_bank(wave).length, level)
    h0c, h1c, h0r, h1r, _, _, _, _ = resolve_2d(wave, x.dtype, x.device)
    return _dwt2d_impl(x, h0c, h1c, h0r, h1r, mode, level)


def idwt2d(coeffs, wave="db1", mode: str = "symmetric") -> torch.Tensor:
    """Inverse 2-D DWT.

    Args:
        coeffs (tuple): ``(yl, yh)`` as returned by :func:`dwt2d` (any
            highpass entry may be ``None``, in which case it is treated as
            zeros).
        wave (str or pywt.Wavelet or tuple): as in :func:`dwt2d`.
        mode (str): as in :func:`dwt2d`.

    Returns:
        torch.Tensor: reconstructed ``(N, C, H', W')`` tensor (dtype matches
        the coefficients).
    """
    yl, highs = coeffs
    if yl is None:
        raise ValueError("The lowpass coefficient tensor yl cannot be None")
    if yl.ndim != 4:
        raise ValueError(f"idwt2d expects 4-D coefficients; got shape {tuple(yl.shape)}")
    check_spatial(yl.shape[2:], ("H", "W"), "idwt2d")
    mode = validate_mode(mode)

    _, _, _, _, g0c, g1c, g0r, g1r = resolve_2d(wave, yl.dtype, yl.device)
    return _idwt2d_impl(yl, highs, g0c, g1c, g0r, g1r, validate_mode(mode))


class DWT2D(torch.nn.Module):
    """2-D DWT module.

    Args:
        J (int): number of decomposition levels.
        wave (str or pywt.Wavelet or tuple): as in :func:`dwt2d`.
        mode (str): as in :func:`dwt2d`.

    The precision follows the input tensor: feed a float16 tensor and the
    output is float16, regardless of the dtype used at construction time.
    The module is ``torch.compile``-compatible (filters are resolved eagerly
    at construction; only torch ops run in ``forward``).
    """

    def __init__(self, J: int = 1, wave="db1", mode: str = "symmetric"):
        super().__init__()
        validate_level(J)
        self.J = J
        self.wave = wave
        self.mode = validate_mode(mode)
        self._bank_length = filter_bank(wave).length
        filts = resolve_2d(wave, torch.float64, torch.device("cpu"))
        for name, t in zip(_DWT2D_NAMES, filts[:4], strict=True):
            self.register_buffer(name, t)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"DWT2D expects a 4-D input (N, C, H, W); got shape {tuple(x.shape)}")
        check_spatial(x.shape[2:], ("H", "W"), "DWT2D")
        check_reflect_length(x.shape[2:], ("H", "W"), self.mode, "DWT2D")
        if not _IS_COMPILING():
            warn_if_too_high(min(x.shape[-2:]), self._bank_length, self.J)
        h0c = self.h0_col.to(dtype=x.dtype, device=x.device)
        h1c = self.h1_col.to(dtype=x.dtype, device=x.device)
        h0r = self.h0_row.to(dtype=x.dtype, device=x.device)
        h1r = self.h1_row.to(dtype=x.dtype, device=x.device)
        return _dwt2d_impl(x, h0c, h1c, h0r, h1r, self.mode, self.J)

    def extra_repr(self) -> str:
        return f"J={self.J}, wave={wave_name(self.wave)!r}, mode={self.mode!r}"


class IDWT2D(torch.nn.Module):
    """2-D inverse DWT module (see :func:`idwt2d`)."""

    def __init__(self, wave="db1", mode: str = "symmetric"):
        super().__init__()
        self.wave = wave
        self.mode = validate_mode(mode)
        filts = resolve_2d(wave, torch.float64, torch.device("cpu"))
        for name, t in zip(_IDWT2D_NAMES, filts[4:], strict=True):
            self.register_buffer(name, t)

    def forward(self, coeffs):
        yl, _ = coeffs
        if yl is None:
            raise ValueError("The lowpass coefficient tensor yl cannot be None")
        if yl.ndim != 4:
            raise ValueError(f"IDWT2D expects 4-D coefficients; got shape {tuple(yl.shape)}")
        check_spatial(yl.shape[2:], ("H", "W"), "IDWT2D")
        g0c = self.g0_col.to(dtype=yl.dtype, device=yl.device)
        g1c = self.g1_col.to(dtype=yl.dtype, device=yl.device)
        g0r = self.g0_row.to(dtype=yl.dtype, device=yl.device)
        g1r = self.g1_row.to(dtype=yl.dtype, device=yl.device)
        return _idwt2d_impl(yl, coeffs[1], g0c, g1c, g0r, g1r, self.mode)

    def extra_repr(self) -> str:
        return f"wave={wave_name(self.wave)!r}, mode={self.mode!r}"
