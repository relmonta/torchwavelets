"""1-D discrete wavelet transform (functional and module API).

The module API resolves its filters into registered buffers at construction
time (eager) and only performs torch ops (``.to(dtype, device)`` + convs) in
``forward``, so it is fully traceable by ``torch.compile``.
"""

from __future__ import annotations

import torch

from ..core.filters import filter_bank, resolve_1d, wave_name
from ..core.modes import validate_mode
from . import lowlevel
from ._validation import check_reflect_length, check_spatial, validate_level, warn_if_too_high

__all__ = ["dwt1d", "idwt1d", "DWT1D", "IDWT1D"]

_IS_COMPILING = getattr(torch.compiler, "is_compiling", lambda: False)


def _dwt1d_impl(x: torch.Tensor, h0, h1, mode: str, level: int):
    yl = x[:, :, None, :]
    highs = []
    for _ in range(level):
        lohi = lowlevel.afb1d(yl, h0, h1, mode, 3)
        yl = lohi[:, ::2]
        highs.append(lohi[:, 1::2][:, :, 0])
    return yl[:, :, 0], highs


def _idwt1d_impl(yl: torch.Tensor, highs, g0, g1, mode: str) -> torch.Tensor:
    y = yl[:, :, None, :]
    for h in reversed(highs):
        if h is None:
            h = torch.zeros_like(y[:, :, 0])
        elif y.shape[3] == h.shape[2] + 1:
            # PyWavelets convention: drop the extra sample introduced by an
            # odd-length previous level before synthesizing.
            y = y[..., :-1]
        if y.shape[3] != h.shape[2]:
            raise ValueError(
                "coefficient shape mismatch: approximation "
                f"{tuple(y.shape)} vs detail {tuple(h.shape)} at the same level"
            )
        y = lowlevel.sfb1d(y, h[:, :, None, :], g0, g1, mode, 3)
    return y[:, :, 0]


def dwt1d(
    x: torch.Tensor, wave="db1", mode: str = "symmetric", level: int = 1
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Forward 1-D DWT.

    Args:
        x (torch.Tensor): ``(N, C, L)`` input.
        wave (str or pywt.Wavelet or tuple): wavelet name, ``pywt.Wavelet`` or
            custom filter bank.
        mode (str): boundary mode.
        level (int): number of decomposition levels (``0`` returns ``(x, [])``).

    Returns:
        tuple[torch.Tensor, list[torch.Tensor]]: ``(yl, yh)`` where ``yl`` is
        the final lowpass of shape ``(N, C, L')`` and ``yh`` is a list of
        ``level`` highpass tensors, finest first. The output dtype matches the
        input dtype.
    """
    if x.ndim != 3:
        raise ValueError(f"dwt1d expects a 3-D input (N, C, L); got shape {tuple(x.shape)}")
    check_spatial(x.shape[-1:], ("L",), "dwt1d")
    validate_level(level)
    mode = validate_mode(mode)
    check_reflect_length(x.shape[-1:], ("L",), mode, "dwt1d")

    warn_if_too_high(x.shape[-1], filter_bank(wave).length, level)
    h0, h1, _, _ = resolve_1d(wave, x.dtype, x.device)
    return _dwt1d_impl(x, h0, h1, mode, level)


def idwt1d(coeffs, wave="db1", mode: str = "symmetric") -> torch.Tensor:
    """Inverse 1-D DWT.

    Args:
        coeffs (tuple): ``(yl, yh)`` as returned by :func:`dwt1d` (any
            highpass entry may be ``None``, in which case it is treated as
            zeros).
        wave (str or pywt.Wavelet or tuple): as in :func:`dwt1d`.
        mode (str): as in :func:`dwt1d`.

    Returns:
        torch.Tensor: reconstructed ``(N, C, L')`` tensor (dtype matches the
        coefficients).

    This is a ``pywt.waverec``-style inverse: the coefficient pyramid from
    :func:`dwt1d` is inverted level by level, and (like ``pywt.waverec``) an
    odd-length previous level's extra sample is dropped inside the
    reconstruction. Unlike ``pywt.idwt`` (single-level), the lowpass/detail
    pair of a level does not need to match in length.
    """
    yl, highs = coeffs
    if yl is None:
        raise ValueError("The lowpass coefficient tensor yl cannot be None")
    if yl.ndim != 3:
        raise ValueError(f"idwt1d expects 3-D coefficients; got shape {tuple(yl.shape)}")
    check_spatial(yl.shape[-1:], ("L",), "idwt1d")
    mode = validate_mode(mode)

    _, _, g0, g1 = resolve_1d(wave, yl.dtype, yl.device)
    return _idwt1d_impl(yl, highs, g0, g1, mode)


class DWT1D(torch.nn.Module):
    """1-D DWT module.

    Args:
        J (int): number of decomposition levels.
        wave (str or pywt.Wavelet or tuple): as in :func:`dwt1d`.
        mode (str): as in :func:`dwt1d`.

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
        h0, h1, _, _ = resolve_1d(wave, torch.float64, torch.device("cpu"))
        self.register_buffer("h0", h0)
        self.register_buffer("h1", h1)

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"DWT1D expects a 3-D input (N, C, L); got shape {tuple(x.shape)}")
        check_spatial(x.shape[-1:], ("L",), "DWT1D")
        check_reflect_length(x.shape[-1:], ("L",), self.mode, "DWT1D")
        if not _IS_COMPILING():
            warn_if_too_high(x.shape[-1], self._bank_length, self.J)
        h0 = self.h0.to(dtype=x.dtype, device=x.device)
        h1 = self.h1.to(dtype=x.dtype, device=x.device)
        return _dwt1d_impl(x, h0, h1, self.mode, self.J)

    def extra_repr(self) -> str:
        return f"J={self.J}, wave={wave_name(self.wave)!r}, mode={self.mode!r}"


class IDWT1D(torch.nn.Module):
    """1-D inverse DWT module (see :func:`idwt1d`)."""

    def __init__(self, wave="db1", mode: str = "symmetric"):
        super().__init__()
        self.wave = wave
        self.mode = validate_mode(mode)
        _, _, g0, g1 = resolve_1d(wave, torch.float64, torch.device("cpu"))
        self.register_buffer("g0", g0)
        self.register_buffer("g1", g1)

    def forward(self, coeffs):
        yl, _ = coeffs
        if yl is None:
            raise ValueError("The lowpass coefficient tensor yl cannot be None")
        if yl.ndim != 3:
            raise ValueError(f"IDWT1D expects 3-D coefficients; got shape {tuple(yl.shape)}")
        check_spatial(yl.shape[-1:], ("L",), "IDWT1D")
        g0 = self.g0.to(dtype=yl.dtype, device=yl.device)
        g1 = self.g1.to(dtype=yl.dtype, device=yl.device)
        return _idwt1d_impl(yl, coeffs[1], g0, g1, self.mode)

    def extra_repr(self) -> str:
        return f"wave={wave_name(self.wave)!r}, mode={self.mode!r}"
