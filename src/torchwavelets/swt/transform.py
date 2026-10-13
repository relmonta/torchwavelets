"""Stationary (undecimated) wavelet transform: functional and module API.

The SWT (algorithm a trous) keeps every subband at the input resolution by
dilating the analysis filters by ``2**(level-1)`` and skipping decimation.
Coefficients follow the PyWavelets conventions exactly (coarsest first), and
``iswt`` / ``iswt2`` are exact inverses (verified bit-for-bit against pywt
by the parity tests).

Inputs must be divisible by ``2**level`` along every transformed axis, as
required by the stationary transform (PyWavelets raises the same error).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from ..core.filters import resolve_2d
from ..dwt import lowlevel as dwt_ll
from ..dwt._validation import check_spatial
from .lowlevel import swt_axis

__all__ = ["swt1d", "iswt1d", "swt2d", "iswt2d", "SWT1D", "ISWT1D", "SWT2D", "ISWT2D"]


def _check_level(level: int) -> None:
    if isinstance(level, bool) or not isinstance(level, int) or level < 1:
        raise ValueError(f"level must be a positive int; got {level!r}")


def _check_divisible(n: int, level: int, name: str) -> None:
    if n % (2**level) != 0:
        raise ValueError(
            f"{name} length {n} is not divisible by 2**level = {2**level}; "
            "the stationary wavelet transform requires it (pad the input)"
        )


def _iswt_idwt(lo: torch.Tensor, hi: torch.Tensor, g0r, g1r) -> torch.Tensor:
    """Single-level periodic inverse DWT on 3-D ``(N, C, L)`` coefficients."""
    lo4 = lo.unsqueeze(2)
    hi4 = hi.unsqueeze(2)
    return dwt_ll.sfb1d(lo4, hi4, g0r, g1r, "periodization", 3).squeeze(2)


def swt1d(
    x: torch.Tensor, wave, level: int, norm: bool = False
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """1-D stationary wavelet transform.

    Args:
        x (torch.Tensor): ``(N, C, L)`` input.
        wave (str or pywt.Wavelet or tuple): wavelet name / bank (see
            :func:`torchwavelets.core.filters.filter_bank`).
        level (int): number of scales.
        norm (bool): normalize so the coefficient energy equals the input
            energy (orthogonal wavelets only, like PyWavelets).

    Returns:
        list[tuple[torch.Tensor, torch.Tensor]]: ``[(cA_n, cD_n), ...,
        (cA_1, cD_1)]`` (coarsest first), every subband of shape ``(N, C, L)``.
    """
    _check_level(level)
    if x.ndim != 3:
        raise ValueError(f"swt1d expects (N, C, L); got {tuple(x.shape)}")
    check_spatial(x.shape[-1:], ("L",), "swt1d")
    _check_divisible(x.shape[-1], level, "signal")
    h0c, h1c, _, _, g0c, g1c, _, _ = resolve_2d(wave, x.dtype, x.device)
    if norm:
        h0c, h1c = h0c / 2**0.5, h1c / 2**0.5
    x4 = x.unsqueeze(-1)  # (N, C, L, 1)
    ret = []
    a = x4
    for j in range(1, level + 1):
        fstep = 2 ** (j - 1)
        ca = swt_axis(a, h0c, 2, fstep)
        cd = swt_axis(a, h1c, 2, fstep)
        ret.append((ca.squeeze(-1), cd.squeeze(-1)))
        a = ca
    ret.reverse()
    return ret


def iswt1d(coeffs, wave, norm: bool = False) -> torch.Tensor:
    """Inverse 1-D stationary wavelet transform (exact inverse of :func:`swt1d`)."""
    for cA, cD in coeffs:
        if cA.shape != cD.shape:
            raise ValueError(
                "iswt1d requires cA and cD of the same shape; got "
                f"{tuple(cA.shape)} vs {tuple(cD.shape)}"
            )
    _, _, _, _, _, _, g0r, g1r = resolve_2d(wave, coeffs[0][0].dtype, coeffs[0][0].device)
    if norm:
        g0r, g1r = g0r * 2**0.5, g1r * 2**0.5
    n_levels = len(coeffs)
    output = coeffs[0][0].clone()
    N = output.shape[-1]
    for j in range(n_levels, 0, -1):
        step = 2 ** (j - 1)
        cD = coeffs[-j][1]
        for first in range(step):
            idx = torch.arange(first, N, step, device=output.device)
            even = idx[0::2]
            odd = idx[1::2]
            x1 = _iswt_idwt(
                output.index_select(-1, even),
                cD.index_select(-1, even),
                g0r,
                g1r,
            )
            x2 = _iswt_idwt(
                output.index_select(-1, odd),
                cD.index_select(-1, odd),
                g0r,
                g1r,
            )
            x2 = torch.roll(x2, 1, dims=-1)
            output[..., idx] = (x1 + x2) / 2
    return output


def swt2d(
    x: torch.Tensor, wave, level: int, norm: bool = False
) -> list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """2-D stationary wavelet transform.

    Args:
        x (torch.Tensor): ``(N, C, H, W)`` input.
        wave (str or pywt.Wavelet or tuple): as in :func:`swt1d`.
        level (int): as in :func:`swt1d`.
        norm (bool): as in :func:`swt1d`.

    Returns:
        list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        ``[(cA_j, (cH_j, cV_j, cD_j)), ...]`` (coarsest first), all subbands
        of shape ``(N, C, H, W)``.
    """
    _check_level(level)
    if x.ndim != 4:
        raise ValueError(f"swt2d expects (N, C, H, W); got {tuple(x.shape)}")
    check_spatial(x.shape[2:], ("H", "W"), "swt2d")
    _check_divisible(x.shape[-2], level, "height")
    _check_divisible(x.shape[-1], level, "width")
    h0c, h1c, h0r, h1r, _, _, _, _ = resolve_2d(wave, x.dtype, x.device)
    if norm:
        h0c, h1c, h0r, h1r = h0c / 2**0.5, h1c / 2**0.5, h0r / 2**0.5, h1r / 2**0.5
    ret = []
    a = x
    for j in range(1, level + 1):
        fstep = 2 ** (j - 1)
        # axis 0 (rows, dim 2), then axis 1 (cols, dim 3), like pywt.swt2
        low = swt_axis(a, h0c, 2, fstep)
        high = swt_axis(a, h1c, 2, fstep)
        aa = swt_axis(low, h0r, 3, fstep)
        ad = swt_axis(low, h1r, 3, fstep)
        da = swt_axis(high, h0r, 3, fstep)
        dd = swt_axis(high, h1r, 3, fstep)
        ret.append((aa, (da, ad, dd)))
        a = aa
    ret.reverse()
    return ret


def iswt2d(coeffs, wave, norm: bool = False) -> torch.Tensor:
    """Inverse 2-D stationary wavelet transform (exact inverse of :func:`swt2d`)."""
    for cA, cs in coeffs:
        if any(c.shape != cA.shape for c in cs):
            raise ValueError(
                "iswt2d requires all subbands of a level to share the shape "
                f"{tuple(cA.shape)}; got {[tuple(c.shape) for c in cs]}"
            )
    _, _, _, _, g0c, g1c, g0r, g1r = resolve_2d(wave, coeffs[0][0].dtype, coeffs[0][0].device)
    if norm:
        g0c, g1c, g0r, g1r = g0c * 2**0.5, g1c * 2**0.5, g0r * 2**0.5, g1r * 2**0.5
    n_levels = len(coeffs)
    output = coeffs[0][0].clone()
    H, W = output.shape[-2:]
    for j in range(n_levels):
        step = 2 ** (n_levels - j - 1)
        cH, cV, cD = coeffs[j][1]
        for fh in range(step):
            for fw in range(step):
                approx = output.clone()
                out_slice = output[..., fh::step, fw::step]
                out_slice.zero_()
                for oh in (0, 1):
                    for ow in (0, 1):
                        h_eo = slice(fh + oh * step, H, 2 * step)
                        w_eo = slice(fw + ow * step, W, 2 * step)
                        ll = approx[..., h_eo, w_eo]
                        lh = cH[..., h_eo, w_eo]
                        hl = cV[..., h_eo, w_eo]
                        hh = cD[..., h_eo, w_eo]
                        lo = dwt_ll.sfb1d(ll, lh, g0c, g1c, "periodization", 2)
                        hi = dwt_ll.sfb1d(hl, hh, g0c, g1c, "periodization", 2)
                        xr = dwt_ll.sfb1d(lo, hi, g0r, g1r, "periodization", 3)
                        if oh:
                            xr = torch.roll(xr, 1, dims=-2)
                        if ow:
                            xr = torch.roll(xr, 1, dims=-1)
                        out_slice += xr
                out_slice /= 4
    return output


class SWT1D(nn.Module):
    """1-D stationary wavelet transform module (see :func:`swt1d`)."""

    def __init__(self, wave, level: int, norm: bool = False):
        super().__init__()
        _check_level(level)
        self.wave = wave
        self.level = level
        self.norm = norm
        h0c, h1c, _, _, _, _, _, _ = resolve_2d(wave, torch.float64, torch.device("cpu"))
        self.register_buffer("h0c", h0c)
        self.register_buffer("h1c", h1c)

    def forward(self, x: torch.Tensor):
        h0c = cast(torch.Tensor, self.h0c).to(dtype=x.dtype, device=x.device)
        h1c = cast(torch.Tensor, self.h1c).to(dtype=x.dtype, device=x.device)
        if self.norm:
            h0c, h1c = h0c / 2**0.5, h1c / 2**0.5
        _check_level(self.level)
        if x.ndim != 3:
            raise ValueError(f"SWT1D expects (N, C, L); got {tuple(x.shape)}")
        _check_divisible(x.shape[-1], self.level, "signal")
        x4 = x.unsqueeze(-1)
        ret = []
        a = x4
        for j in range(1, self.level + 1):
            fstep = 2 ** (j - 1)
            ca = swt_axis(a, h0c, 2, fstep)
            cd = swt_axis(a, h1c, 2, fstep)
            ret.append((ca.squeeze(-1), cd.squeeze(-1)))
            a = ca
        ret.reverse()
        return ret

    def extra_repr(self) -> str:
        return f"wave={self.wave!r}, level={self.level}, norm={self.norm}"


class ISWT1D(nn.Module):
    """1-D inverse stationary wavelet transform module (see :func:`iswt1d`)."""

    def __init__(self, wave, norm: bool = False):
        super().__init__()
        self.wave = wave
        self.norm = norm
        _, _, _, _, _, _, g0r, g1r = resolve_2d(wave, torch.float64, torch.device("cpu"))
        self.register_buffer("g0r", g0r)
        self.register_buffer("g1r", g1r)

    def forward(self, coeffs):
        g0r = self.g0r.to(dtype=coeffs[0][0].dtype, device=coeffs[0][0].device)
        g1r = self.g1r.to(dtype=coeffs[0][0].dtype, device=coeffs[0][0].device)
        if self.norm:
            g0r, g1r = g0r * 2**0.5, g1r * 2**0.5
        n_levels = len(coeffs)
        output = coeffs[0][0].clone()
        N = output.shape[-1]
        for j in range(n_levels, 0, -1):
            step = 2 ** (j - 1)
            cD = coeffs[-j][1]
            for first in range(step):
                idx = torch.arange(first, N, step, device=output.device)
                even = idx[0::2]
                odd = idx[1::2]
                x1 = _iswt_idwt(
                    output.index_select(-1, even),
                    cD.index_select(-1, even),
                    g0r,
                    g1r,
                )
                x2 = _iswt_idwt(
                    output.index_select(-1, odd),
                    cD.index_select(-1, odd),
                    g0r,
                    g1r,
                )
                x2 = torch.roll(x2, 1, dims=-1)
                output[..., idx] = (x1 + x2) / 2
        return output

    def extra_repr(self) -> str:
        return f"wave={self.wave!r}, norm={self.norm}"


class SWT2D(nn.Module):
    """2-D stationary wavelet transform module (see :func:`swt2d`)."""

    def __init__(self, wave, level: int, norm: bool = False):
        super().__init__()
        _check_level(level)
        self.wave = wave
        self.level = level
        self.norm = norm
        h0c, h1c, h0r, h1r, _, _, _, _ = resolve_2d(wave, torch.float64, torch.device("cpu"))
        for name, t in (("h0c", h0c), ("h1c", h1c), ("h0r", h0r), ("h1r", h1r)):
            self.register_buffer(name, t)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"SWT2D expects (N, C, H, W); got {tuple(x.shape)}")
        _check_divisible(x.shape[-2], self.level, "height")
        _check_divisible(x.shape[-1], self.level, "width")
        f = {
            n: getattr(self, n).to(dtype=x.dtype, device=x.device)
            for n in ("h0c", "h1c", "h0r", "h1r")
        }
        if self.norm:
            for n in f:
                f[n] = f[n] / 2**0.5
        ret = []
        a = x
        for j in range(1, self.level + 1):
            fstep = 2 ** (j - 1)
            low = swt_axis(a, f["h0c"], 2, fstep)
            high = swt_axis(a, f["h1c"], 2, fstep)
            aa = swt_axis(low, f["h0r"], 3, fstep)
            ad = swt_axis(low, f["h1r"], 3, fstep)
            da = swt_axis(high, f["h0r"], 3, fstep)
            dd = swt_axis(high, f["h1r"], 3, fstep)
            ret.append((aa, (da, ad, dd)))
            a = aa
        ret.reverse()
        return ret

    def extra_repr(self) -> str:
        return f"wave={self.wave!r}, level={self.level}, norm={self.norm}"


class ISWT2D(nn.Module):
    """2-D inverse stationary wavelet transform module (see :func:`iswt2d`)."""

    def __init__(self, wave, norm: bool = False):
        super().__init__()
        self.wave = wave
        self.norm = norm
        _, _, _, _, g0c, g1c, g0r, g1r = resolve_2d(wave, torch.float64, torch.device("cpu"))
        for name, t in (("g0c", g0c), ("g1c", g1c), ("g0r", g0r), ("g1r", g1r)):
            self.register_buffer(name, t)

    def forward(self, coeffs):
        f = {
            n: getattr(self, n).to(dtype=coeffs[0][0].dtype, device=coeffs[0][0].device)
            for n in ("g0c", "g1c", "g0r", "g1r")
        }
        if self.norm:
            for n in f:
                f[n] = f[n] * 2**0.5
        n_levels = len(coeffs)
        output = coeffs[0][0].clone()
        H, W = output.shape[-2:]
        for j in range(n_levels):
            step = 2 ** (n_levels - j - 1)
            cH, cV, cD = coeffs[j][1]
            for fh in range(step):
                for fw in range(step):
                    approx = output.clone()
                    out_slice = output[..., fh::step, fw::step]
                    out_slice.zero_()
                    for oh in (0, 1):
                        for ow in (0, 1):
                            h_eo = slice(fh + oh * step, H, 2 * step)
                            w_eo = slice(fw + ow * step, W, 2 * step)
                            ll = approx[..., h_eo, w_eo]
                            lh = cH[..., h_eo, w_eo]
                            hl = cV[..., h_eo, w_eo]
                            hh = cD[..., h_eo, w_eo]
                            lo = dwt_ll.sfb1d(ll, lh, f["g0c"], f["g1c"], "periodization", 2)
                            hi = dwt_ll.sfb1d(hl, hh, f["g0c"], f["g1c"], "periodization", 2)
                            xr = dwt_ll.sfb1d(lo, hi, f["g0r"], f["g1r"], "periodization", 3)
                            if oh:
                                xr = torch.roll(xr, 1, dims=-2)
                            if ow:
                                xr = torch.roll(xr, 1, dims=-1)
                            out_slice += xr
                    out_slice /= 4
        return output

    def extra_repr(self) -> str:
        return f"wave={self.wave!r}, norm={self.norm}"
