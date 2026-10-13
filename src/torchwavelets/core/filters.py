"""Filter-bank handling: PyWavelets-compatible wavelets and custom banks.

Custom banks are validated for perfect reconstruction at creation time, so a
biorthogonal bank passed as a 2-tuple (which assumes orthogonality) raises
immediately instead of silently producing a wrong inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pywt
import torch
import torch.nn.functional as F

__all__ = [
    "FilterBank",
    "filter_bank",
    "wave_name",
    "resolve_1d",
    "resolve_2d",
]


@dataclass(frozen=True)
class FilterBank:
    """A 1-D two-channel filter bank in PyWavelets' natural (non-reversed) order.

    Attributes:
        h0: analysis lowpass (``pywt.Wavelet.dec_lo``).
        h1: analysis highpass (``pywt.Wavelet.dec_hi``).
        g0: synthesis lowpass (``pywt.Wavelet.rec_lo``).
        g1: synthesis highpass (``pywt.Wavelet.rec_hi``).
    """

    h0: np.ndarray
    h1: np.ndarray
    g0: np.ndarray
    g1: np.ndarray

    @property
    def length(self) -> int:
        """Filter length (number of taps)."""
        return int(self.h0.size)


def _normalize(wave: object):
    """Return a hashable cache key for ``wave``."""
    if isinstance(wave, pywt.Wavelet):
        # Key on the actual filter bank, not just the name: pywt allows
        # ``pywt.Wavelet("db2", filter_bank=<custom bank>)``.
        return (
            "pywt",
            wave.name,
            tuple(np.asarray(wave.dec_lo, dtype=np.float64).ravel().tolist()),
            tuple(np.asarray(wave.dec_hi, dtype=np.float64).ravel().tolist()),
            tuple(np.asarray(wave.rec_lo, dtype=np.float64).ravel().tolist()),
            tuple(np.asarray(wave.rec_hi, dtype=np.float64).ravel().tolist()),
        )
    if isinstance(wave, str):
        return wave
    if not isinstance(wave, (tuple, list, np.ndarray)):
        raise ValueError(f"Unsupported wavelet specification: {wave!r}")
    arrs = tuple(tuple(np.asarray(w, dtype=np.float64).ravel()) for w in wave)
    if len(arrs) not in (2, 4):
        raise ValueError(
            "A custom filter bank must be a 2-tuple (h0, h1) or "
            f"4-tuple (h0, h1, g0, g1); got {len(arrs)} arrays."
        )
    return ("custom", arrs)


def wave_name(wave: object) -> str:
    """A short human-readable name for a wavelet specification."""
    if isinstance(wave, pywt.Wavelet):
        return wave.name
    if isinstance(wave, str):
        return wave
    return "custom"


def _check_pr(bank: FilterBank, tol: float = 1e-5) -> None:
    """Verify perfect reconstruction of a custom bank on a random fp64 signal.

    Raises ``ValueError`` if the bank is not perfectly reconstructing (for
    example, a biorthogonal pair given as a 2-tuple, where the synthesis
    filters would wrongly equal the analysis filters).
    """
    if len(bank.h0) != len(bank.h1) or len(bank.g0) != len(bank.g1):
        raise ValueError(
            "Custom filter banks must have equal-length analysis filters "
            f"(got {len(bank.h0)} vs {len(bank.h1)}) and equal-length "
            f"synthesis filters (got {len(bank.g0)} vs {len(bank.g1)})."
        )
    if len(bank.h0) != len(bank.g0):
        raise NotImplementedError(
            "Custom banks with different analysis/synthesis filter lengths "
            "(e.g. LeGall 5/3) are not supported yet; got analysis length "
            f"{len(bank.h0)} and synthesis length {len(bank.g0)}."
        )

    n = 64
    L = len(bank.h0)
    x = torch.randn(n, dtype=torch.float64)
    h0 = torch.tensor(bank.h0[::-1].copy(), dtype=torch.float64)[None, None, :]
    h1 = torch.tensor(bank.h1[::-1].copy(), dtype=torch.float64)[None, None, :]
    g0 = torch.tensor(bank.g0.copy(), dtype=torch.float64)[None, None, :]
    g1 = torch.tensor(bank.g1.copy(), dtype=torch.float64)[None, None, :]
    xp = F.pad(x[None, None, :], (L - 2, L - 2))
    lo = F.conv1d(xp, h0, stride=2)
    hi = F.conv1d(xp, h1, stride=2)
    xr = F.conv_transpose1d(lo, g0, stride=2, padding=(L - 2,))
    xr = xr + F.conv_transpose1d(hi, g1, stride=2, padding=(L - 2,))
    xr = xr[0, 0, :n]
    rel_err = (x - xr).norm().item() / x.norm().item()
    if rel_err > tol:
        raise ValueError(
            "The custom filter bank is not perfectly reconstructing "
            f"(relative error {rel_err:.2e}). For a biorthogonal bank pass the "
            "4-tuple (h0, h1, g0, g1) with explicit synthesis filters; a "
            "2-tuple assumes an orthogonal bank with g = h."
        )


@lru_cache(maxsize=256)
def _bank_from_key(key) -> FilterBank:
    if isinstance(key, str):
        w = pywt.Wavelet(key)
        return FilterBank(
            np.array(w.dec_lo),
            np.array(w.dec_hi),
            np.array(w.rec_lo),
            np.array(w.rec_hi),
        )
    if key[0] == "pywt":
        _, name, dec_lo, dec_hi, rec_lo, rec_hi = key
        w = pywt.Wavelet(name, filter_bank=(dec_lo, dec_hi, rec_lo, rec_hi))
        return FilterBank(
            np.array(w.dec_lo),
            np.array(w.dec_hi),
            np.array(w.rec_lo),
            np.array(w.rec_hi),
        )
    _, arrs = key
    if len(arrs) == 2:
        # Custom orthogonal bank: synthesis filters are the time-reversed
        # analysis filters (this is exactly pywt's rec_lo convention for
        # orthogonal wavelets, e.g. db2.rec_lo == db2.dec_lo[::-1]).
        h0, h1 = arrs
        bank = FilterBank(np.array(h0), np.array(h1), np.array(h0[::-1]), np.array(h1[::-1]))
    else:
        bank = FilterBank(
            np.array(arrs[0]), np.array(arrs[1]), np.array(arrs[2]), np.array(arrs[3])
        )
    _check_pr(bank)
    return bank


def filter_bank(wave) -> FilterBank:
    """Resolve ``wave`` to a :class:`FilterBank`.

    ``wave`` may be a PyWavelets wavelet name (``"db4"``), a
    :class:`pywt.Wavelet`, a 2-tuple ``(h0, h1)`` (orthogonal custom bank), or
    a 4-tuple ``(h0, h1, g0, g1)`` (general bank). Custom banks are validated
    for perfect reconstruction and raise ``ValueError`` otherwise.
    """
    return _bank_from_key(_normalize(wave))


@lru_cache(maxsize=1024)
def _resolve_1d_cached(key, dtype, device):
    bank = _bank_from_key(key)
    h0 = torch.tensor(bank.h0[::-1].copy(), dtype=dtype, device=device).reshape(1, 1, 1, -1)
    h1 = torch.tensor(bank.h1[::-1].copy(), dtype=dtype, device=device).reshape(1, 1, 1, -1)
    g0 = torch.tensor(bank.g0.copy(), dtype=dtype, device=device).reshape(1, 1, 1, -1)
    g1 = torch.tensor(bank.g1.copy(), dtype=dtype, device=device).reshape(1, 1, 1, -1)
    return (h0, h1, g0, g1)


def resolve_1d(
    wave, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analysis (reversed, for correlation) and synthesis (natural) 1-D filters.

    Returns tensors of shape ``(1, 1, 1, L)``, ready to filter a signal along
    the width dimension (``dim=3``) of a 4-D tensor. The analysis filters are
    reversed so ``conv2d`` implements correlation like PyWavelets' ``dwt``.
    Cached per ``(wave, dtype, device)``.
    """
    return _resolve_1d_cached(_normalize(wave), dtype, device)


def resolve_2d(wave, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Like :func:`resolve_1d` but returns column and row orientations.

    Returns ``(h0_col, h1_col, h0_row, h1_row, g0_col, g1_col, g0_row, g1_row)``
    with column filters shaped ``(1, 1, L, 1)`` and row filters ``(1, 1, 1, L)``.
    """
    h0, h1, g0, g1 = resolve_1d(wave, dtype, device)
    return (
        h0.reshape(1, 1, -1, 1),
        h1.reshape(1, 1, -1, 1),
        h0,
        h1,
        g0.reshape(1, 1, -1, 1),
        g1.reshape(1, 1, -1, 1),
        g0,
        g1,
    )
