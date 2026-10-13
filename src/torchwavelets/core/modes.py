"""Boundary modes: PyWavelets-exact padding for all supported modes."""

from __future__ import annotations

from functools import lru_cache

import torch

__all__ = [
    "SUPPORTED_MODES",
    "validate_mode",
    "output_length",
    "padding_sizes",
    "pad_indices",
]

#: Modes supported by torchwavelets.  ``zero``, ``symmetric``, ``reflect``,
#: ``periodization``, ``periodic``, ``constant`` and ``smooth`` match
#: PyWavelets exactly (``smooth`` is a linear extrapolation from the edges).
#: ``replicate`` (edge extension) is an alias for ``constant``.
SUPPORTED_MODES = (
    "zero",
    "symmetric",
    "reflect",
    "periodization",
    "periodic",
    "constant",
    "replicate",
    "smooth",
)

_ALIASES = {
    "per": "periodization",
    "wrap": "periodic",
    "edge": "constant",
    "replicate": "constant",
}


def validate_mode(mode: str) -> str:
    """Normalize ``mode`` (accepts pywt aliases like ``per``/``wrap``/``edge``)."""
    if mode in _ALIASES:
        return _ALIASES[mode]
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Supported modes: {SUPPORTED_MODES}")
    return mode


def output_length(n: int, length: int, mode: str) -> int:
    """Length of a single-level decimated transform, as PyWavelets computes it.

    Equivalent to ``pywt.dwt_coeff_len`` for the modes PyWavelets supports.
    """
    if mode == "periodization":
        return (n + 1) // 2
    return (n + length - 1) // 2


def padding_sizes(n: int, length: int, mode: str) -> tuple[int, int]:
    """Total padding ``(before, after)`` needed so the strided convolution
    reproduces PyWavelets' output length exactly.

    The padding is ``(L - 2)`` before the signal and the remainder after —
    this is the distribution PyWavelets uses for every extension mode, and it
    holds for arbitrary ``n`` (no limit relative to the filter length).
    """
    out = output_length(n, length, mode)
    p = 2 * (out - 1) - n + length
    before = length - 2
    return before, p - before


@lru_cache(maxsize=512)
def _pad_indices_cached(n, before, after, mode, device):
    """Long index tensor of length ``n + before + after`` extending a signal of
    length ``n`` with the given boundary mode (for ``mode != 'zero'``).
    Cached per ``(n, before, after, mode, device)``; do not mutate the result.
    """
    idx = torch.arange(-before, n + after, device=device)
    if mode == "symmetric":
        # Half-sample symmetric extension (period 2n, mirrors at -0.5 and n-0.5).
        rel = idx + 0.5
        rng = 2 * n
        normed = torch.remainder(rel, rng)
        ref = torch.where(normed >= n, rng - normed, normed)
        idx = (ref - 0.5).long()
    elif mode == "reflect":
        # Whole-sample symmetric extension (period 2n-2, mirrors at 0 and n-1).
        if n <= 1:
            idx = torch.zeros_like(idx)
        else:
            rng = 2 * n - 2
            normed = torch.remainder(idx, rng)
            idx = torch.where(normed >= n, rng - normed, normed).long()
    elif mode == "periodic":
        idx = torch.remainder(idx, n)
    elif mode == "constant":
        # Edge extension (PyWavelets' 'constant' mode pads with the boundary
        # value; this is the same as torch's 'replicate' padding).
        idx = torch.clamp(idx, 0, n - 1)
    else:  # pragma: no cover - validated upstream
        raise ValueError(f"pad_indices does not support mode {mode!r}")
    return idx


def pad_indices(n: int, before: int, after: int, mode: str, device: torch.device) -> torch.Tensor:
    """Long index tensor extending a signal of length ``n`` with the given
    boundary mode (for ``mode != 'zero'``)."""
    return _pad_indices_cached(n, before, after, mode, device)
