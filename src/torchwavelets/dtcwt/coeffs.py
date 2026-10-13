"""Dual-tree complex wavelet filter banks (Kingsbury's design).

The filter coefficients are the standard DTCWT banks:

- Level 1 (biorthogonal): ``antonini``, ``farras``, ``legall``, ``near_sym_a``,
  ``near_sym_b``, ``near_sym_a2``, ``near_sym_b_bp``
- Levels 2+ (quarter-sample shift): ``qshift_06``, ``qshift_32``, ``qshift_a``,
  ``qshift_b``, ``qshift_c``, ``qshift_d``, ``qshift_b_bp``

The coefficients originate from Nick Kingsbury's dual-tree complex wavelet
transform toolbox (Cambridge University), which is free for research use only.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

__all__ = ["level1", "qshift", "biort", "pywt_coeffs"]


@lru_cache(maxsize=32)
def _load_from_file(basename: str, varnames: tuple) -> tuple[np.ndarray, ...]:
    """Load filter vectors from the bundled ``.npz`` data files."""
    data = files("torchwavelets.dtcwt.data").joinpath(basename + ".npz")
    with data.open("rb") as f:
        mat = dict(np.load(f))
    try:
        return tuple(mat[k] for k in varnames)
    except KeyError as e:  # pragma: no cover - data files are fixed
        raise ValueError(
            f"Wavelet {basename!r} does not define ({', '.join(varnames)}) "
            f"coefficients; available keys: {sorted(mat)}"
        ) from e


def level1(name: str, compact: bool = False) -> tuple[np.ndarray, ...]:
    """Load level-1 (biorthogonal) DTCWT filters by name.

    Returns ``(h0o, g0o, h1o, g1o)`` (compact) or the 8-tuple
    ``(h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b)``. ``near_sym_b_bp`` also
    returns the bandpass ``h2o``/``g2o`` (compact) or ``h2a, h2b, g2a, g2b``.
    """
    if compact:
        if name == "near_sym_b_bp":
            return _load_from_file(name, ("h0o", "g0o", "h1o", "g1o", "h2o", "g2o"))
        return _load_from_file(name, ("h0o", "g0o", "h1o", "g1o"))
    return _load_from_file(name, ("h0a", "h0b", "g0a", "g0b", "h1a", "h1b", "g1a", "g1b"))


def qshift(name: str) -> tuple[np.ndarray, ...]:
    """Load level >= 2 (quarter-sample shift) DTCWT filters by name.

    Returns ``(h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b)`` (``qshift_b_bp``
    also returns the bandpass ``h2a, h2b, g2a, g2b``).
    """
    if name == "qshift_b_bp":
        return _load_from_file(
            name,
            ("h0a", "h0b", "g0a", "g0b", "h1a", "h1b", "g1a", "g1b", "h2a", "h2b", "g2a", "g2b"),
        )
    return _load_from_file(name, ("h0a", "h0b", "g0a", "g0b", "h1a", "h1b", "g1a", "g1b"))


def biort(name: str) -> tuple[np.ndarray, ...]:
    """Deprecated alias of :func:`level1` (``compact=True``)."""
    return level1(name, compact=True)


def pywt_coeffs(name: str):
    """Wrap a PyWavelets wavelet (for custom level-1 banks)."""
    import pywt

    return pywt.Wavelet(name)
