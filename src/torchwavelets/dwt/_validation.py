"""Shared input validation for the transforms.

Everything here is pure Python / math on purpose: no PyWavelets call, so the
module API stays fully traceable under ``torch.compile``.
"""

from __future__ import annotations

import math
import warnings


def check_spatial(shape: tuple[int, ...], names: tuple[str, ...], where: str) -> None:
    """Raise a clear error for zero-size spatial dimensions."""
    for name, size in zip(names, shape, strict=True):
        if size == 0:
            raise ValueError(
                f"{where} requires non-empty spatial dimensions; got "
                f"zero-size {name!r} in {tuple(shape)}"
            )


def check_reflect_length(
    shape: tuple[int, ...], names: tuple[str, ...], mode: str, where: str
) -> None:
    """Mirror PyWavelets: reflect needs a length > 1 along every transformed axis."""
    if mode != "reflect":
        return
    for name, size in zip(names, shape, strict=True):
        if size == 1:
            raise ValueError(
                f"Input data length must be greater than 1 for [anti]reflect "
                f"mode ({where}: {name!r} has length 1)."
            )


def validate_level(level: int) -> None:
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError(f"level must be a non-negative int; got {level!r}")


def dwt_max_level(data_len: int, filter_len: int) -> int:
    """Maximum useful decomposition level (matches ``pywt.dwt_max_level``)."""
    if filter_len < 2:
        raise ValueError(f"invalid filter length: {filter_len}")
    if data_len < filter_len - 1:
        return 0
    return int(math.floor(math.log2(data_len / (filter_len - 1))))


def warn_if_too_high(n: int, length: int, level: int) -> None:
    """Mirror PyWavelets' warning when the level is too high."""
    if level > 0 and level > dwt_max_level(n, length):
        warnings.warn(
            f"Level value of {level} is too high: all coefficients will "
            "experience boundary effects.",
            UserWarning,
            stacklevel=3,
        )
