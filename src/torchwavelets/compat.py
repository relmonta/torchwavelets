"""Drop-in compatibility with the ``pytorch_wavelets`` API.

The upstream class names are plain aliases: the constructor signatures and
call semantics are identical, only the semantics are fixed (precision follows
the input dtype, autocast works, and boundary handling matches PyWavelets
exactly). For the DTCWT, the bandpass coefficients are converted from the
native complex format to the upstream ``(..., 2)`` real/imag layout.
"""

from __future__ import annotations

import torch

from .dtcwt.transform2d import DTCWT2D, IDTCWT2D
from .dwt.transform1d import DWT1D, IDWT1D
from .dwt.transform2d import DWT2D, IDWT2D

__all__ = [
    "DWTForward",
    "DWTInverse",
    "DWT1DForward",
    "DWT1DInverse",
    "DTCWTForward",
    "DTCWTInverse",
    "DWT",
    "IDWT",
    "DWT1D",
    "IDWT1D",
    "DTCWT",
    "IDTCWT",
]


class DWTForward(DWT2D):
    """``pytorch_wavelets``-compatible 2-D DWT forward (upstream defaults)."""

    def __init__(self, J=1, wave="db1", mode="zero"):
        super().__init__(J=J, wave=wave, mode=mode)


class DWTInverse(IDWT2D):
    """``pytorch_wavelets``-compatible 2-D DWT inverse (upstream defaults)."""

    def __init__(self, wave="db1", mode="zero"):
        super().__init__(wave=wave, mode=mode)


class DWT1DForward(DWT1D):
    """``pytorch_wavelets``-compatible 1-D DWT forward (upstream defaults)."""

    def __init__(self, J=1, wave="db1", mode="zero"):
        super().__init__(J=J, wave=wave, mode=mode)


class DWT1DInverse(IDWT1D):
    """``pytorch_wavelets``-compatible 1-D DWT inverse (upstream defaults)."""

    def __init__(self, wave="db1", mode="zero"):
        super().__init__(wave=wave, mode=mode)


class DTCWTForward(DTCWT2D):
    """``pytorch_wavelets``-compatible DTCWT forward.

    Returns bandpasses in the upstream layout ``(N, C, 6, H, W, 2)``
    (real and imaginary parts along the last dimension). ``ri_dim`` must be
    the default ``-1`` (the native complex layout is fixed).
    """

    def __init__(
        self,
        biort="near_sym_a",
        qshift="qshift_a",
        J=3,
        skip_hps=False,
        include_scale=False,
        o_dim=2,
        ri_dim=-1,
        mode="symmetric",
    ):
        if ri_dim != -1:
            raise ValueError("torchwavelets DTCWT only supports ri_dim=-1")
        if mode != "symmetric":
            raise ValueError(
                "torchwavelets DTCWT only supports mode='symmetric' (Kingsbury boundary handling)"
            )
        super().__init__(
            biort=biort,
            qshift=qshift,
            J=J,
            skip_hps=skip_hps,
            include_scale=include_scale,
            o_dim=o_dim,
        )

    def forward(self, x: torch.Tensor):
        yl, yh = super().forward(x)
        if yh is None:
            return yl, yh
        return yl, [None if h is None else torch.stack((h.real, h.imag), dim=-1) for h in yh]


class DTCWTInverse(IDTCWT2D):
    """``pytorch_wavelets``-compatible DTCWT inverse (accepts the upstream
    ``(..., 2)`` real/imag layout or the native complex format)."""

    def __init__(self, biort="near_sym_a", qshift="qshift_a", o_dim=2, ri_dim=-1, mode="symmetric"):
        if ri_dim != -1:
            raise ValueError("torchwavelets DTCWT only supports ri_dim=-1")
        if mode != "symmetric":
            raise ValueError(
                "torchwavelets DTCWT only supports mode='symmetric' (Kingsbury boundary handling)"
            )
        super().__init__(biort=biort, qshift=qshift, o_dim=o_dim)

    def forward(self, coeffs):
        yl, yh = coeffs
        yh = [None if h is None else _to_complex(h) for h in yh]
        return super().forward((yl, yh))


def _to_complex(h: torch.Tensor) -> torch.Tensor:
    if h.dtype.is_complex:
        return h
    if h.shape[-1] == 2:
        return torch.complex(h[..., 0], h[..., 1])
    raise ValueError(
        f"Expected a complex tensor or a (..., 2) real/imag tensor; got shape {tuple(h.shape)}"
    )


# pytorch_wavelets' historical aliases
DWT = DWTForward
IDWT = DWTInverse
DTCWT = DTCWTForward
IDTCWT = DTCWTInverse
