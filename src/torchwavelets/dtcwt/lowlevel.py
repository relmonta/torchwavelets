"""Low-level dual-tree complex wavelet filter banks.

These are the exact Kingsbury DTCWT operations (ported from the original
MATLAB toolbox via fbcotter/pytorch_wavelets): column/row filtering with
biorthogonal filters at level 1 and quarter-sample-shift filters at level 2+,
plus the quad-to-complex conversions. Everything is built from torch ops, so
gradients flow through all dtypes and ``torch.compile`` works.

Boundary handling follows Kingsbury's ``symmetric`` (half-sample) extension.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..core.modes import pad_indices

__all__ = [
    "colfilter",
    "rowfilter",
    "coldfilt",
    "rowdfilt",
    "colifilt",
    "rowifilt",
    "q2c",
    "c2q",
    "highs_to_orientations",
    "orientations_to_highs",
]


def colfilter(X: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Column filter (filter along the rows) with decimation by 2."""
    b, ch, row, col = X.shape
    m = h.shape[2] // 2
    xe = pad_indices(row, m, m, "symmetric", X.device)
    return F.conv2d(X[:, :, xe], h.repeat(ch, 1, 1, 1), groups=ch)


def rowfilter(X: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Row filter (filter along the columns) with decimation by 2.

    ``h`` is a column filter ``(1, 1, L, 1)``; it is transposed internally.
    """
    b, ch, row, col = X.shape
    m = h.shape[2] // 2
    h = h.transpose(2, 3).contiguous()
    xe = pad_indices(col, m, m, "symmetric", X.device)
    return F.conv2d(X[:, :, :, xe], h.repeat(ch, 1, 1, 1), groups=ch)


def coldfilt(X: torch.Tensor, ha: torch.Tensor, hb: torch.Tensor, highpass: bool) -> torch.Tensor:
    """Filter two-phase column filter bank with decimation by 2.

    ``X`` must have a row count divisible by 4. Combines the a/b tree filters
    and interleaves their outputs.
    """
    batch, ch, r, c = X.shape
    r2 = r // 2
    if r % 4 != 0:
        raise ValueError(f"No. of rows in X must be a multiple of 4\nX was {X.shape}")

    m = ha.shape[2]
    xe = pad_indices(r, m, m, "symmetric", X.device)
    X = torch.cat((X[:, :, xe[2::2]], X[:, :, xe[3::2]]), dim=1)
    h = torch.cat((ha.repeat(ch, 1, 1, 1), hb.repeat(ch, 1, 1, 1)), dim=0)
    X = F.conv2d(X, h, stride=(2, 1), groups=ch * 2)

    if highpass:
        X = torch.stack((X[:, ch:], X[:, :ch]), dim=-2).view(batch, ch, r2, c)
    else:
        X = torch.stack((X[:, :ch], X[:, ch:]), dim=-2).view(batch, ch, r2, c)
    return X


def rowdfilt(X: torch.Tensor, ha: torch.Tensor, hb: torch.Tensor, highpass: bool) -> torch.Tensor:
    """Filter two-phase row filter bank with decimation by 2.

    ``ha``/``hb`` are column filters ``(1, 1, L, 1)``, reshaped internally.
    """
    batch, ch, r, c = X.shape
    c2 = c // 2
    if c % 4 != 0:
        raise ValueError(f"No. of cols in X must be a multiple of 4\nX was {X.shape}")

    m = ha.shape[2]
    xe = pad_indices(c, m, m, "symmetric", X.device)
    X = torch.cat((X[:, :, :, xe[2::2]], X[:, :, :, xe[3::2]]), dim=1)
    h = torch.cat(
        (ha.reshape(1, 1, 1, m).repeat(ch, 1, 1, 1), hb.reshape(1, 1, 1, m).repeat(ch, 1, 1, 1)),
        dim=0,
    )
    X = F.conv2d(X, h, stride=(1, 2), groups=ch * 2)

    if highpass:
        Y = torch.stack((X[:, ch:], X[:, :ch]), dim=-1).view(batch, ch, r, c2)
    else:
        Y = torch.stack((X[:, :ch], X[:, ch:]), dim=-1).view(batch, ch, r, c2)
    return Y


def colifilt(X: torch.Tensor, ha: torch.Tensor, hb: torch.Tensor, highpass: bool) -> torch.Tensor:
    """Inverse of :func:`coldfilt` (with upsampling)."""
    m = ha.shape[2]
    m2 = m // 2
    hao, hae = ha[:, :, 1::2], ha[:, :, ::2]
    hbo, hbe = hb[:, :, 1::2], hb[:, :, ::2]
    batch, ch, r, c = X.shape
    if r % 2 != 0:
        raise ValueError(f"No. of rows in X must be a multiple of 2.\nX was {X.shape}")
    xe = pad_indices(r, m2, m2, "symmetric", X.device)

    if m2 % 2 == 0:
        h1, h2, h3, h4 = hae, hbe, hao, hbo
        if highpass:
            X = torch.cat(
                (X[:, :, xe[1:-2:2]], X[:, :, xe[:-2:2]], X[:, :, xe[3::2]], X[:, :, xe[2::2]]),
                dim=1,
            )
        else:
            X = torch.cat(
                (X[:, :, xe[:-2:2]], X[:, :, xe[1:-2:2]], X[:, :, xe[2::2]], X[:, :, xe[3::2]]),
                dim=1,
            )
    else:
        h1, h2, h3, h4 = hao, hbo, hae, hbe
        if highpass:
            X = torch.cat(
                (
                    X[:, :, xe[2:-1:2]],
                    X[:, :, xe[1:-1:2]],
                    X[:, :, xe[2:-1:2]],
                    X[:, :, xe[1:-1:2]],
                ),
                dim=1,
            )
        else:
            X = torch.cat(
                (
                    X[:, :, xe[1:-1:2]],
                    X[:, :, xe[2:-1:2]],
                    X[:, :, xe[1:-1:2]],
                    X[:, :, xe[2:-1:2]],
                ),
                dim=1,
            )
    h = torch.cat(
        (
            h1.repeat(ch, 1, 1, 1),
            h2.repeat(ch, 1, 1, 1),
            h3.repeat(ch, 1, 1, 1),
            h4.repeat(ch, 1, 1, 1),
        ),
        dim=0,
    )
    X = F.conv2d(X, h, groups=4 * ch)
    # Stack 4 tensors of shape [batch, ch, r2, c] into [batch, ch, r*2, c]
    X = torch.stack(
        [X[:, :ch], X[:, ch : 2 * ch], X[:, 2 * ch : 3 * ch], X[:, 3 * ch :]], dim=3
    ).view(batch, ch, r * 2, c)
    return X


def rowifilt(X: torch.Tensor, ha: torch.Tensor, hb: torch.Tensor, highpass: bool) -> torch.Tensor:
    """Inverse of :func:`rowdfilt` (with upsampling)."""
    m = ha.shape[2]
    m2 = m // 2
    hao, hae = ha[:, :, 1::2], ha[:, :, ::2]
    hbo, hbe = hb[:, :, 1::2], hb[:, :, ::2]
    batch, ch, r, c = X.shape
    if c % 2 != 0:
        raise ValueError(f"No. of cols in X must be a multiple of 2.\nX was {X.shape}")
    xe = pad_indices(c, m2, m2, "symmetric", X.device)

    if m2 % 2 == 0:
        h1, h2, h3, h4 = hae, hbe, hao, hbo
        if highpass:
            X = torch.cat(
                (
                    X[:, :, :, xe[1:-2:2]],
                    X[:, :, :, xe[:-2:2]],
                    X[:, :, :, xe[3::2]],
                    X[:, :, :, xe[2::2]],
                ),
                dim=1,
            )
        else:
            X = torch.cat(
                (
                    X[:, :, :, xe[:-2:2]],
                    X[:, :, :, xe[1:-2:2]],
                    X[:, :, :, xe[2::2]],
                    X[:, :, :, xe[3::2]],
                ),
                dim=1,
            )
    else:
        h1, h2, h3, h4 = hao, hbo, hae, hbe
        if highpass:
            X = torch.cat(
                (
                    X[:, :, :, xe[2:-1:2]],
                    X[:, :, :, xe[1:-1:2]],
                    X[:, :, :, xe[2:-1:2]],
                    X[:, :, :, xe[1:-1:2]],
                ),
                dim=1,
            )
        else:
            X = torch.cat(
                (
                    X[:, :, :, xe[1:-1:2]],
                    X[:, :, :, xe[2:-1:2]],
                    X[:, :, :, xe[1:-1:2]],
                    X[:, :, :, xe[2:-1:2]],
                ),
                dim=1,
            )
    h = torch.cat(
        (
            h1.repeat(ch, 1, 1, 1),
            h2.repeat(ch, 1, 1, 1),
            h3.repeat(ch, 1, 1, 1),
            h4.repeat(ch, 1, 1, 1),
        ),
        dim=0,
    ).reshape(4 * ch, 1, 1, m2)
    X = F.conv2d(X, h, groups=4 * ch)
    X = torch.stack(
        [X[:, :ch], X[:, ch : 2 * ch], X[:, 2 * ch : 3 * ch], X[:, 3 * ch :]], dim=4
    ).view(batch, ch, r, c * 2)
    return X


def q2c(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert quad-number subimages to two complex subimages.

    Returns ``(z1, z2)`` complex tensors (each of half the spatial size).
    """
    y = y / np.sqrt(2)
    a, b = y[:, :, 0::2, 0::2], y[:, :, 0::2, 1::2]
    c, d = y[:, :, 1::2, 0::2], y[:, :, 1::2, 1::2]
    z1 = torch.complex(a - d, b + c)
    z2 = torch.complex(a + d, b - c)
    return z1, z2


def c2q(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Convert two complex subimages to a quad-number image (inverse of q2c)."""
    w1r, w1i = w1.real, w1.imag
    w2r, w2i = w2.real, w2.imag

    x1 = w1r + w2r
    x2 = w1i + w2i
    x3 = w1i - w2i
    x4 = -w1r + w2r

    b, ch, r, c = w1r.shape
    y = w1r.new_zeros((b, ch, r * 2, c * 2))
    y[:, :, ::2, ::2] = x1
    y[:, :, ::2, 1::2] = x2
    y[:, :, 1::2, ::2] = x3
    y[:, :, 1::2, 1::2] = x4
    y /= np.sqrt(2)
    return y


def highs_to_orientations(lh, hl, hh, o_dim: int = 2) -> torch.Tensor:
    """Combine the three quad bandpasses into 6 complex orientations.

    Returns a complex tensor with the 6 orientations stacked along ``o_dim``,
    in the order 15, 45, 75, 105, 135, 165 degrees.
    """
    (z15, z165), (z45, z135), (z75, z105) = q2c(lh), q2c(hh), q2c(hl)
    return torch.stack([z15, z45, z75, z105, z135, z165], dim=o_dim)


def orientations_to_highs(highs: torch.Tensor, o_dim: int = 2):
    """Inverse of :func:`highs_to_orientations`."""
    dev = highs.device
    z15, z165 = torch.unbind(
        torch.index_select(highs, o_dim, torch.tensor([0, 5], device=dev)), dim=o_dim
    )
    z45, z135 = torch.unbind(
        torch.index_select(highs, o_dim, torch.tensor([1, 4], device=dev)), dim=o_dim
    )
    z75, z105 = torch.unbind(
        torch.index_select(highs, o_dim, torch.tensor([2, 3], device=dev)), dim=o_dim
    )

    lh = c2q(z15, z165)
    hl = c2q(z75, z105)
    hh = c2q(z45, z135)
    return lh, hl, hh
