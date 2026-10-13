"""Dual-tree complex wavelet scattering layers.

A scattering layer applies one (or two) DTCWT scale(s) with the smooth
modulus ``sqrt(|Wx|^2 + magbias^2) - magbias`` and stacks the results along
the channel dimension. With the DTCWT's near shift-invariance this gives
locally shift-invariant feature maps for a single, cheap layer — the classic
DTCWT scattering front-end.

The implementation is built directly on the complex-native DTCWT, so no
custom ``autograd.Function`` is needed: gradients flow through the complex
ops, autocast works, and the module is ``torch.compile``-compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dtcwt._autocast import autocast_off
from ..dtcwt.coeffs import biort as _biort
from ..dtcwt.coeffs import qshift as _qshift
from ..dtcwt.transform2d import _compute_dtype, _prep_filt, _validate_names
from ..dtcwt.transform_funcs import fwd_j1, fwd_j2plus

__all__ = ["ScatLayer", "ScatLayerj2"]


def _smooth_mag(highs: torch.Tensor, magbias: float) -> torch.Tensor:
    """Smooth modulus ``sqrt(|z|^2 + b^2) - b`` of complex bandpasses."""
    return torch.sqrt(highs.abs().pow(2) + magbias**2) - magbias


def _pad_to(x: torch.Tensor, mult: int) -> torch.Tensor:
    """Reflectively pad ``x`` so H and W are multiples of ``mult``."""
    _, _, r, c = x.shape
    rem = r % mult
    if rem != 0:
        before = (mult - rem) // 2
        after = mult - rem - before
        if before > 0:
            x = torch.cat((x[:, :, :before], x), dim=2)
        if after > 0:
            x = torch.cat((x, x[:, :, -after:]), dim=2)
    rem = c % mult
    if rem != 0:
        before = (mult - rem) // 2
        after = mult - rem - before
        if before > 0:
            x = torch.cat((x[:, :, :, :before], x), dim=3)
        if after > 0:
            x = torch.cat((x, x[:, :, :, -after:]), dim=3)
    return x


def _check_mode(mode: str) -> None:
    if mode != "symmetric":
        raise ValueError(
            f"scattering supports only mode='symmetric' (Kingsbury boundary handling); got {mode!r}"
        )


def _scatter_j1_impl(x, f, magbias: float, combine_colour: bool) -> torch.Tensor:
    """One scale of scattering: (lowpass, |bandpass|) stacked in channels."""
    ll, highs = fwd_j1(x, f["h0o"], f["h1o"], False, 2)
    ll = F.avg_pool2d(ll, 2)
    if combine_colour:
        # combine the colour channels of the magnitude
        mag = torch.sqrt(highs.abs().pow(2).sum(dim=1) + magbias**2) - magbias
        return torch.cat((ll, mag), dim=1)
    mag = _smooth_mag(highs, magbias)
    b, ch, _, h, w = mag.shape
    return torch.cat((ll, mag.reshape(b, ch * 6, h, w)), dim=1)


def _scatter_j2_impl(x, f, magbias: float, combine_colour: bool) -> torch.Tensor:
    """Two scales of scattering with second-order terms."""
    # ---- first order, scale 1 ----
    s0, h1 = fwd_j1(x, f["h0o"], f["h1o"], False, 2)
    if combine_colour:
        s1_j1 = torch.sqrt(h1.abs().pow(2).sum(dim=1) + magbias**2) - magbias  # (N,6,H,W)
    else:
        s1_j1 = _smooth_mag(h1, magbias)  # (N,C,6,H,W)

    # ---- first order, scale 2 ----
    s0, h2 = fwd_j2plus(s0, f["h0a"], f["h1a"], f["h0b"], f["h1b"], False, 2)
    if combine_colour:
        s1_j2 = torch.sqrt(h2.abs().pow(2).sum(dim=1) + magbias**2) - magbias
    else:
        s1_j2 = _smooth_mag(h2, magbias)
    s0 = F.avg_pool2d(s0, 2)

    if combine_colour:
        # second order on the 6 combined-orientation channels
        s1_j1_flat = s1_j1
        s0_2, h2nd = fwd_j1(s1_j1_flat, f["h0o"], f["h1o"], False, 2)
        s1_j1_pool = F.avg_pool2d(s0_2, 2)  # smoothed first-order map
        s2_j1 = torch.sqrt(h2nd.abs().pow(2) + magbias**2) - magbias  # (N,6,6,H,W)
        b, _, _, h, w = s2_j1.shape
        # upstream orders second-order channels (o2nd, o1st)
        s2_j1 = s2_j1.permute(0, 2, 1, 3, 4).reshape(b, 36, h, w)
        return torch.cat((s0, s1_j1_pool, s1_j2, s2_j1), dim=1)

    # ---- second order on the 6*C orientation channels ----
    b, ch, _, h, w = s1_j1.shape
    s1_j1_flat = s1_j1.reshape(b, ch * 6, h, w)  # c-major channel order
    s0_2, h2nd = fwd_j1(s1_j1_flat, f["h0o"], f["h1o"], False, 2)
    s1_j1_pool = F.avg_pool2d(s0_2, 2)  # smoothed first-order map
    s2_j1 = _smooth_mag(h2nd, magbias)  # (N, 6C, 6, H', W')  dims: (C, o1st, o2nd)
    b, ch6, _, h2_, w2_ = s2_j1.shape
    # upstream orders second-order channels (o2nd, o1st); keep C outer
    s2_j1 = s2_j1.reshape(b, ch, 6, 6, h2_, w2_).permute(0, 1, 3, 2, 4, 5)
    s2_j1 = s2_j1.reshape(b, ch * 36, h2_, w2_)
    # pool s1_j1 and flatten s1_j2 for stacking
    b2, ch2, _, h3, w3 = s1_j2.shape
    s1_j2_flat = s1_j2.reshape(b2, ch2 * 6, h3, w3)
    return torch.cat((s0, s1_j1_pool, s1_j2_flat, s2_j1), dim=1)


class ScatLayer(nn.Module):
    """One order of scattering at a single scale (level-1 DTCWT).

    Args:
        biort (str): level-1 biorthogonal filter bank name.
        mode (str): boundary handling (only ``'symmetric'`` is supported).
        magbias (float): smoothing bias of the modulus.
        combine_colour (bool): for 3-channel input, combine colour channels
            into a single set of 6 orientation magnitudes (output has 3+6
            channels instead of 1+6*C).

    Returns a feature map ``(N, 1 + 6*C, H/2, W/2)`` (or ``(N, 9, H/2, W/2)``
    with ``combine_colour``), the lowpass followed by the invariant terms.
    """

    def __init__(self, biort="near_sym_a", mode="symmetric", magbias=1e-2, combine_colour=False):
        super().__init__()
        _validate_names(biort, "qshift_a")
        _check_mode(mode)
        self.biort = biort
        self.mode = mode
        self.magbias = float(magbias)
        self.combine_colour = combine_colour
        h0o, _, h1o, _ = _biort(biort)
        self.register_buffer("h0o", _prep_filt(h0o, torch.float64, torch.device("cpu")))
        self.register_buffer("h1o", _prep_filt(h1o, torch.float64, torch.device("cpu")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ScatLayer expects (N, C, H, W); got {tuple(x.shape)}")
        if self.combine_colour and x.shape[1] != 3:
            raise ValueError(f"combine_colour requires 3 input channels; got {x.shape[1]}")
        dtype = _compute_dtype(x)
        f = {
            "h0o": self.h0o.to(dtype=dtype, device=x.device),
            "h1o": self.h1o.to(dtype=dtype, device=x.device),
        }
        with autocast_off(x.device):
            return _scatter_j1_impl(_pad_to(x.to(dtype), 2), f, self.magbias, self.combine_colour)

    def extra_repr(self) -> str:
        return f"biort={self.biort!r}, mode={self.mode!r}, magbias={self.magbias}"


class ScatLayerj2(nn.Module):
    """Two scales of scattering with second-order terms.

    Applies the level-1 biorthogonal and level-2 qshift DTCWT filters (the
    correct two-scale filter set, unlike stacking two :class:`ScatLayer`).

    Returns ``(N, 49*C, H/4, W/4)``: ``C`` lowpass + ``12*C`` first-order
    invariants (scales 1 and 2) + ``36*C`` second-order invariants (or
    ``(N, 3 + 12 + 36, ...)`` with ``combine_colour``).
    """

    def __init__(
        self,
        biort="near_sym_a",
        qshift="qshift_a",
        mode="symmetric",
        magbias=1e-2,
        combine_colour=False,
    ):
        super().__init__()
        _validate_names(biort, qshift)
        _check_mode(mode)
        self.biort = biort
        self.qshift = qshift
        self.mode = mode
        self.magbias = float(magbias)
        self.combine_colour = combine_colour
        h0o, _, h1o, _ = _biort(biort)
        h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
        dev = torch.device("cpu")
        for name, arr in (
            ("h0o", h0o),
            ("h1o", h1o),
            ("h0a", h0a),
            ("h1a", h1a),
            ("h0b", h0b),
            ("h1b", h1b),
        ):
            self.register_buffer(name, _prep_filt(arr, torch.float64, dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ScatLayerj2 expects (N, C, H, W); got {tuple(x.shape)}")
        if self.combine_colour and x.shape[1] != 3:
            raise ValueError(f"combine_colour requires 3 input channels; got {x.shape[1]}")
        dtype = _compute_dtype(x)
        f = {
            name: getattr(self, name).to(dtype=dtype, device=x.device)
            for name in ("h0o", "h1o", "h0a", "h1a", "h0b", "h1b")
        }
        with autocast_off(x.device):
            return _scatter_j2_impl(_pad_to(x.to(dtype), 8), f, self.magbias, self.combine_colour)

    def extra_repr(self) -> str:
        return (
            f"biort={self.biort!r}, qshift={self.qshift!r}, "
            f"mode={self.mode!r}, magbias={self.magbias}"
        )
