"""The pytorch_wavelets compatibility layer: same API, fixed semantics."""

import torch
from conftest import make_1d, make_2d, rel_err

from torchwavelets.compat import DWT, IDWT, DWT1DForward, DWT1DInverse, DWTForward, DWTInverse


def test_upstream_signatures():
    # these are the exact signatures pytorch_wavelets exposes
    f = DWTForward(J=3, wave="db3", mode="zero")
    inv = DWTInverse(wave="db3", mode="zero")
    f1 = DWT1DForward(J=3, wave="db6", mode="zero")
    inv1 = DWT1DInverse(wave="db6", mode="zero")
    assert f.J == 3 and inv.mode == "zero" and f1.J == 3 and inv1.mode == "zero"


def test_dwtforward_roundtrip():
    x = make_2d((2, 3, 32, 32))
    xfm = DWTForward(J=3, wave="db3", mode="zero")
    ifm = DWTInverse(wave="db3", mode="zero")
    yl, yh = xfm(x)
    assert len(yl.shape) == 4 and len(yh) == 3
    xr = ifm((yl, yh))
    assert rel_err(x, xr) < 1e-12


def test_dwt1dforward_roundtrip():
    x = make_1d((2, 3, 64))
    xfm = DWT1DForward(J=3, wave="db6", mode="zero")
    ifm = DWT1DInverse(wave="db6", mode="zero")
    yl, yh = xfm(x)
    assert len(yl.shape) == 3 and len(yh) == 3
    xr = ifm((yl, yh))
    assert rel_err(x, xr) < 1e-12


def test_compat_float16_input_fp32_module():
    """The upstream bug: a fp32-built inverse transform must accept fp16."""
    x = make_2d((1, 1, 16, 16), dtype=torch.float16)
    xfm = DWTForward(J=2, wave="db2", mode="symmetric")
    ifm = DWTInverse(wave="db2", mode="symmetric")
    yl, yh = xfm(x)
    assert yl.dtype == torch.float16
    xr = ifm((yl, yh))
    assert xr.dtype == torch.float16
    assert rel_err(x, xr) < 2e-2


def test_historical_aliases():
    assert DWT is DWTForward
    assert IDWT is DWTInverse


def test_compat_with_autocast():
    x = make_2d((1, 1, 16, 16), dtype=torch.float32)
    xfm = DWTForward(J=2, wave="db2", mode="symmetric")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        yl, _ = xfm(x)
    assert yl.dtype == torch.bfloat16
