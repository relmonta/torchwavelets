"""DTCWT compatibility layer: upstream pytorch_wavelets API and semantics."""

import pytest
import torch
from conftest import make_2d

from torchwavelets.compat import DTCWT, IDTCWT, DTCWTForward, DTCWTInverse


def test_forward_ri_layout():
    x = make_2d((1, 2, 32, 32), dtype=torch.float32)
    xfm = DTCWTForward(J=3, biort="near_sym_a", qshift="qshift_a")
    yl, yh = xfm(x)
    assert yl.dtype == torch.float32
    assert len(yh) == 3
    assert yh[0].shape == (1, 2, 6, 16, 16, 2)
    assert yh[0].dtype == torch.float32


def test_roundtrip_ri_layout():
    x = make_2d((1, 1, 32, 32), dtype=torch.float32)
    xfm = DTCWTForward(J=3)
    ifm = DTCWTInverse()
    yl, yh = xfm(x)
    xr = ifm((yl, yh))
    assert xr.shape == x.shape
    assert (x - xr).norm().item() / x.norm().item() < 1e-5


def test_inverse_accepts_native_complex():
    x = make_2d((1, 1, 32, 32), dtype=torch.float32)
    import torchwavelets as tw

    yl, yh = tw.dtcwt2d(x, J=3)
    xr = DTCWTInverse()((yl, yh))
    assert (x - xr).norm().item() / x.norm().item() < 1e-5


def test_unsupported_params_raise():
    with pytest.raises(ValueError):
        DTCWTForward(ri_dim=1)
    with pytest.raises(ValueError):
        DTCWTForward(mode="zero")
    with pytest.raises(ValueError):
        DTCWTInverse(ri_dim=0)


def test_historical_aliases():
    assert DTCWT is DTCWTForward
    assert IDTCWT is DTCWTInverse


def test_module_matches_functional():
    x = make_2d((1, 1, 32, 32))
    m = torchwavelets_module_dtcwt()
    yl1, yh1 = m(x)
    yl2, yh2 = torchwavelets_functional_dtcwt(x)
    assert torch.equal(yl1, yl2)
    for a, b in zip(yh1, yh2, strict=True):
        assert torch.equal(a, b)


def torchwavelets_module_dtcwt():
    import torchwavelets as tw

    return tw.DTCWT2D(J=3)


def torchwavelets_functional_dtcwt(x):
    import torchwavelets as tw

    return tw.dtcwt2d(x, J=3)
