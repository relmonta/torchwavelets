"""3-D DTCWT: parity with the numpy dtcwt reference, roundtrips, dtypes and
gradients."""

import numpy as np
import pytest
import torch
from conftest import rel_err

import torchwavelets as tw

pytest.importorskip("dtcwt")
from dtcwt import Transform3d as Transform3d_np  # noqa: E402

BANKS = [
    ("near_sym_a", "qshift_a"),
    ("near_sym_b", "qshift_b"),
    ("antonini", "qshift_06"),
]


def _x3d(n=16, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, n, n, n, dtype=dtype, generator=g)


@pytest.mark.parametrize("biort,qshift", BANKS)
@pytest.mark.parametrize("J", [1, 2, 3])
def test_parity(biort, qshift, J):
    x = _x3d(32 if J > 1 else 16)
    ref = Transform3d_np(biort=biort, qshift=qshift)
    pyr = ref.forward(x[0, 0].numpy(), nlevels=J)

    yl, yh = tw.dtcwt3d(x, biort=biort, qshift=qshift, J=J)
    assert np.allclose(yl[0, 0].numpy(), pyr.lowpass, atol=1e-8)
    assert len(yh) == J
    for a, b in zip(yh, pyr.highpasses, strict=True):
        assert a.shape[2:] == b.shape
        assert np.allclose(a[0, 0].numpy(), b, atol=1e-8)


@pytest.mark.parametrize("shape", [(16, 16, 16), (32, 32, 32)])
def test_roundtrip(shape):
    x = _x3d(shape[0])
    yl, yh = tw.dtcwt3d(x, J=2)
    xr = tw.idtcwt3d((yl, yh))
    assert xr.shape == x.shape
    assert rel_err(x, xr) < 1e-10


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_promotion(dtype):
    x = _x3d(16, dtype=dtype)
    yl, yh = tw.dtcwt3d(x, J=2)
    assert yl.dtype == dtype
    assert all(
        h.dtype == (torch.complex64 if dtype == torch.float32 else torch.complex128) for h in yh
    )
    xr = tw.idtcwt3d((yl, yh))
    assert xr.dtype == dtype
    assert rel_err(x, xr) < (1e-5 if dtype == torch.float32 else 1e-10)


def test_gradients_flow():
    x = _x3d(16, seed=3).requires_grad_()
    yl, yh = tw.dtcwt3d(x, J=2)
    (yl.sum() + sum(h.abs().sum() for h in yh)).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_module_matches_functional():
    x = _x3d(16, dtype=torch.float32)
    yl1, yh1 = tw.DTCWT3D(J=2)(x)
    yl2, yh2 = tw.dtcwt3d(x, J=2)
    assert torch.equal(yl1, yl2)
    for a, b in zip(yh1, yh2, strict=True):
        assert torch.equal(a, b)
    r1 = tw.IDTCWT3D()((yl1, yh1))
    r2 = tw.idtcwt3d((yl2, yh2))
    assert torch.equal(r1, r2)
    assert rel_err(x, r1) < 1e-5


def test_compile_fullgraph():
    x = _x3d(16, dtype=torch.float32)
    m = torch.compile(tw.DTCWT3D(J=2), backend="eager", fullgraph=True)
    yl, yh = m(x)
    yl2, yh2 = tw.dtcwt3d(x, J=2)
    assert torch.allclose(yl, yl2, atol=1e-6)
    for a, b in zip(yh, yh2, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_invalid_names_raise():
    x = _x3d(16)
    with pytest.raises(ValueError):
        tw.dtcwt3d(x, biort="nope")
    with pytest.raises(ValueError):
        tw.dtcwt3d(x, qshift="nope")


def test_batched():
    x = torch.randn(2, 3, 16, 16, 16, dtype=torch.float64)
    yl, yh = tw.dtcwt3d(x, J=2)
    assert yl.shape[:2] == (2, 3)
    xr = tw.idtcwt3d((yl, yh))
    assert rel_err(x, xr) < 1e-10
