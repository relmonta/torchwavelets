"""1-D DTCWT: parity with the numpy dtcwt reference, roundtrips, dtypes,
gradients and compile compatibility."""

import numpy as np
import pytest
import torch
from conftest import make_1d, rel_err

import torchwavelets as tw

pytest.importorskip("dtcwt")
from dtcwt import Transform1d as Transform1d_np  # noqa: E402

BANKS = [
    ("near_sym_a", "qshift_a"),
    ("near_sym_b", "qshift_b"),
    ("antonini", "qshift_06"),
]


@pytest.mark.parametrize("biort,qshift", BANKS)
@pytest.mark.parametrize("J", [1, 2, 3])
def test_parity(biort, qshift, J):
    x = make_1d((1, 1, 64), dtype=torch.float64)
    ref = Transform1d_np(biort=biort, qshift=qshift)
    pyr = ref.forward(x[0, 0].numpy(), nlevels=J)

    yl, yh = tw.dtcwt1d(x, biort=biort, qshift=qshift, J=J)
    assert np.allclose(yl[0, 0].numpy(), pyr.lowpass.ravel(), atol=1e-8)
    assert len(yh) == J
    for a, b in zip(yh, pyr.highpasses, strict=True):
        assert a.shape[2:] == b.shape[:1]
        assert np.allclose(a[0, 0].numpy(), b.ravel(), atol=1e-8)


@pytest.mark.parametrize("shape", [(64,), (63,), (32,)])
def test_roundtrip(shape):
    x = make_1d((1, 1, *shape))
    yl, yh = tw.dtcwt1d(x, J=3)
    xr = tw.idtcwt1d((yl, yh))
    assert xr.shape[-1] >= shape[0]
    assert rel_err(x[..., : shape[0]], xr[..., : shape[0]]) < 1e-10


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_promotion(dtype):
    x = make_1d((1, 1, 32), dtype=dtype)
    yl, yh = tw.dtcwt1d(x, J=2)
    assert yl.dtype == dtype
    assert all(
        h.dtype == (torch.complex64 if dtype == torch.float32 else torch.complex128) for h in yh
    )
    xr = tw.idtcwt1d((yl, yh))
    assert xr.dtype == dtype
    assert rel_err(x, xr) < (1e-5 if dtype == torch.float32 else 1e-10)


def test_half_computed_in_float32():
    x = make_1d((1, 1, 32), dtype=torch.float16)
    yl, yh = tw.dtcwt1d(x, J=2)
    assert yl.dtype == torch.float32
    assert all(h.dtype == torch.complex64 for h in yh)


def test_include_scale():
    x = make_1d((1, 1, 32))
    scales, yh = tw.dtcwt1d(x, J=2, include_scale=True)
    assert len(scales) == 2 and len(yh) == 2
    assert scales[0].shape == x.shape  # level-1 lowpass is full resolution
    assert scales[1].shape == (1, 1, 16)


def test_gradcheck():
    from torch.autograd import gradcheck

    x = make_1d((1, 1, 8), dtype=torch.float64, seed=7).requires_grad_()

    def f(t):
        yl, yh = tw.dtcwt1d(t, biort="near_sym_a", qshift="qshift_a", J=1)
        return yl, yh[0]

    assert gradcheck(f, (x,), atol=1e-6, rtol=1e-4)


def test_gradients_flow():
    x = make_1d((1, 1, 16), dtype=torch.float64, seed=9).requires_grad_()
    yl, yh = tw.dtcwt1d(x, J=2)
    (yl.sum() + sum(h.abs().sum() for h in yh)).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_module_matches_functional():
    x = make_1d((1, 1, 32))
    m = tw.DTCWT1D(J=3)
    yl1, yh1 = m(x)
    yl2, yh2 = tw.dtcwt1d(x, J=3)
    assert torch.equal(yl1, yl2)
    for a, b in zip(yh1, yh2, strict=True):
        assert torch.equal(a, b)
    r1 = tw.IDTCWT1D()((yl1, yh1))
    r2 = tw.idtcwt1d((yl2, yh2))
    assert torch.equal(r1, r2)


def test_compile_fullgraph():
    x = make_1d((1, 1, 32))
    m = torch.compile(tw.DTCWT1D(J=2), backend="eager", fullgraph=True)
    yl, yh = m(x)
    yl2, yh2 = tw.dtcwt1d(x, J=2)
    assert torch.allclose(yl, yl2, atol=1e-6)
    for a, b in zip(yh, yh2, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_invalid_names_raise():
    x = make_1d((1, 1, 16))
    with pytest.raises(ValueError):
        tw.dtcwt1d(x, biort="nope")
    with pytest.raises(ValueError):
        tw.dtcwt1d(x, qshift="nope")
