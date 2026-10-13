"""Stationary wavelet transform: PyWavelets parity, roundtrips, dtypes,
gradients and compile compatibility."""

import numpy as np
import pytest
import torch
from conftest import make_1d, make_2d, rel_err

import torchwavelets as tw

WAVELETS = ["haar", "db2", "db4", "sym4", "bior2.4"]


@pytest.mark.parametrize("wave", WAVELETS)
@pytest.mark.parametrize("level", [1, 2, 3])
def test_swt1d_parity(wave, level):
    import pywt

    x = make_1d((1, 2, 64), dtype=torch.float64)
    ref = pywt.swt(x[0].numpy(), wave, level=level)
    mine = tw.swt1d(x, wave, level=level)
    assert len(mine) == level
    for i in range(level):
        assert np.allclose(mine[i][0][0].numpy(), ref[i][0], atol=1e-10)
        assert np.allclose(mine[i][1][0].numpy(), ref[i][1], atol=1e-10)


@pytest.mark.parametrize("wave", WAVELETS)
@pytest.mark.parametrize("level", [1, 2, 3])
def test_swt1d_roundtrip(wave, level):
    x = make_1d((1, 2, 64), dtype=torch.float64)
    c = tw.swt1d(x, wave, level=level)
    xr = tw.iswt1d(c, wave)
    assert rel_err(x, xr) < 1e-10


@pytest.mark.parametrize("wave", ["db2", "haar", "bior2.4"])
@pytest.mark.parametrize("level", [1, 2])
def test_swt2d_parity(wave, level):
    import pywt

    x = make_2d((1, 1, 64, 64), dtype=torch.float64)
    ref = pywt.swt2(x[0, 0].numpy(), wave, level=level)
    mine = tw.swt2d(x, wave, level=level)
    for i in range(level):
        assert np.allclose(mine[i][0][0, 0].numpy(), ref[i][0], atol=1e-10)
        for a, b in zip(mine[i][1], ref[i][1], strict=True):
            assert np.allclose(a[0, 0].numpy(), b, atol=1e-10)


@pytest.mark.parametrize("wave", ["db2", "haar", "sym3"])
@pytest.mark.parametrize("level", [1, 2])
def test_swt2d_roundtrip(wave, level):
    x = make_2d((1, 1, 64, 64), dtype=torch.float64)
    c = tw.swt2d(x, wave, level=level)
    xr = tw.iswt2d(c, wave)
    assert rel_err(x, xr) < 1e-10


def test_norm_matches_pywt():
    """norm=True must reproduce PyWavelets' rescaled filters exactly."""
    import pywt

    x = make_1d((1, 1, 64), dtype=torch.float64)
    ref = pywt.swt(x[0, 0].numpy(), "db2", level=2, norm=True)
    mine = tw.swt1d(x, "db2", level=2, norm=True)
    for i in range(2):
        assert np.allclose(mine[i][0][0].numpy(), ref[i][0], atol=1e-10)
        assert np.allclose(mine[i][1][0].numpy(), ref[i][1], atol=1e-10)
    xr = tw.iswt1d(mine, "db2", norm=True)
    assert rel_err(x, xr) < 1e-10


def test_norm_roundtrip_2d():
    x = make_2d((1, 1, 64, 64), dtype=torch.float64)
    c = tw.swt2d(x, "db2", level=2, norm=True)
    xr = tw.iswt2d(c, "db2", norm=True)
    assert rel_err(x, xr) < 1e-10


def test_all_subbands_same_size():
    x = make_2d((2, 3, 32, 32))
    c = tw.swt2d(x, "db2", level=2)
    for pair in c:
        assert pair[0].shape == x.shape
        for h in pair[1]:
            assert h.shape == x.shape


def test_divisibility_required():
    with pytest.raises(ValueError):
        tw.swt1d(make_1d((1, 1, 12)), "db2", level=3)
    with pytest.raises(ValueError):
        tw.swt2d(make_2d((1, 1, 12, 16)), "db2", level=3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtypes(dtype):
    x = make_1d((1, 1, 32), dtype=dtype)
    c = tw.swt1d(x, "db2", level=2)
    assert all(v.dtype == dtype for pair in c for v in pair)
    assert tw.iswt1d(c, "db2").dtype == dtype


def test_fp16():
    x = make_1d((1, 1, 32), dtype=torch.float16)
    c = tw.swt1d(x, "db2", level=2)
    assert all(v.dtype == torch.float16 for pair in c for v in pair)
    xr = tw.iswt1d(c, "db2")
    assert xr.dtype == torch.float16
    assert rel_err(x, xr) < 2e-2


def test_gradients_flow():
    x = make_1d((1, 1, 32), dtype=torch.float64, seed=3).requires_grad_()
    c = tw.swt1d(x, "db2", level=2)
    (sum(v.sum() for pair in c for v in pair)).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    x2 = make_2d((1, 1, 16, 16), dtype=torch.float64, seed=4).requires_grad_()
    c2 = tw.swt2d(x2, "db2", level=2)
    loss = sum(v.sum() for pair in c2 for v in (pair[0], *pair[1]))
    loss.backward()
    assert x2.grad is not None
    assert torch.isfinite(x2.grad).all()


def test_module_matches_functional():
    x = make_1d((1, 1, 32))
    c1 = tw.SWT1D("db2", level=2)(x)
    c2 = tw.swt1d(x, "db2", level=2)
    for (a1, b1), (a2, b2) in zip(c1, c2, strict=True):
        assert torch.equal(a1, a2)
        assert torch.equal(b1, b2)
    xr1 = tw.ISWT1D("db2")(c1)
    xr2 = tw.iswt1d(c2, "db2")
    assert torch.equal(xr1, xr2)


def test_module_2d_matches_functional():
    x = make_2d((1, 1, 32, 32))
    c1 = tw.SWT2D("db2", level=2)(x)
    c2 = tw.swt2d(x, "db2", level=2)
    for (a1, d1), (a2, d2) in zip(c1, c2, strict=True):
        assert torch.equal(a1, a2)
        for h1, h2 in zip(d1, d2, strict=True):
            assert torch.equal(h1, h2)
    xr1 = tw.ISWT2D("db2")(c1)
    assert rel_err(x, xr1) < 1e-10


def test_compile_fullgraph_forward():
    x = make_1d((1, 1, 64))
    c = torch.compile(tw.SWT1D("db2", level=2), backend="eager", fullgraph=True)
    out = c(x)
    ref = tw.swt1d(x, "db2", level=2)
    for (a1, b1), (a2, b2) in zip(out, ref, strict=True):
        assert torch.allclose(a1, a2, atol=1e-6)
        assert torch.allclose(b1, b2, atol=1e-6)


def test_compile_fullgraph_inverse():
    x = make_1d((1, 1, 64))
    c = tw.swt1d(x, "db2", level=2)
    f = torch.compile(tw.ISWT1D("db2"), backend="eager", fullgraph=True)
    xr = f(c)
    assert rel_err(x, xr) < 1e-6
