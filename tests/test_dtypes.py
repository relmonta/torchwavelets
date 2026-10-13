"""Dtype, device and autocast behavior: the transform must respect torch
precision (output dtype == input dtype) and never mutate module state."""

import pytest
import torch
from conftest import make_2d, rel_err

import torchwavelets as tw

ALL_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_forward_dtype_follows_input(dtype):
    x = make_2d((2, 2, 16, 16), dtype=dtype)
    dwt = tw.DWT2D(wave="db2", J=2)
    yl, yh = dwt(x)
    assert yl.dtype == dtype
    assert all(h.dtype == dtype for h in yh)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_inverse_dtype_follows_coeffs(dtype):
    x = make_2d((2, 2, 16, 16), dtype=dtype)
    yl, yh = tw.dwt2d(x, wave="db2", mode="zero", level=2)
    xr = tw.idwt2d((yl, yh), wave="db2", mode="zero")
    assert xr.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_roundtrip(dtype):
    x = make_2d((2, 2, 32, 32), dtype=dtype)
    yl, yh = tw.dwt2d(x, wave="db4", mode="symmetric", level=3)
    xr = tw.idwt2d((yl, yh), wave="db4", mode="symmetric")
    assert rel_err(x, xr) < 2e-2


def test_float16_module_built_at_float32():
    """The classic pytorch_wavelets failure: fp32-built module, fp16 input."""
    dwt = tw.DWT2D(wave="db4", J=2)
    idwt = tw.IDWT2D(wave="db4")
    x = make_2d((1, 2, 32, 32), dtype=torch.float16)
    yl, yh = dwt(x)
    assert yl.dtype == torch.float16
    xr = idwt((yl, yh))
    assert xr.dtype == torch.float16
    assert rel_err(x, xr) < 2e-2


def test_mixed_dtype_calls_do_not_mutate_module():
    dwt = tw.DWT2D(wave="db2", J=2)
    for dtype in ALL_DTYPES:
        yl, _ = dwt(make_2d((1, 1, 16, 16), dtype=dtype))
        assert yl.dtype == dtype
    # The module still processes any dtype afterwards, and its cache is keyed
    # per dtype: results are identical to a fresh module.
    ref = tw.DWT2D(wave="db2", J=2)
    for dtype in ALL_DTYPES:
        a = make_2d((1, 1, 16, 16), dtype=dtype, seed=7)
        assert torch.equal(dwt(a)[0], ref(a)[0])


def test_backward_dtype_matches_input():
    x = make_2d((1, 1, 16, 16), dtype=torch.float16, seed=3).requires_grad_()
    yl, yh = tw.dwt2d(x, wave="db2", mode="symmetric", level=2)
    (yl.sum() + sum(h.sum() for h in yh)).backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.float16


def test_autocast_bf16_forward_and_backward():
    x = make_2d((2, 2, 16, 16), dtype=torch.float32).requires_grad_()
    dwt = tw.DWT2D(wave="db4", J=2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        yl, yh = dwt(x)
        assert yl.dtype == torch.bfloat16
        loss = yl.sum() + sum(h.sum() for h in yh)
        loss.backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.float32


def test_grad_scaler_style_scale_invariance():
    """Wavelets are linear: scaling the input scales the coefficients."""
    x = make_2d((1, 1, 16, 16))
    yl, yh = tw.dwt2d(x, wave="db4", mode="symmetric", level=2)
    yl2, yh2 = tw.dwt2d(2.0 * x, wave="db4", mode="symmetric", level=2)
    assert torch.allclose(2.0 * yl, yl2, atol=1e-10)
    for a, b in zip(yh, yh2, strict=True):
        assert torch.allclose(2.0 * a, b, atol=1e-10)
