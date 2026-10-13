"""DTCWT parity with the reference numpy implementation (``dtcwt`` package)."""

import numpy as np
import pytest
import torch
from conftest import make_2d

import torchwavelets as tw

pytest.importorskip("dtcwt")
from dtcwt import Transform2d as Transform2d_np  # noqa: E402

BANKS = [
    ("near_sym_a", "qshift_a"),
    ("near_sym_b", "qshift_b"),
    ("antonini", "qshift_06"),
]


def _to_np(yl, yh):
    """(N=1, C=1) tensors -> numpy (H, W) / list of (H, W, 6)."""
    lows = yl.detach().cpu().numpy()[0, 0]
    highs = [np.transpose(h.detach().cpu().numpy()[0, 0], (1, 2, 0)) for h in yh]
    return lows, highs


@pytest.mark.parametrize("biort,qshift", BANKS)
@pytest.mark.parametrize("J", [1, 2, 3])
def test_parity(biort, qshift, J):
    """Coefficients must match the numpy dtcwt reference exactly (fp64)."""
    x = make_2d((1, 1, 64, 64))
    ref = Transform2d_np(biort=biort, qshift=qshift)
    pyr = ref.forward(x[0, 0].numpy(), nlevels=J)

    yl, yh = tw.dtcwt2d(x, biort=biort, qshift=qshift, J=J)
    yl_np, yh_np = _to_np(yl, yh)

    assert np.allclose(yl_np, pyr.lowpass, atol=1e-8)
    assert len(yh_np) == J
    for a, b in zip(yh_np, pyr.highpasses, strict=True):
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=1e-8)


@pytest.mark.parametrize("shape", [(64, 64), (65, 63), (32, 32)])
def test_roundtrip(shape):
    x = make_2d((1, 1, *shape))
    yl, yh = tw.dtcwt2d(x, J=3)
    xr = tw.idtcwt2d((yl, yh))
    # odd inputs are padded to even; the original is recovered in the prefix
    assert xr.shape[-2] >= shape[0] and xr.shape[-1] >= shape[1]
    err = (xr[..., : shape[0], : shape[1]] - x).norm().item() / x.norm().item()
    assert err < 1e-10


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_promotion(dtype):
    x = make_2d((1, 1, 32, 32), dtype=dtype)
    yl, yh = tw.dtcwt2d(x, J=2)
    assert yl.dtype == dtype
    assert all(
        h.dtype == torch.complex64 if dtype == torch.float32 else torch.complex128 for h in yh
    )
    xr = tw.idtcwt2d((yl, yh))
    assert xr.dtype == dtype
    assert (x - xr).norm().item() / x.norm().item() < (1e-5 if dtype == torch.float32 else 1e-10)


def test_half_input_computed_in_float32():
    x = make_2d((1, 1, 32, 32), dtype=torch.float16)
    yl, yh = tw.dtcwt2d(x, J=2)
    assert yl.dtype == torch.float32
    assert all(h.dtype == torch.complex64 for h in yh)


def test_skip_hps():
    x = make_2d((1, 1, 32, 32))
    yl, yh = tw.dtcwt2d(x, J=3, skip_hps=[True, False, True])
    assert yh[0] is None and yh[1] is not None and yh[2] is None
    xr = tw.idtcwt2d((yl, yh))
    assert xr.shape == x.shape


def test_include_scale():
    x = make_2d((1, 1, 32, 32))
    scales, yh = tw.dtcwt2d(x, J=2, include_scale=True)
    assert len(scales) == 2 and len(yh) == 2
    # level-1 lowpass is full resolution; level-2 lowpass is 2x the bandpass
    assert scales[0].shape == (1, 1, 32, 32)
    assert scales[1].shape == (1, 1, 16, 16)
    assert yh[1].shape[:2] + yh[1].shape[3:] == (1, 1, 8, 8)


def test_gradcheck():
    from torch.autograd import gradcheck

    x = make_2d((1, 1, 8, 8), dtype=torch.float64, seed=7).requires_grad_()

    def f(t):
        yl, yh = tw.dtcwt2d(t, biort="near_sym_a", qshift="qshift_a", J=1)
        return yl, yh[0]

    assert gradcheck(f, (x,), atol=1e-6, rtol=1e-4)


def test_gradients_flow():
    x = make_2d((1, 1, 16, 16), dtype=torch.float64, seed=9).requires_grad_()
    yl, yh = tw.dtcwt2d(x, J=2)
    loss = yl.abs().sum() + sum(h.abs().sum() for h in yh)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_invalid_names_raise():
    x = make_2d((1, 1, 16, 16))
    with pytest.raises(ValueError):
        tw.dtcwt2d(x, biort="nope")
    with pytest.raises(ValueError):
        tw.dtcwt2d(x, qshift="nope")
