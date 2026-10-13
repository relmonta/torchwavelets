"""Perfect-reconstruction (round-trip) tests."""

import numpy as np
import pytest
import pywt
import torch
from conftest import make_1d, make_2d, rel_err

import torchwavelets as tw


@pytest.mark.parametrize("n", [16, 32])
def test_roundtrip_1d_even(wavelet, mode, float_dtype, n):
    x = make_1d((2, 3, n), dtype=float_dtype)
    yl, yh = tw.dwt1d(x, wave=wavelet, mode=mode, level=3)
    xr = tw.idwt1d((yl, yh), wave=wavelet, mode=mode)
    tol = 1e-3 if float_dtype == torch.float32 else 1e-10
    assert xr.shape == x.shape
    assert rel_err(x, xr) < tol


@pytest.mark.parametrize("shape", [(16, 16), (32, 32)])
def test_roundtrip_2d_even(wavelet, mode, float_dtype, shape):
    x = make_2d((2, 3, *shape), dtype=float_dtype)
    yl, yh = tw.dwt2d(x, wave=wavelet, mode=mode, level=3)
    xr = tw.idwt2d((yl, yh), wave=wavelet, mode=mode)
    tol = 1e-3 if float_dtype == torch.float32 else 1e-10
    assert xr.shape == x.shape
    assert rel_err(x, xr) < tol


@pytest.mark.parametrize("wavelet", ["haar", "db2", "db4", "bior2.4"])
@pytest.mark.parametrize("mode", ["zero", "symmetric", "periodization"])
@pytest.mark.parametrize("n", [17, 33])
def test_roundtrip_1d_odd(wavelet, mode, n):
    """Odd sizes follow PyWavelets: the reconstruction is 1 sample larger and
    the original signal is recovered in the leading samples."""
    x = make_1d((1, 1, n))
    yl, yh = tw.dwt1d(x, wave=wavelet, mode=mode, level=2)
    xr = tw.idwt1d((yl, yh), wave=wavelet, mode=mode)
    assert xr.shape[-1] >= n
    assert rel_err(x, xr[..., :n]) < 1e-10


@pytest.mark.parametrize("wavelet", ["db2", "sym4", "bior2.4"])
@pytest.mark.parametrize("mode", ["zero", "periodization"])
@pytest.mark.parametrize("shape", [(17, 16), (17, 17)])
def test_roundtrip_2d_odd(wavelet, mode, shape):
    x = make_2d((1, 1, *shape))
    yl, yh = tw.dwt2d(x, wave=wavelet, mode=mode, level=2)
    xr = tw.idwt2d((yl, yh), wave=wavelet, mode=mode)
    assert xr.shape[-2] >= shape[0] and xr.shape[-1] >= shape[1]
    assert rel_err(x, xr[..., : shape[0], : shape[1]]) < 1e-10


def test_roundtrip_highpass_none_means_zero():
    """None highpasses behave exactly like PyWavelets' ``waverec`` with None."""
    x = make_1d((1, 1, 16))
    yl, yh = tw.dwt1d(x, wave="db2", mode="zero", level=2)
    xr = tw.idwt1d((yl, [None, None]), wave="db2", mode="zero")
    ref = pywt.waverec([yl[0, 0].numpy(), None, None], "db2", mode="zero")
    assert np.allclose(xr[0, 0].numpy(), ref, atol=1e-10)


def test_roundtrip_2d_highpass_none_means_zero():
    x = make_2d((1, 1, 16, 16))
    yl, _ = tw.dwt2d(x, wave="db2", mode="zero", level=2)
    xr = tw.idwt2d((yl, [None, None]), wave="db2", mode="zero")
    ref = pywt.waverec2(
        [yl[0, 0].numpy(), (None, None, None), (None, None, None)], "db2", mode="zero"
    )
    assert np.allclose(xr[0, 0].numpy(), ref, atol=1e-10)


def test_small_signals_reconstruct(wavelet):
    """Signals smaller than the filter support must still reconstruct."""
    for n in [2, 3, 4, 5, 6]:
        x = make_1d((1, 1, n))
        yl, yh = tw.dwt1d(x, wave=wavelet, mode="zero", level=1)
        xr = tw.idwt1d((yl, yh), wave=wavelet, mode="zero")
        assert rel_err(x, xr[..., :n]) < 1e-10
