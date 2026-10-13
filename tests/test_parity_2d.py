"""Numerical parity of the 2-D transform with PyWavelets (fp64, exact)."""

import numpy as np
import pytest
import pywt
import torch
from conftest import make_2d
from test_parity_1d import PYWT_MODES

import torchwavelets as tw


@pytest.mark.parametrize("shape", [(16, 16), (16, 17), (17, 17)])
@pytest.mark.parametrize("mode", PYWT_MODES)
def test_single_level_parity(wavelet, mode, shape):
    """dwt2d/idwt2d single level must match pywt.dwt2 / pywt.idwt2 exactly."""
    x = make_2d((1, 1, *shape))
    yl, (yh,) = tw.dwt2d(x, wave=wavelet, mode=mode, level=1)
    ref = pywt.dwt2(x[0, 0].numpy(), wavelet, mode=mode)
    cA, (cH, cV, cD) = ref

    ours = [yl[0, 0].numpy(), yh[0, 0, 0].numpy(), yh[0, 0, 1].numpy(), yh[0, 0, 2].numpy()]
    for o, r in zip(ours, [cA, cH, cV, cD], strict=True):
        assert o.shape == r.shape
        assert np.allclose(o, r, atol=1e-10)

    xr = tw.idwt2d((yl, [yh]), wave=wavelet, mode=mode)
    ref_rec = pywt.idwt2(ref, wavelet, mode=mode)
    assert np.allclose(xr[0, 0].numpy(), ref_rec, atol=1e-10)


@pytest.mark.parametrize(
    "shape,level",
    [((33, 33), 2), ((32, 32), 3), ((33, 32), 2), ((17, 17), 2)],
)
@pytest.mark.parametrize("mode", PYWT_MODES)
def test_multilevel_parity(wavelet, mode, shape, level):
    x = make_2d((1, 1, *shape))
    yl, yh = tw.dwt2d(x, wave=wavelet, mode=mode, level=level)
    ref = pywt.wavedec2(x[0, 0].numpy(), wavelet, mode=mode, level=level)

    assert np.allclose(yl[0, 0].numpy(), ref[0], atol=1e-10)
    for j, (dh, dv, dd) in enumerate(ref[1:]):
        h = yh[len(yh) - 1 - j]
        assert np.allclose(h[0, 0, 0].numpy(), dh, atol=1e-10)
        assert np.allclose(h[0, 0, 1].numpy(), dv, atol=1e-10)
        assert np.allclose(h[0, 0, 2].numpy(), dd, atol=1e-10)

    xr = tw.idwt2d((yl, yh), wave=wavelet, mode=mode)
    ref_rec = pywt.waverec2(ref, wavelet, mode=mode)
    assert np.allclose(xr[0, 0].numpy(), ref_rec, atol=1e-10)


@pytest.mark.parametrize("shape", [(16, 16), (17, 17)])
@pytest.mark.parametrize("mode", PYWT_MODES)
def test_idwt_from_pywt_coefficients(wavelet, mode, shape):
    """Our synthesis must invert pywt's own analysis coefficients."""
    x = make_2d((1, 1, *shape))
    cA, (cH, cV, cD) = pywt.dwt2(x[0, 0].numpy(), wavelet, mode=mode)
    yl = torch.tensor(cA, dtype=torch.float64)[None, None]
    h = torch.stack([torch.tensor(c) for c in (cH, cV, cD)]).unsqueeze(0).unsqueeze(0)
    xr = tw.idwt2d((yl, [h]), wave=wavelet, mode=mode)
    ref = pywt.idwt2((cA, (cH, cV, cD)), wavelet, mode=mode)
    assert np.allclose(xr[0, 0].numpy(), ref, atol=1e-10)


@pytest.mark.parametrize("mode", PYWT_MODES)
def test_multilevel_cross_inverse_2d(wavelet, mode):
    """idwt2d of pywt's wavedec2 coefficients == pywt.waverec2."""
    x = make_2d((1, 1, 33, 33))
    coeffs = pywt.wavedec2(x[0, 0].numpy(), wavelet, mode=mode, level=2)
    yl = torch.tensor(coeffs[0], dtype=torch.float64)[None, None]
    yh = []
    for dh, dv, dd in reversed(coeffs[1:]):
        h = (
            torch.stack([torch.tensor(dh), torch.tensor(dv), torch.tensor(dd)])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        yh.append(h)
    xr = tw.idwt2d((yl, yh), wave=wavelet, mode=mode)
    ref = pywt.waverec2(coeffs, wavelet, mode=mode)
    assert np.allclose(xr[0, 0].numpy(), ref, atol=1e-10)


def test_custom_biorthogonal_4tuple_parity_2d():
    """A biorthogonal custom bank (4-tuple) must match pywt's bior2.4."""
    w = pywt.Wavelet("bior2.4")
    bank = (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)
    x = make_2d((1, 1, 16, 16))
    yl, (yh,) = tw.dwt2d(x, wave=bank, mode="zero", level=1)
    ref = pywt.dwt2(x[0, 0].numpy(), w, mode="zero")
    ours = [yl[0, 0].numpy(), yh[0, 0, 0].numpy(), yh[0, 0, 1].numpy(), yh[0, 0, 2].numpy()]
    for o, r in zip(ours, [ref[0], *ref[1]], strict=True):
        assert np.allclose(o, r, atol=1e-10)
    # and it must perfectly reconstruct
    xr = tw.idwt2d((yl, [yh]), wave=bank, mode="zero")
    assert np.allclose(xr[0, 0].numpy(), x[0, 0].numpy(), atol=1e-10)
