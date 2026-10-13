"""Numerical parity of the 1-D transform with PyWavelets (fp64, exact)."""

import numpy as np
import pytest
import pywt
import torch
from conftest import make_1d

import torchwavelets as tw

#: Modes PyWavelets supports (``replicate`` is a torchwavelets extension).
PYWT_MODES = ["zero", "symmetric", "reflect", "periodization", "periodic", "constant", "smooth"]


@pytest.mark.parametrize("n", [8, 9, 16, 17])
@pytest.mark.parametrize("mode", PYWT_MODES)
def test_single_level_parity(wavelet, mode, n):
    """dwt1d/idwt1d single level must match pywt.dwt / pywt.idwt exactly."""
    x = make_1d((1, 1, n))
    yl, (yh,) = tw.dwt1d(x, wave=wavelet, mode=mode, level=1)
    ref = pywt.dwt(x[0, 0].numpy(), wavelet, mode=mode)

    assert np.allclose(yl[0, 0].numpy(), ref[0], atol=1e-10)
    assert np.allclose(yh[0, 0].numpy(), ref[1], atol=1e-10)

    xr = tw.idwt1d((yl, [yh]), wave=wavelet, mode=mode)
    ref_rec = pywt.idwt(ref[0], ref[1], wavelet, mode=mode)
    assert np.allclose(xr[0, 0].numpy(), ref_rec, atol=1e-10)


@pytest.mark.parametrize("n,level", [(33, 2), (32, 3), (17, 2)])
@pytest.mark.parametrize("mode", PYWT_MODES)
def test_multilevel_parity(wavelet, mode, n, level):
    x = make_1d((1, 1, n))
    yl, yh = tw.dwt1d(x, wave=wavelet, mode=mode, level=level)
    ref = pywt.wavedec(x[0, 0].numpy(), wavelet, mode=mode, level=level)

    assert np.allclose(yl[0, 0].numpy(), ref[0], atol=1e-10)
    # our yh is finest-first; pywt is coarsest-first
    for j, r in enumerate(reversed(ref[1:])):
        assert np.allclose(yh[j][0, 0].numpy(), r, atol=1e-10)

    xr = tw.idwt1d((yl, yh), wave=wavelet, mode=mode)
    ref_rec = pywt.waverec(ref, wavelet, mode=mode)
    assert np.allclose(xr[0, 0].numpy(), ref_rec, atol=1e-10)


def test_output_length_matches_pywt(wavelet):
    w = pywt.Wavelet(wavelet)
    L = len(w.dec_lo)
    for n in [2, 3, 7, 8, 9, 16, 17]:
        for mode in PYWT_MODES:
            assert tw.dwt1d(make_1d((1, 1, n)), wave=wavelet, mode=mode, level=1)[0].shape[
                -1
            ] == pywt.dwt_coeff_len(n, L, mode)


def test_custom_orthogonal_bank_matches_pywt():
    w = pywt.Wavelet("db2")
    bank = (np.array(w.dec_lo), np.array(w.dec_hi))
    x = make_1d((1, 1, 16))
    yl, (yh,) = tw.dwt1d(x, wave=bank, mode="zero", level=1)
    ref = pywt.dwt(x[0, 0].numpy(), w, mode="zero")
    assert np.allclose(yl[0, 0].numpy(), ref[0], atol=1e-10)
    assert np.allclose(yh[0, 0].numpy(), ref[1], atol=1e-10)
    # and it must perfectly reconstruct
    xr = tw.idwt1d((yl, [yh]), wave=bank, mode="zero")
    assert np.allclose(xr[0, 0].numpy(), x[0, 0].numpy(), atol=1e-10)


@pytest.mark.parametrize("n", [16, 17])
@pytest.mark.parametrize("mode", PYWT_MODES)
def test_idwt_from_pywt_coefficients(wavelet, mode, n):
    """Our synthesis must invert pywt's own analysis coefficients."""
    x = make_1d((1, 1, n))
    cA, cD = pywt.dwt(x[0, 0].numpy(), wavelet, mode=mode)
    yl = torch.tensor(cA, dtype=torch.float64)[None, None]
    yh = torch.tensor(cD, dtype=torch.float64)[None, None]
    xr = tw.idwt1d((yl, [yh]), wave=wavelet, mode=mode)
    ref = pywt.idwt(cA, cD, wavelet, mode=mode)
    assert np.allclose(xr[0, 0].numpy(), ref, atol=1e-10)


@pytest.mark.parametrize("mode", PYWT_MODES)
def test_multilevel_cross_inverse_1d(wavelet, mode):
    """idwt1d of pywt's wavedec coefficients == pywt.waverec."""
    x = make_1d((1, 1, 33))
    coeffs = pywt.wavedec(x[0, 0].numpy(), wavelet, mode=mode, level=2)
    yl = torch.tensor(coeffs[0], dtype=torch.float64)[None, None]
    yh = [torch.tensor(c, dtype=torch.float64)[None, None] for c in reversed(coeffs[1:])]
    xr = tw.idwt1d((yl, yh), wave=wavelet, mode=mode)
    ref = pywt.waverec(coeffs, wavelet, mode=mode)
    assert np.allclose(xr[0, 0].numpy(), ref, atol=1e-10)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("mode", PYWT_MODES)
def test_tiny_sizes_parity(wavelet, mode, n):
    """Signals smaller than the filter support (n < L-2) must still match pywt."""
    if mode == "reflect" and n == 1:
        pytest.skip("pywt rejects reflect mode with n=1 (length must be > 1)")
    if mode == "smooth" and n == 1:
        pytest.skip(
            "pywt's smooth extension reads out-of-bounds memory at n=1 "
            "(degenerate; our linear extrapolation uses slope = x0)"
        )
    x = make_1d((1, 1, n))
    yl, (yh,) = tw.dwt1d(x, wave=wavelet, mode=mode, level=1)
    ref = pywt.dwt(x[0, 0].numpy(), wavelet, mode=mode)
    assert np.allclose(yl[0, 0].numpy(), ref[0], atol=1e-10)
    assert np.allclose(yh[0, 0].numpy(), ref[1], atol=1e-10)
    xr = tw.idwt1d((yl, [yh]), wave=wavelet, mode=mode)
    ref_rec = pywt.idwt(ref[0], ref[1], wavelet, mode=mode)
    assert np.allclose(xr[0, 0].numpy(), ref_rec, atol=1e-10)
