"""Edge cases: level=0, invalid arguments, small signals, batches and
channels, custom filter banks, aliases."""

import numpy as np
import pytest
import pywt
import torch
from conftest import make_1d, make_2d

import torchwavelets as tw


def test_level_zero_returns_input():
    x = make_2d((1, 1, 8, 8))
    yl, yh = tw.dwt2d(x, wave="db2", mode="zero", level=0)
    assert torch.equal(yl, x)
    assert yh == []
    x1 = make_1d((1, 1, 8))
    yl, yh = tw.dwt1d(x1, wave="db2", mode="zero", level=0)
    assert torch.equal(yl, x1)
    assert yh == []


def test_invalid_level_raises():
    x = make_2d((1, 1, 8, 8))
    with pytest.raises(ValueError):
        tw.dwt2d(x, wave="db2", level=-1)
    with pytest.raises(ValueError):
        tw.dwt2d(x, wave="db2", level=1.5)


def test_invalid_mode_raises():
    x = make_2d((1, 1, 8, 8))
    with pytest.raises(ValueError):
        tw.dwt2d(x, wave="db2", mode="banana")
    with pytest.raises(ValueError):
        tw.DWT2D(wave="db2", mode="banana")


def test_invalid_wave_raises():
    x = make_2d((1, 1, 8, 8))
    with pytest.raises(ValueError):
        tw.dwt2d(x, wave="definitely-not-a-wavelet")


def test_ndim_validation():
    with pytest.raises(ValueError):
        tw.dwt2d(torch.randn(2, 2, 8), wave="db2")
    with pytest.raises(ValueError):
        tw.dwt1d(torch.randn(2, 2, 8, 8), wave="db2")


def test_batch_and_channels():
    x = make_2d((5, 7, 32, 32))
    yl, yh = tw.dwt2d(x, wave="db2", mode="symmetric", level=2)
    assert yl.shape == (5, 7, 10, 10)
    assert yh[0].shape == (5, 7, 3, 17, 17)
    assert yh[1].shape == (5, 7, 3, 10, 10)


def test_mode_aliases():
    x = make_2d((1, 1, 16, 16))
    for alias, canonical in [
        ("per", "periodization"),
        ("wrap", "periodic"),
        ("edge", "constant"),
        ("replicate", "constant"),
    ]:
        a = tw.dwt2d(x, wave="db2", mode=alias, level=1)
        b = tw.dwt2d(x, wave="db2", mode=canonical, level=1)
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1][0], b[1][0])


def test_pywt_wavelet_object_accepted():
    x = make_2d((1, 1, 16, 16))
    yl, yh = tw.dwt2d(x, wave=pywt.Wavelet("db3"), mode="zero", level=1)
    ref = pywt.dwt2(x[0, 0].numpy(), "db3", mode="zero")
    assert np.allclose(yl[0, 0].numpy(), ref[0], atol=1e-10)


def test_custom_bank_4tuple():
    w = pywt.Wavelet("db2")
    bank = (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)
    x = make_2d((1, 1, 16, 16))
    yl, yh = tw.dwt2d(x, wave=bank, mode="zero", level=1)
    ref = pywt.dwt2(x[0, 0].numpy(), w, mode="zero")
    assert np.allclose(yl[0, 0].numpy(), ref[0], atol=1e-10)


def test_module_repr():
    dwt = tw.DWT2D(J=3, wave="db4", mode="symmetric")
    assert "J=3" in repr(dwt)
    assert "db4" in repr(dwt)


def test_module_pickle_roundtrip():
    import pickle

    dwt = tw.DWT2D(J=2, wave="db4", mode="symmetric")
    dwt2 = pickle.loads(pickle.dumps(dwt))
    x = make_2d((1, 1, 16, 16))
    a, b = dwt(x)
    a2, b2 = dwt2(x)
    assert torch.equal(a, a2)
    assert torch.equal(b[0], b2[0])


def test_module_validation():
    with pytest.raises(ValueError):
        tw.DWT2D(J=-1, wave="db2")
    with pytest.raises(ValueError):
        tw.DWT1D(J=1.5, wave="db2")
    with pytest.raises(ValueError):
        tw.DWT2D(wave="db2", mode="banana")
    # wrong ndim through the module API must raise, not IndexError
    with pytest.raises(ValueError):
        tw.DWT2D(wave="db2")(torch.randn(2, 2, 8))
    with pytest.raises(ValueError):
        tw.DWT1D(wave="db2")(torch.randn(2, 2, 8, 8))


def test_idwt_shape_mismatch_raises():
    yl = make_2d((1, 1, 8, 8))
    h = make_2d((1, 1, 3, 6, 6))  # 6 is not a valid pairing for 8 (only 7 is, via crop)
    with pytest.raises(ValueError, match="shape mismatch"):
        tw.idwt2d((yl, [h]), wave="db2", mode="zero")
    yl1 = make_1d((1, 1, 8))
    h1 = make_1d((1, 1, 6))
    with pytest.raises(ValueError, match="shape mismatch"):
        tw.idwt1d((yl1, [h1]), wave="db2", mode="zero")


def test_biorthogonal_custom_2tuple_rejected():
    w = pywt.Wavelet("bior2.4")
    with pytest.raises(ValueError, match="perfectly reconstructing"):
        tw.dwt2d(make_2d((1, 1, 8, 8)), wave=(w.dec_lo, w.dec_hi))


def test_unequal_length_custom_bank_rejected():
    w = pywt.Wavelet("db2")
    with pytest.raises(ValueError, match="equal-length"):
        tw.dwt1d(make_1d((1, 1, 16)), wave=(w.dec_lo[:-1], w.dec_hi, w.rec_lo, w.rec_hi))


def test_level_too_high_warns():
    x = make_1d((1, 1, 16))  # db4 (L=8): max level = 1
    with pytest.warns(UserWarning, match="too high"):
        tw.dwt1d(x, wave="db4", mode="zero", level=3)


def test_tiny_signal_2d():
    """A 2x2 image with db4 (L=8) must still run and reconstruct."""
    x = make_2d((1, 1, 2, 2))
    yl, yh = tw.dwt2d(x, wave="db4", mode="zero", level=1)
    xr = tw.idwt2d((yl, yh), wave="db4", mode="zero")
    assert xr.shape[2:] >= (2, 2)
