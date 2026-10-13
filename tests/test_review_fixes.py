"""Regression tests for the issues found by the code review."""

import numpy as np
import pytest
import pywt
import torch
from conftest import make_1d, make_2d

import torchwavelets as tw

# --- packaging: bundled DTCWT filter data must be present -------------------


def test_dtcwt_data_files_shipped():
    import importlib.resources as res

    files = list(res.files("torchwavelets.dtcwt.data").iterdir())
    npz = [f for f in files if f.name.endswith(".npz")]
    assert len(npz) >= 14, "DTCWT filter .npz files missing from the install"


# --- autocast: the whole DTCWT family must not crash ------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtcwt_autocast(dtype):
    x2 = make_2d((1, 1, 16, 16), dtype=torch.float32)
    x1 = make_1d((1, 1, 32), dtype=torch.float32)
    x3 = torch.randn(1, 1, 8, 8, 8, dtype=torch.float32)
    with torch.autocast("cpu", dtype=dtype):
        yl, yh = tw.dtcwt2d(x2, J=1)
        assert yl.dtype == torch.float32 and yh[0].dtype == torch.complex64
        yl, yh = tw.DTCWT2D(J=1)(x2)
        assert yl.dtype == torch.float32 and yh[0].dtype == torch.complex64
        yl, yh = tw.dtcwt1d(x1, J=1)
        assert yl.dtype == torch.float32
        yl, yh = tw.dtcwt3d(x3, J=1)
        assert yl.dtype == torch.float32
        z = tw.ScatLayer()(x2)
        assert z.dtype == torch.float32
        z = tw.ScatLayerj2()(x2)
        assert z.dtype == torch.float32
        s0, invs = tw.DTCWTScattering(J=1)(x2)
        assert s0.dtype == torch.float32
        # inverse under autocast
        yl, yh = tw.dtcwt2d(x2, J=1)
        r = tw.idtcwt2d((yl, yh))
        assert r.dtype == torch.float32 and r.shape == x2.shape


# --- packets: pywt-compatible __setitem__ -----------------------------------


def test_packet_setitem_creates_subnodes():
    x = make_1d((1, 1, 32))
    wp = tw.WaveletPacket(x, "db2", mode="symmetric", maxlevel=2)
    wp["aa"] = torch.zeros_like(wp["aa"].data)
    wp["ad"] = torch.zeros_like(wp["ad"].data)
    assert wp["aa"] is not None and wp["ad"] is not None
    # root path "" sets the root data (pywt behavior)
    wp[""] = torch.zeros_like(x)
    assert torch.equal(wp.data, torch.zeros_like(x))
    # maxlevel overflow raises IndexError like pywt
    with pytest.raises(IndexError):
        wp["aaa"] = 0


def test_packet_axis_supported():
    x = torch.randn(64, 3, 1)  # signal along dim 0
    wp = tw.WaveletPacket(x, "db2", mode="symmetric", maxlevel=2, axis=0)
    wp.get_level(2)
    r = wp.reconstruct(update=False)
    assert torch.allclose(r, x, atol=1e-6)
    # pywt parity of the node shape
    p = pywt.WaveletPacket(x.numpy(), "db2", mode="symmetric", maxlevel=2, axis=0)
    p.get_level(2)
    assert tuple(wp["aa"].data.shape) == p["aa"].data.shape


def test_packet2d_axes_validated():
    with pytest.raises(ValueError):
        tw.WaveletPacket2D(make_2d((1, 1, 32, 32)), "db2", axes=(0, 1))


# --- custom pywt.Wavelet filter banks ---------------------------------------


def test_custom_pywt_wavelet_bank():
    haar = pywt.Wavelet("haar")
    w2 = pywt.Wavelet("db2", filter_bank=(haar.dec_lo, haar.dec_hi, haar.rec_lo, haar.rec_hi))
    x = make_1d((1, 1, 64))
    ta, td = tw.dwt1d(x, wave=w2, mode="periodization", level=1)
    pa, pd = pywt.dwt(x.numpy()[0, 0], w2, mode="periodization")
    assert np.allclose(pa, ta[0, 0].numpy(), atol=1e-12)
    assert np.allclose(pd, td[0][0, 0].numpy(), atol=1e-12)
    # and it must differ from the stock db2 (the old cache bug)
    sa, _ = tw.dwt1d(x, wave="db2", mode="periodization", level=1)
    assert not torch.allclose(sa[0], ta[0], atol=1e-12)


# --- idtcwt1d None bandpasses -----------------------------------------------


def test_idtcwt1d_none_bandpass():
    x = make_1d((1, 1, 32))
    yl, yh = tw.dtcwt1d(x, J=2)
    r = tw.idtcwt1d((yl, [None, None]))
    assert r.shape == x.shape
    assert torch.isfinite(r).all()
    # J=0 forward -> inverse returns the lowpass unchanged
    yl0, yh0 = tw.dtcwt1d(x, J=0)
    assert yh0 is None
    r0 = tw.idtcwt1d((yl0, yh0))
    assert torch.equal(r0, yl0)


# --- SWT inverse validation --------------------------------------------------


def test_iswt1d_shape_mismatch_raises():
    x = make_1d((1, 1, 64))
    coeffs = tw.swt1d(x, "db2", level=2)
    bad = [(cA, cD[..., :-1]) for cA, cD in coeffs]
    with pytest.raises(ValueError):
        tw.iswt1d(bad, "db2")


def test_iswt2d_shape_mismatch_raises():
    x = make_2d((1, 1, 32, 32))
    coeffs = tw.swt2d(x, "db2", level=1)
    cA, cs = coeffs[0]
    bad = [(cA, [c[..., :-1] for c in cs])]
    with pytest.raises(ValueError):
        tw.iswt2d(bad, "db2")


# --- empty batch / zero-size / validation -----------------------------------


def test_empty_batch():
    x = make_2d((0, 1, 16, 16))
    yl, yh = tw.dwt2d(x, level=1)
    assert yl.shape[0] == 0 and yh[0].shape[0] == 0
    yl, yh = tw.dwt2d(x, level=2)
    r = tw.idwt2d((yl, yh))
    assert r.shape[0] == 0
    yl, yh = tw.dtcwt3d(torch.randn(0, 1, 8, 8, 8), J=1)
    assert yl.shape[0] == 0 and yh[0].shape[0] == 0
    yl, yh = tw.dtcwt1d(make_1d((0, 1, 16)), J=1)
    assert yl.shape[0] == 0
    yl, yh = tw.dtcwt2d(x, J=1)
    assert yl.shape[0] == 0


@pytest.mark.parametrize(
    "fn",
    [
        lambda: tw.dwt1d(make_1d((1, 1, 0))),
        lambda: tw.dwt2d(make_2d((1, 1, 0, 16))),
        lambda: tw.idwt1d((make_1d((1, 1, 0)), [])),
        lambda: tw.swt1d(make_1d((1, 1, 0)), "db2", level=1),
        lambda: tw.dtcwt1d(make_1d((1, 1, 0))),
        lambda: tw.dtcwt2d(make_2d((1, 1, 0, 16))),
        lambda: tw.dtcwt3d(torch.randn(1, 1, 0, 8, 8)),
    ],
)
def test_zero_size_raises(fn):
    with pytest.raises(ValueError):
        fn()


@pytest.mark.parametrize(
    "fn",
    [
        lambda: tw.dwt1d(make_1d((1, 1, 16)), level=True),
        lambda: tw.swt1d(make_1d((1, 1, 16)), "db2", level=True),
        lambda: tw.DTCWT2D(J=True),
        lambda: tw.DTCWTScattering(J=True),
        lambda: tw.DTCWTScattering(max_order=True),
    ],
)
def test_bool_level_rejected(fn):
    with pytest.raises(ValueError):
        fn()


def test_module_inverse_none_yl():
    with pytest.raises(ValueError):
        tw.IDWT1D()((None, []))
    with pytest.raises(ValueError):
        tw.IDWT2D()((None, []))
    with pytest.raises(ValueError):
        tw.IDTCWT1D()((None, []))
    with pytest.raises(ValueError):
        tw.IDTCWT2D()((None, []))
    with pytest.raises(ValueError):
        tw.IDTCWT3D()((None, []))


def test_dtcwt3d_odd_dims_raises():
    with pytest.raises(ValueError):
        tw.dtcwt3d(torch.randn(1, 1, 7, 8, 8))
    with pytest.raises(ValueError):
        tw.DTCWT3D()(torch.randn(1, 1, 8, 8, 7))


# --- default boundary modes match pywt (math review findings) ---------------


def test_default_modes_match_pywt():
    x = make_1d((1, 1, 33))
    yl, (yh,) = tw.dwt1d(x, wave="db2")  # default mode
    ref = pywt.dwt(x[0, 0].numpy(), "db2")  # pywt default is 'symmetric'
    assert np.allclose(yl[0, 0].numpy(), ref[0], atol=1e-12)
    assert np.allclose(yh[0, 0].numpy(), ref[1], atol=1e-12)
    # idwt default symmetric
    r = tw.idwt1d((yl, [yh]), wave="db2")
    assert np.allclose(r[0, 0].numpy(), pywt.idwt(*ref, "db2"), atol=1e-10)
    # 2D
    x2 = make_2d((1, 1, 32, 32))
    yl2, (h2,) = tw.dwt2d(x2, wave="db2", level=1)
    ref2 = pywt.dwt2(x2[0, 0].numpy(), "db2")
    assert np.allclose(yl2[0, 0].numpy(), ref2[0], atol=1e-12)


def test_packet2d_default_mode_smooth():
    x = make_2d((1, 1, 16, 16))
    wp = tw.WaveletPacket2D(x, "db2")  # pywt's default is 'smooth'
    wp.get_level(2)
    p = pywt.WaveletPacket2D(x[0, 0].numpy(), "db2")
    p.get_level(2)
    assert tuple(wp["aa"].data.shape) == (1, 1, *p["aa"].data.shape)
    assert np.allclose(wp["aa"].data[0, 0].numpy(), p["aa"].data, atol=1e-12)
    # 1D packet default stays 'symmetric' (pywt default)
    x1 = make_1d((1, 1, 32))
    wp1 = tw.WaveletPacket(x1, "db2")
    wp1.get_level(2)
    p1 = pywt.WaveletPacket(x1[0, 0].numpy(), "db2")
    p1.get_level(2)
    assert np.allclose(wp1["aa"].data[0, 0].numpy(), p1["aa"].data, atol=1e-12)


def test_reflect_length_one_raises():
    with pytest.raises(ValueError):
        tw.dwt1d(make_1d((1, 1, 1)), mode="reflect")
    with pytest.raises(ValueError):
        tw.dwt2d(make_2d((1, 1, 1, 32)), mode="reflect")
    with pytest.raises(ValueError):
        tw.DWT1D(mode="reflect")(make_1d((1, 1, 1)))
    # symmetric at n=1 is fine (pywt allows it)
    yl, _ = tw.dwt1d(make_1d((1, 1, 1)), mode="symmetric")
    assert yl.shape[-1] >= 1


def test_smooth_matches_pywt():
    x = make_1d((1, 1, 32))
    yl, (yh,) = tw.dwt1d(x, wave="db4", mode="smooth", level=1)
    ref = pywt.dwt(x[0, 0].numpy(), "db4", mode="smooth")
    assert np.allclose(yl[0, 0].numpy(), ref[0], atol=1e-12)
    r = tw.idwt1d((yl, [yh]), wave="db4", mode="smooth")
    assert np.allclose(r[0, 0].numpy(), pywt.idwt(*ref, "db4", mode="smooth"), atol=1e-10)


def test_unusable_biorts_clear_error():
    with pytest.raises(ValueError, match="farras"):
        tw.dtcwt2d(make_2d((1, 1, 16, 16)), biort="farras")
    with pytest.raises(ValueError, match="near_sym_a2"):
        tw.DTCWT2D(biort="near_sym_a2")


# --- compat keeps the pytorch_wavelets defaults -----------------------------


def test_compat_default_mode_zero():
    x = make_2d((1, 1, 32, 32))
    xf = tw.compat.DWTForward()(x)  # upstream default mode='zero'
    ref = tw.dwt2d(x, wave="db1", mode="zero", level=1)
    assert torch.equal(xf[0], ref[0])
    assert torch.equal(xf[1][0], ref[1][0])


# --- compat is importable from the package root -----------------------------


def test_compat_reexport():
    assert hasattr(tw, "compat")
    assert hasattr(tw.compat, "DWTForward")
