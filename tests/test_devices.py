"""Device portability: the transforms must run on every available device —
CPU, every CUDA device, hybrid CPU/GPU nodes and NVIDIA Grace (GH200,
ARM64) machines — with identical results.
"""

import pytest
import torch
from conftest import make_1d, make_2d, rel_err

import torchwavelets as tw


@pytest.mark.parametrize("wavelet", ["haar", "db4", "bior2.4"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_roundtrip_2d_every_device(any_device, wavelet, dtype):
    x = make_2d((2, 3, 32, 32), dtype=dtype, device=any_device)
    yl, yh = tw.dwt2d(x, wave=wavelet, mode="symmetric", level=2)
    assert yl.device == any_device
    assert all(h.device == any_device for h in yh)
    xr = tw.idwt2d((yl, yh), wave=wavelet, mode="symmetric")
    assert xr.device == any_device
    tol = 1e-3 if dtype == torch.float32 else 1e-10
    assert rel_err(x, xr) < tol


@pytest.mark.parametrize("wavelet", ["db2", "sym4"])
def test_roundtrip_1d_every_device(any_device, wavelet):
    x = make_1d((2, 3, 128), device=any_device)
    yl, yh = tw.dwt1d(x, wave=wavelet, mode="periodization", level=3)
    xr = tw.idwt1d((yl, yh), wave=wavelet, mode="periodization")
    assert xr.device == any_device
    assert rel_err(x, xr) < 1e-10


@pytest.mark.cuda
def test_all_cuda_devices_match_cpu(cuda_device):
    """Every CUDA device (including multi-GPU and Grace) matches CPU exactly."""
    x = make_2d((1, 2, 24, 24), dtype=torch.float64)
    yl_cpu, yh_cpu = tw.dwt2d(x, wave="db4", mode="periodization", level=2)
    yl_g, yh_g = tw.dwt2d(x.to(cuda_device), wave="db4", mode="periodization", level=2)
    assert torch.allclose(yl_cpu, yl_g.cpu(), atol=1e-10)
    for a, b in zip(yh_cpu, yh_g, strict=True):
        assert torch.allclose(a, b.cpu(), atol=1e-10)
    xr_cpu = tw.idwt2d((yl_cpu, yh_cpu), wave="db4", mode="periodization")
    xr_g = tw.idwt2d((yl_g, yh_g), wave="db4", mode="periodization")
    assert torch.allclose(xr_cpu, xr_g.cpu(), atol=1e-10)


@pytest.mark.cuda
def test_half_precision_on_cuda(cuda_device):
    x = make_2d((1, 1, 64, 64), dtype=torch.float16, device=cuda_device)
    yl, yh = tw.dwt2d(x, wave="db4", mode="symmetric", level=3)
    assert yl.dtype == torch.float16
    assert yl.device == cuda_device
    xr = tw.idwt2d((yl, yh), wave="db4", mode="symmetric")
    assert rel_err(x, xr) < 2e-2


@pytest.mark.cuda
def test_autocast_on_cuda(cuda_device):
    x = make_2d((1, 1, 32, 32), dtype=torch.float32, device=cuda_device)
    dwt = tw.DWT2D(wave="db4", J=2, mode="symmetric")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        yl, _ = dwt(x)
    assert yl.dtype == torch.bfloat16


def test_roundtrip_dtcwt_every_device(any_device):
    x = make_2d((1, 2, 32, 32), device=any_device)
    yl, yh = tw.dtcwt2d(x, J=3)
    assert yl.device == any_device
    assert all(h.device == any_device for h in yh)
    assert all(h.dtype.is_complex for h in yh)
    xr = tw.idtcwt2d((yl, yh))
    assert xr.device == any_device
    assert rel_err(x, xr) < 1e-5
    # 1-D DTCWT on the same device
    x1 = make_1d((1, 2, 32), device=any_device)
    yl1, yh1 = tw.dtcwt1d(x1, J=3)
    assert yl1.device == any_device
    assert all(h.device == any_device for h in yh1)
    xr1 = tw.idtcwt1d((yl1, yh1))
    assert rel_err(x1, xr1) < 1e-5


@pytest.mark.cuda
def test_dtcwt_cuda_matches_cpu(cuda_device):
    """DTCWT coefficients on every CUDA device match CPU exactly."""
    x = make_2d((1, 1, 32, 32), dtype=torch.float64)
    yl_cpu, yh_cpu = tw.dtcwt2d(x, J=2)
    yl_g, yh_g = tw.dtcwt2d(x.to(cuda_device), J=2)
    assert torch.allclose(yl_cpu, yl_g.cpu(), atol=1e-10)
    for a, b in zip(yh_cpu, yh_g, strict=True):
        assert torch.allclose(a, b.cpu(), atol=1e-10)
    xr = tw.idtcwt2d((yl_g, yh_g))
    assert rel_err(x.to(cuda_device), xr) < 1e-10


def test_swt_roundtrip_every_device(any_device):
    x = make_1d((1, 2, 64), device=any_device)
    c = tw.swt1d(x, "db2", level=2)
    assert all(v.device == any_device for pair in c for v in pair)
    xr = tw.iswt1d(c, "db2")
    assert xr.device == any_device
    assert rel_err(x, xr) < 1e-10
    x2 = make_2d((1, 1, 32, 32), device=any_device)
    c2 = tw.swt2d(x2, "db2", level=2)
    xr2 = tw.iswt2d(c2, "db2")
    assert xr2.device == any_device
    assert rel_err(x2, xr2) < 1e-10


def test_packet_roundtrip_every_device(any_device):
    x = make_1d((1, 1, 32), device=any_device)
    wp = tw.WaveletPacket(x, "db2", mode="symmetric", maxlevel=2)
    for path in ("aa", "ad", "da", "dd"):
        wp.get_path(path)
    r = wp.reconstruct(update=False)
    assert r.device == any_device
    assert rel_err(x, r) < 1e-10
    x2 = make_2d((1, 1, 32, 32), device=any_device)
    wp2 = tw.WaveletPacket2D(x2, "db2", mode="symmetric", maxlevel=2)
    for path in (
        "aa",
        "ah",
        "av",
        "ad",
        "ha",
        "hh",
        "hv",
        "hd",
        "va",
        "vh",
        "vv",
        "vd",
        "da",
        "dh",
        "dv",
        "dd",
    ):
        wp2.get_path(path)
    r2 = wp2.reconstruct(update=False)
    assert r2.device == any_device
    assert rel_err(x2, r2) < 1e-10


def test_scattering_every_device(any_device):
    x = make_2d((1, 1, 32, 32), device=any_device)
    s0, invs = tw.DTCWTScattering(J=2, max_order=2)(x)
    assert s0.device == any_device
    assert all(t.device == any_device for _, t in invs)
    z = tw.ScatLayerj2()(x)
    assert z.device == any_device
