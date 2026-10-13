"""General DTCWT scattering network: reference parity, shapes, invariance,
dtypes, gradients and compile compatibility."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import make_2d

import torchwavelets as tw


def test_parity_with_scatlayerj2():
    """(J=2, max_order=2) must reproduce ScatLayerj2 exactly."""
    x = make_2d((1, 1, 64, 64), dtype=torch.float64, seed=1)
    net = tw.DTCWTScattering(J=2, max_order=2, magbias=1e-2)
    s0, invs = net(x)
    z = tw.ScatLayerj2(magbias=1e-2)(x)
    blocks = {
        (1,): z[:, 1:7],
        (2,): z[:, 7:13],
        (1, 2): z[:, 13:49],
    }
    assert torch.allclose(z[:, 0:1], s0, atol=1e-10)
    bypath = dict(invs)
    assert set(bypath) == set(blocks)
    for path, ref in blocks.items():
        assert torch.allclose(ref, bypath[path], atol=1e-10)


def test_parity_with_scatlayerj2_multichannel():
    """Channel ordering must match ScatLayerj2 for C > 1 (not just C == 1)."""
    x = make_2d((1, 3, 64, 64), dtype=torch.float64, seed=1)
    net = tw.DTCWTScattering(J=2, max_order=2, magbias=1e-2)
    s0, invs = net(x)
    z = tw.ScatLayerj2(magbias=1e-2)(x)
    C = 3
    blocks = {
        (1,): z[:, C : C + 6 * C],
        (2,): z[:, C + 6 * C : C + 12 * C],
        (1, 2): z[:, C + 12 * C : 49 * C],
    }
    assert torch.allclose(z[:, :C], s0, atol=1e-10)
    bypath = dict(invs)
    for path, ref in blocks.items():
        assert torch.allclose(ref, bypath[path], atol=1e-10)
    # and ScatLayer parity at J=1
    z1 = tw.ScatLayer(magbias=1e-2)(x)
    s01, invs1 = tw.DTCWTScattering(J=1, max_order=1, magbias=1e-2)(x)
    assert torch.allclose(z1[:, :C], s01, atol=1e-10)
    assert torch.allclose(z1[:, C:], dict(invs1)[(1,)], atol=1e-10)


@pytest.mark.parametrize("J,max_order", [(3, 1), (3, 2), (3, 3), (4, 2)])
def test_paths_and_shapes(J, max_order):
    x = make_2d((1, 1, 64, 64), dtype=torch.float64, seed=2)
    s0, invs = tw.DTCWTScattering(J=J, max_order=max_order)(x)
    paths = [p for p, _ in invs]
    # all strictly increasing paths of length 1..max_order, ordered by
    # (length, lexicographic)
    expected = sorted(
        (
            p
            for m in range(1, max_order + 1)
            for p in __import__("itertools").combinations(range(1, J + 1), m)
        ),
        key=lambda p: (len(p), p),
    )
    assert paths == expected
    # invariant spatial size: smoothed (extendable) paths live at 2^(jm+1),
    # terminal paths at 2^jm
    assert s0.shape[-2:] == (64 // 2**J,) * 2
    for path, t in invs:
        extendable = path[-1] < J and len(path) < max_order
        eff = path[-1] + (1 if extendable else 0)
        assert t.shape[-2:] == (64 // 2**eff,) * 2


def test_multi_channel():
    x = make_2d((1, 3, 64, 64), dtype=torch.float64, seed=3)
    s0, invs = tw.DTCWTScattering(J=2, max_order=2)(x)
    assert s0.shape[1] == 3
    bypath = dict(invs)
    assert bypath[(1,)].shape[1] == 6 * 3
    assert bypath[(1, 2)].shape[1] == 36 * 3


def test_shift_invariance():
    """Invariants are shift-invariant at dyadic integer shifts."""
    g = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128] / 128
    v = (
        0.5 * np.sin(2 * np.pi * 8 * xx)
        + 0.3 * np.sin(2 * np.pi * 4 * yy + 1.0)
        + 0.4 * np.exp(-((xx - 0.3) ** 2 + (yy - 0.7) ** 2) / 0.02)
    )
    x = torch.tensor(v + 0.05 * g.standard_normal((128, 128)))[None, None]
    net = tw.DTCWTScattering(J=3, max_order=2, magbias=1e-2)
    s0, invs = net(x)
    shift = 8
    pad = F.pad(x, (0, shift, 0, shift), mode="reflect")
    xs = pad[..., shift : shift + 128, shift : shift + 128]
    s0s, invss = net(xs)
    assert torch.allclose(
        torch.roll(s0s, (1, 1), (-1, -2))[:, :, 6:-6, 6:-6],
        s0[:, :, 6:-6, 6:-6],
        atol=1e-5,
    )
    for (pa, a), (_, b) in zip(invs, invss, strict=True):
        # smoothed (extendable) invariants live at the effective scale jm+1
        eff = pa[-1] + (1 if pa[-1] < 3 and len(pa) < 2 else 0)
        k = shift // 2**eff
        assert torch.allclose(
            torch.roll(b, (k, k), (-1, -2))[:, :, 6:-6, 6:-6],
            a[:, :, 6:-6, 6:-6],
            atol=1e-4,
        )


def test_gradients_flow():
    x = make_2d((1, 1, 16, 16), dtype=torch.float64, seed=5).requires_grad_()
    s0, invs = tw.DTCWTScattering(J=2, max_order=2)(x)
    (s0.sum() + sum(v.sum() for _, v in invs)).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtypes(dtype):
    x = make_2d((1, 1, 32, 32), dtype=dtype)
    s0, invs = tw.DTCWTScattering(J=2, max_order=2)(x)
    assert s0.dtype == dtype
    assert all(t.dtype == dtype for _, t in invs)


def test_half_computed_in_float32():
    x = make_2d((1, 1, 32, 32), dtype=torch.float16)
    s0, invs = tw.DTCWTScattering(J=2, max_order=2)(x)
    assert s0.dtype == torch.float32
    assert all(t.dtype == torch.float32 for _, t in invs)


def test_module_matches_functional():
    x = make_2d((1, 1, 32, 32))
    m = tw.DTCWTScattering(J=3, max_order=2)
    s1, i1 = m(x)
    s2, i2 = tw.dtcwt_scattering(x, J=3, max_order=2)
    assert torch.equal(s1, s2)
    assert [p for p, _ in i1] == [p for p, _ in i2]
    for (_, a), (_, b) in zip(i1, i2, strict=True):
        assert torch.equal(a, b)


def test_compile_fullgraph():
    x = make_2d((1, 1, 32, 32))
    m = torch.compile(tw.DTCWTScattering(J=2, max_order=2), backend="eager", fullgraph=True)
    s1, i1 = m(x)
    s2, i2 = tw.DTCWTScattering(J=2, max_order=2)(x)
    assert torch.allclose(s1, s2, atol=1e-6)
    for (_, a), (_, b) in zip(i1, i2, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_invalid_args():
    with pytest.raises(ValueError):
        tw.DTCWTScattering(J=0)
    with pytest.raises(ValueError):
        tw.DTCWTScattering(max_order=0)
    with pytest.raises(ValueError):
        tw.DTCWTScattering(biort="nope")
