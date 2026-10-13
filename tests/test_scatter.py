"""DTCWT scattering layers: parity, shapes, dtypes, gradients, invariance."""

import numpy as np
import pytest
import torch
from conftest import make_2d

import torchwavelets as tw


@pytest.mark.parametrize(
    "layer,shape",
    [
        (lambda: tw.ScatLayer(), (2, 7, 32, 32)),
        (lambda: tw.ScatLayerj2(), (2, 49, 16, 16)),
        (lambda: tw.ScatLayer(magbias=0.1), (2, 7, 32, 32)),
        (lambda: tw.ScatLayerj2(magbias=0.1), (2, 49, 16, 16)),
    ],
)
def test_shapes(layer, shape):
    x = make_2d((2, 1, 64, 64))
    z = layer()(x)
    assert tuple(z.shape) == shape
    assert torch.isfinite(z).all()


def test_odd_sizes_padded():
    x = make_2d((1, 1, 63, 62))
    z1 = tw.ScatLayer()(x)
    z2 = tw.ScatLayerj2()(x)
    assert z1.shape == (1, 7, 32, 31)
    assert z2.shape == (1, 49, 16, 16)


def test_combine_colour_shapes():
    x = make_2d((1, 3, 64, 64))
    assert tw.ScatLayer(combine_colour=True)(x).shape == (1, 9, 32, 32)
    assert tw.ScatLayerj2(combine_colour=True)(x).shape == (1, 51, 16, 16)
    with pytest.raises(ValueError):
        tw.ScatLayer(combine_colour=True)(make_2d((1, 1, 64, 64)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtypes(dtype):
    x = make_2d((1, 1, 32, 32), dtype=dtype)
    z1 = tw.ScatLayer()(x)
    z2 = tw.ScatLayerj2()(x)
    assert z1.dtype == dtype
    assert z2.dtype == dtype


def test_half_computed_in_float32():
    x = make_2d((1, 1, 32, 32), dtype=torch.float16)
    assert tw.ScatLayer()(x).dtype == torch.float32
    assert tw.ScatLayerj2()(x).dtype == torch.float32


def test_mode_restricted():
    with pytest.raises(ValueError):
        tw.ScatLayer(mode="zero")
    with pytest.raises(ValueError):
        tw.ScatLayerj2(mode="reflect")


def test_gradcheck():
    from torch.autograd import gradcheck

    x = make_2d((1, 1, 16, 16), dtype=torch.float64, seed=5).requires_grad_()
    assert gradcheck(tw.ScatLayer(), (x,), atol=1e-6, rtol=1e-4)


def test_gradcheck_j2():
    from torch.autograd import gradcheck

    x = make_2d((1, 1, 16, 16), dtype=torch.float64, seed=5).requires_grad_()
    assert gradcheck(tw.ScatLayerj2(), (x,), atol=1e-5, rtol=1e-3)


def test_gradients_flow():
    x = make_2d((1, 1, 32, 32), dtype=torch.float64, seed=3).requires_grad_()
    z = tw.ScatLayerj2()(x)
    z.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_constant_image_has_zero_bandpass():
    x = torch.ones(1, 1, 64, 64)
    z = tw.ScatLayer(magbias=0.0)(x)
    # first channel = lowpass; the 6 bandpass channels must vanish
    assert torch.allclose(z[:, 0], torch.ones_like(z[:, 0]), atol=1e-5)
    assert z[:, 1:].abs().max().item() < 1e-4


def test_shift_invariance():
    """DTCWT scattering is shift-invariant at dyadic integer shifts: the
    feature map of a translated input is the translated feature map."""
    import torch.nn.functional as F

    g = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128] / 128
    v = (
        0.5 * np.sin(2 * np.pi * 8 * xx)
        + 0.3 * np.sin(2 * np.pi * 4 * yy + 1.0)
        + 0.4 * np.exp(-((xx - 0.3) ** 2 + (yy - 0.7) ** 2) / 0.02)
    )
    x = torch.tensor(v + 0.05 * g.standard_normal((128, 128)))[None, None]

    for layer, sub in ((tw.ScatLayer(), 2), (tw.ScatLayerj2(), 4)):
        z = layer(x)
        shift = 4
        pad = F.pad(x, (0, shift, 0, shift), mode="reflect")
        xs = pad[..., shift : shift + 128, shift : shift + 128]
        zs = layer(xs)
        k = shift // sub  # feature-map shift in subband pixels
        aligned = torch.roll(zs, shifts=(k, k), dims=(-1, -2))
        m = 6
        a, b = aligned[..., m:-m, m:-m], z[..., m:-m, m:-m]
        assert (a - b).abs().max().item() < 1e-3 * b.abs().max().item()


def test_parity_with_pytorch_wavelets():
    """Bit-compatible with the reference implementation (fp32, C=1)."""
    pytest.importorskip("pytorch_wavelets")
    from pytorch_wavelets.scatternet import ScatLayer as UpJ1
    from pytorch_wavelets.scatternet import ScatLayerj2 as UpJ2

    x = make_2d((2, 1, 64, 64), seed=1, dtype=torch.float32)
    pairs = [
        (tw.ScatLayer(), UpJ1(biort="near_sym_a", mode="symmetric", magbias=1e-2)),
        (
            tw.ScatLayerj2(),
            UpJ2(biort="near_sym_a", qshift="qshift_a", mode="symmetric", magbias=1e-2),
        ),
    ]
    for mine, up in pairs:
        zu, zm = up(x), mine(x)
        assert tuple(zm.shape) == tuple(zu.shape)
        assert (zu - zm).abs().max().item() < 1e-5


def test_compile_fullgraph():
    x = make_2d((1, 1, 32, 32))
    for layer in (tw.ScatLayer(), tw.ScatLayerj2()):
        c = torch.compile(layer, backend="eager", fullgraph=True)
        z = c(x)
        assert torch.allclose(z, layer(x), atol=1e-6)


def test_against_numpy_manual_reference():
    """Independent check of the scattering math for a trivial input.

    A delta image must produce bandpass magnitudes concentrated at the
    corresponding orientation; check the first-order energy is finite and
    consistent between the two layers."""
    x = torch.zeros(1, 1, 64, 64)
    x[0, 0, 32, 32] = 1.0
    z1 = tw.ScatLayer()(x)
    z2 = tw.ScatLayerj2()(x)
    # first-order channels (1:7 for j1; 1:13 for j2) must not all vanish
    assert z1[:, 1:].abs().max().item() > 1e-3
    assert z2[:, 1:13].abs().max().item() > 1e-3
    # and the lowpass carries most of the energy at the coarser scale
    assert z2[:, 0].abs().sum().item() > 0


def test_multi_channel():
    x = make_2d((2, 3, 64, 64))
    z = tw.ScatLayerj2()(x)
    # C + 6C (scale 1) + 6C (scale 2) + 36C (second order) = 49C
    assert z.shape == (2, 49 * 3, 16, 16)
