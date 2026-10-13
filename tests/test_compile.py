"""torch.compile compatibility: the module API must be fully traceable
(no graph breaks) with ``fullgraph=True``."""

import pytest
import torch
from conftest import make_1d, make_2d

import torchwavelets as tw


@pytest.mark.parametrize("mode", ["zero", "symmetric", "periodization"])
def test_compile_dwt2d_fullgraph(mode):
    dwt = tw.DWT2D(wave="db4", J=2, mode=mode)
    x = make_2d((2, 2, 32, 32), dtype=torch.float32)
    yl, yh = dwt(x)
    yl2, yh2 = torch.compile(dwt, backend="eager", fullgraph=True)(x)
    assert torch.allclose(yl, yl2, atol=1e-6)
    for a, b in zip(yh, yh2, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_compile_idwt2d_fullgraph():
    idwt = tw.IDWT2D(wave="db4", mode="symmetric")
    yl = make_2d((2, 2, 6, 6), dtype=torch.float32)
    h = make_2d((2, 2, 3, 6, 6), dtype=torch.float32)
    xr = idwt((yl, [h]))
    xr2 = torch.compile(idwt, backend="eager", fullgraph=True)((yl, [h]))
    assert torch.allclose(xr, xr2, atol=1e-6)


def test_compile_dwt1d_fullgraph_backward():
    dwt = tw.DWT1D(wave="db6", J=3, mode="zero")
    x = make_1d((2, 3, 128), dtype=torch.float32).requires_grad_()
    c = torch.compile(dwt, backend="eager", fullgraph=True)
    yl, yh = c(x)
    (yl.sum() + sum(h.sum() for h in yh)).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_compile_fp16_fullgraph():
    x = make_2d((2, 2, 16, 16), dtype=torch.float16)
    c = torch.compile(tw.DWT2D(wave="db2", J=2), backend="eager", fullgraph=True)
    yl, _ = c(x)
    assert yl.dtype == torch.float16


def test_compile_results_match_across_dtypes():
    """A compiled module must still honor the input dtype (no dtype baking)."""
    dwt = tw.DWT2D(wave="db2", J=2)
    c = torch.compile(dwt, backend="eager", fullgraph=True)
    for dtype in [torch.float32, torch.float16, torch.float64]:
        x = make_2d((1, 1, 16, 16), dtype=dtype, seed=3)
        yl, _ = c(x)
        assert yl.dtype == dtype
        assert torch.allclose(yl, dwt(x)[0], atol=1e-3)


def test_compile_dtcwt2d_fullgraph():
    dtcwt = tw.DTCWT2D(J=2)
    x = make_2d((1, 1, 32, 32), dtype=torch.float32)
    yl, yh = dtcwt(x)
    yl2, yh2 = torch.compile(dtcwt, backend="eager", fullgraph=True)(x)
    assert torch.allclose(yl, yl2, atol=1e-6)
    for a, b in zip(yh, yh2, strict=True):
        assert b.dtype.is_complex
        assert torch.allclose(a, b, atol=1e-6)


def test_compile_dtcwt2d_fullgraph_backward():
    dtcwt = tw.DTCWT2D(J=2)
    x = make_2d((1, 1, 32, 32), dtype=torch.float32).requires_grad_()
    c = torch.compile(dtcwt, backend="eager", fullgraph=True)
    yl, yh = c(x)
    (yl.sum() + sum(h.abs().sum() for h in yh)).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# --- the contract: EVERY module compiles with fullgraph=True ----------------

def _f32_1d(shape=(1, 1, 32)):
    return make_1d(shape, dtype=torch.float32)


def _f32_2d(shape=(1, 1, 32, 32)):
    return make_2d(shape, dtype=torch.float32)


def _coeffs1d():
    return tw.dwt1d(_f32_1d(), wave="db2", level=2)


def _coeffs2d():
    return tw.dwt2d(_f32_2d(), wave="db2", level=2)


def _swt_coeffs1d():
    return tw.swt1d(_f32_1d(), "db2", level=2)


def _swt_coeffs2d():
    return tw.swt2d(_f32_2d(), "db2", level=1)


def _dtcwt_coeffs1d():
    return tw.dtcwt1d(_f32_1d(), J=1)


def _dtcwt_coeffs2d():
    return tw.dtcwt2d(_f32_2d(), J=1)


def _dtcwt_coeffs3d():
    return tw.dtcwt3d(torch.randn(1, 1, 8, 8, 8), J=1)


ALL_MODULES = [
    (lambda: tw.DWT1D(J=2, wave="db2"), _f32_1d),
    (lambda: tw.IDWT1D(wave="db2"), _coeffs1d),
    (lambda: tw.DWT2D(J=2, wave="db2"), _f32_2d),
    (lambda: tw.IDWT2D(wave="db2"), _coeffs2d),
    (lambda: tw.DTCWT1D(J=1), _f32_1d),
    (lambda: tw.IDTCWT1D(), _dtcwt_coeffs1d),
    (lambda: tw.DTCWT2D(J=1), _f32_2d),
    (lambda: tw.IDTCWT2D(), _dtcwt_coeffs2d),
    (lambda: tw.DTCWT3D(J=1), lambda: torch.randn(1, 1, 8, 8, 8)),
    (lambda: tw.IDTCWT3D(), _dtcwt_coeffs3d),
    (lambda: tw.ScatLayer(), _f32_2d),
    (lambda: tw.ScatLayerj2(), _f32_2d),
    (lambda: tw.DTCWTScattering(J=2), _f32_2d),
    (lambda: tw.SWT1D("db2", level=2), _f32_1d),
    (lambda: tw.ISWT1D("db2"), _swt_coeffs1d),
    (lambda: tw.SWT2D("db2", level=1), _f32_2d),
    (lambda: tw.ISWT2D("db2"), _swt_coeffs2d),
]

_ALL_IDS = [
    "DWT1D", "IDWT1D", "DWT2D", "IDWT2D",
    "DTCWT1D", "IDTCWT1D", "DTCWT2D", "IDTCWT2D", "DTCWT3D", "IDTCWT3D",
    "ScatLayer", "ScatLayerj2", "DTCWTScattering",
    "SWT1D", "ISWT1D", "SWT2D", "ISWT2D",
]


def _flatten(out):
    """Flatten a nested (tuple/list) output into a list of tensors."""
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        items = []
        for o in out:
            items.extend(_flatten(o))
        return items
    return []  # non-tensor leaves (e.g. scattering paths) are not compared


@pytest.mark.parametrize("make_module,make_input", ALL_MODULES, ids=_ALL_IDS)
def test_compile_every_module_fullgraph(make_module, make_input):
    """torch.jit.script is not supported; torch.compile(fullgraph=True) is
    the contract: every module must compile with zero graph breaks and match
    the eager output bit-for-bit."""
    m = make_module()
    x = make_input()
    eager = m(x)
    compiled = torch.compile(m, backend="eager", fullgraph=True)(x)
    for a, b in zip(_flatten(eager), _flatten(compiled), strict=True):
        assert a.dtype == b.dtype
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
