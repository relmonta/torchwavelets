"""Gradient correctness: gradcheck on forward and inverse, 1-D and 2-D,
including a second-order (double-backward) check."""

import pytest
import torch
from conftest import make_1d, make_2d

import torchwavelets as tw


@pytest.mark.parametrize("mode", ["zero", "symmetric", "periodization"])
def test_gradcheck_dwt2d(mode):
    x = make_2d((1, 1, 8, 8), dtype=torch.float64, seed=11).requires_grad_()

    def f(t):
        yl, yh = tw.dwt2d(t, wave="db2", mode=mode, level=1)
        return yl, yh[0]

    assert torch.autograd.gradcheck(f, (x,), atol=1e-8, rtol=1e-5)


@pytest.mark.parametrize("mode", ["zero", "symmetric", "periodization"])
def test_gradcheck_idwt2d(mode):
    yl = make_2d((1, 1, 6, 6), dtype=torch.float64, seed=5).requires_grad_()
    h = make_2d((1, 1, 3, 6, 6), dtype=torch.float64, seed=6).requires_grad_()

    def f(lo, hi):
        return tw.idwt2d((lo, [hi]), wave="db2", mode=mode)

    assert torch.autograd.gradcheck(f, (yl, h), atol=1e-8, rtol=1e-5)


def test_gradcheck_dwt1d():
    x = make_1d((1, 1, 16), dtype=torch.float64, seed=13).requires_grad_()

    def f(t):
        yl, yh = tw.dwt1d(t, wave="db2", mode="symmetric", level=2)
        return yl, yh[0], yh[1]

    assert torch.autograd.gradcheck(f, (x,), atol=1e-8, rtol=1e-5)


def test_gradcheck_idwt1d():
    yl = make_1d((1, 1, 9), dtype=torch.float64, seed=21).requires_grad_()
    h = make_1d((1, 1, 9), dtype=torch.float64, seed=22).requires_grad_()

    def f(lo, hi):
        return tw.idwt1d((lo, [hi]), wave="db2", mode="zero")

    assert torch.autograd.gradcheck(f, (yl, h), atol=1e-8, rtol=1e-5)


def test_double_backward_2d():
    """The transform is built from plain torch ops: second derivatives work."""
    x = make_2d((1, 1, 8, 8), dtype=torch.float64, seed=31).requires_grad_()

    def f(t):
        yl, yh = tw.dwt2d(t, wave="db2", mode="symmetric", level=1)
        return (yl**2).sum() + sum((h**2).sum() for h in yh)

    assert torch.autograd.gradgradcheck(f, (x,), atol=1e-6, rtol=1e-4)


def test_gradients_flow_through_all_levels():
    x = make_2d((1, 1, 16, 16), dtype=torch.float64, seed=41).requires_grad_()
    yl, yh = tw.dwt2d(x, wave="db4", mode="symmetric", level=3)
    loss = yl.mean() + sum(h.mean() for h in yh)
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
