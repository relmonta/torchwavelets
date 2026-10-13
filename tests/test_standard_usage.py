"""Standard PyTorch-ecosystem usage: serialization, DataLoader, torch.func,
inference mode and a distributed (DDP-style) smoke test."""

import io

import pytest
import torch
from conftest import make_1d, make_2d

import torchwavelets as tw

# --- serialization -----------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: tw.DWT2D(J=2, wave="db4", mode="symmetric"),
        lambda: tw.DTCWT2D(J=2),
        lambda: tw.ScatLayerj2(),
        lambda: tw.DTCWTScattering(J=2, max_order=2),
        lambda: tw.SWT1D("db2", level=2),
    ],
)
def test_state_dict_roundtrip(factory):
    """Modules are plain nn.Modules: state_dict save/load must roundtrip."""
    m = factory()
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    m2 = factory()
    m2.load_state_dict(torch.load(buf, weights_only=False))
    if isinstance(m, tw.SWT1D):
        x = make_1d((1, 1, 32))
        a_ca, b_ca = m(x)[0][0], m2(x)[0][0]
        assert torch.equal(a_ca, b_ca)
        return
    x = make_2d((1, 1, 32, 32))
    a, b = m(x), m2(x)
    assert torch.equal(a[0], b[0])


def test_whole_module_save():
    m = tw.DWT2D(J=2)
    buf = io.BytesIO()
    torch.save(m, buf)
    buf.seek(0)
    m2 = torch.load(buf, weights_only=False)
    x = make_2d((1, 1, 16, 16))
    assert torch.equal(m(x)[0], m2(x)[0])


# --- DataLoader integration ---------------------------------------------------


def test_dataloader_pipeline():
    """The transforms work inside a Dataset/DataLoader pipeline."""
    x = make_2d((16, 1, 16, 16))

    class DS(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return self.data[i]

    dl = torch.utils.data.DataLoader(DS(x), batch_size=4)
    dwt = tw.DWT2D(J=1, wave="db2", mode="periodization")
    for xb in dl:
        yl, yh = dwt(xb)
        assert yl.shape[0] == 4 and yl.shape == (4, 1, 8, 8)
        break


# --- torch.func (jacrev) ------------------------------------------------------


def test_jacrev_through_dwt():
    """Higher-order derivatives flow through the transforms (torch.func)."""
    f = lambda t: tw.DWT2D(J=1)(t)[0].sum()  # noqa: E731
    x = make_2d((1, 1, 8, 8), dtype=torch.float64)
    jac = torch.func.jacrev(f)(x)
    assert jac.shape == x.shape
    assert torch.isfinite(jac).all()


def test_gradgrad_through_scattering():
    x = make_2d((1, 1, 16, 16), dtype=torch.float64).requires_grad_()
    y = tw.DTCWTScattering(J=1)(x)[0].sum()
    (g,) = torch.autograd.grad(y, x, create_graph=True)
    (gg,) = torch.autograd.grad(g.sum(), x)
    assert torch.isfinite(gg).all()


# --- inference mode -----------------------------------------------------------


def test_inference_mode_dwt():
    with torch.inference_mode():
        yl, _ = tw.dwt2d(make_2d((1, 1, 16, 16)), level=1)
    assert torch.isfinite(yl).all()


def test_inference_mode_dtcwt():
    with torch.inference_mode():
        yl, _ = tw.dtcwt2d(make_2d((1, 1, 16, 16)), J=1)
    assert torch.isfinite(yl).all()


def test_inference_mode_swt():
    with torch.inference_mode():
        coeffs = tw.swt1d(make_1d((1, 1, 16)), "db2", level=1)
    assert torch.isfinite(coeffs[0][0]).all()


def test_inference_mode_scattering():
    with torch.inference_mode():
        s0, _ = tw.DTCWTScattering(J=1)(make_2d((1, 1, 16, 16)))
    assert torch.isfinite(s0).all()


# --- distributed (DDP-style) smoke test ---------------------------------------


def _ddp_worker(rank, world, init_file, queue):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    x = make_2d((2, 1, 16, 16)).requires_grad_()
    yl, _ = tw.DWT2D(J=1)(x)
    yl.sum().backward()
    dist.all_reduce(x.grad, op=dist.ReduceOp.SUM)
    queue.put(float(x.grad.sum()))
    dist.destroy_process_group()


@pytest.mark.slow
def test_distributed_smoke():
    """Two gloo processes run the transforms and synchronize gradients."""
    import os
    import tempfile

    import torch.multiprocessing as mp

    with tempfile.NamedTemporaryFile(delete=False) as fh:
        init_file = fh.name
    os.unlink(init_file)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_ddp_worker, args=(r, 2, init_file, q)) for r in range(2)]
    for p in procs:
        p.start()
    grads = [q.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(60)
    assert all(p.exitcode == 0 for p in procs)
    assert abs(grads[0] - grads[1]) < 1e-6
