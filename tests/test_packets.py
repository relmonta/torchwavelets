"""Wavelet packets: PyWavelets parity, tree API, reconstruction, dtypes."""

import numpy as np
import pytest
import torch
from conftest import make_1d, make_2d, rel_err

import torchwavelets as tw


@pytest.mark.parametrize("wave", ["db2", "haar", "sym3", "bior2.4"])
def test_packet1d_full_tree_parity(wave):
    import pywt

    x = make_1d((1, 1, 32), dtype=torch.float64, seed=1)[0, 0]
    wp_py = pywt.WaveletPacket(x.numpy(), wave, mode="symmetric", maxlevel=3)
    for _ in range(3):
        wp_py.decompose()
    wp_me = tw.WaveletPacket(x, wave, mode="symmetric", maxlevel=3)
    for _ in range(3):
        wp_me.decompose()
    for n_py in wp_py.get_level(3):
        n_me = wp_me.get_path(n_py.path)
        assert np.allclose(n_py.data, n_me.data.numpy()[0, 0], atol=1e-10)
    r_py = wp_py.reconstruct(update=False)
    r_me = wp_me.reconstruct(update=False)[0, 0].numpy()
    assert np.allclose(r_py, r_me, atol=1e-10)
    assert np.allclose(r_me, x.numpy(), atol=1e-10)


@pytest.mark.parametrize("wave", ["db2", "haar"])
def test_packet2d_full_tree_parity(wave):
    import pywt

    x = make_2d((1, 1, 32, 32), dtype=torch.float64, seed=2)[0, 0]
    wp_py = pywt.WaveletPacket2D(x.numpy(), wave, mode="symmetric", maxlevel=2)
    for _ in range(2):
        wp_py.decompose()
    wp_me = tw.WaveletPacket2D(x, wave, mode="symmetric", maxlevel=2)
    for _ in range(2):
        wp_me.decompose()
    for n_py in wp_py.get_level(2):
        n_me = wp_me.get_path(n_py.path)
        assert np.allclose(n_py.data, n_me.data.numpy()[0, 0], atol=1e-10)
    r_py = wp_py.reconstruct(update=False)
    r_me = wp_me.reconstruct(update=False)[0, 0].numpy()
    assert np.allclose(r_py, r_me, atol=1e-10)
    assert np.allclose(r_me, x.numpy(), atol=1e-10)


def test_freq_order():
    import pywt

    x = make_2d((1, 1, 32, 32), dtype=torch.float64, seed=3)[0, 0]
    wp_py = pywt.WaveletPacket2D(x.numpy(), "db2", mode="symmetric", maxlevel=2)
    wp_py.decompose()
    wp_py.decompose()
    wp_me = tw.WaveletPacket2D(x, "db2", mode="symmetric", maxlevel=2)
    wp_me.decompose()
    wp_me.decompose()
    for order in ("natural", "freq"):
        a = wp_py.get_level(2, order=order)
        b = wp_me.get_level(2, order=order)
        paths_a = (
            [[n.path for n in g] for g in a] if isinstance(a[0], list) else [n.path for n in a]
        )
        paths_b = (
            [[n.path for n in g] for g in b] if isinstance(b[0], list) else [n.path for n in b]
        )
        assert paths_a == paths_b
    # 1D gray-code order
    wp_py1 = pywt.WaveletPacket(x.numpy()[0], "db2", mode="symmetric", maxlevel=3)
    wp_me1 = tw.WaveletPacket(x[0], "db2", mode="symmetric", maxlevel=3)
    for _ in range(3):
        wp_py1.decompose()
        wp_me1.decompose()
    a = [n.path for n in wp_py1.get_level(3, order="freq")]
    b = [n.path for n in wp_me1.get_level(3, order="freq")]
    assert a == b


def test_getitem_and_setitem():
    x = make_1d((1, 1, 32))
    wp = tw.WaveletPacket(x, "db2", mode="symmetric", maxlevel=3)
    wp.decompose()
    node = wp["ad"]
    assert node.path == "ad"
    # setting data on a (possibly new) node
    wp["ad"] = torch.zeros_like(node.data)
    assert wp["ad"].data.abs().sum().item() == 0.0
    # setting by path of a node that must be created
    wp["ada"] = torch.zeros_like(wp["ada"].data)
    assert wp["ada"].data.abs().sum().item() == 0.0


def test_reconstruct_update():
    x = make_1d((1, 1, 32), dtype=torch.float64)
    wp = tw.WaveletPacket(x, "db2", mode="symmetric", maxlevel=2)
    wp.decompose()
    wp.decompose()
    r = wp.reconstruct(update=True)
    assert rel_err(x, r) < 1e-10
    assert rel_err(x, wp.data) < 1e-10


def test_leaf_nodes_and_decompose_limit():
    x = make_1d((1, 1, 32))
    wp = tw.WaveletPacket(x, "db2", mode="symmetric", maxlevel=2)
    wp.decompose()
    # decompose() only decomposes the node itself (like pywt); the tree is
    # filled by path access / get_level
    for path in ("aa", "ad", "da", "dd"):
        wp.get_path(path)
    leaves = wp.get_leaf_nodes()
    assert sorted(n.path for n in leaves) == ["aa", "ad", "da", "dd"]
    with pytest.raises(ValueError):
        wp["aa"].decompose()  # already at maxlevel
    with pytest.raises(IndexError):
        wp["aaa"]


def test_partial_tree_reconstruction():
    """A partially decomposed tree must still reconstruct exactly."""
    x = make_1d((1, 1, 32), dtype=torch.float64)
    wp = tw.WaveletPacket(x, "db2", mode="symmetric", maxlevel=3)
    wp.decompose()
    wp["a"].decompose()  # only the 'a' branch further decomposed
    r = wp.reconstruct(update=False)
    assert rel_err(x, r) < 1e-10


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtypes(dtype):
    x = make_1d((1, 1, 32), dtype=dtype)
    wp = tw.WaveletPacket(x, "db2", mode="symmetric", maxlevel=2)
    wp.decompose()
    wp.decompose()
    for n in wp.get_level(2):
        assert n.data.dtype == dtype
    assert wp.reconstruct(update=False).dtype == dtype


def test_gradients_flow():
    x = make_1d((1, 1, 32), dtype=torch.float64, seed=5).requires_grad_()
    wp = tw.WaveletPacket(x, "db2", mode="symmetric", maxlevel=2)
    wp.decompose()
    wp.decompose()
    loss = sum(n.data.sum() for n in wp.get_level(2))
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_custom_bank():
    x = make_1d((1, 1, 32), dtype=torch.float64)
    wp = tw.WaveletPacket(x, ("db2"), mode="symmetric", maxlevel=2)
    wp.decompose()
    wp.decompose()
    r = wp.reconstruct(update=False)
    assert rel_err(x, r) < 1e-10


def test_batched_data():
    x = make_1d((2, 3, 32), dtype=torch.float64)
    wp = tw.WaveletPacket(x, "db2", mode="symmetric", maxlevel=2)
    wp.decompose()
    wp.decompose()
    r = wp.reconstruct(update=False)
    assert r.shape == x.shape
    assert rel_err(x, r) < 1e-10


def test_maxlevel_default():
    x = make_1d((1, 1, 32))
    wp = tw.WaveletPacket(x, "db2", mode="symmetric")
    # pywt.dwt_max_level(32, db2) = 3
    assert wp.maxlevel == 3
