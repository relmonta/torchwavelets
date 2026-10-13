"""Shared fixtures for the torchwavelets test suite."""

import os

import pytest
import torch

# Tiny CPU convs are pathological on oversubscribed many-core nodes (the
# thread pool setup dominates); pin a small pool for fast, reproducible tests.
torch.set_num_threads(int(os.environ.get("TW_TEST_THREADS", "4")))

WAVELETS = ["haar", "db2", "db4", "db6", "sym4", "coif2", "bior2.4", "rbio2.4"]
MODES = ["zero", "symmetric", "reflect", "periodization", "periodic", "constant", "replicate"]
FLOAT_DTYPES = [torch.float32, torch.float64]

_CPU = torch.device("cpu")


def available_cuda_devices() -> list[torch.device]:
    """All CUDA devices of this machine (multi-GPU, hybrid and Grace nodes)."""
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


def available_devices() -> list[torch.device]:
    """Every device the transforms can run on (CPU plus all CUDA devices)."""
    return [_CPU] + available_cuda_devices()


def _device_id(dev: torch.device | None) -> str:
    if dev is None:
        return "no-cuda"
    return "cpu" if dev.type == "cpu" else f"cuda:{dev.index}"


@pytest.fixture(params=available_devices(), ids=_device_id)
def any_device(request):
    return request.param


@pytest.fixture(params=available_cuda_devices() or [None], ids=_device_id)
def cuda_device(request):
    if request.param is None:
        pytest.skip("CUDA not available")
    return request.param


@pytest.fixture(params=WAVELETS)
def wavelet(request):
    return request.param


@pytest.fixture(params=MODES)
def mode(request):
    return request.param


@pytest.fixture(params=FLOAT_DTYPES)
def float_dtype(request):
    return request.param


def make_1d(shape, dtype=torch.float64, device=None, seed=0):
    dev = device or _CPU
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.randn(shape, dtype=dtype, device=dev, generator=g)


def make_2d(shape, dtype=torch.float64, device=None, seed=0):
    dev = device or _CPU
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.randn(shape, dtype=dtype, device=dev, generator=g)


def rel_err(a, b):
    return (a - b).norm().item() / a.norm().item()
