"""Shared data accessors for the torchwavelets examples.

Downloads (once) and loads:

* ``peppers`` — a real 512x512 photograph (USC-SIPI).
* CIFAR-10 — 60k 32x32 colour images, 10 classes (Krizhevsky 2009).
* MIT-BIH Arrhythmia record 100 — a real 360 Hz ECG (PhysioNet).

All data is cached under ``~/data/torchwavelets``.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request

import numpy as np

DATA_DIR = os.path.expanduser("~/data/torchwavelets")

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR_BATCHES = {
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
    "test_batch.bin",
}
MITDB_BASE = "https://physionet.org/files/mitdb/1.0.0/100"
PEPPERS_URL = "https://sipi.usc.edu/database/misc/4.2.07.tiff"
CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"[data] downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


def load_photo(size: int = 512) -> np.ndarray:
    """Centred ``size x size`` grayscale crop of the peppers photo, float 0-1."""
    from PIL import Image

    path = _fetch(PEPPERS_URL, os.path.join(DATA_DIR, "peppers.tiff"))
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    h, w = arr.shape
    r, c = (h - size) // 2, (w - size) // 2
    return arr[r : r + size, c : c + size]


def _unpickle(fh):
    """Load one CIFAR-10 batch (raw 3073-byte records: 1 label byte + 3072
    RGB bytes per sample, the format shipped in the binary tar)."""
    raw = np.frombuffer(fh.read(), dtype=np.uint8).reshape(-1, 3073)
    labels = raw[:, 0].astype(np.int64)
    pixels = raw[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC uint8
    return pixels, labels


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_train, y_train, x_test, y_test)`` as uint8 NHWC arrays."""
    tarpath = _fetch(CIFAR_URL, os.path.join(DATA_DIR, "cifar-10-binary.tar.gz"))
    with tarfile.open(tarpath, "r:gz") as tf:
        names = {n.split("/")[-1] for n in tf.getnames()}
        missing = CIFAR_BATCHES - names
        if missing:
            raise FileNotFoundError(f"cifar tar missing batches: {missing}")
        x_tr, y_tr, x_te, y_te = [], [], [], []
        for name in sorted(CIFAR_BATCHES):
            member = next(m for m in tf.getmembers() if m.name.endswith(name))
            x, y = _unpickle(tf.extractfile(member))
            if name == "test_batch.bin":
                x_te, y_te = x, y
            else:
                x_tr.append(x)
                y_tr.append(y)
    return np.concatenate(x_tr), np.concatenate(y_tr), x_te, y_te


def load_mitdb_ecg(channel: int = 0, nsamples: int = 8192) -> np.ndarray:
    """MIT-BIH record 100, channel ``channel``, first ``nsamples`` (mV, 360 Hz)."""
    dat = _fetch(f"{MITDB_BASE}.dat", os.path.join(DATA_DIR, "mitdb-100.dat"))
    _fetch(f"{MITDB_BASE}.hea", os.path.join(DATA_DIR, "mitdb-100.hea"))
    with open(os.path.join(DATA_DIR, "mitdb-100.hea")) as fh:
        fields = fh.readline().split()
        chan_spec = fh.readline().split()
    n_chan, _fs = int(fields[1]), int(fields[2])
    gain, baseline = float(chan_spec[2]), int(chan_spec[4])

    with open(dat, "rb") as fh:
        raw = np.frombuffer(fh.read(), dtype=np.uint8).reshape(-1, 3)
    s0 = raw[:, 0].astype(np.int32) | ((raw[:, 1] & 0x0F).astype(np.int32) << 8)
    s1 = (raw[:, 1] >> 4).astype(np.int32) | (raw[:, 2].astype(np.int32) << 4)
    for s in (s0, s1):
        s[s >= 2048] -= 4096
    samples = np.empty(2 * len(s0), dtype=np.int32)
    samples[0::2] = s0
    samples[1::2] = s1
    sig = samples[channel::n_chan][:nsamples]
    return (sig.astype(np.float64) - baseline) / gain  # mV
