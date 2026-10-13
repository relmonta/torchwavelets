"""Autocast handling for the DTCWT family.

The DTCWT assembles complex bandpasses with ``torch.complex``, which rejects
half and bfloat16. Under ``torch.autocast`` the convolution outputs are cast
to the autocast dtype, so the whole computation runs in an autocast-disabled
scope: inputs are promoted to float32 internally (the documented half/bf16
behavior) and the result keeps that precision.
"""

from __future__ import annotations

import torch


def autocast_off(device: torch.device) -> torch.autocast:
    """Return an autocast-disabled context for ``device``'s type.

    A plain factory rather than a ``contextlib.contextmanager``: Dynamo
    inlines regular function calls and natively supports ``torch.autocast``,
    but generator-based context managers cannot be traced, so they break
    ``torch.compile(fullgraph=True)``.
    """
    device_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.autocast(device_type=device_type, enabled=False)
