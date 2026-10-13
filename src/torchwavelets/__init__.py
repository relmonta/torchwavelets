"""torchwavelets: differentiable discrete wavelet transforms for PyTorch."""

from . import compat  # noqa: F401  (drop-in pytorch_wavelets API)
from ._version import __version__
from .core.modes import SUPPORTED_MODES
from .dtcwt.transform1d import DTCWT1D, IDTCWT1D, dtcwt1d, idtcwt1d
from .dtcwt.transform2d import DTCWT2D, IDTCWT2D, dtcwt2d, idtcwt2d
from .dtcwt.transform3d import DTCWT3D, IDTCWT3D, dtcwt3d, idtcwt3d
from .dwt.transform1d import DWT1D, IDWT1D, dwt1d, idwt1d
from .dwt.transform2d import DWT2D, IDWT2D, dwt2d, idwt2d
from .packets import WaveletPacket, WaveletPacket2D
from .scatter.lowlevel import ScatLayer, ScatLayerj2
from .scatter.network import DTCWTScattering, dtcwt_scattering
from .swt.transform import (
    ISWT1D,
    ISWT2D,
    SWT1D,
    SWT2D,
    iswt1d,
    iswt2d,
    swt1d,
    swt2d,
)

__all__ = [
    "__version__",
    "SUPPORTED_MODES",
    "dwt1d",
    "idwt1d",
    "dwt2d",
    "idwt2d",
    "dtcwt2d",
    "idtcwt2d",
    "DWT1D",
    "IDWT1D",
    "DWT2D",
    "IDWT2D",
    "DTCWT2D",
    "IDTCWT2D",
    "DTCWT1D",
    "IDTCWT1D",
    "DTCWT3D",
    "IDTCWT3D",
    "dtcwt1d",
    "idtcwt1d",
    "dtcwt3d",
    "idtcwt3d",
    "ScatLayer",
    "ScatLayerj2",
    "DTCWTScattering",
    "dtcwt_scattering",
    "WaveletPacket",
    "WaveletPacket2D",
    "swt1d",
    "iswt1d",
    "swt2d",
    "iswt2d",
    "SWT1D",
    "ISWT1D",
    "SWT2D",
    "ISWT2D",
]
