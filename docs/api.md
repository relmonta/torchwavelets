# API reference

The package exports the full functional and module APIs at the top level
(`import torchwavelets as tw`). The `compat` submodule mirrors the
`pytorch_wavelets` API.

!!! note "Precision contract"

    Every transform preserves the input dtype (fp16/bf16/fp32/fp64). The
    DTCWT family is the documented exception: half and bfloat16 inputs are
    computed in float32 (no complex16 kernels) and returned as
    float32/complex64; it also runs in an autocast-disabled scope.

## Discrete wavelet transform

### 1-D

::: torchwavelets.dwt.transform1d
    options:
      members: [dwt1d, idwt1d, DWT1D, IDWT1D]

### 2-D

::: torchwavelets.dwt.transform2d
    options:
      members: [dwt2d, idwt2d, DWT2D, IDWT2D]

## Dual-tree complex wavelet transform

### 1-D

::: torchwavelets.dtcwt.transform1d
    options:
      members: [dtcwt1d, idtcwt1d, DTCWT1D, IDTCWT1D]

### 2-D

::: torchwavelets.dtcwt.transform2d
    options:
      members: [dtcwt2d, idtcwt2d, DTCWT2D, IDTCWT2D]

### 3-D

::: torchwavelets.dtcwt.transform3d
    options:
      members: [dtcwt3d, idtcwt3d, DTCWT3D, IDTCWT3D]

## DTCWT scattering

::: torchwavelets.scatter.lowlevel
    options:
      members: [ScatLayer, ScatLayerj2]

::: torchwavelets.scatter.network
    options:
      members: [dtcwt_scattering, DTCWTScattering]

## Stationary (undecimated) wavelet transform

::: torchwavelets.swt.transform
    options:
      members: [swt1d, iswt1d, swt2d, iswt2d, SWT1D, ISWT1D, SWT2D, ISWT2D]

## Wavelet packets

::: torchwavelets.packets
    options:
      members: [WaveletPacket, WaveletPacket2D]

## `pytorch_wavelets` compatibility

::: torchwavelets.compat
    options:
      members: [DWTForward, DWTInverse, DWT1DForward, DWT1DInverse,
                DTCWTForward, DTCWTInverse]
      show_submodules: false

## Boundary modes and filter banks

::: torchwavelets.core.modes
    options:
      members: [validate_mode, SUPPORTED_MODES]

::: torchwavelets.core.filters
    options:
      members: [filter_bank, FilterBank]
