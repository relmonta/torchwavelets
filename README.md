# torchwavelets

[![CI](https://github.com/relmonta/torchwavelets/actions/workflows/ci.yml/badge.svg)](https://github.com/relmonta/torchwavelets/actions/workflows/ci.yml)
[![Docs](https://github.com/relmonta/torchwavelets/actions/workflows/docs.yml/badge.svg)](https://relmonta.github.io/torchwavelets/)
[![PyPI version](https://img.shields.io/pypi/v/torchwavelets.svg)](https://pypi.org/project/torchwavelets/)
[![Python versions](https://img.shields.io/pypi/pyversions/torchwavelets.svg)](https://pypi.org/project/torchwavelets/)
[![License](https://img.shields.io/pypi/l/torchwavelets.svg)](https://github.com/relmonta/torchwavelets/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Differentiable discrete wavelet transforms for PyTorch.

`torchwavelets` provides 1D and 2D discrete wavelet transforms (DWT), their
inverses, the 2D dual-tree complex wavelet transform (DTCWT) with native
complex tensors, DTCWT scattering networks, the stationary wavelet transform
(SWT) and wavelet packets — built to respect PyTorch's precision, device and
autocast semantics.

## Why another wavelet library?

The reference implementation, [`fbcotter/pytorch_wavelets`](https://github.com/fbcotter/pytorch_wavelets),
is unmaintained (last commit 2023) and has well-known problems:

- **AMP / mixed precision is broken** — the custom `autograd.Function`s run
  their backward in float32 against float16 gradients (`expected scalar type
  Half but found Float`, issue #63).
- **Filters are baked in at construction time** via `torch.get_default_dtype()`;
  feeding a float16 tensor to a float32-built transform fails.
- **Boundary handling** does not match PyWavelets exactly and fails for small
  signals; the stationary wavelet inverse is dead code.
- No 1D/3D DTCWT, no wavelet packets, no native complex tensors, stale CI.

`torchwavelets` fixes these at the design level:

## Design principles

1. **Output dtype == input dtype.** fp16/bf16/fp32/fp64 in → the same out.
   Filters are resolved lazily per `(dtype, device)` and cached; nothing is
   mutated during forward, so `torch.compile`, DDP and mixed-precision
   training all behave. (One documented exception: the DTCWT family computes
   half and bfloat16 inputs in float32 — complex16 kernels do not exist — and
   runs in an autocast-disabled scope, returning float32.)
2. **Autocast-native.** The transforms are built from plain `conv2d` /
   `conv_transpose2d` ops (no custom `autograd.Function`), so `torch.autocast`
   and `torch.amp.GradScaler` work without any special-casing. The DTCWT
   family computes in an autocast-disabled scope (its complex bandpasses have
   no bf16 kernels) and returns float32 for half/bf16 inputs.
3. **PyWavelets-exact boundary handling.** All modes (`zero`, `symmetric`,
   `reflect`, `periodization`, `periodic`, `constant`, `replicate`) match
   `pywt.dwt` / `pywt.idwt` bit-for-bit, including odd-length and tiny signals
   (index-based padding, no size limits).
4. **Pure torch ops.** Index-based padding keeps everything differentiable,
   higher-order differentiable, and `torch.compile`-friendly.

## Installation

```bash
pip install -e .[test,examples]
```

## Usage

```python
import torch
import torchwavelets as tw

x = torch.randn(4, 3, 64, 64)          # (N, C, H, W)

# Functional API
yl, yh = tw.dwt2d(x, wave="db4", mode="symmetric", level=3)
xr = tw.idwt2d((yl, yh), wave="db4", mode="symmetric")

# Module API
dwt = tw.DWT2D(wave="db4", J=3, mode="symmetric")
yl, yh = dwt(x)
xr = tw.IDWT2D(wave="db4", mode="symmetric")((yl, yh))

# 1D
y = torch.randn(4, 3, 1024)
yl, yh = tw.dwt1d(y, wave="db6", level=3)

# Mixed precision just works
x16 = x.half()
yl16, _ = dwt(x16)          # -> float16 output
with torch.autocast("cuda", dtype=torch.bfloat16):
    yl_bf, _ = dwt(x.float())  # -> bfloat16 output (on CUDA)
```

## Compatibility with `pytorch_wavelets`

```python
from torchwavelets.compat import DWTForward, DWTInverse, DWT1DForward, DWT1DInverse

xfm = DWTForward(J=3, wave="db3", mode="zero")
```

The compat layer keeps the old API but fixes the old semantics: a float16
input works regardless of the dtype used when the transform was constructed.

## Status

- [x] DWT 1D / 2D, analysis + synthesis, all boundary modes
- [x] float16 / bfloat16 / float32 / float64
- [x] PyWavelets numerical parity tests, gradient checks, AMP tests
- [x] `torch.compile`-compatible module API (`fullgraph=True`)
- [x] Device-portable: CPU, CUDA, multi-GPU, hybrid and Grace (GH200) nodes
- [x] DTCWT 1D / 2D / 3D (native complex tensors, `near_sym_*` / `qshift_*` banks)
- [x] DTCWT scattering layers (`ScatLayer`, `ScatLayerj2`)
- [x] General arbitrary-order scattering network (`DTCWTScattering`)
- [x] Stationary (undecimated) wavelet transform 1D / 2D
- [x] Wavelet packets 1D / 2D (`WaveletPacket`, `WaveletPacket2D`)
- [x] Scattering network (deeper, arbitrary-order)

## Examples and applications

Self-contained, reproducible demos with real data (photograph, CIFAR-10,
MIT-BIH ECG) — run `python examples/<script>.py` from the repository root
(data is cached in `~/data/torchwavelets`, figures land in `docs/figures/`).
See [`examples/README.md`](examples/README.md) for the quick reference and
[`docs/examples.md`](docs/examples.md) for the full illustrated walkthrough
with results and figures.

<p align="center">
  <a href="https://github.com/relmonta/torchwavelets/blob/main/docs/figures/transform_gallery.png">
    <img src="https://raw.githubusercontent.com/relmonta/torchwavelets/main/docs/figures/transform_gallery.png" alt="transform gallery" width="98%">
  </a>
</p>

**Application 1 — image denoising (DWT vs DTCWT).** A photograph corrupted
with Gaussian noise (sigma = 25/255, PSNR 20.3 dB) is restored by per-level
3-sigma soft thresholding of the wavelet coefficients. The redundant, nearly
shift-invariant DTCWT gains **+3.1 dB PSNR** over the critically sampled DWT
(29.3 dB vs 26.2 dB).

<p align="center">
  <a href="https://github.com/relmonta/torchwavelets/blob/main/docs/figures/denoising.png">
    <img src="https://raw.githubusercontent.com/relmonta/torchwavelets/main/docs/figures/denoising.png" alt="image denoising" width="98%">
  </a>
</p>

**Application 2 — ECG denoising (SWT vs DWT).** Real 360 Hz ECG (MIT-BIH
Arrhythmia record 100) plus 50 uV noise (SNR 10.6 dB), restored by the same
per-level soft thresholding. The undecimated SWT gains **+1.9 dB SNR** over
the DWT (14.9 dB vs 13.0 dB).

<p align="center">
  <a href="https://github.com/relmonta/torchwavelets/blob/main/docs/figures/ecg_denoising.png">
    <img src="https://raw.githubusercontent.com/relmonta/torchwavelets/main/docs/figures/ecg_denoising.png" alt="ecg denoising" width="98%">
  </a>
</p>

**Application 3 — CIFAR-10 classification with scattering features.** A fixed
(untrained) DTCWT scattering front-end (`DTCWTScattering(J=2, max_order=2)`)
extracts 588 shift-invariant dimensions per 32x32 image (147 invariants,
2x2-average pooled); a logistic regression reaches **56.5 %** test accuracy,
versus **31.6 %** on 3072 raw pixels — with 5x fewer dimensions and no
learned front-end.

<p align="center">
  <a href="https://github.com/relmonta/torchwavelets/blob/main/docs/figures/cifar10_scattering.png">
    <img src="https://raw.githubusercontent.com/relmonta/torchwavelets/main/docs/figures/cifar10_scattering.png" alt="cifar10 scattering" width="98%">
  </a>
</p>

## `torch.compile`

The module API (`DWT2D`, `IDWT2D`, `DWT1D`, `IDWT1D`) resolves its filters
into buffers at construction time and runs only torch ops in `forward`, so it
is fully traceable:

```python
import torch
import torchwavelets as tw

dwt = torch.compile(tw.DWT2D(wave="db4", J=3, mode="symmetric"), fullgraph=True)
yl, yh = dwt(x)
```

The functional API (`tw.dwt2d(x, wave=...)`) is eager-only for the filter
resolution step; use the module API under `torch.compile`.

## Ecosystem notes

* **Serialization**: all transforms are plain `nn.Module`s — `state_dict`
  save/load and `torch.save`/`torch.load` work as usual.
* **DataLoader / datasets**: the module API takes batched `(N, C, ...)`
  tensors, so it slots directly into `DataLoader` pipelines.
* **`torch.func`**: `jacrev`/`jacfwd` and double-backward work through the
  transforms (all plain torch ops).
* **`torch.distributed` (DDP)**: modules hold no state beyond filter buffers
  and never mutate inputs, so DDP training works (smoke-tested with gloo).
* **`torch.jit.script` is not supported** (the transforms use dynamic loops
  over levels); use `torch.compile` instead, which is fully supported.
* **`torch.vmap`**: the API is already batched (`N, C, ...` required), so
  vmap is unnecessary; the batch dimension plays that role.
* **`torch.inference_mode`**: supported everywhere.

## Device portability

Tests parametrize over every available device (`CPU` plus all CUDA devices),
so the suite automatically covers hybrid CPU/GPU nodes and NVIDIA
Grace-Hopper (ARM64) machines; GPU results are reported in the CHANGELOG.

## Credits and acknowledgements

`torchwavelets` is an independent implementation, but it stands on the
shoulders of the wavelet community. The parts of the code that are ports or
adaptations are credited here; everything else was written from the
mathematics and the public conventions.

* **Nick Kingsbury (Cambridge University)** — the dual-tree complex wavelet
  transform itself. The filter banks (`near_sym_a/b`, `qshift_a/b`, ...,
  shipped in `torchwavelets/dtcwt/data/`), the quarter-sample-shift design,
  the orientation convention and the `symmetric` (half-sample) boundary
  handling originate from his DTCWT toolbox, which is **free for research
  use only**. The 3-D DTCWT follows the same work-grid construction.
* **[`fbcotter/pytorch_wavelets`](https://github.com/fbcotter/pytorch_wavelets)** —
  the unmaintained reference implementation this package replaces. Its
  low-level DTCWT operations (`colfilter`, `rowfilter`, `coldfilt`,
  `rowdfilt`, `colifilt`, `rowifilt`, `q2c`, `c2q`, orientation assembly),
  the two-scale scattering layers (`ScatLayer`, `ScatLayerj2`) and the
  `pytorch_wavelets`-compatible API (`torchwavelets.compat`) are ports of
  that codebase, rewritten on plain torch ops to fix its AMP, dtype and
  boundary problems.
* **[PyWavelets](https://pywavelets.readthedocs.io/)** — the parity oracle
  for the DWT, SWT and wavelet packets. The boundary modes, coefficient
  shapes, `swt` conventions, packet paths/sequency ordering and the
  "drop extra sample" reconstruction rule all mirror PyWavelets exactly;
  the test suite is bit-exact against it.
* **[`dtcwt`](https://github.com/rgommers/dtcwt)** (numpy reference
  implementation) — the DTCWT parity oracle for 1-D/2-D/3-D; the test suite
  is bit-exact against it to ~1e-15.
* **Stéphane Mallat** — the scattering-transform theory. The scattering
  layers and the arbitrary-order `DTCWTScattering` network implement the
  Mallat recipe (wavelet modulus + smoothing) with a DTCWT front-end.
* **PyTorch ecosystem** — `torch.compile`, autocast and `nn.Module`
  conventions; the tests follow the standard PyTorch-ecosystem usage
  patterns (serialization, DataLoader, `torch.func`, DDP).

If you use this package in research, please cite it — see
[`CITATION.cff`](CITATION.cff).

### Primary references

1. N. G. Kingsbury, *The dual-tree complex wavelet transform: a new efficient
   tool for image restoration and enhancement*, EUSIPCO 1998.
2. N. G. Kingsbury, *Complex wavelets for shift invariant analysis and
   filtering of signals*, Applied and Computational Harmonic Analysis, 2001.
3. I. W. Selesnick, R. G. Baraniuk, N. G. Kingsbury, *The dual-tree complex
   wavelet transform*, IEEE Signal Processing Magazine, 2005.
4. S. Mallat, *Group invariant scattering*, Communications on Pure and
   Applied Mathematics, 2012.
5. I. Daubechies, *Ten Lectures on Wavelets*, SIAM, 1992.

## License

Apache-2.0 (see `LICENSE`). See
[Credits and acknowledgements](#credits-and-acknowledgements) for the
upstream sources: the DTCWT filter banks originate from Nick Kingsbury's
MATLAB toolbox, which is free for **research use only** — relevant if you
plan commercial use of the DTCWT parts.
