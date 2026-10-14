# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-10-14

### Changed

- License changed from BSD-3-Clause to Apache-2.0 (`LICENSE`, package
  metadata, citation and docs updated).

## [0.1.0] - 2026-10-13

First public release.

### Added

- **Discrete wavelet transform (DWT)** 1D and 2D, analysis and synthesis,
  functional and `nn.Module` APIs, with all seven PyWavelets boundary modes
  (`zero`, `symmetric`, `reflect`, `periodization`, `periodic`, `constant`,
  `replicate`) plus PyWavelets' `smooth` (linear extrapolation) — bit-exact
  vs `pywt.dwt` / `pywt.idwt` / `pywt.dwt2` / `pywt.idwt2`.
- **Dual-tree complex wavelet transform (DTCWT)** 1D, 2D and 3D with native
  complex tensors (`near_sym_*` / `qshift_*` filter banks), bit-exact vs the
  reference numpy `dtcwt` package.
- **DTCWT scattering**: `ScatLayer`, `ScatLayerj2` (parity-verified against
  `pytorch_wavelets`) and a general arbitrary-order network
  `DTCWTScattering(J, max_order)` with exactly shift-invariant smoothed
  invariants.
- **Stationary (undecimated) wavelet transform (SWT)** 1D and 2D with exact
  inverses, bit-exact vs `pywt.swt` / `pywt.iswt` / `pywt.swt2` / `pywt.iswt2`.
- **Wavelet packets** 1D and 2D (`WaveletPacket`, `WaveletPacket2D`) with
  PyWavelets-compatible paths, sequency ordering and reconstruction.
- **`pytorch_wavelets` compatibility layer** (`torchwavelets.compat`) with
  bit-exact numerics but fixed precision/AMP/boundary semantics.
- **Precision contract**: output dtype == input dtype (fp16/bf16/fp32/fp64),
  autocast-native DWT/SWT, `torch.compile(fullgraph=True)`-compatible module
  API, DTCWT computed in float32 for half/bf16 (documented exception).
- **Supported torch versions**: `torch>=2.6` — the full test matrix
  (including the `torch.compile(fullgraph=True)` contract and NumPy 2.x
  compatibility) is validated on torch 2.6, 2.8 and current stable. Older
  torch releases are end-of-life and broken with NumPy 2.x.
- **Examples with real data**: transform gallery, DWT-vs-DTCWT image
  denoising, SWT-vs-DWT ECG denoising (MIT-BIH), CIFAR-10 scattering
  classification, and PyTorch Lightning mixed-precision training with a
  precision audit.
- **Documentation site** (MkDocs Material + mkdocstrings): home, examples and
  full API reference; CI for GitHub Pages.
- **Tests**: 2100+ tests including PyWavelets / numpy-dtcwt /
  pytorch_wavelets parity matrices, gradient checks, `torch.compile`
  fullgraph, AMP/autocast, devices (CPU + all CUDA), and a PyTorch Lightning
  precision-contract suite.

### Fixed / verified

- Independently reviewed (strict math review and strict code review); all
  findings addressed: packaging of the DTCWT filter data, autocast safety of
  the DTCWT family, PyWavelets-faithful defaults and `smooth` mode,
  edge-case parity (odd/tiny sizes, zero-size validation), custom
  `pywt.Wavelet` filter banks, and multi-channel scattering parity.
- GPU-validated on NVIDIA Grace-Hopper (GH200, ARM64, CUDA 12.8) and AMD
  MI210 (ROCm 6.4) via SLURM.
