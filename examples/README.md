# Examples and applications

Runnable, self-contained demos. All of them download their (real) data on
first use into `~/data/torchwavelets` and render figures into
`docs/figures/`.

| script | what it shows |
|---|---|
| `plot_transform_gallery.py` | every transform of the package applied to a real photograph (DWT 2D, DTCWT 2D, SWT 2D, wavelet packets 2D, DTCWT scattering) |
| `application_denoising.py` | real-world image denoising: DWT vs DTCWT complex soft thresholding on a noisy photo (PSNR) |
| `application_ecg.py` | real 360 Hz ECG (MIT-BIH 100) denoising: SWT (à-trous) vs DWT (RMSE / SNR) |
| `application_cifar10.py` | CIFAR-10 classification with a fixed 588-d DTCWT scattering front-end + logistic regression, vs a 3072-d raw-pixel baseline |
| `application_lightning.py` | mixed-precision training with PyTorch Lightning: wavelet layers follow the Trainer precision (DWT -> bf16, DTCWT -> fp32) |

Run them from the repository root, e.g.:

```bash
python examples/plot_transform_gallery.py
python examples/application_denoising.py
python examples/application_ecg.py
python examples/application_cifar10.py      # ~3-5 min (downloads CIFAR-10)
pip install ".[lightning]"                  # for the Lightning example
python examples/application_lightning.py    # ~1-2 min (bf16-mixed on CPU,
                                            # 16-mixed on CUDA)
```

Reference results (fixed seeds, CPU):

* denoising, peppers 512x512, sigma = 25/255: noisy 20.3 dB -> DWT 26.2 dB,
  DTCWT **29.3 dB** (+3.1 dB)
* ECG, MIT-BIH 100, sigma = 50 uV: noisy 10.6 dB SNR -> DWT 13.0 dB,
  SWT **14.9 dB** (+1.9 dB)
* CIFAR-10: raw pixels (3072-d) 31.6 %, scattering (588-d) **56.5 %** (+24.9
  points, 5x fewer dimensions, untrained front-end)
* Lightning (10k train / 2k test, 2 epochs, bf16-mixed): test accuracy
  **45.4 %**; precision audit: DWT2D lowpass **bfloat16**, DTCWT scattering
  **float32** (documented exception), head logits **bfloat16**
