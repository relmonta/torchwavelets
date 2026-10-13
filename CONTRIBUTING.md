# Contributing to torchwavelets

Thanks for your interest! This is a small, high-quality package — we keep the
bar high but the process light.

## Getting started

```bash
git clone <your-fork> && cd torchwavelets
pip install -e ".[test,examples,docs]"
make test        # python -m pytest tests/ -q
make lint        # ruff check src tests examples
make docs        # mkdocs build --strict (must be warning-free)
```

## What makes a good contribution

* **Parity first.** `torchwavelets` is bit-exact against PyWavelets and the
  reference numpy `dtcwt`. Any change to numerics must keep those parity
  tests green (they are extensive: wavelets x modes x odd/tiny sizes x
  dtypes).
* **The precision contract is sacred.** Output dtype == input dtype,
  autocast-native, `torch.compile(fullgraph=True)`-compatible. If you change
  the DTCWT (float32 for half/bf16, autocast-disabled scope), update the docs.
* **No custom `autograd.Function`** unless absolutely necessary — plain torch
  ops keep AMP, compile and higher-order gradients working.
* **Boundary modes must match PyWavelets exactly**, including `smooth`.

## Pull request checklist

1. `ruff check src tests examples` clean.
2. `mypy src/torchwavelets` clean.
3. `pytest tests/ -q` green.
4. `mkdocs build --strict` warning-free (docstrings are rendered by
   mkdocstrings — keep Google style with typed parameters).
5. New behavior comes with a regression test and a CHANGELOG entry.
6. Keep the public API documented: function docstrings + `docs/api.md`.

## Testing

* CPU: `pytest tests/ -q` (fast, ~20 s).
* GPU: the suite auto-parametrizes over every CUDA device
  (`tests/conftest.py`); run it on a CUDA-capable machine to validate
  (e.g. via SLURM on a hybrid or Grace-Hopper node).
* Lightning tests (`tests/test_lightning.py`) skip when `lightning` is not
  installed; `pip install ".[lightning]"` to run them.

## Code of conduct

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Be kind, assume good
faith, and keep discussions technical.
