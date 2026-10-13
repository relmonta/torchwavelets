"""PyTorch Lightning integration: wavelet layers follow the Trainer precision.

A simple model contains a DWT layer, a DTCWT layer and a linear head. For
each ``Trainer(precision=...)`` mode the test asserts the dtype the layers
actually produced inside the trainer's autocast context:

* ``32-true`` / ``64-true``: no autocast -> float32 / float64 end to end.
* ``bf16-mixed`` (CPU): the DWT follows the autocast dtype (bfloat16); the
  DTCWT family computes in float32 (documented: no complex bf16 kernels) and
  returns complex64; the head runs in the autocast dtype.
* ``16-mixed`` (CUDA only): same contract with float16.

Skipped when ``lightning`` is not installed.
"""

import pytest

torch = pytest.importorskip("torch")
lightning = pytest.importorskip("lightning")

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import torchwavelets as tw  # noqa: E402


class WaveletPrecisionNet(lightning.LightningModule):
    """Minimal model: DWT2D + DTCWT2D layers and a linear head."""

    def __init__(self):
        super().__init__()
        self.dwt = tw.DWT2D(J=2, wave="db2", mode="symmetric")
        self.dtcwt = tw.DTCWT2D(J=1)
        self.head = nn.Linear(2, 3)
        self.dtypes: dict[str, torch.dtype] = {}
        self.losses: list[torch.Tensor] = []

    def forward(self, x):
        yl, _ = self.dwt(x)
        yl2, yh = self.dtcwt(x)
        z = torch.cat([yl.mean(dim=(-1, -2)), yl2.mean(dim=(-1, -2))], dim=1)
        logits = self.head(z)
        self.dtypes = {
            "DWT2D lowpass": yl.dtype,
            "DTCWT2D lowpass": yl2.dtype,
            "DTCWT2D bandpass": yh[0].dtype,
            "head logits": logits.dtype,
        }
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.losses.append(loss.detach())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def _data(n=32):
    x = torch.randn(n, 1, 32, 32)
    y = torch.randint(0, 3, (n,))
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=16)


def _train(model, precision, batches=3, accelerator="cpu"):
    trainer = lightning.Trainer(
        max_epochs=1,
        limit_train_batches=batches,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator=accelerator,
    )
    trainer.fit(model, _data())
    return model


@pytest.mark.parametrize(
    "precision,accelerator,expect",
    [
        (
            "32-true",
            "cpu",
            {
                "DWT2D lowpass": torch.float32,
                "DTCWT2D lowpass": torch.float32,
                "DTCWT2D bandpass": torch.complex64,
                "head logits": torch.float32,
            },
        ),
        (
            "64-true",
            "cpu",
            {
                "DWT2D lowpass": torch.float64,
                "DTCWT2D lowpass": torch.float64,
                "DTCWT2D bandpass": torch.complex128,
                "head logits": torch.float64,
            },
        ),
        (
            "bf16-mixed",
            "cpu",
            {
                "DWT2D lowpass": torch.bfloat16,
                "DTCWT2D lowpass": torch.float32,
                "DTCWT2D bandpass": torch.complex64,
                "head logits": torch.bfloat16,
            },
        ),
    ],
)
def test_wavelet_layers_follow_trainer_precision(precision, accelerator, expect):
    """The wavelet layers' dtypes inside the Trainer match its precision mode."""
    model = _train(WaveletPrecisionNet(), precision, accelerator=accelerator)
    for stage, dtype in expect.items():
        assert model.dtypes[stage] == dtype, (
            f"{stage} produced {model.dtypes[stage]} under "
            f"precision={precision!r}, expected {dtype}"
        )
    assert torch.isfinite(torch.stack(model.losses)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="16-mixed needs CUDA")
def test_wavelet_layers_follow_16_mixed():
    model = _train(WaveletPrecisionNet(), "16-mixed", accelerator="cuda")
    assert model.dtypes["DWT2D lowpass"] == torch.float16
    assert model.dtypes["DTCWT2D lowpass"] == torch.float32
    assert model.dtypes["DTCWT2D bandpass"] == torch.complex64
    assert model.dtypes["head logits"] == torch.float16


def test_wavelet_layers_are_differentiable_in_lightning():
    """Gradients flow through both wavelet layers inside a Lightning step."""
    model = WaveletPrecisionNet()
    x, y = next(iter(_data(16)))
    loss = model.training_step((x, y), 0)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
