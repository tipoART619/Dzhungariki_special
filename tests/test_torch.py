"""Проверка эквивалентности PyTorch и NumPy реализаций (пропускается без torch)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from skimage.metrics import structural_similarity

from doe.losses import doe_loss, pcc, ssim_map_mean
from doe.model import DOENet
from doe.optics_np import reconstruct
from doe.optics_torch import FresnelPropagator


def test_propagator_matches_numpy():
    rng = np.random.default_rng(0)
    ph = rng.uniform(-np.pi, np.pi, (128, 128))
    a_np = reconstruct(ph)
    a_t = FresnelPropagator()(torch.tensor(ph, dtype=torch.float32)[None]).abs()[0].numpy()
    assert np.allclose(a_np, a_t, atol=1e-4)


def test_backward_is_inverse():
    prop = FresnelPropagator()
    ph = torch.rand(2, 128, 128) * 6.28 - 3.14
    u = prop(ph)
    v = prop.backward_field(u)
    assert torch.allclose(v, torch.polar(torch.ones_like(ph), ph)[:, None].expand_as(v), atol=1e-4)


def test_ssim_matches_skimage():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 255, (2, 64, 64))
    s = ssim_map_mean(torch.tensor(x)[None], torch.tensor(y)[None]).item()
    assert np.isclose(s, structural_similarity(x, y, data_range=255), atol=1e-6)


def test_model_forward_backward():
    model = DOENet(FresnelPropagator(), base=8, depth=3)
    t = torch.rand(2, 4, 128, 128)
    ph = model(t)
    assert ph.shape == (2, 128, 128) and ph.abs().max() <= np.pi + 1e-5
    loss, st = doe_loss(model.prop(ph).abs(), t)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.obj_phase.grad is not None and torch.isfinite(model.obj_phase.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
