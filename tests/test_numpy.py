"""Проверки физики, кодирования и градиентов (NumPy). Запуск: python -m pytest tests"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from skimage.metrics import structural_similarity

from doe.baselines import loss_and_grad
from doe.config import DISTANCES, min_distance_single_fft
from doe.optics_np import (chirps, fresnel_backward, fresnel_forward, fresnel_full, phase_to_uint8,
                           uint8_to_phase)
from doe.ssim_np import ssim_grad


def test_sampling_condition():
    assert min(DISTANCES) >= min_distance_single_fft()


def test_roundtrip_and_energy():
    rng = np.random.default_rng(0)
    f = np.exp(1j * rng.uniform(-np.pi, np.pi, (128, 128)))
    ch = chirps()
    u = fresnel_forward(f, ch[0])
    assert np.allclose(fresnel_backward(u, ch[0]), f)
    assert np.isclose((np.abs(u) ** 2).sum(), (np.abs(f) ** 2).sum())


def test_full_fresnel_same_modulus():
    rng = np.random.default_rng(1)
    f = np.exp(1j * rng.uniform(-np.pi, np.pi, (128, 128)))
    u, _ = fresnel_full(f, DISTANCES[0])
    r = np.abs(u) / np.abs(fresnel_forward(f, chirps()[0]))
    assert np.std(r) / np.mean(r) < 1e-10


def test_phase_encoding():
    assert phase_to_uint8(np.array([-np.pi + 1e-9]))[0] == 0
    assert phase_to_uint8(np.array([np.pi]))[0] == 255
    assert np.isclose(uint8_to_phase(np.array([0]))[0], -np.pi)
    assert np.isclose(uint8_to_phase(np.array([255]))[0], np.pi)


def test_ssim_matches_skimage_and_grad():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, 255, (2, 32, 32))
    v, g = ssim_grad(x, y)
    assert np.isclose(v, structural_similarity(x, y, data_range=255))
    e = 1e-4
    x2 = x.copy(); x2[10, 12] += e
    assert np.isclose((ssim_grad(x2, y)[0] - v) / e, g[10, 12], rtol=1e-3)


def test_loss_gradient_fd():
    rng = np.random.default_rng(3)
    t = rng.uniform(0, 1, (4, 128, 128))
    ch = chirps()
    ph = rng.uniform(-np.pi, np.pi, (128, 128))
    L, g = loss_and_grad(ph, t, ch, w_mse=1.0, w_ssim=0.0)
    e = 1e-6
    p2 = ph.copy(); p2[40, 70] += e
    assert np.isclose((loss_and_grad(p2, t, ch, 1.0, 0.0)[0] - L) / e, g[40, 70], rtol=1e-3)
