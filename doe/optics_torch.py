"""Дифференцируемый расчёт дифракции Френеля (одно БПФ) на PyTorch.

Полностью эквивалентен doe/optics_np.py (проверяется тестом tests/test_torch.py).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import DISTANCES, N, PIXEL, WAVELENGTH
from .optics_np import chirps as chirps_np


def _fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def _ifft2c(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


class FresnelPropagator(nn.Module):
    """Фаза ДОЭ (B, n, n) -> комплексные поля (B, D, n, n) на D расстояниях."""

    def __init__(self, distances=DISTANCES, n: int = N, pitch: float = PIXEL, wl: float = WAVELENGTH):
        super().__init__()
        ch = chirps_np(distances, n, pitch, wl).astype(np.complex64)
        self.register_buffer("chirps", torch.from_numpy(ch))  # (D, n, n)
        self.num_planes = len(distances)
        self.n = n

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        field = torch.polar(torch.ones_like(phase), phase)  # exp(i*phase)
        return _fft2c(field[:, None] * self.chirps[None])

    def backward_field(self, u: torch.Tensor) -> torch.Tensor:
        """Сопряжённый переход: поля (B, D, n, n) в плоскостях восстановления -> плоскость ДОЭ."""
        return _ifft2c(u) * torch.conj(self.chirps)[None]
