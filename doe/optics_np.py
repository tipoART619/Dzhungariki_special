"""Численный расчёт дифракции Френеля (одно преобразование Фурье), NumPy.

Поле в плоскости z:
    U(x, y) = exp(ikz)/(i*lambda*z) * exp(i*pi*(x^2+y^2)/(lambda*z))
              * FFT{ U0(xi, eta) * exp(i*pi*(xi^2+eta^2)/(lambda*z)) }
Внешние множители влияют только на фазу, поэтому для модуля (восстановленного
изображения) вычисляется |FFT{U0 * chirp_z}|. Используется унитарное БПФ
(norm="ortho"), чтобы энергия сохранялась и обратный переход был сопряжённым.
"""
from __future__ import annotations

import numpy as np

from .config import DISTANCES, N, PIXEL, WAVELENGTH, output_pitch


def coords(n: int = N, pitch: float = PIXEL) -> np.ndarray:
    return (np.arange(n) - n // 2) * pitch


def input_chirp(z: float, n: int = N, pitch: float = PIXEL, wl: float = WAVELENGTH) -> np.ndarray:
    x = coords(n, pitch)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    return np.exp(1j * np.pi * (xx ** 2 + yy ** 2) / (wl * z)).astype(np.complex128)


def chirps(distances=DISTANCES, n: int = N, pitch: float = PIXEL, wl: float = WAVELENGTH) -> np.ndarray:
    """Массив (D, n, n) входных чирпов для всех расстояний."""
    return np.stack([input_chirp(z, n, pitch, wl) for z in distances])


def _fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def _ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def fresnel_forward(field: np.ndarray, chirp: np.ndarray) -> np.ndarray:
    """Поле ДОЭ (..., n, n) -> поле в плоскости восстановления (без внешних фазовых множителей)."""
    return _fft2c(field * chirp)


def fresnel_backward(u: np.ndarray, chirp: np.ndarray) -> np.ndarray:
    """Сопряжённый (обратный) переход: плоскость восстановления -> плоскость ДОЭ."""
    return _ifft2c(u) * np.conj(chirp)


def fresnel_full(field: np.ndarray, z: float, pitch: float = PIXEL, wl: float = WAVELENGTH):
    """Полное выражение Френеля с одним БПФ, включая внешние множители и физические координаты.

    Возвращает (U, dx_out). Нужна для документации/проверки; для модуля результат
    совпадает с fresnel_forward с точностью до постоянного множителя.
    """
    n = field.shape[-1]
    k = 2 * np.pi / wl
    dx_out = output_pitch(z, n, pitch, wl)
    x = (np.arange(n) - n // 2) * dx_out
    xx, yy = np.meshgrid(x, x, indexing="xy")
    outer = np.exp(1j * k * z) / (1j * wl * z) * np.exp(1j * np.pi * (xx ** 2 + yy ** 2) / (wl * z))
    u = outer * _fft2c(field * input_chirp(z, n, pitch, wl)) * (n * pitch ** 2)
    return u, dx_out


def reconstruct(phase: np.ndarray, distances=DISTANCES) -> np.ndarray:
    """Фазовый ДОЭ (n, n) -> модули полей (D, n, n) на всех расстояниях."""
    ch = chirps(distances, n=phase.shape[-1])
    field = np.exp(1j * phase)
    return np.abs(fresnel_forward(field[None], ch))


# ----------------------------------------------------------------- кодирование
def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Приведение фазы к диапазону [-pi, pi]."""
    return np.angle(np.exp(1j * phase))


def phase_to_uint8(phase: np.ndarray) -> np.ndarray:
    """-pi -> 0, +pi -> 255 (линейно)."""
    p = wrap_phase(phase)
    return np.clip(np.round((p + np.pi) / (2 * np.pi) * 255.0), 0, 255).astype(np.uint8)


def uint8_to_phase(img: np.ndarray) -> np.ndarray:
    """0 -> -pi, 255 -> +pi."""
    return img.astype(np.float64) / 255.0 * 2 * np.pi - np.pi


def amplitude_to_uint8(amp: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Модуль поля -> 8-битное изображение.

    Нормировка по перцентилю (а не по максимуму), чтобы единичные яркие спеклы
    не затемняли всё изображение. Значения выше перцентиля насыщаются в 255.
    """
    a = np.asarray(amp, dtype=np.float64)
    ref = np.percentile(a, percentile)
    if ref <= 0:
        ref = a.max() if a.max() > 0 else 1.0
    return np.clip(np.round(a / ref * 255.0), 0, 255).astype(np.uint8)
