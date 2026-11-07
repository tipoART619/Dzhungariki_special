"""Параметры задачи (из постановки)."""

N = 128                      # размер ДОЭ и изображений, пикселей
PIXEL = 10e-6                # шаг пиксела ПВМС, м
WAVELENGTH = 532e-9          # длина волны, м
DISTANCES = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)  # расстояния восстановления, м
NUM_PLANES = len(DISTANCES)


def min_distance_single_fft(n: int = N, pitch: float = PIXEL, wl: float = WAVELENGTH) -> float:
    """Минимальное z, при котором входной чирп exp(i*pi*r^2/(lambda*z)) дискретизирован без наложения.

    z_min = N * dx^2 / lambda. Для параметров задачи z_min = 24.06 мм.
    """
    return n * pitch ** 2 / wl


def output_pitch(z: float, n: int = N, pitch: float = PIXEL, wl: float = WAVELENGTH) -> float:
    """Шаг дискретизации в плоскости восстановления для Френеля с одним БПФ: lambda*z/(N*dx)."""
    return wl * z / (n * pitch)
