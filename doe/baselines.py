"""Классические (итеративные) методы синтеза многоплоскостного ДОЭ — для сравнения с нейросетью.

1. gerchberg_saxton — многоплоскостной алгоритм Гершберга–Сакстона (усреднение
   обратных полей от всех плоскостей).
2. gradient_phase_opt — прямая градиентная оптимизация фазы (Adam) по той же
   функции потерь, что и у нейросети (1 - PCC + нормированная MSE + 1 - SSIM), с
   аналитическим градиентом через сопряжённый оператор Френеля.
"""
from __future__ import annotations

import numpy as np

from .config import DISTANCES
from .optics_np import chirps, fresnel_backward, fresnel_forward
from .ssim_np import ssim_grad


def _targets(targets_u8):
    t = np.asarray(targets_u8, dtype=np.float64) / 255.0
    return t


def gerchberg_saxton(targets_u8, distances=DISTANCES, iters: int = 200, seed: int = 0, init_phase=None):
    t = _targets(targets_u8)
    ch = chirps(distances, n=t.shape[-1])
    rng = np.random.default_rng(seed)
    if init_phase is None:
        # стартовая фаза: суперпозиция обратно распространённых полей со случайной фазой объекта
        u0 = t * np.exp(1j * rng.uniform(-np.pi, np.pi, size=t.shape))
        phase = np.angle(fresnel_backward(u0, ch).sum(0))
    else:
        phase = np.array(init_phase, dtype=np.float64)
    for _ in range(iters):
        u = fresnel_forward(np.exp(1j * phase)[None], ch)
        u = t * np.exp(1j * np.angle(u))
        phase = np.angle(fresnel_backward(u, ch).sum(0))
    return phase


def loss_and_grad(phase, t, ch, w_mse: float = 1.0, w_ssim: float = 0.0, pct: float = 99.5):
    """L = sum_j [ (1 - PCC(A_j, t_j)) + w_mse * min_s ||s A_j - t_j||^2 / ||t_j||^2 ],  A_j = |F_j(exp(i phase))|.

    Возвращает (L, dL/dphase). Градиент по фазе: Im( conj(v_j) * F_j^H(G_j) ), где
    v_j = exp(i phase) * chirp_j,  G_j = dL/dA_j * U_j / |U_j|.
    """
    field = np.exp(1j * phase)
    u = fresnel_forward(field[None], ch)
    a = np.abs(u)
    ax = (-2, -1)
    ac = a - a.mean(ax, keepdims=True)
    tc = t - t.mean(ax, keepdims=True)
    na = np.sqrt((ac ** 2).sum(ax, keepdims=True)) + 1e-12
    nt = np.sqrt((tc ** 2).sum(ax, keepdims=True)) + 1e-12
    p = (ac * tc).sum(ax, keepdims=True) / (na * nt)
    dl_da = -(tc / (na * nt) - p * ac / na ** 2)
    loss = float((1 - p).sum())

    if w_mse > 0:
        s = (a * t).sum(ax, keepdims=True) / ((a * a).sum(ax, keepdims=True) + 1e-12)
        tt = (t * t).sum(ax, keepdims=True) + 1e-12
        r = s * a - t
        loss += float(w_mse * ((r ** 2).sum(ax, keepdims=True) / tt).sum())
        dl_da = dl_da + w_mse * 2 * s * r / tt  # теорема об огибающей: s оптимально

    if w_ssim > 0:
        # SSIM считается так же, как при оценке: амплитуда -> [0, 255] по 99.5-перцентилю
        for j in range(a.shape[0]):
            ref = np.percentile(a[j], pct) + 1e-12
            x = 255.0 * a[j] / ref
            mk = x < 255.0
            val, gx = ssim_grad(np.minimum(x, 255.0), 255.0 * t[j])
            loss += float(w_ssim * (1 - val))
            dl_da[j] = dl_da[j] - w_ssim * gx * mk * 255.0 / ref

    g = dl_da * u / (a + 1e-12)
    v_grad = fresnel_backward(g, ch)  # = F^H(G) * conj(chirp)
    grad = np.imag(np.conj(field)[None] * v_grad).sum(0)
    return loss, grad


def gradient_phase_opt(targets_u8, distances=DISTANCES, iters: int = 300, lr: float = 0.05,
                       w_mse: float = 1.0, w_ssim: float = 2.0, init_phase=None, seed: int = 0, history: list | None = None):
    t = _targets(targets_u8)
    ch = chirps(distances, n=t.shape[-1])
    if init_phase is None:
        rng = np.random.default_rng(seed)
        u0 = t * np.exp(1j * rng.uniform(-np.pi, np.pi, size=t.shape))
        phase = np.angle(fresnel_backward(u0, ch).sum(0))
    else:
        phase = np.array(init_phase, dtype=np.float64)
    m = np.zeros_like(phase)
    v = np.zeros_like(phase)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(1, iters + 1):
        loss, g = loss_and_grad(phase, t, ch, w_mse, w_ssim)
        if history is not None:
            history.append(loss)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        phase = phase - lr * (m / (1 - b1 ** k)) / (np.sqrt(v / (1 - b2 ** k)) + eps)
    return np.angle(np.exp(1j * phase))
