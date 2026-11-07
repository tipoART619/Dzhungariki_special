"""Функции потерь (PyTorch): 1 - PCC, нормированная MSE с оптимальным масштабом, 1 - SSIM.

SSIM реализован так же, как в scikit-image (structural_similarity по умолчанию):
равномерное окно 7x7, несмещённая ковариация, data_range=255, отражение на
границах и отбрасывание полос шириной 3 пикселя. Перед SSIM амплитуда
нормируется так же, как при сохранении PNG (по 99.5-перцентилю, с насыщением).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

W = 7
CN = W * W / (W * W - 1)
PAD = (W - 1) // 2


def _uniform(x):  # x: (M, 1, n, n)
    x = F.pad(x, (PAD, PAD, PAD, PAD), mode="reflect")
    k = torch.full((1, 1, W, W), 1.0 / (W * W), device=x.device, dtype=x.dtype)
    return F.conv2d(x, k)


def ssim_map_mean(x, y, L: float = 255.0):
    """x, y: (M, n, n) в шкале [0, L]. Возвращает SSIM (M,), совпадающий со skimage."""
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    x = x[:, None]
    y = y[:, None]
    mx, my = _uniform(x), _uniform(y)
    sxx = CN * (_uniform(x * x) - mx * mx)
    syy = CN * (_uniform(y * y) - my * my)
    sxy = CN * (_uniform(x * y) - mx * my)
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s[..., PAD:-PAD, PAD:-PAD].mean((-3, -2, -1))


def pcc(a, t, eps: float = 1e-8):
    ac = a - a.mean((-2, -1), keepdim=True)
    tc = t - t.mean((-2, -1), keepdim=True)
    return (ac * tc).sum((-2, -1)) / torch.sqrt((ac * ac).sum((-2, -1)) * (tc * tc).sum((-2, -1)) + eps)


def to_display(a, pct: float = 0.995):
    """Амплитуда (..., n, n) -> шкала [0, 255] как при сохранении (перцентиль, насыщение)."""
    flat = a.flatten(-2)
    ref = torch.quantile(flat, pct, dim=-1).clamp_min(1e-8).detach()
    return torch.clamp(255.0 * a / ref[..., None, None], max=255.0)


def doe_loss(amp, targets, w_pcc: float = 1.0, w_mse: float = 1.0, w_ssim: float = 2.0):
    """amp, targets: (B, D, n, n); targets в [0, 1]. Возвращает (loss, dict метрик)."""
    p = pcc(amp, targets)
    s = (amp * targets).sum((-2, -1), keepdim=True) / (amp * amp).sum((-2, -1), keepdim=True).clamp_min(1e-8)
    mse = ((s * amp - targets) ** 2).sum((-2, -1)) / (targets * targets).sum((-2, -1)).clamp_min(1e-8)
    loss = w_pcc * (1 - p).sum(1).mean() + w_mse * mse.sum(1).mean()
    stats = {"pcc": p.mean().item(), "mse": mse.mean().item()}
    if w_ssim > 0:
        b, d, n, _ = amp.shape
        x = to_display(amp).reshape(b * d, n, n)
        y = (255.0 * targets).reshape(b * d, n, n)
        ss = ssim_map_mean(x, y).reshape(b, d)
        loss = loss + w_ssim * (1 - ss).sum(1).mean()
        stats["ssim"] = ss.mean().item()
    stats["loss"] = loss.item()
    return loss, stats
