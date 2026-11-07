"""Метрики качества: коэффициент корреляции Пирсона и SSIM."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM для 8-битных изображений (scikit-image, параметры по умолчанию, data_range=255)."""
    return float(structural_similarity(np.asarray(a, dtype=np.float64),
                                       np.asarray(b, dtype=np.float64), data_range=255.0))


def evaluate(recons_u8, targets_u8, distances=None) -> list[dict]:
    """Список словарей с метриками для каждой пары (восстановленное, исходное)."""
    rows = []
    for i, (r, t) in enumerate(zip(recons_u8, targets_u8)):
        row = {"plane": i + 1, "PCC": pcc(r, t), "SSIM": ssim(r, t)}
        if distances is not None:
            row["z_mm"] = round(distances[i] * 1e3, 3)
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    return {"PCC_mean": float(np.mean([r["PCC"] for r in rows])),
            "SSIM_mean": float(np.mean([r["SSIM"] for r in rows]))}
