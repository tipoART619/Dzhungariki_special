"""Генератор обучающих данных.

Обучение физически-информированное (без учителя): правильные ДОЭ не нужны,
нужны только наборы по 4 изображения. Изображения генерируются процедурно
(буквы/цифры, геометрические фигуры, решётки, пятна, их комбинации), поэтому
датасет не требует скачивания. Дополнительно можно подмешать свои изображения
из папки (например, MNIST/EMNIST/тестовые изображения организаторов).
"""
from __future__ import annotations

import glob
import os
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import N

CHARS = string.ascii_uppercase + string.digits + "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ"

_FONT_DIRS = [
    "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
    "/Library/Fonts", "/System/Library/Fonts", "C:/Windows/Fonts",
]
_FONT_CACHE: list[str] | None = None


def _font_files() -> list[str]:
    global _FONT_CACHE
    if _FONT_CACHE is None:
        prefer = ("dejavu", "liberation", "arial", "verdana", "freesans", "free", "noto", "times", "cour")
        found = []
        for d in _FONT_DIRS:
            if os.path.isdir(d):
                found += glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True)
        good = [f for f in found if any(p in os.path.basename(f).lower() for p in prefer)]
        _FONT_CACHE = (good or found)[:40]
    return _FONT_CACHE


def _font(rng, size):
    files = _font_files()
    for _ in range(5):
        if not files:
            break
        try:
            return ImageFont.truetype(files[rng.integers(len(files))], size)
        except Exception:
            continue
    return ImageFont.load_default(size=size)  # Pillow >= 10.1


def _draw_text(img, rng):
    d = ImageDraw.Draw(img)
    k = int(rng.integers(1, 4))
    txt = "".join(CHARS[rng.integers(len(CHARS))] for _ in range(k))
    size = int(rng.integers(40, 110) / (1 + 0.45 * (k - 1)))
    font = _font(rng, size)
    try:
        bbox = d.textbbox((0, 0), txt, font=font)
    except Exception:
        bbox = (0, 0, size * k, size)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = rng.integers(0, max(1, N - w)) - bbox[0]
    y = rng.integers(0, max(1, N - h)) - bbox[1]
    d.text((int(x), int(y)), txt, fill=int(rng.integers(150, 256)), font=font)
    if rng.random() < 0.3:
        img = img.rotate(float(rng.uniform(-30, 30)), resample=Image.BILINEAR)
    return img


def _draw_shapes(img, rng):
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(1, 5))):
        c = int(rng.integers(100, 256))
        x0, y0 = rng.integers(0, N - 20, size=2)
        w, h = rng.integers(15, 80, size=2)
        box = [int(x0), int(y0), int(min(N - 1, x0 + w)), int(min(N - 1, y0 + h))]
        kind = rng.integers(5)
        width = int(rng.integers(2, 8))
        fill = c if rng.random() < 0.6 else None
        if kind == 0:
            d.ellipse(box, fill=fill, outline=c, width=width)
        elif kind == 1:
            d.rectangle(box, fill=fill, outline=c, width=width)
        elif kind == 2:
            pts = [tuple(int(v) for v in rng.integers(0, N, size=2)) for _ in range(int(rng.integers(3, 7)))]
            d.polygon(pts, fill=fill, outline=c)
        elif kind == 3:
            d.line([tuple(int(v) for v in rng.integers(0, N, size=2)) for _ in range(2)], fill=c, width=width)
        else:
            d.arc(box, float(rng.uniform(0, 360)), float(rng.uniform(0, 360)), fill=c, width=width)
    return img


def _gratings(rng):
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    th = rng.uniform(0, np.pi)
    f = rng.uniform(2, 12) / N
    g = 0.5 + 0.5 * np.cos(2 * np.pi * f * (xx * np.cos(th) + yy * np.sin(th)) + rng.uniform(0, 2 * np.pi))
    if rng.random() < 0.5:
        g = (g > 0.5).astype(np.float64)
    cx, cy = rng.uniform(0.25, 0.75, size=2) * N
    r = rng.uniform(0.2, 0.5) * N
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
    return Image.fromarray((g * mask * rng.uniform(150, 255)).astype(np.uint8))


def _blobs(rng):
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    img = np.zeros((N, N))
    for _ in range(int(rng.integers(1, 6))):
        cx, cy = rng.uniform(10, N - 10, size=2)
        sx, sy = rng.uniform(4, 25, size=2)
        img += rng.uniform(0.3, 1.0) * np.exp(-((xx - cx) ** 2 / (2 * sx ** 2) + (yy - cy) ** 2 / (2 * sy ** 2)))
    img = img / max(img.max(), 1e-8)
    return Image.fromarray((img * 255).astype(np.uint8))


def random_image(rng: np.random.Generator) -> np.ndarray:
    """Одно случайное 8-битное изображение 128x128 на тёмном фоне."""
    p = rng.random()
    if p < 0.40:
        img = _draw_text(Image.new("L", (N, N), 0), rng)
    elif p < 0.70:
        img = _draw_shapes(Image.new("L", (N, N), 0), rng)
    elif p < 0.80:
        img = _gratings(rng)
    elif p < 0.88:
        img = _blobs(rng)
    else:
        img = _draw_shapes(_draw_text(Image.new("L", (N, N), 0), rng), rng)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.0))))
    return np.asarray(img, dtype=np.uint8)


class ImageSource:
    """Источник изображений: процедурный генератор + (опционально) изображения из папки."""

    def __init__(self, extra_dir: str | None = None, extra_prob: float = 0.5):
        self.extra = []
        if extra_dir:
            from .io import IMG_EXT
            self.extra = sorted(p for p in Path(extra_dir).rglob("*") if p.suffix.lower() in IMG_EXT)
        self.extra_prob = extra_prob if self.extra else 0.0

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if self.extra and rng.random() < self.extra_prob:
            from .io import load_gray
            img = load_gray(self.extra[rng.integers(len(self.extra))])
            if rng.random() < 0.5:
                img = np.ascontiguousarray(np.fliplr(img))
            return img
        return random_image(rng)

    def sample_set(self, rng: np.random.Generator, k: int = 4) -> np.ndarray:
        """Набор из k изображений (k, N, N) uint8."""
        return np.stack([self.sample(rng) for _ in range(k)])
