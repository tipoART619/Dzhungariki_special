"""Создаёт 4 демонстрационных входных изображения в examples/input/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from doe.config import N
from doe.dataset import _font
from doe.io import save_gray


def make():
    rng = np.random.default_rng(0)
    out = ROOT / "examples" / "input"
    imgs = []
    # 1: буква
    im = Image.new("L", (N, N), 0); d = ImageDraw.Draw(im)
    f = _font(np.random.default_rng(3), 96)
    bb = d.textbbox((0, 0), "Ф", font=f)
    d.text(((N - bb[2] - bb[0]) // 2, (N - bb[3] - bb[1]) // 2), "Ф", fill=255, font=f)
    imgs.append(im)
    # 2: цифра
    im = Image.new("L", (N, N), 0); d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), "7", font=f)
    d.text(((N - bb[2] - bb[0]) // 2, (N - bb[3] - bb[1]) // 2), "7", fill=255, font=f)
    imgs.append(im)
    # 3: кольцо и квадрат
    im = Image.new("L", (N, N), 0); d = ImageDraw.Draw(im)
    d.ellipse([24, 24, 104, 104], outline=255, width=10)
    d.rectangle([52, 52, 76, 76], fill=255)
    imgs.append(im)
    # 4: звезда (многоугольник)
    im = Image.new("L", (N, N), 0); d = ImageDraw.Draw(im)
    pts = []
    for k in range(10):
        r = 52 if k % 2 == 0 else 22
        a = np.pi / 2 + k * np.pi / 5
        pts.append((64 + r * np.cos(a), 66 - r * np.sin(a)))
    d.polygon(pts, fill=255)
    imgs.append(im)
    for i, im in enumerate(imgs):
        save_gray(np.asarray(im), out / f"{i + 1}.png")
    print("saved to", out)


if __name__ == "__main__":
    make()
