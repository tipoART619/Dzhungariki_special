"""Генерирует фиксированные наборы для валидации/теста: data/<split>/<0000>/{1..4}.png

Пример:
    python scripts/make_dataset.py --split val --count 64 --seed 1
    python scripts/make_dataset.py --split test --count 100 --seed 2
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from doe.dataset import ImageSource
from doe.io import save_gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--extra-dir", default=None, help="папка со своими изображениями (необязательно)")
    ap.add_argument("--out", default=str(ROOT / "data"))
    a = ap.parse_args()
    src = ImageSource(a.extra_dir)
    rng = np.random.default_rng(a.seed)
    for i in range(a.count):
        imgs = src.sample_set(rng)
        for j, im in enumerate(imgs):
            save_gray(im, Path(a.out) / a.split / f"{i:04d}" / f"{j + 1}.png")
    print(f"{a.count} наборов сохранено в {Path(a.out) / a.split}")


if __name__ == "__main__":
    main()
