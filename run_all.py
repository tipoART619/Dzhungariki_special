"""Кроссплатформенный полный цикл (Windows / Linux / macOS). Запуск из любой папки:

    python run_all.py                    # данные -> обучение -> синтез -> сравнение
    python run_all.py --steps 2000       # короткое обучение
    python run_all.py --pretrained       # без обучения: скачать готовые веса и пересчитать результаты
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print(">>", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4)
    ap.add_argument("--pretrained", action="store_true", help="не обучать, использовать weights/doe_nn.pt")
    a = ap.parse_args()

    run("scripts/make_examples.py")
    run("scripts/make_dataset.py", "--split", "val", "--count", 64, "--seed", 1)
    run("scripts/make_dataset.py", "--split", "test", "--count", 100, "--seed", 2)
    if a.pretrained:
        run("scripts/download_weights.py")
        ckpt = "weights/doe_nn.pt"
    else:
        run("scripts/train.py", "--steps", a.steps, "--batch", a.batch, "--workers", a.workers, "--out", "runs/main")
        ckpt = "runs/main/best.pt"
    run("scripts/synthesize.py", "--input-dir", "examples/input", "--ckpt", ckpt, "--out", "results/nn")
    run("scripts/synthesize.py", "--input-dir", "examples/input", "--ckpt", ckpt, "--out", "results/nn_refine", "--refine", 100)
    run("scripts/synthesize_classic.py", "--method", "gs", "--out", "results/gs")
    run("scripts/synthesize_classic.py", "--method", "adam", "--out", "results/adam")
    run("scripts/evaluate.py", "--test-dir", "data/test", "--ckpt", ckpt, "--out", "results/benchmark")


if __name__ == "__main__":
    main()
