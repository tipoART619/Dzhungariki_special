"""Сравнение методов на тестовом наборе: нейросеть vs GS vs Adam (качество и время).

Пример:
    python scripts/make_dataset.py --split test --count 50 --seed 2
    python scripts/evaluate.py --test-dir data/test --ckpt runs/main/best.pt --out results/benchmark
Без --ckpt сравниваются только классические методы.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from doe.baselines import gerchberg_saxton, gradient_phase_opt
from doe.config import DISTANCES
from doe.io import list_images, load_gray
from doe.metrics import evaluate, summarize
from doe.optics_np import amplitude_to_uint8, phase_to_uint8, reconstruct, uint8_to_phase


def score(phase, imgs):
    amps = reconstruct(uint8_to_phase(phase_to_uint8(phase)), DISTANCES)
    return summarize(evaluate([amplitude_to_uint8(x) for x in amps], imgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "test"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark"))
    ap.add_argument("--gs-iters", type=int, default=200)
    ap.add_argument("--adam-iters", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    sets = sorted(p for p in Path(a.test_dir).iterdir() if p.is_dir())[: a.limit]
    methods = {
        "GS": lambda im: gerchberg_saxton(im, DISTANCES, iters=a.gs_iters),
        "Adam": lambda im: gradient_phase_opt(im, DISTANCES, iters=a.adam_iters),
    }
    if a.ckpt:
        import torch
        from scripts.synthesize import load_model  # noqa
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(a.ckpt, dev)

        def nn_method(im):
            with torch.no_grad():
                t = torch.from_numpy(im.astype(np.float32) / 255.0)[None].to(dev)
                ph = model(t)
                if dev == "cuda":
                    torch.cuda.synchronize()
            return ph[0].cpu().numpy().astype(np.float64)

        nn_method(np.zeros((4, 128, 128), np.uint8))  # прогрев
        methods = {"NN": nn_method, **methods}

    rows = []
    for s in sets:
        imgs = np.stack([load_gray(p) for p in list_images(s)[:4]])
        for name, f in methods.items():
            t0 = time.perf_counter()
            ph = f(imgs)
            dt = time.perf_counter() - t0
            rows.append({"set": s.name, "method": name, "time_s": dt, **score(ph, imgs)})
        print(s.name, {r["method"]: round(r["PCC_mean"], 3) for r in rows[-len(methods):]})
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_set.csv", index=False)
    summary = df.groupby("method")[["PCC_mean", "SSIM_mean", "time_s"]].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
