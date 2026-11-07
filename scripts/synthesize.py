"""Синтез ДОЭ обученной нейросетью для 4 изображений + восстановление + метрики.

Пример:
    python scripts/synthesize.py --input-dir examples/input --ckpt runs/main/best.pt --out results/nn
    python scripts/synthesize.py --inputs a.png b.png c.png d.png --ckpt runs/main/best.pt --refine 100

Порядок изображений = порядок расстояний (24.1, 25.3, 26.5, 27.7 мм); при --input-dir
файлы сортируются по имени.

--refine N : после нейросети N шагов донастройки фазы (Adam) на конкретных
             изображениях — опционально, для максимального качества.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from doe.config import DISTANCES
from doe.io import export_results, load_four, save_preview
from doe.losses import doe_loss
from doe.model import DOENet
from doe.optics_torch import FresnelPropagator


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    cfg = ck["cfg"]
    model = DOENet(FresnelPropagator(cfg.get("distances", DISTANCES)), base=cfg["base"], depth=cfg["depth"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def refine(model, phase, t, steps, lr=0.05):
    p = phase.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(steps):
        loss, _ = doe_loss(model.prop(p).abs(), t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return p.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs=4, default=None)
    ap.add_argument("--input-dir", default=str(ROOT / "examples" / "input"))
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "main" / "best.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "nn"))
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    imgs, names = load_four(a.inputs, a.input_dir)
    model = load_model(a.ckpt, a.device)
    t = torch.from_numpy(imgs.astype(np.float32) / 255.0)[None].to(a.device)

    with torch.no_grad():
        model(t)  # прогрев
        if a.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        phase = model(t)
        if a.device == "cuda":
            torch.cuda.synchronize()
        t_nn = time.perf_counter() - t0
    extra = {"method": "neural network", "time_s": t_nn, "ckpt": str(a.ckpt), "device": a.device}
    if a.refine > 0:
        t1 = time.perf_counter()
        phase = refine(model, phase, t, a.refine)
        extra.update(method=f"neural network + {a.refine} refine steps", time_s=t_nn + time.perf_counter() - t1)

    rep = export_results(phase[0].cpu().numpy().astype(np.float64), imgs, a.out, DISTANCES, extra, names)
    save_preview(a.out, imgs, extra["method"])
    print(f"PCC={rep['PCC_mean']:.4f}  SSIM={rep['SSIM_mean']:.4f}  time={extra['time_s'] * 1e3:.1f} мс  -> {a.out}")


if __name__ == "__main__":
    main()
