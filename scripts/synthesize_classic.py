"""Классический синтез ДОЭ (без нейросети, только NumPy) — для сравнения.

Пример:
    python scripts/synthesize_classic.py --method gs   --input-dir examples/input --out results/gs
    python scripts/synthesize_classic.py --method adam --input-dir examples/input --out results/adam
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doe.baselines import gerchberg_saxton, gradient_phase_opt
from doe.config import DISTANCES
from doe.io import export_results, load_four, save_preview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["gs", "adam"], default="adam")
    ap.add_argument("--inputs", nargs=4, default=None)
    ap.add_argument("--input-dir", default=str(ROOT / "examples" / "input"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--w-ssim", type=float, default=2.0)
    a = ap.parse_args()
    out = a.out or str(ROOT / "results" / a.method)
    imgs, names = load_four(a.inputs, a.input_dir)
    t0 = time.perf_counter()
    if a.method == "gs":
        iters = a.iters or 200
        phase = gerchberg_saxton(imgs, DISTANCES, iters=iters)
        label = f"Gerchberg-Saxton, {iters} iter"
    else:
        iters = a.iters or 300
        phase = gradient_phase_opt(imgs, DISTANCES, iters=iters, w_ssim=a.w_ssim)
        label = f"Adam phase optimization, {iters} iter"
    dt = time.perf_counter() - t0
    rep = export_results(phase, imgs, out, DISTANCES, {"method": label, "time_s": dt}, names)
    save_preview(out, imgs, label)
    print(f"{label}: PCC={rep['PCC_mean']:.4f}  SSIM={rep['SSIM_mean']:.4f}  time={dt:.2f} с  -> {out}")


if __name__ == "__main__":
    main()
