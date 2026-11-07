"""Облегчённая копия чекпойнта для публикации: только веса модели и конфиг (без оптимизатора).

    python scripts/export_weights.py --ckpt runs/main/best.pt --out weights/doe_nn.pt
"""
import argparse
import hashlib
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/main/best.pt")
    ap.add_argument("--out", default="weights/doe_nn.pt")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": ck["model"], "cfg": ck["cfg"], "step": ck.get("step")}, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out}: {out.stat().st_size / 1e6:.1f} МБ, sha256={sha}")


if __name__ == "__main__":
    main()
