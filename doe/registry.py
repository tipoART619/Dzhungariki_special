"""Реестр версий модели (weights/registry.json) и официальный бенчмарк нейросети.

Реестр хранится в git; сами веса — в GitHub Releases (тег = версия, файл doe_nn.pt).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "weights" / "registry.json"
REPO = "tipoART619/Dzhungariki_special"
ASSET = "doe_nn.pt"
# допустимое падение PCC относительно опубликованной модели, после которого публикация блокируется
DEGRADATION_TOL = 0.01


def release_url(tag: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{tag}/{ASSET}"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"repo": REPO, "latest": None, "versions": []}


def save(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get(reg: dict, version: str | None = None) -> dict | None:
    version = version or reg.get("latest")
    return next((v for v in reg["versions"] if v["version"] == version), None)


def next_version(reg: dict) -> str:
    nums = [int(v["version"].split("-v")[-1]) for v in reg["versions"] if v["version"].startswith("model-v")]
    return f"model-v{max(nums, default=0) + 1}"


def today() -> str:
    return _dt.date.today().isoformat()


# ---------- официальный бенчмарк: 8-битный ДОЭ -> восстановление -> PCC/SSIM (как у жюри) ----------

def load_sets(test_dir, limit: int | None = None) -> np.ndarray:
    from .io import list_images, load_gray
    sets = sorted(p for p in Path(test_dir).iterdir() if p.is_dir())[:limit]
    return np.stack([np.stack([load_gray(p) for p in list_images(s)[:4]]) for s in sets])


def benchmark_model(model, sets_u8: np.ndarray, device, batch: int = 16) -> dict:
    """PCC/SSIM нейросети на фиксированных наборах; sets_u8: (S, 4, N, N) uint8."""
    import torch
    from .config import DISTANCES
    from .metrics import evaluate, summarize
    from .optics_np import amplitude_to_uint8, phase_to_uint8, reconstruct, uint8_to_phase

    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(sets_u8), batch):
            t = torch.from_numpy(sets_u8[i:i + batch].astype(np.float32) / 255.0).to(device)
            phases = model(t).detach().cpu().numpy().astype(np.float64)
            for ph, imgs in zip(phases, sets_u8[i:i + batch]):
                amps = reconstruct(uint8_to_phase(phase_to_uint8(ph)), DISTANCES)
                rows.append(summarize(evaluate([amplitude_to_uint8(x) for x in amps], imgs)))
    if was_training:
        model.train()
    return {"PCC": float(np.mean([r["PCC_mean"] for r in rows])),
            "SSIM": float(np.mean([r["SSIM_mean"] for r in rows])),
            "sets": len(rows)}


def load_model(ckpt, device):
    import torch
    from .config import DISTANCES
    from .model import DOENet
    from .optics_torch import FresnelPropagator
    ck = torch.load(ckpt, map_location=device)
    cfg = ck["cfg"]
    model = DOENet(FresnelPropagator(cfg.get("distances", DISTANCES)), base=cfg["base"], depth=cfg["depth"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
