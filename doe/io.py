"""Чтение/запись 8-битных одноканальных PNG и полный цикл «ДОЭ -> восстановление -> метрики»."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DISTANCES, N
from .metrics import evaluate, summarize
from .optics_np import amplitude_to_uint8, phase_to_uint8, reconstruct, uint8_to_phase

IMG_EXT = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_gray(path, size: int = N) -> np.ndarray:
    """Загружает изображение как 8-бит, 1 канал, size x size."""
    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def save_gray(arr: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L").save(path)


def list_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def load_four(paths=None, folder=None) -> tuple[np.ndarray, list[str]]:
    if paths is None:
        paths = list_images(folder)
    paths = [Path(p) for p in paths]
    if len(paths) != 4:
        raise ValueError(f"Нужно ровно 4 изображения, получено {len(paths)}: {paths}")
    return np.stack([load_gray(p) for p in paths]), [p.name for p in paths]


def export_results(phase: np.ndarray, targets_u8: np.ndarray, out_dir, distances=DISTANCES,
                   extra: dict | None = None, input_names=None) -> dict:
    """Сохраняет ДОЭ и 4 восстановления, считает метрики.

    Восстановление выполняется из *сохранённого 8-битного* ДОЭ (как это будет
    делать жюри), а не из непрерывной фазы.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doe_u8 = phase_to_uint8(phase)
    save_gray(doe_u8, out / "doe.png")

    amps = reconstruct(uint8_to_phase(doe_u8), distances)
    recons = []
    for i, (a, z) in enumerate(zip(amps, distances)):
        r = amplitude_to_uint8(a)
        recons.append(r)
        save_gray(r, out / f"recon_{i + 1}_z{z * 1e3:.1f}mm.png")
    for i, t in enumerate(targets_u8):
        save_gray(t, out / f"input_{i + 1}.png")

    rows = evaluate(recons, targets_u8, distances)
    report = {"per_plane": rows, **summarize(rows)}
    if input_names:
        report["inputs"] = list(input_names)
    if extra:
        report.update(extra)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "metrics.csv", "w", encoding="utf-8") as f:
        f.write("plane,z_mm,PCC,SSIM\n")
        for r in rows:
            f.write(f"{r['plane']},{r['z_mm']},{r['PCC']:.4f},{r['SSIM']:.4f}\n")
        f.write(f"mean,,{report['PCC_mean']:.4f},{report['SSIM_mean']:.4f}\n")
    return report


def save_preview(out_dir, targets_u8, title: str = "") -> None:
    """Сводная картинка: исходные, восстановленные, ДОЭ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    rep = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    recon_files = sorted(out.glob("recon_*.png"))
    fig, ax = plt.subplots(2, 5, figsize=(15, 7.2))
    for i in range(4):
        ax[0, i].imshow(targets_u8[i], cmap="gray", vmin=0, vmax=255)
        ax[0, i].set_title(f"Исходное {i + 1}")
        ax[1, i].imshow(np.asarray(Image.open(recon_files[i])), cmap="gray", vmin=0, vmax=255)
        r = rep["per_plane"][i]
        ax[1, i].set_title(f"z={r['z_mm']} мм\nPCC={r['PCC']:.3f} SSIM={r['SSIM']:.3f}")
    ax[0, 4].imshow(np.asarray(Image.open(out / "doe.png")), cmap="twilight", vmin=0, vmax=255)
    ax[0, 4].set_title("ДОЭ (фаза)")
    ax[1, 4].axis("off")
    ax[1, 4].text(0.0, 0.5, f"Среднее\nPCC  = {rep['PCC_mean']:.3f}\nSSIM = {rep['SSIM_mean']:.3f}"
                  + (f"\nВремя = {rep['time_s']:.3f} с" if "time_s" in rep else ""), fontsize=14)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "preview.png", dpi=110)
    plt.close(fig)
