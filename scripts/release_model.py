"""Регистрация и публикация новой версии модели.

1. Считает метрики чекпойнта на фиксированном тесте data/test (8-битный ДОЭ, как у жюри).
2. Сравнивает с последней опубликованной версией: при падении PCC > 0.01 — отказ (если нет --force).
3. Сохраняет облегчённые веса в weights/doe_nn.pt и weights/<версия>.pt,
   дописывает weights/registry.json и CHANGELOG.md.
4. С --publish создаёт GitHub Release (нужен gh: https://cli.github.com, `gh auth login`).

    python scripts/release_model.py --ckpt runs/full/best.pt --notes "полное обучение 20k шагов" --publish
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from doe import registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--version", default=None, help="по умолчанию следующая model-vN")
    ap.add_argument("--notes", default="", help="что изменилось (попадёт в реестр, CHANGELOG и релиз)")
    ap.add_argument("--data", default="", help="на каких данных обучалась/дообучалась")
    ap.add_argument("--parent", default=None, help="от какой версии дообучена")
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "test"))
    ap.add_argument("--extra-test", default=None, help="доп. тестовый набор (новые данные), только для отчёта")
    ap.add_argument("--force", action="store_true", help="зарегистрировать даже при деградации")
    ap.add_argument("--publish", action="store_true", help="создать GitHub Release через gh")
    a = ap.parse_args()

    reg = registry.load()
    version = a.version or registry.next_version(reg)
    if registry.get(reg, version):
        sys.exit(f"версия {version} уже есть в реестре")

    dev = registry.default_device()
    model = registry.load_model(a.ckpt, dev)
    m = registry.benchmark_model(model, registry.load_sets(a.test_dir), dev)
    print(f"{version}: PCC={m['PCC']:.4f} SSIM={m['SSIM']:.4f} на {m['sets']} тестовых наборах")
    extra = None
    if a.extra_test:
        extra = registry.benchmark_model(model, registry.load_sets(a.extra_test), dev)
        print(f"   доп. тест {a.extra_test}: PCC={extra['PCC']:.4f} SSIM={extra['SSIM']:.4f}")

    prev = registry.get(reg)
    if prev:
        d = m["PCC"] - prev["metrics"]["PCC"]
        print(f"   относительно {prev['version']}: ΔPCC={d:+.4f}")
        if d < -registry.DEGRADATION_TOL and not a.force:
            sys.exit(f"ОТКАЗ: модель хуже {prev['version']} на {-d:.4f} PCC (порог {registry.DEGRADATION_TOL}). "
                     "Используйте --force, если это осознанно.")

    ck = torch.load(a.ckpt, map_location="cpu")
    wdir = ROOT / "weights"
    wdir.mkdir(exist_ok=True)
    vfile = wdir / f"{version}.pt"
    torch.save({"model": ck["model"], "cfg": ck["cfg"], "step": ck.get("step"), "version": version}, vfile)
    shutil.copyfile(vfile, wdir / registry.ASSET)

    entry = {
        "version": version,
        "date": registry.today(),
        "parent": a.parent or (prev["version"] if prev else None),
        "url": registry.release_url(version),
        "sha256": registry.sha256(vfile),
        "size_mb": round(vfile.stat().st_size / 1e6, 1),
        "steps": ck.get("step"),
        "cfg": ck["cfg"],
        "data": a.data,
        "metrics": {**m, "test_dir": Path(a.test_dir).name},
        "extra_metrics": extra,
        "notes": a.notes,
    }
    reg["versions"].append(entry)
    reg["latest"] = version
    registry.save(reg)

    cl = ROOT / "CHANGELOG.md"
    old = cl.read_text(encoding="utf-8") if cl.exists() else "# История версий модели\n"
    head, _, rest = old.partition("\n")
    line = (f"\n## {version} — {entry['date']}\n\n"
            f"- PCC {m['PCC']:.4f}, SSIM {m['SSIM']:.4f} (data/test, {m['sets']} наборов)"
            + (f", ΔPCC {m['PCC'] - prev['metrics']['PCC']:+.4f} к {prev['version']}" if prev else "") + "\n"
            + (f"- основа: {entry['parent']}\n" if entry["parent"] else "")
            + (f"- данные: {a.data}\n" if a.data else "")
            + (f"- {a.notes}\n" if a.notes else ""))
    cl.write_text(head + "\n" + line + rest, encoding="utf-8")
    print(f"реестр обновлён: weights/registry.json, CHANGELOG.md; веса: {vfile}")

    notes = f"PCC {m['PCC']:.4f}, SSIM {m['SSIM']:.4f}. {a.notes}".strip()
    cmd = ["gh", "release", "create", version, f"{vfile}#{registry.ASSET}", "--title", version, "--notes", notes]
    if a.publish:
        # gh загружает файл под именем из пути, поэтому копируем во временную папку с именем doe_nn.pt
        tmp = ROOT / "weights" / "_upload"
        tmp.mkdir(exist_ok=True)
        up = tmp / registry.ASSET
        shutil.copyfile(vfile, up)
        cmd[4] = str(up)
        try:
            subprocess.run(cmd, check=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print("релиз опубликован")
    else:
        print("\nДля публикации:")
        print(f"  gh release create {version} weights/{registry.ASSET} --title {version} --notes \"{notes}\"")
    print("Не забудьте закоммитить реестр:")
    print(f"  git add weights/registry.json CHANGELOG.md && git commit -m \"{version}\" && git push")


if __name__ == "__main__":
    main()
