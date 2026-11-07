"""Обновление «знаний» модели на новых данных одной командой.

    python scripts/update_model.py --data data/extra/letters_v2 --steps 5000 --notes "добавлены рукописные буквы"

Что происходит:
  1. берётся текущая модель (weights/doe_nn.pt или --from model-vN);
  2. дообучение: новые изображения подмешиваются к процедурному генератору (--mix, по умолчанию 50 %),
     чтобы модель не «забыла» старое; каждые 5 валидаций — чекпойнт и проверка на data/test;
  3. лучший чекпойнт проверяется на старом тесте (деградация) и, если указан, на новом (--new-test);
  4. если качество не упало — регистрируется новая версия (weights/registry.json, CHANGELOG.md),
     с --publish — сразу GitHub Release.
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doe import registry  # noqa: E402


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print(">>", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="папка с новыми изображениями (ищутся рекурсивно)")
    ap.add_argument("--from", dest="base", default=None, help="версия-основа (по умолчанию latest / weights/doe_nn.pt)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4, help="для дообучения меньше, чем с нуля")
    ap.add_argument("--mix", type=float, default=0.5, help="доля новых изображений в обучении")
    ap.add_argument("--new-test", default=None, help="тестовые наборы из новых данных (папка с подпапками по 4 картинки)")
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4)
    ap.add_argument("--notes", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()

    data = Path(a.data)
    if not data.exists():
        sys.exit(f"нет папки {data}")
    reg = registry.load()
    base_entry = registry.get(reg, a.base)
    if a.base and base_entry is None:
        sys.exit(f"версии {a.base} нет в реестре")
    base = ROOT / "weights" / (f"{base_entry['version']}.pt" if base_entry else registry.ASSET)
    if not base.exists():
        if run("scripts/download_weights.py", *(["--version", a.base] if a.base else []), "--out", base):
            sys.exit("не удалось получить базовую модель")

    out = ROOT / "runs" / f"update_{dt.datetime.now():%Y%m%d_%H%M%S}"
    rc = run("scripts/train.py", "--init", base, "--extra-dir", data, "--extra-prob", a.mix,
             "--steps", a.steps, "--batch", a.batch, "--lr", a.lr, "--workers", a.workers, "--out", out)
    if rc:
        sys.exit("обучение завершилось с ошибкой")

    rel = ["scripts/release_model.py", "--ckpt", out / "best.pt", "--data", str(data),
           "--parent", base_entry["version"] if base_entry else "", "--notes", a.notes]
    if a.new_test:
        rel += ["--extra-test", a.new_test]
    if a.force:
        rel.append("--force")
    if a.publish:
        rel.append("--publish")
    sys.exit(run(*rel))


if __name__ == "__main__":
    main()
