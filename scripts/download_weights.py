"""Скачивает веса модели из GitHub Releases по реестру weights/registry.json.

    python scripts/download_weights.py                    # последняя версия -> weights/doe_nn.pt
    python scripts/download_weights.py --version model-v1 # конкретная версия (откат)
    python scripts/download_weights.py --list             # список версий с метриками
"""
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doe import registry  # noqa: E402


def fetch(url, out: Path, sha=None):
    tmp = out.with_suffix(".part")
    print(f"скачиваю {url}")
    urllib.request.urlretrieve(url, tmp)
    if sha and registry.sha256(tmp) != sha:
        tmp.unlink()
        sys.exit("ошибка: контрольная сумма не совпала")
    tmp.replace(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="версия из реестра (по умолчанию latest)")
    ap.add_argument("--url", default=None, help="прямая ссылка на файл весов")
    ap.add_argument("--out", default=str(ROOT / "weights" / registry.ASSET))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="скачать, даже если файл уже есть")
    a = ap.parse_args()

    reg = registry.load()
    if a.list:
        for v in reg["versions"]:
            star = "*" if v["version"] == reg.get("latest") else " "
            print(f"{star} {v['version']:10s} {v['date']}  PCC={v['metrics']['PCC']:.4f} "
                  f"SSIM={v['metrics']['SSIM']:.4f}  {v.get('notes', '')}")
        return

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.url:
        fetch(a.url, out)
        return
    entry = registry.get(reg, a.version)
    if entry is None:
        if out.exists() and not a.version:
            print(f"реестр пуст, используется имеющийся файл {out}")
            return
        sys.exit(f"версия {a.version or 'latest'} не найдена в weights/registry.json")

    local = ROOT / "weights" / f"{entry['version']}.pt"
    if local.exists() and registry.sha256(local) == entry["sha256"]:
        if local.resolve() != out.resolve():
            shutil.copyfile(local, out)
        print(f"{entry['version']}: локальная копия -> {out}")
        return
    if out.exists() and not a.force and registry.sha256(out) == entry["sha256"]:
        print(f"уже актуально: {out} ({entry['version']})")
        return
    fetch(entry["url"], out, entry["sha256"])
    print(f"готово: {out} ({entry['version']}, PCC={entry['metrics']['PCC']:.4f})")


if __name__ == "__main__":
    main()
