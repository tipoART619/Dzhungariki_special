"""Скачивает обученные веса из GitHub Release в weights/doe_nn.pt.

    python scripts/download_weights.py
    python scripts/download_weights.py --url <прямая ссылка на файл>
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# заполнить после создания релиза на GitHub
WEIGHTS_URL = "https://github.com/tipoART619/Dzhungariki_special/releases/download/v1.0/doe_nn.pt"
SHA256 = ""  # если задан — файл проверяется


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=WEIGHTS_URL)
    ap.add_argument("--out", default=str(ROOT / "weights" / "doe_nn.pt"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"уже есть: {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"скачиваю {a.url}")
    tmp = out.with_suffix(".part")
    urllib.request.urlretrieve(a.url, tmp)
    if SHA256 and hashlib.sha256(tmp.read_bytes()).hexdigest() != SHA256:
        tmp.unlink()
        sys.exit("ошибка: контрольная сумма не совпала")
    tmp.replace(out)
    print(f"готово: {out} ({out.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
