#!/bin/sh
# Команды контейнера. Папки: /data/input (4 изображения), /data/output (результаты)
set -e
cmd="${1:-synth}"; [ $# -gt 0 ] && shift
IN=/data/input; OUT=/data/output
WEIGHTS=/app/weights/doe_nn.pt

need_input() {
  n=$(find "$IN" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.tif*' \) 2>/dev/null | wc -l)
  if [ "$n" -ne 4 ]; then
    echo "В папке input должно быть ровно 4 изображения (найдено: $n)."
    echo "Порядок по имени файла = порядок плоскостей 24.1 / 25.3 / 26.5 / 27.7 мм (например 1.png..4.png)."
    exit 1
  fi
}

case "$cmd" in
  synth)    # нейросеть; доп. аргументы, напр. --refine 100
    need_input
    python scripts/synthesize.py --input-dir "$IN" --ckpt "$WEIGHTS" --out "$OUT/nn" "$@" ;;
  classic)  # gs или adam
    need_input
    m="${1:-adam}"; [ $# -gt 0 ] && shift
    python scripts/synthesize_classic.py --method "$m" --input-dir "$IN" --out "$OUT/$m" "$@" ;;
  demo)     # на встроенных примерах, input не нужен
    python scripts/synthesize.py --input-dir examples/input --ckpt "$WEIGHTS" --out "$OUT/demo" "$@" ;;
  test)
    python -m pytest tests -q ;;
  shell)
    exec /bin/bash ;;
  *)
    exec "$cmd" "$@" ;;
esac
echo "Готово. Результаты в папке output."
