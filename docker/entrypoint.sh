#!/bin/sh
# Команды контейнера. Папки: /data/input (4 изображения), /data/output (результаты)
set -e
cmd="${1:-synth}"; [ $# -gt 0 ] && shift
IN=/data/input; OUT=/data/output
HOSTW=/data/weights
WEIGHTS=/app/weights/doe_nn.pt

# Модель: 1) папка weights на компьютере, 2) копия в образе, 3) скачивание из GitHub Release
get_weights() {
  if [ -f "$HOSTW/doe_nn.pt" ]; then
    WEIGHTS="$HOSTW/doe_nn.pt"
  elif [ ! -f "$WEIGHTS" ]; then
    mkdir -p "$HOSTW" 2>/dev/null || true
    target="$WEIGHTS"
    [ -w "$HOSTW" ] && target="$HOSTW/doe_nn.pt"
    echo "Модель не найдена, скачиваю с GitHub..."
    if python scripts/download_weights.py --out "$target"; then
      WEIGHTS="$target"
    else
      echo ""
      echo "ОШИБКА: не удалось получить модель."
      echo "Скачайте doe_nn.pt со страницы https://github.com/tipoART619/Dzhungariki_special/releases"
      echo "и положите в папку weights проекта, затем повторите запуск."
      exit 1
    fi
  fi
  echo "Модель: $WEIGHTS"
}

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
    get_weights
    python scripts/synthesize.py --input-dir "$IN" --ckpt "$WEIGHTS" --out "$OUT/nn" "$@" ;;
  classic)  # gs или adam
    need_input
    m="${1:-adam}"; [ $# -gt 0 ] && shift
    python scripts/synthesize_classic.py --method "$m" --input-dir "$IN" --out "$OUT/$m" "$@" ;;
  demo)     # на встроенных примерах, input не нужен
    get_weights
    python scripts/synthesize.py --input-dir examples/input --ckpt "$WEIGHTS" --out "$OUT/demo" "$@" ;;
  test)
    python -m pytest tests -q ;;
  shell)
    exec /bin/bash ;;
  *)
    exec "$cmd" "$@" ;;
esac
echo "Готово. Результаты в папке output."
