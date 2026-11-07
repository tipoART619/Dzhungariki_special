#!/usr/bin/env bash
# Linux / macOS. Нужен Docker. Использование:
#   ./run.sh                 нейросеть на 4 картинках из ./input  -> ./output/nn
#   ./run.sh synth --refine 100
#   ./run.sh demo            пример без своих картинок
#   ./run.sh classic gs      Гершберг–Сакстон (или adam)
#   ./run.sh test            тесты
#   ./run.sh build           пересобрать образ
set -e
cd "$(dirname "$0")"
IMAGE=doe-nn
command -v docker >/dev/null || { echo "Docker не найден: https://docs.docker.com/get-docker/"; exit 1; }
if [ "$1" = "build" ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" .
  [ "$1" = "build" ] && exit 0
fi
mkdir -p input output weights
docker run --rm -v "$PWD/input:/data/input:ro" -v "$PWD/output:/data/output" -v "$PWD/weights:/data/weights" "$IMAGE" "$@"
