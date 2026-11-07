#!/usr/bin/env bash
# Полный цикл: данные -> обучение -> синтез -> сравнение. Запуск из корня проекта.
set -e
python scripts/make_examples.py
python scripts/make_dataset.py --split val  --count 64  --seed 1
python scripts/make_dataset.py --split test --count 100 --seed 2
python scripts/train.py --steps ${STEPS:-20000} --batch ${BATCH:-16} --out runs/main
python scripts/synthesize.py --input-dir examples/input --ckpt runs/main/best.pt --out results/nn
python scripts/synthesize.py --input-dir examples/input --ckpt runs/main/best.pt --out results/nn_refine --refine 100
python scripts/synthesize_classic.py --method gs   --out results/gs
python scripts/synthesize_classic.py --method adam --out results/adam
python scripts/evaluate.py --test-dir data/test --ckpt runs/main/best.pt --out results/benchmark
