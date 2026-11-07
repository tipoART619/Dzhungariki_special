# Образ для синтеза ДОЭ готовой нейросетью (CPU). Сборка: docker build -t doe-nn .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 MPLBACKEND=Agg PIP_NO_CACHE_DIR=1
WORKDIR /app

# torch для CPU (≈200 МБ вместо ≈2,5 ГБ CUDA-сборки); работает на x86_64 и ARM (Apple Silicon)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# веса: берутся из weights/doe_nn.pt, если он есть рядом с Dockerfile, иначе скачиваются из GitHub Release
RUN python scripts/download_weights.py && sed -i 's/\r$//' docker/entrypoint.sh && chmod +x docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["synth"]
