# Runpod Serverless worker for fusion-embedding-2 text embeddings.
#
# Design for the Hub validator: keep the IMAGE small so it pulls fast (a big baked image blew
# the validator's container-startup window at ~8 min). The ~5 GB of weights are downloaded on
# the first request instead, covered by tests.json's 15-min timeout, and hf_transfer makes that
# download fast. Slim base keeps the image lean; torch's cu124 wheels bring the CUDA runtime.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
        runpod numpy "transformers>=4.46" huggingface_hub hf_transfer accelerate \
        safetensors einops pillow soundfile librosa \
        "git+https://github.com/Eximius-Labs/fusion-embedding"

COPY .runpod/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
