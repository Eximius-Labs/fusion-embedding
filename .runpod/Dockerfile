# Runpod Serverless worker for fusion-embedding-2 text embeddings.
#
# Keep the IMAGE as small as possible: the Hub validator pod pulls at only ~8 MB/s, and startup
# time scales with image size (a baked ~8 GB image took ~14 min to start; this ~3 GB image ~7 min).
# So don't bake the weights, download them on the first request (hf_transfer for speed), covered by
# tests.json's timeout. Slim base; torch's cu124 wheels bring the CUDA runtime.
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
