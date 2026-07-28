# Runpod Serverless worker for fusion-embedding-2 text embeddings.
#
# Transport matters: Runpod's registry pulls at ~120 MB/s with no rate limit, while an
# unauthenticated HuggingFace download at request time is throttled and slow (and stalled the
# Hub test). So bake the weights into the image (fast registry pull) and load them fully offline
# at request time (zero HF calls). Slim base keeps the image lean; torch's cu124 wheels bring
# the CUDA runtime.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
        runpod numpy "transformers>=4.46" huggingface_hub accelerate \
        safetensors einops pillow soundfile librosa \
        "git+https://github.com/Eximius-Labs/fusion-embedding"

# Bake both (public) weight repos so the worker gets them from the fast registry, not HF.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-VL-Embedding-2B'); \
snapshot_download('EximiusLabs/fusion-embedding-2-2b-preview')"

# After baking, load only from the baked cache: no huggingface.co calls, no rate limits.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY .runpod/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
