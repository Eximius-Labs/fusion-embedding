# Runpod Serverless worker for fusion-embedding-2 text embeddings.
# Model weights are baked into the image (HF_HOME=/opt/hf) so there is no runtime download;
# the first request just loads from local disk. Install and run with the same python3.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV HF_HOME=/opt/hf \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ffmpeg libgl1 libglib2.0-0 libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir \
        runpod numpy "transformers>=4.46" huggingface_hub accelerate \
        safetensors einops pillow soundfile librosa \
        "git+https://github.com/Eximius-Labs/fusion-embedding"

# Bake the weights (both public) so cold starts don't download at runtime.
RUN python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-VL-Embedding-2B'); \
snapshot_download('EximiusLabs/fusion-embedding-2-2b-preview')"

COPY .runpod/handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
