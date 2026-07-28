# Runpod Serverless worker for fusion-embedding-2 text embeddings.
# Weights download at runtime into the mounted volume (HF_HOME=/runpod-volume), so the image
# stays small. All text-path deps are installed with the SAME interpreter that runs the
# handler (python3), so imports resolve.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV HF_HOME=/runpod-volume \
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

COPY .runpod/handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
