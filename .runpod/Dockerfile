# Runpod Serverless worker for fusion-embedding-2 text embeddings.
# Weights download at runtime into the mounted network volume (HF_HOME=/runpod-volume),
# so the image stays small and models cache across cold starts.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV HF_HOME=/runpod-volume \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip git && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
        runpod numpy "transformers>=4.46" huggingface_hub \
        "git+https://github.com/Eximius-Labs/fusion-embedding"

COPY .runpod/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
