"""Runpod Serverless handler: text embeddings from fusion-embedding-2.

Loads only the text path (the frozen base + connector), not the audio tower, so it fits a
small GPU. Request/response follow the familiar OpenAI embeddings shape.

Request:  {"input": "some text"}  or  {"input": ["text a", "text b"], "dim": 512}
Response: {"object": "list", "model": ..., "dim": D, "data": [{"index": i, "embedding": [...]}]}
"""
import os

import numpy as np
import runpod
from fusion_embedding import FusionTextEmbedder

MODEL_REPO = os.environ.get("MODEL_REPO", "EximiusLabs/fusion-embedding-2-2b-preview")
MODEL_REVISION = os.environ.get("MODEL_REVISION") or None

# Load once at cold start; a bad load should crash the worker with a clear message.
_model = FusionTextEmbedder.from_pretrained(MODEL_REPO, device="cuda", revision=MODEL_REVISION)


def handler(event):
    job_input = event.get("input") or {}
    payload = job_input.get("input")
    dim = job_input.get("dim")
    if payload is None:
        return {"error": "missing 'input' (a string or a list of strings)"}
    texts = payload if isinstance(payload, list) else [payload]
    try:
        vecs = _model.encode(texts, dim=dim)          # np.ndarray, [N, D]
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    data = [{"object": "embedding", "index": i, "embedding": vecs[i].tolist()}
            for i in range(len(texts))]
    return {"object": "list", "model": MODEL_REPO, "dim": int(vecs.shape[-1]), "data": data}


runpod.serverless.start({"handler": handler})
