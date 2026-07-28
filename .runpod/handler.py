"""Runpod Serverless handler: text embeddings from fusion-embedding-2.

The model is loaded lazily on the first request, not at import, so the worker becomes ready
immediately and the (one-time) weight download + load happens inside the first request rather
than blocking container startup. Only the text path is loaded (no audio tower), so it fits a
small GPU. Request/response follow the familiar OpenAI embeddings shape.

Request:  {"input": "some text"}  or  {"input": ["text a", "text b"], "dim": 512}
Response: {"object": "list", "model": ..., "dim": D, "data": [{"index": i, "embedding": [...]}]}
"""
import os

import numpy as np
import runpod

MODEL_REPO = os.environ.get("MODEL_REPO", "EximiusLabs/fusion-embedding-2-2b-preview")
MODEL_REVISION = os.environ.get("MODEL_REVISION") or None

_model = None


def _get_model():
    global _model
    if _model is None:
        from fusion_embedding import FusionTextEmbedder
        _model = FusionTextEmbedder.from_pretrained(
            MODEL_REPO, device="cuda", revision=MODEL_REVISION
        )
    return _model


def handler(event):
    job_input = event.get("input") or {}
    payload = job_input.get("input")
    dim = job_input.get("dim")
    if payload is None:
        return {"error": "missing 'input' (a string or a list of strings)"}
    texts = payload if isinstance(payload, list) else [payload]
    try:
        vecs = _get_model().encode(texts, dim=dim)     # np.ndarray, [N, D]
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    data = [{"object": "embedding", "index": i, "embedding": vecs[i].tolist()}
            for i in range(len(texts))]
    return {"object": "list", "model": MODEL_REPO, "dim": int(vecs.shape[-1]), "data": data}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
