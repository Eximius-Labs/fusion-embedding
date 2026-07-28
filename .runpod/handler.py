"""Runpod Serverless handler: text embeddings from fusion-embedding-2.

The model is loaded lazily on the first request, not at import, so the worker becomes ready
immediately and the (one-time) weight download + load happens inside the first request rather
than blocking container startup. Only the text path is loaded (no audio tower), so it fits a
small GPU. Request/response follow the familiar OpenAI embeddings shape.

Request:  {"input": "some text"}  or  {"input": ["text a", "text b"], "dim": 512}
Response: {"object": "list", "model": ..., "dim": D, "data": [{"index": i, "embedding": [...]}]}
"""
import os
import threading

import numpy as np
import runpod

MODEL_REPO = os.environ.get("MODEL_REPO", "EximiusLabs/fusion-embedding-2-2b-preview")
MODEL_REVISION = os.environ.get("MODEL_REVISION") or None

_model = None
_model_lock = threading.Lock()


def _get_model():
    # Double-checked lock: the background warm thread and the first request can both call this.
    # Loading a transformers model onto CUDA from two threads at once can deadlock, so serialize.
    global _model
    if _model is None:
        with _model_lock:
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


def _safe_warm():
    try:
        _get_model()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    # Warm the model in the background so it is ready (or nearly so) by the time the first
    # job arrives, without blocking worker readiness. Weights are baked into the image, so
    # this is a local disk load, not a download. _get_model() is lock-guarded, so a request
    # that arrives mid-warm waits for the same load instead of racing a second one.
    threading.Thread(target=_safe_warm, daemon=True).start()
    runpod.serverless.start({"handler": handler})
