"""Runpod Serverless handler: text and image embeddings from fusion-embedding-2.

Text and image share one vector space. The base vision-language model handles both, so no
audio tower is loaded and it still fits a small GPU. The model loads lazily on the first
request (warmed in the background at boot). Response follows the OpenAI embeddings shape.

Request (text):   {"input": {"text": "a dog", "dim": 512}}    (list of strings also works)
Request (image):  {"input": {"image": "<https url | data-uri | base64>"}}
Response:         {"object": "list", "model": ..., "dim": D, "data": [{"index": i, "embedding": [...]}]}
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


def _load_image(spec):
    """spec: an http(s) URL, a data: URI, or raw base64 -> PIL.Image."""
    import base64
    import io

    from PIL import Image
    if isinstance(spec, (bytes, bytearray)):
        data = bytes(spec)
    elif spec.startswith(("http://", "https://")):
        import urllib.request
        req = urllib.request.Request(spec, headers={"User-Agent": "fusion-embedding-worker"})
        data = urllib.request.urlopen(req, timeout=30).read()
    else:
        if spec.startswith("data:"):
            spec = spec.split(",", 1)[1]
        data = base64.b64decode(spec)
    return Image.open(io.BytesIO(data))


def handler(event):
    job_input = event.get("input") or {}
    dim = job_input.get("dim")
    image = job_input.get("image")
    text = job_input.get("text", job_input.get("input"))   # 'input' kept for back-compat
    try:
        model = _get_model()
    except Exception as e:  # noqa: BLE001
        return {"error": f"model load failed: {e}"}
    try:
        if image is not None:
            vec = model.embed_image(_load_image(image), dim=dim).numpy()
            data = [{"object": "embedding", "index": 0, "embedding": vec.tolist()}]
        elif text is not None:
            texts = text if isinstance(text, list) else [text]
            vecs = model.encode(texts, dim=dim)            # np.ndarray, [N, D]
            if vecs.ndim == 1:
                vecs = vecs[None, :]
            data = [{"object": "embedding", "index": i, "embedding": vecs[i].tolist()}
                    for i in range(len(texts))]
        else:
            return {"error": "provide 'text' (string or list) or 'image' (url / data-uri / base64)"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"object": "list", "model": MODEL_REPO, "dim": len(data[0]["embedding"]), "data": data}


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
