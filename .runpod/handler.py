"""Runpod Serverless handler: text, image, video, and audio embeddings from fusion-embedding-2.

All modalities share one vector space. By default a light embedder loads the frozen
vision-language base only (text / image / video — no 7B audio tower), so it still fits a
small GPU. Set FE2_ENABLE_AUDIO=1 to load the full FusionEmbedder (adds the audio tower);
that endpoint additionally serves audio. The model loads lazily on the first request
(warmed in the background at boot). Response follows the OpenAI embeddings shape.

Request (text):   {"input": {"text": "a dog", "dim": 512}}    (list of strings also works)
Request (image):  {"input": {"image": "<https url | data-uri | base64>", "dim": 512}}
Request (video):  {"input": {"video": "<https url | data-uri | base64>", "dim": 512}}
Request (audio):  {"input": {"audio": "<https url | data-uri | base64>", "dim": 512}}   (needs FE2_ENABLE_AUDIO=1)
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
                if os.environ.get("FE2_ENABLE_AUDIO"):
                    # Full embedder: loads the 7B audio tower and serves all four modalities.
                    from fusion_embedding import FusionEmbedder
                    _model = FusionEmbedder.from_pretrained(
                        MODEL_REPO, device="cuda", revision=MODEL_REVISION
                    )
                else:
                    # Light embedder: text / image / video only, no audio tower (small GPU).
                    from fusion_embedding import FusionTextEmbedder
                    _model = FusionTextEmbedder.from_pretrained(
                        MODEL_REPO, device="cuda", revision=MODEL_REVISION
                    )
    return _model


def _load_bytes(spec):
    """spec: an http(s) URL, a data: URI, or raw base64 -> bytes."""
    import base64

    if isinstance(spec, (bytes, bytearray)):
        return bytes(spec)
    if spec.startswith(("http://", "https://")):
        import urllib.request
        req = urllib.request.Request(spec, headers={"User-Agent": "fusion-embedding-worker"})
        return urllib.request.urlopen(req, timeout=30).read()
    if spec.startswith("data:"):
        spec = spec.split(",", 1)[1]
    return base64.b64decode(spec)


def handler(event):
    job_input = event.get("input") or {}
    dim = job_input.get("dim")
    image = job_input.get("image")
    video = job_input.get("video")
    audio = job_input.get("audio")
    text = job_input.get("text", job_input.get("input"))   # 'input' kept for back-compat
    try:
        model = _get_model()
    except Exception as e:  # noqa: BLE001
        return {"error": f"model load failed: {e}"}
    try:
        if image is not None:
            import io

            from PIL import Image
            vec = model.embed_image(Image.open(io.BytesIO(_load_bytes(image))), dim=dim).numpy()
            data = [{"object": "embedding", "index": 0, "embedding": vec.tolist()}]
        elif video is not None:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            try:
                tmp.write(_load_bytes(video))
                tmp.close()
                vec = model.embed_video(tmp.name, dim=dim).numpy()
            finally:
                os.unlink(tmp.name)
            data = [{"object": "embedding", "index": 0, "embedding": vec.tolist()}]
        elif audio is not None:
            if not hasattr(model, "embed_audio"):
                return {"error": "audio requires FE2_ENABLE_AUDIO=1 on this endpoint"}
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                tmp.write(_load_bytes(audio))
                tmp.close()
                vec = model.embed_audio(tmp.name, dim=dim).numpy()
            finally:
                os.unlink(tmp.name)
            data = [{"object": "embedding", "index": 0, "embedding": vec.tolist()}]
        elif text is not None:
            texts = text if isinstance(text, list) else [text]
            if hasattr(model, "encode"):
                vecs = model.encode(texts, dim=dim)        # FusionTextEmbedder, np.ndarray [N, D]
                if vecs.ndim == 1:
                    vecs = vecs[None, :]
            else:
                # FusionEmbedder has no .encode: embed each string and stack.
                vecs = np.vstack([model.embed_text(t, dim=dim).numpy() for t in texts])
            data = [{"object": "embedding", "index": i, "embedding": vecs[i].tolist()}
                    for i in range(len(texts))]
        else:
            return {"error": "provide 'text' (string or list), 'image', 'video', or 'audio' "
                             "(url / data-uri / base64)"}
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
    # job arrives, without blocking worker readiness. Weights download on first load, so
    # this hides the download behind boot. _get_model() is lock-guarded, so a request that
    # arrives mid-warm waits for the same load instead of racing a second one.
    threading.Thread(target=_safe_warm, daemon=True).start()
    runpod.serverless.start({"handler": handler})
