"""Modal smoke for the out-of-tree SGLang plugin (fusion_embedding.sglang_plugin).

Stage-gated parity checks of the SGLang-served FusionEmbeddingModel against the
reference embedder (fusion_embedding.UnifiedEmbedder) on identical inputs:

* stage check — GPU-free structural gate: plugin entry point, config merge, model
  registry resolution, processor registration.
* stage a — frozen base only (adapters + audio disabled in the plugin): text and
  image embeddings must match the reference at cosine > 0.999.
* stage b — adapters loaded, gates closed: stage-a vectors must be unchanged
  (allclose atol 1e-5; both runs embed one prompt at a time so batching is identical).
* stage c — full model on A100-80GB: audio via the SGLang multimodal path (audio
  tower + resampler + token-gated adapters) vs the reference embed_audio, per clip;
  text/image parity re-checked with the full configuration loaded.
* stage serve — boots the literal launch contract
  (``python -m sglang.launch_server --model-path <repo> --is-embedding`` plus the
  three SGLANG_EXTERNAL_* env hooks) and embeds the text prompts over the
  OpenAI-compatible /v1/embeddings API (bf16).

Inputs are built deterministically INSIDE each container from a shared spec
(seeded numpy), so no media bytes cross the wire; real audio files on the
fusion-data volume are passed with --audio-paths (comma-separated /vol paths).

Run from the repo root (billable GPU job; do not run as part of any automated flow):
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/sglang_plugin_smoke.py --stage check
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/sglang_plugin_smoke.py --stage a
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/sglang_plugin_smoke.py --stage b
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/sglang_plugin_smoke.py --stage c
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/sglang_plugin_smoke.py --stage serve

Stages a/b/c gate at --dtype float32 (both sides fp32: isolates implementation
correctness from bf16 kernel-rounding differences, which the text-side whitening
amplifies); pass --dtype bfloat16 to measure served-precision parity. fp32 runs use
the torch_native attention backend (the fused backends are half-precision only). On
Windows shells prefix MSYS_NO_PATHCONV=1 when passing --audio-paths (Git Bash
mangles /vol/... paths).
"""
import math
import os as _os

import modal

app = modal.App("fusion-sglang-plugin-smoke")

REPO = "EximiusLabs/fusion-embedding-2-2b-preview"
SGLANG_VERSION = "0.5.16"

# Reference image: the environment the released model is exercised in (matches the
# phase-B/C Modal images: torch 2.5.1 cu124 + transformers 5.12.1 — the same
# transformers sglang 0.5.16 pins).
ref_img = (modal.Image.debian_slim(python_version="3.11")
           .apt_install("git", "ffmpeg", "libsndfile1")
           .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "scipy", "pillow",
                        "transformers==5.12.1", "accelerate", "huggingface_hub",
                        "safetensors", "soundfile>=0.12", "librosa>=0.10",
                        extra_index_url="https://download.pytorch.org/whl/cu124")
           .add_local_python_source("fusion_embedding"))

# SGLang image: the official pinned image (cu129 build runs on Modal's cu12 drivers)
# + the plugin pip-installed, so its sglang.srt.plugins entry point auto-loads in
# every SGLang process.
# The plugin ships inside the fusion-embedding package (fusion_embedding.sglang_plugin);
# installed with --no-deps because the sglang image already carries torch/numpy/librosa.
sgl_img = (modal.Image.from_registry(f"lmsysorg/sglang:v{SGLANG_VERSION}-cu129")
           .pip_install("soundfile>=0.12", "librosa>=0.10")
           .add_local_file("pyproject.toml", "/pkg/fe/pyproject.toml", copy=True)
           .add_local_file("README.md", "/pkg/fe/README.md", copy=True)
           .add_local_dir("fusion_embedding", "/pkg/fe/fusion_embedding", copy=True)
           .run_commands("pip install --no-deps /pkg/fe"))

volume = modal.Volume.from_name("fusion-data")
if _os.environ.get("HF_TOKEN"):
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": _os.environ["HF_TOKEN"],
                                        "HUGGING_FACE_HUB_TOKEN": _os.environ["HF_TOKEN"]})
else:
    hf_secret = modal.Secret.from_name("huggingface")

EXTERNAL_ENV = {
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "fusion_embedding.sglang_plugin.models",
    "SGLANG_EXTERNAL_MM_MODEL_ARCH": "FusionEmbeddingModel",
    "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "fusion_embedding.sglang_plugin.processors",
}

CHAT = ("<|im_start|>system\n{instruction}<|im_end|>\n"
        "<|im_start|>user\n{content}<|im_end|>\n"
        "<|im_start|>assistant\n")
TEXT_INSTRUCTION = "Retrieve images or text relevant to the user's query."
DOC_INSTRUCTION = "Represent the user's input."
IMAGE_CONTENT = "<|vision_start|><|image_pad|><|vision_end|>"
AUDIO_PROMPT = "<|vision_pad|><|im_end|>"

TEXTS = [
    "a dog barks in the distance while rain falls on a tin roof",
    "an orchestra tuning up before a performance",
    "close-up photo of a rusty bicycle leaning against a brick wall",
    "how do I reset the firmware on this robot arm?",
]
N_IMAGES = 2
N_AUDIOS = 3
SR = 16000
DEFAULT_AUDIO_PATHS = [
    "/vol/demo_space/audio_wav/104251.wav",
    "/vol/demo_space/audio_wav/105386.wav",
    "/vol/demo_space/audio_wav/107188.wav",
]


# --------------------------------------------------------------------------- #
# Deterministic test inputs (identical construction in every container)
# --------------------------------------------------------------------------- #
def make_image(idx: int):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(1000 + idx)
    h, w = 384, 512
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx / w * 255).astype(np.uint8)
    g = (yy / h * 255).astype(np.uint8)
    b = ((np.sin(xx / 17.0) * np.cos(yy / 23.0) * 0.5 + 0.5) * 255).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    noise = rng.integers(0, 60, size=(h, w, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise - 30, 0, 255).astype(np.uint8)
    arr[80 * (idx + 1):120 * (idx + 1), 60:200] = (255, 40, 40)
    return Image.fromarray(arr, "RGB")


def make_image_png(idx: int) -> bytes:
    import io

    buf = io.BytesIO()
    make_image(idx).save(buf, format="PNG")
    return buf.getvalue()


def make_audio(idx: int):
    import numpy as np

    rng = np.random.default_rng(2000 + idx)
    dur = 8.0
    t = np.arange(int(dur * SR), dtype=np.float64) / SR
    if idx % 3 == 0:      # rising chirp + hum
        wav = 0.4 * np.sin(2 * np.pi * (200 + 150 * t) * t) + 0.1 * np.sin(2 * np.pi * 60 * t)
    elif idx % 3 == 1:    # AM tone bursts
        wav = 0.5 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 2 * t)))
    else:                 # filtered-ish noise swells
        n = rng.standard_normal(t.shape)
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        wav = 0.3 * n * env
    wav = wav + 0.02 * rng.standard_normal(t.shape)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


def make_audio_wav(idx: int) -> bytes:
    """Float32 WAV bytes — a lossless container for the synthetic array, so the
    served side decodes the exact samples the reference embeds directly."""
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, make_audio(idx), SR, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def load_volume_audio(path: str):
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


# --------------------------------------------------------------------------- #
# Reference embedder
# --------------------------------------------------------------------------- #
@app.function(image=ref_img, gpu="A10G", timeout=3600,
              volumes={"/vol": volume}, secrets=[hf_secret])
def ref_embed(payload: dict) -> dict:
    _os.environ["HF_HOME"] = "/vol/hf"
    import torch

    from fusion_embedding.unified import UnifiedEmbedder

    dtype = torch.float32 if payload.get("dtype") == "float32" else torch.bfloat16
    emb = UnifiedEmbedder.from_pretrained(REPO, device="cuda", dtype=dtype)
    out: dict = {"text": [], "image": [], "audio": []}
    with torch.no_grad():
        for t in payload["texts"]:
            out["text"].append(emb.embed_text(t).tolist())
        for i in range(payload["n_images"]):
            out["image"].append(emb.embed_image(make_image(i)).tolist())
        if payload.get("with_audio"):
            for i in range(payload["n_audios"]):
                out["audio"].append(emb.embed_audio(make_audio(i), sr=SR).tolist())
            for p in payload.get("audio_paths", []):
                wav, sr = load_volume_audio(p)
                out["audio"].append(emb.embed_audio(wav, sr=sr).tolist())
    return out


# --------------------------------------------------------------------------- #
# SGLang side
# --------------------------------------------------------------------------- #
def _sgl_embed(payload: dict, with_audio: bool) -> dict:
    import sglang as sgl

    dtype = payload.get("dtype", "float32")
    kwargs = dict(
        model_path=REPO,
        is_embedding=True,
        dtype=dtype,
        mem_fraction_static=0.75,
        # Pin the KV pool size: stage A/B/C differ in loaded weights (adapters,
        # audio tower), and a memory-derived pool size would change the fused
        # attention kernel plans between configurations — a deterministic
        # engine-level bf16 drift that has nothing to do with the gating logic.
        max_total_tokens=20000,
        log_level="info",
    )
    if dtype == "float32":
        # The fused attention backends and sgl-kernel norm/activation ops are
        # half-precision only; fp32 (the structural parity precision) runs on
        # torch_native SDPA + the plugin's native-op parity mode.
        kwargs["attention_backend"] = "torch_native"
        # The default ViT backend (triton_attn on Ampere) is half-precision;
        # sdpa is dtype-generic and is what the HF reference runs.
        kwargs["mm_attention_backend"] = "sdpa"
        _os.environ["FUSION_SGLANG_NATIVE_OPS"] = "1"
    engine = sgl.Engine(**kwargs)

    def one(prompt: str, image: bytes = None, audio=None) -> list:
        enc = {}
        if image is not None:
            enc["image_data"] = [image]
        if audio is not None:
            enc["audio_data"] = [audio]
        ret = engine.encode(prompt, **enc)
        if isinstance(ret, list):
            ret = ret[0]
        return list(ret["embedding"])

    out: dict = {"text": [], "image": [], "audio": []}
    try:
        for t in payload["texts"]:
            out["text"].append(one(CHAT.format(instruction=TEXT_INSTRUCTION, content=t)))
        for i in range(payload["n_images"]):
            out["image"].append(one(
                CHAT.format(instruction=DOC_INSTRUCTION, content=IMAGE_CONTENT),
                image=make_image_png(i),
            ))
        if with_audio:
            for i in range(payload["n_audios"]):
                out["audio"].append(one(AUDIO_PROMPT, audio=make_audio_wav(i)))
            for p in payload.get("audio_paths", []):
                out["audio"].append(one(AUDIO_PROMPT, audio=p))
    finally:
        engine.shutdown()
    return out


@app.function(image=sgl_img, gpu="A10G", timeout=3600,
              volumes={"/vol": volume}, secrets=[hf_secret])
def sgl_stage_a(payload: dict) -> dict:
    _os.environ.update(HF_HOME="/vol/hf", **EXTERNAL_ENV,
                       FUSION_SGLANG_DISABLE_AUDIO="1",
                       FUSION_SGLANG_DISABLE_ADAPTERS="1")
    return _sgl_embed(payload, with_audio=False)


@app.function(image=sgl_img, gpu="A10G", timeout=3600,
              volumes={"/vol": volume}, secrets=[hf_secret])
def sgl_stage_b(payload: dict) -> dict:
    _os.environ.update(HF_HOME="/vol/hf", **EXTERNAL_ENV,
                       FUSION_SGLANG_DISABLE_AUDIO="1")
    return _sgl_embed(payload, with_audio=False)


@app.function(image=sgl_img, gpu="A100-80GB", timeout=3600,
              volumes={"/vol": volume}, secrets=[hf_secret])
def sgl_stage_c(payload: dict) -> dict:
    _os.environ.update(HF_HOME="/vol/hf", **EXTERNAL_ENV)
    return _sgl_embed(payload, with_audio=True)


@app.function(image=sgl_img, gpu="A100-80GB", timeout=3600,
              volumes={"/vol": volume}, secrets=[hf_secret])
def sgl_serve_smoke(payload: dict) -> dict:
    """Boot the literal deliverable command (env hooks +
    ``python -m sglang.launch_server --model-path <repo> --is-embedding``) and embed
    the text prompts through the OpenAI-compatible /v1/embeddings API."""
    import json
    import subprocess
    import sys
    import threading
    import time
    import urllib.request

    env = dict(_os.environ, HF_HOME="/vol/hf", **EXTERNAL_ENV)
    proc = subprocess.Popen(
        [sys.executable, "-m", "sglang.launch_server",
         "--model-path", REPO, "--is-embedding"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    base = "http://127.0.0.1:30000"
    deadline = time.time() + 1800
    log_lines: list[str] = []

    def _drain():
        for line in proc.stdout:  # type: ignore[union-attr]
            log_lines.append(line.rstrip())

    threading.Thread(target=_drain, daemon=True).start()
    while time.time() < deadline:
        if proc.poll() is not None:
            return {"error": "server exited", "log": log_lines[-80:]}
        try:
            with urllib.request.urlopen(base + "/health", timeout=3):
                break
        except Exception:
            time.sleep(3)
    else:
        proc.kill()
        return {"error": "server never became healthy", "log": log_lines[-80:]}

    out: dict = {"text": []}
    try:
        for t in payload["texts"]:
            body = json.dumps({
                "model": REPO,
                "input": CHAT.format(instruction=TEXT_INSTRUCTION, content=t),
            }).encode()
            req = urllib.request.Request(
                base + "/v1/embeddings", data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            out["text"].append(data["data"][0]["embedding"])
    finally:
        proc.terminate()
    return out


@app.function(image=sgl_img, gpu="T4", timeout=1800,
              volumes={"/vol": volume}, secrets=[hf_secret])
def check_registration() -> str:
    """Structural sanity (no model load): plugin entry point loads against sglang
    0.5.16, the shell config merges, the model registry and processor registry
    resolve our arch. Runs on the smallest GPU only because importing the sglang
    model zoo needs the driver libraries present."""
    _os.environ["HF_HOME"] = "/vol/hf"
    _os.environ.update(EXTERNAL_ENV)
    lines = []
    from importlib.metadata import version as _pkg_version
    lines.append(f"sglang {_pkg_version('sglang')}")
    import transformers
    lines.append(f"transformers {transformers.__version__}")

    from importlib.metadata import entry_points
    eps = [e.name for e in entry_points(group="sglang.srt.plugins")]
    lines.append(f"entry points (sglang.srt.plugins): {eps}")
    assert "fusion_embedding" in eps

    from fusion_embedding.sglang_plugin import register
    register()

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(REPO)
    lines.append(f"config class: {type(cfg).__name__} model_type={cfg.model_type}")
    assert cfg.architectures == ["FusionEmbeddingModel"], cfg.architectures
    assert cfg.text_config.hidden_size == 2048
    assert cfg.adapter_rank == 384 and cfg.n_query == 64 and cfg.d_audio == 3584
    lines.append(f"text_config layers={cfg.text_config.num_hidden_layers} "
                 f"matryoshka={cfg.matryoshka_dimensions}")

    import fusion_embedding.sglang_plugin.models.fusion_embedding as m
    lines.append(f"model module import OK: EntryClass={m.EntryClass.__name__}")

    from sglang.srt.models.registry import ModelRegistry
    model_cls, arch = ModelRegistry.resolve_model_cls(["FusionEmbeddingModel"])
    lines.append(f"registry resolves: {arch} -> {model_cls.__module__}.{model_cls.__name__}")
    assert model_cls is m.FusionEmbeddingModel

    from sglang.srt.configs.model_config import is_multimodal_model
    assert is_multimodal_model(["FusionEmbeddingModel"]), "external mm arch hook failed"
    lines.append("is_multimodal_model: True (SGLANG_EXTERNAL_MM_MODEL_ARCH)")

    from sglang.srt.managers.multimodal_processor import (
        PROCESSOR_MAPPING, import_processors,
    )
    import_processors("sglang.srt.multimodal.processors")
    import_processors(_os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"], overwrite=True)
    hits = [c.__name__ for mc, c in PROCESSOR_MAPPING.items()
            if mc.__name__ == "FusionEmbeddingModel"]
    lines.append(f"processor mapping for FusionEmbeddingModel: {hits}")
    assert hits == ["FusionMultimodalProcessor"], hits
    return "\n".join(lines)


@app.function(image=ref_img, timeout=900, volumes={"/vol": volume})
def find_audio_clips(max_hits: int = 20) -> list:
    """List candidate real audio files on the fusion-data volume (shallow scan)."""
    import pathlib
    hits = []
    root = pathlib.Path("/vol/demo_space/audio_wav")
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if p.suffix.lower() in (".wav", ".flac"):
                hits.append(str(p))
                if len(hits) >= max_hits:
                    break
    return hits


# --------------------------------------------------------------------------- #
# Comparison + orchestration
# --------------------------------------------------------------------------- #
def _cos(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return num / (da * db)


def _max_abs_diff(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _report(name: str, ref_vecs, test_vecs, cos_gate: float):
    ok = True
    for i, (r, v) in enumerate(zip(ref_vecs, test_vecs)):
        c = _cos(r, v)
        d = _max_abs_diff(r, v)
        status = "PASS" if c > cos_gate else "FAIL"
        if c <= cos_gate:
            ok = False
        print(f"  {name}[{i}]: cosine={c:.6f} max_abs_diff={d:.6f}  {status}")
    return ok


@app.local_entrypoint()
def main(stage: str = "check", audio_paths: str = "", dtype: str = "float32"):
    payload = {
        "texts": TEXTS,
        "n_images": N_IMAGES,
        "n_audios": N_AUDIOS,
        "audio_paths": [p for p in audio_paths.split(",") if p] or
                       (DEFAULT_AUDIO_PATHS if stage == "c" else []),
        "with_audio": stage == "c",
        "dtype": dtype,
    }
    print(f"[smoke] comparing at dtype={dtype}")

    if stage == "check":
        print(check_registration.remote())
        return
    if stage == "find-audio":
        for p in find_audio_clips.remote():
            print(p)
        return

    if stage == "a":
        print("== reference (frozen base readout) ==")
        ref = ref_embed.remote(payload)
        print("== SGLang stage A (no adapters, no audio) ==")
        va = sgl_stage_a.remote(payload)
        ok = _report("text", ref["text"], va["text"], 0.999)
        ok &= _report("image", ref["image"], va["image"], 0.999)
        print("STAGE A:", "PASS" if ok else "FAIL")

    elif stage == "b":
        print("== SGLang stage A (no adapters) ==")
        va = sgl_stage_a.remote(payload)
        print("== SGLang stage B (adapters loaded, gates closed) ==")
        vb = sgl_stage_b.remote(payload)
        ok = True
        for name in ("text", "image"):
            for i, (x, y) in enumerate(zip(va[name], vb[name])):
                d = _max_abs_diff(x, y)
                same = d <= 1e-5
                ok &= same
                print(f"  {name}[{i}]: max_abs_diff={d:.3e} "
                      f"(allclose atol=1e-5: {'PASS' if same else 'FAIL'})")
        print("STAGE B:", "PASS" if ok else "FAIL")

    elif stage == "acontrol":
        # Determinism control: the SAME stage-A configuration booted twice.
        # Bounds what any a-vs-b comparison can resolve at this dtype.
        print("== SGLang stage A, boot 1 ==")
        r1 = sgl_stage_a.remote(payload)
        print("== SGLang stage A, boot 2 ==")
        r2 = sgl_stage_a.remote(payload)
        for name in ("text", "image"):
            for i, (x, y) in enumerate(zip(r1[name], r2[name])):
                print(f"  {name}[{i}]: boot-to-boot max_abs_diff={_max_abs_diff(x, y):.3e}")

    elif stage == "c":
        print("== reference (full, incl. audio) ==")
        ref = ref_embed.remote(payload)
        print("== SGLang stage C (full model, A100-80GB) ==")
        vc = sgl_stage_c.remote(payload)
        ok = _report("text", ref["text"], vc["text"], 0.999)
        ok &= _report("image", ref["image"], vc["image"], 0.999)
        ok &= _report("audio", ref["audio"], vc["audio"], 0.999)
        print("STAGE C:", "PASS" if ok else "FAIL")

    elif stage == "serve":
        # The literal deliverable command over HTTP (served precision = bf16), so
        # the parity target is the bf16 text gate.
        payload["dtype"] = "bfloat16"
        print("== reference (bf16) ==")
        ref = ref_embed.remote(payload)
        print("== sglang.launch_server --is-embedding (HTTP /v1/embeddings) ==")
        vs = sgl_serve_smoke.remote(payload)
        if "error" in vs:
            print("SERVE FAIL:", vs["error"])
            for line in vs.get("log", []):
                print("   ", line)
            raise SystemExit(1)
        ok = _report("text", ref["text"], vs["text"], 0.999)
        print("SERVE:", "PASS" if ok else "FAIL")

    else:
        raise SystemExit(f"unknown stage {stage!r} (use check|find-audio|a|b|c|serve)")
