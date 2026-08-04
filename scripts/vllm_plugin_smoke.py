"""Modal smoke for the out-of-tree vLLM plugin (fusion_embedding.vllm_plugin).

Stage-gated parity checks of the vLLM-served FusionEmbeddingModel against the
reference embedder (fusion_embedding.UnifiedEmbedder) on identical inputs:

* stage a — frozen base only (adapters + audio disabled in the plugin): text and
  image embeddings must match the reference at cosine > 0.999.
* stage b — adapters loaded, gates closed: stage-a vectors must be unchanged
  (allclose; both runs embed one prompt at a time so batching is identical).
* stage c — full model on A100-80GB: audio via the vLLM multimodal path (audio
  tower + resampler + token-gated adapters) vs the reference embed_audio, per clip;
  text/image parity re-checked with the full configuration loaded.

Inputs are built deterministically INSIDE each container from a shared spec
(seeded numpy), so no media bytes cross the wire; optionally, real audio files on
the fusion-data volume can be passed with --audio-paths (comma-separated /vol paths).

Run from the repo root (billable GPU job; do not run as part of any automated flow):
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/vllm_plugin_smoke.py --stage check
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/vllm_plugin_smoke.py --stage a
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/vllm_plugin_smoke.py --stage b
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/vllm_plugin_smoke.py --stage c
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/vllm_plugin_smoke.py --stage serve
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/vllm_plugin_smoke.py --stage find-audio

Stages a/b/c gate at --dtype float32 (both sides fp32: isolates implementation
correctness from bf16 kernel-rounding differences, which the text-side whitening
amplifies); pass --dtype bfloat16 to measure served-precision parity. ``serve`` boots
the literal `vllm serve <repo> --runner pooling` command on an A100 and embeds the
text prompts over HTTP /v1/embeddings (bf16). On Windows shells prefix
MSYS_NO_PATHCONV=1 when passing --audio-paths (Git Bash mangles /vol/... paths).
"""
import math
import os as _os

import modal

app = modal.App("fusion-vllm-plugin-smoke")

REPO = "EximiusLabs/fusion-embedding-2-2b-preview"

# Reference image: the environment the released model is exercised in (matches the
# phase-B/C Modal images: torch 2.5.1 cu124 + current transformers 5.x).
ref_img = (modal.Image.debian_slim(python_version="3.11")
           .apt_install("git", "ffmpeg", "libsndfile1")
           .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "scipy", "pillow",
                        "transformers==5.12.1", "accelerate", "huggingface_hub",
                        "safetensors", "soundfile>=0.12", "librosa>=0.10",
                        extra_index_url="https://download.pytorch.org/whl/cu124")
           .add_local_python_source("fusion_embedding"))

# vLLM image: pinned vllm 0.26.0 + the fusion-embedding package pip-installed (the
# plugin ships inside it as fusion_embedding.vllm_plugin; the entry point must be
# registered through pip metadata so every vLLM process auto-loads it).
vllm_img = (modal.Image.debian_slim(python_version="3.12")
            .apt_install("git", "ffmpeg", "libsndfile1")
            .pip_install("vllm[audio]==0.26.0", "soundfile>=0.12", "librosa>=0.10")
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
# vLLM side
# --------------------------------------------------------------------------- #
def _vllm_embed(payload: dict, with_audio: bool) -> dict:
    from vllm import LLM

    limits = {"image": 1, "video": 0}
    if with_audio:
        limits["audio"] = 1
    # num_gpu_blocks_override pins the KV-cache allocation across configurations.
    # Without it, loading the adapters (~350MB fp32) shrinks free memory, vLLM sizes
    # the KV pool differently, and fused-attention kernel plans change — a
    # deterministic ~1e-4 drift between adapter-free and adapters-loaded runs that
    # has nothing to do with the adapter math (root-caused on the SGLang port,
    # where pinning max_total_tokens restored bitwise equality).
    llm = LLM(model=REPO, runner="pooling", max_model_len=8192,
              dtype=payload.get("dtype", "auto"),
              gpu_memory_utilization=0.85, limit_mm_per_prompt=limits,
              num_gpu_blocks_override=2048)

    def one(prompt) -> list:
        res = llm.embed(prompt)
        return list(res[0].outputs.embedding)

    out: dict = {"text": [], "image": [], "audio": []}
    for t in payload["texts"]:
        out["text"].append(one(CHAT.format(instruction=TEXT_INSTRUCTION, content=t)))
    for i in range(payload["n_images"]):
        out["image"].append(one({
            "prompt": CHAT.format(instruction=DOC_INSTRUCTION, content=IMAGE_CONTENT),
            "multi_modal_data": {"image": make_image(i)},
        }))
    if with_audio:
        for i in range(payload["n_audios"]):
            out["audio"].append(one({
                "prompt": AUDIO_PROMPT,
                "multi_modal_data": {"audio": (make_audio(i), SR)},
            }))
        for p in payload.get("audio_paths", []):
            wav, sr = load_volume_audio(p)
            out["audio"].append(one({
                "prompt": AUDIO_PROMPT,
                "multi_modal_data": {"audio": (wav, sr)},
            }))
    return out


@app.function(image=vllm_img, gpu="A10G", timeout=3600,
              volumes={"/vol": volume}, secrets=[hf_secret])
def vllm_stage_a(payload: dict) -> dict:
    _os.environ.update(HF_HOME="/vol/hf",
                       FUSION_VLLM_DISABLE_AUDIO="1",
                       FUSION_VLLM_DISABLE_ADAPTERS="1")
    return _vllm_embed(payload, with_audio=False)


@app.function(image=vllm_img, gpu="A10G", timeout=3600,
              volumes={"/vol": volume}, secrets=[hf_secret])
def vllm_stage_b(payload: dict) -> dict:
    _os.environ.update(HF_HOME="/vol/hf", FUSION_VLLM_DISABLE_AUDIO="1")
    _os.environ.update(payload.get("extra_env", {}))
    return _vllm_embed(payload, with_audio=False)


@app.function(image=vllm_img, gpu="A100-80GB", timeout=3600,
              volumes={"/vol": volume}, secrets=[hf_secret])
def vllm_stage_c(payload: dict) -> dict:
    _os.environ.update(HF_HOME="/vol/hf")
    return _vllm_embed(payload, with_audio=True)


@app.function(image=vllm_img, gpu="A100-80GB", timeout=3600,
              volumes={"/vol": volume}, secrets=[hf_secret])
def vllm_serve_smoke(payload: dict) -> dict:
    """Boot the literal deliverable command (`vllm serve <repo> --runner pooling`)
    and embed the text prompts through the OpenAI-compatible /v1/embeddings API."""
    import json
    import subprocess
    import time
    import urllib.request

    _os.environ["HF_HOME"] = "/vol/hf"
    proc = subprocess.Popen(
        ["vllm", "serve", REPO, "--runner", "pooling"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = "http://127.0.0.1:8000"
    deadline = time.time() + 1500
    log_lines: list[str] = []

    import threading

    def _drain():
        for line in proc.stdout:  # type: ignore[union-attr]
            log_lines.append(line.rstrip())

    threading.Thread(target=_drain, daemon=True).start()
    while time.time() < deadline:
        if proc.poll() is not None:
            return {"error": "server exited", "log": log_lines[-60:]}
        try:
            with urllib.request.urlopen(base + "/health", timeout=3):
                break
        except Exception:
            time.sleep(3)
    else:
        proc.kill()
        return {"error": "server never became healthy", "log": log_lines[-60:]}

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
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            out["text"].append(data["data"][0]["embedding"])
    finally:
        proc.terminate()
    return out


@app.function(image=vllm_img, timeout=1200, volumes={"/vol": volume}, secrets=[hf_secret])
def check_registration() -> str:
    """GPU-free sanity: plugin imports against vllm 0.26, config merges, registry maps."""
    _os.environ["HF_HOME"] = "/vol/hf"
    lines = []
    import vllm  # noqa: F401
    lines.append(f"vllm {vllm.__version__}")
    import transformers
    lines.append(f"transformers {transformers.__version__}")

    from fusion_embedding.vllm_plugin import register
    register()

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(REPO)
    lines.append(f"config class: {type(cfg).__name__} model_type={cfg.model_type}")
    assert cfg.architectures == ["FusionEmbeddingModel"], cfg.architectures
    assert cfg.text_config.hidden_size == 2048
    assert cfg.adapter_rank == 384 and cfg.n_query == 64 and cfg.d_audio == 3584
    lines.append(f"text_config layers={cfg.text_config.num_hidden_layers} "
                 f"matryoshka={cfg.matryoshka_dimensions}")

    import fusion_embedding.vllm_plugin.model as m
    lines.append(f"model module import OK: {m.FusionEmbeddingModel.__name__} "
                 f"pooling={m.FusionEmbeddingModel.is_pooling_model}")

    from importlib.metadata import entry_points
    eps = [e.name for e in entry_points(group="vllm.general_plugins")]
    lines.append(f"entry points: {eps}")
    assert "fusion_embedding" in eps
    return "\n".join(lines)


@app.function(image=ref_img, timeout=900, volumes={"/vol": volume})
def find_audio_clips(max_hits: int = 20) -> list:
    """List candidate real audio files on the fusion-data volume (shallow scan)."""
    import pathlib
    hits = []
    root = pathlib.Path("/vol")
    skip = {"hf", "frames", "text_cache", "ckpt", "ckpts", "fsr_stag"}
    for top in sorted(root.iterdir()):
        if not top.is_dir() or top.name in skip:
            continue
        for p in top.rglob("*"):
            if p.suffix.lower() in (".wav", ".flac") and p.stat().st_size > 32000:
                hits.append(str(p))
                if len(hits) >= max_hits:
                    return hits
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
def main(stage: str = "a", audio_paths: str = "", dtype: str = "float32"):
    payload = {
        "texts": TEXTS,
        "n_images": N_IMAGES,
        "n_audios": N_AUDIOS,
        "audio_paths": [p for p in audio_paths.split(",") if p],
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
        print("== vLLM stage A (no adapters, no audio) ==")
        va = vllm_stage_a.remote(payload)
        ok = _report("text", ref["text"], va["text"], 0.999)
        ok &= _report("image", ref["image"], va["image"], 0.999)
        print("STAGE A:", "PASS" if ok else "FAIL")

    elif stage == "aa":
        # Determinism control: the same stage-a configuration in two fresh containers.
        # If this is not exactly 0, cross-container drift exists independent of the
        # adapters, and stage b's exact-zero gate must be judged against this floor.
        print("== vLLM stage A, container 1 ==")
        va = vllm_stage_a.remote(payload)
        print("== vLLM stage A, container 2 ==")
        vb = vllm_stage_a.remote(payload)
        for name in ("text", "image"):
            for i, (x, y) in enumerate(zip(va[name], vb[name])):
                print(f"  {name}[{i}]: max_abs_diff={_max_abs_diff(x, y):.3e}")

    elif stage == "b-nohooks":
        # Isolation: adapters loaded (weights resident) but layer hooks not registered.
        # Compares against plain stage a; separates memory-layout effects from hook
        # effects if stage b ever drifts from exact zero.
        payload["extra_env"] = {"FUSION_VLLM_SKIP_ADAPTER_HOOKS": "1"}
        print("== vLLM stage A (no adapters) ==")
        va = vllm_stage_a.remote(payload)
        print("== vLLM stage B variant (weights loaded, hooks skipped) ==")
        vb = vllm_stage_b.remote(payload)
        for name in ("text", "image"):
            for i, (x, y) in enumerate(zip(va[name], vb[name])):
                print(f"  {name}[{i}]: max_abs_diff={_max_abs_diff(x, y):.3e}")

    elif stage == "b":
        payload.setdefault("extra_env", {})["FUSION_VLLM_HOOK_DEBUG"] = "1"
        print("== vLLM stage A (no adapters) ==")
        va = vllm_stage_a.remote(payload)
        print("== vLLM stage B (adapters loaded, gates closed) ==")
        vb = vllm_stage_b.remote(payload)
        ok = True
        for name in ("text", "image"):
            for i, (x, y) in enumerate(zip(va[name], vb[name])):
                d = _max_abs_diff(x, y)
                same = d <= 1e-5
                ok &= same
                print(f"  {name}[{i}]: max_abs_diff={d:.3e} "
                      f"(allclose atol=1e-5: {'PASS' if same else 'FAIL'})")
        print("STAGE B:", "PASS" if ok else "FAIL")

    elif stage == "c":
        print("== reference (full, incl. audio) ==")
        ref = ref_embed.remote(payload)
        print("== vLLM stage C (full model, A100-80GB) ==")
        vc = vllm_stage_c.remote(payload)
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
        print("== vllm serve --runner pooling (HTTP /v1/embeddings) ==")
        vs = vllm_serve_smoke.remote(payload)
        if "error" in vs:
            print("SERVE FAIL:", vs["error"])
            for line in vs.get("log", []):
                print("   ", line)
            raise SystemExit(1)
        ok = _report("text", ref["text"], vs["text"], 0.999)
        print("SERVE:", "PASS" if ok else "FAIL")

    else:
        raise SystemExit(f"unknown stage {stage!r} (use check|find-audio|a|b|c|serve)")
