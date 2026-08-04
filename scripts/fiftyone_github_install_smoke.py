"""Verify the REAL user install path for the FiftyOne zoo source.

The integration smoke staged the source on disk because the test container had the repo as
a local mount. A user instead runs:

    foz.register_zoo_model_source("https://github.com/Eximius-Labs/fusion-embedding")

which downloads the repo from GitHub and scans it for manifest.json files. This test runs
exactly that line, then walks the same downstream path a user would:

  1. the source registers and our four models appear in list_zoo_models()
  2. eximius/tactus-mat loads through the zoo API (weights pulled from the HF Hub)
  3. it embeds a real held-out mat window and can_embed_prompts is advertised

No GPU needed for registration itself, but the model load wants one; A10G suffices
(tactus-mat + the 2B text side, and we skip text here anyway with load_text handled
inside the wrapper's lazy loading).

Run:
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/fiftyone_github_install_smoke.py
"""
import os as _os

import modal

app = modal.App("fiftyone-github-install-smoke")
img = (modal.Image.debian_slim(python_version="3.11")
       .apt_install("git", "curl")
       .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "scipy", "pillow",
                    "transformers>=4.57", "accelerate", "huggingface_hub", "safetensors",
                    extra_index_url="https://download.pytorch.org/whl/cu124")
       .pip_install("fiftyone", "fiftyone-brain")
       .add_local_python_source("fusion_embedding"))
volume = modal.Volume.from_name("fusion-data")
if _os.environ.get("HF_TOKEN"):
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": _os.environ["HF_TOKEN"],
                                        "HUGGING_FACE_HUB_TOKEN": _os.environ["HF_TOKEN"]})
else:
    hf_secret = modal.Secret.from_name("huggingface")

SOURCE = "https://github.com/Eximius-Labs/fusion-embedding"


@app.function(image=img, volumes={"/vol": volume}, secrets=[hf_secret], gpu="A10G",
              memory=32768, timeout=3600,
              env={"HF_HOME": "/vol/hf", "FIFTYONE_DATABASE_DIR": "/tmp/fodb"})
def smoke() -> dict:
    import json

    import numpy as np

    import fiftyone as fo
    import fiftyone.zoo as foz

    fo.config.requirement_error_level = 1  # repo mounted as source, not pip-installed

    # ---- 1. the exact line a user runs ----
    foz.register_zoo_model_source(SOURCE, overwrite=True)
    sources = foz.list_zoo_model_sources()
    print("registered sources:", [s for s in sources if "Eximius" in s], flush=True)

    names = foz.list_zoo_models()
    ours = sorted(n for n in names if n.startswith("eximius/"))
    print("models declared by the GitHub source:", ours, flush=True)
    expected = ["eximius/ember", "eximius/fusion-embedding-2", "eximius/tactus",
                "eximius/tactus-mat"]
    assert ours == expected, f"expected {expected}, got {ours}"

    # ---- 2. load through the zoo API (exercises download_model + load_model) ----
    model = foz.load_zoo_model("eximius/tactus-mat")
    assert model.can_embed_prompts
    assert model.has_embeddings
    print("eximius/tactus-mat loaded via the GitHub-registered source", flush=True)

    # ---- 3. embed a real held-out window ----
    z = np.load("/vol/bedmat/bedmat_expi.npz")
    press, subj, pos, fidx = z["pressure"], z["subject"], z["position"], z["frame_idx"]
    rows = np.where((subj == 13) & (pos == 3))[0]
    rows = rows[np.argsort(fidx[rows])]
    idx = rows[np.linspace(0, len(rows) - 1, 8).astype(int)]
    vec = model.embed_sensor(press[idx].astype(np.float32))
    assert vec.shape == (2048,), vec.shape
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3

    res = {"source": SOURCE, "models": ours, "loaded": "eximius/tactus-mat",
           "embed_shape": list(vec.shape), "can_embed_prompts": True}
    print("GITHUB_INSTALL_SMOKE:", json.dumps(res), flush=True)
    return res


@app.local_entrypoint()
def main():
    smoke.remote()
