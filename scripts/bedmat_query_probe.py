"""Probe free-text queries against the Tactus Mat demo gallery.

Picks honest example queries for the notebook: which paraphrases actually retrieve the
window a reader would expect, and which do not. Prints a table; no assertions.
"""
import os as _os

import modal

app = modal.App("bedmat-query-probe")
img = (modal.Image.debian_slim(python_version="3.11")
       .apt_install("git")
       .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "scipy", "pillow",
                    "transformers>=4.57", "accelerate", "huggingface_hub", "safetensors",
                    extra_index_url="https://download.pytorch.org/whl/cu124")
       .add_local_python_source("fusion_embedding"))
volume = modal.Volume.from_name("fusion-data")
if _os.environ.get("HF_TOKEN"):
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": _os.environ["HF_TOKEN"],
                                        "HUGGING_FACE_HUB_TOKEN": _os.environ["HF_TOKEN"]})
else:
    hf_secret = modal.Secret.from_name("huggingface")

REPO, REV = "EximiusLabs/fusion-embedding-2-tactus-mat", "v0.1-preview"

CANDIDATES = [
    "someone curled up",
    "a person curled up on their side",
    "curled up in a ball",
    "the fetal position",
    "a person in the fetal position",
    "knees pulled up toward the chest",
    "lying flat, arms out",
    "arms and legs spread wide",
    "a person starfished across the bed",
    "a person on their left",
    "a person on their left side",
    "someone sleeping on the left",
    "lying on the right",
    "flat on the back",
    "a person lying straight on their back",
    "knees bent up",
    "both knees raised",
    "the head of the bed is raised",
    "sitting up against the pillows",
    "propped up on an incline",
]


@app.function(image=img, volumes={"/vol": volume}, secrets=[hf_secret], gpu="A10G",
              memory=32768, timeout=3600, env={"HF_HOME": "/vol/hf"})
def probe() -> dict:
    import importlib.util
    import json
    import sys

    import numpy as np
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download

    for f in ("inference.py", "tactile.py"):
        p = hf_hub_download(REPO, f, revision=REV)
    sys.path.insert(0, _os.path.dirname(p))
    spec = importlib.util.spec_from_file_location(
        "tm", hf_hub_download(REPO, "inference.py", revision=REV))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tm = m.TactusMatEmbedder.from_pretrained(REPO, revision=REV, device="cuda")

    d = np.load(hf_hub_download(REPO, "demo_frames.npz", revision=REV))
    names = [str(n) for n in d["position_names"]]
    G = torch.stack([tm.embed_pressure(w.astype(np.float32)) for w in d["gallery"]])
    gpos = d["gallery_pos"]

    rows = []
    for q in CANDIDATES:
        v = F.normalize(tm.embed_text([q]), dim=-1)[0]
        s = (G @ v).numpy()
        o = np.argsort(-s)
        rows.append({"query": q,
                     "top": names[gpos[o[0]] - 1], "score": round(float(s[o[0]]), 3),
                     "second": names[gpos[o[1]] - 1],
                     "margin": round(float(s[o[0]] - s[o[1]]), 3)})
        print(f"{s[o[0]]:+.3f} (+{s[o[0]]-s[o[1]]:.3f})  {q:42s} -> {rows[-1]['top']}",
              flush=True)
    print("PROBE_JSON:", json.dumps(rows), flush=True)
    return {"rows": rows, "gallery": [names[p - 1] for p in gpos]}


@app.local_entrypoint()
def main():
    probe.remote()
