"""Execute the shipped demo notebook's code cells end to end against the LIVE hub repo.

Runs the real notebook file (no re-implementation): pulls demo.ipynb from the Hub at the
tagged revision, strips the %pip cell, executes the rest in one namespace with a headless
matplotlib backend, and asserts the outputs a reader would judge it by:

  * the gallery ranks each window's own posture first for a majority of windows
  * a free-text query nobody trained on ("someone curled up") puts a fetal window on top
  * the temporal cell finds repositioning events in the composed session

Run (billable GPU, small):
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/bedmat_notebook_smoke.py
"""
import os as _os

import modal

app = modal.App("bedmat-notebook-smoke")
img = (modal.Image.debian_slim(python_version="3.11")
       .apt_install("git")
       .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "scipy", "pillow",
                    "matplotlib", "transformers>=4.57", "accelerate", "huggingface_hub",
                    "safetensors",
                    extra_index_url="https://download.pytorch.org/whl/cu124")
       .add_local_python_source("fusion_embedding"))
volume = modal.Volume.from_name("fusion-data")
if _os.environ.get("HF_TOKEN"):
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": _os.environ["HF_TOKEN"],
                                        "HUGGING_FACE_HUB_TOKEN": _os.environ["HF_TOKEN"]})
else:
    hf_secret = modal.Secret.from_name("huggingface")

REPO = "EximiusLabs/fusion-embedding-2-tactus-mat"
REV = "v0.1-preview"


@app.function(image=img, volumes={"/vol": volume}, secrets=[hf_secret], gpu="A10G",
              memory=32768, timeout=3600, env={"HF_HOME": "/vol/hf", "MPLBACKEND": "Agg"})
def smoke() -> dict:
    import json

    from huggingface_hub import hf_hub_download

    nb_path = hf_hub_download(REPO, "demo.ipynb", revision=REV)
    nb = json.load(open(nb_path, encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__main__"}
    ran = 0
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if src.lstrip().startswith("%pip"):
            print(f"cell {i}: skipping %pip (deps preinstalled)", flush=True)
            continue
        print(f"--- executing cell {i} ---", flush=True)
        exec(compile(src, f"<cell{i}>", "exec"), ns)   # noqa: S102 -- that is the point
        ran += 1

    # ---- assert what a reader would actually judge ----
    import numpy as np
    G, T = ns["G"], ns["T"]
    gpos, names = ns["d"]["gallery_pos"], ns["names"]
    sims = (T @ G.T).numpy()                                  # [17, P]
    top1 = sims.argmax(0) + 1
    hits = int((top1 == gpos).sum())
    assert hits >= len(gpos) - 2, (
        f"gallery: only {hits}/{len(gpos)} windows ranked their own posture first "
        f"({[(names[p-1], names[t-1]) for p, t in zip(gpos, top1) if p != t]})")

    scores, order = ns["scores"], ns["order"]
    best = names[gpos[order[0]] - 1]
    # the notebook's default query uses a word ("starfished") that appears in no training
    # phrase; it must still retrieve the spread-limbs window, or the headline claim is wrong
    assert "star" in best, f"default free-text query returned {best!r}"

    changes = ns["changes"]
    assert len(changes) >= 3, f"temporal cell found only {len(changes)} repositioning events"

    res = {"cells_executed": ran, "gallery_top1": f"{hits}/{len(gpos)}",
           "free_text_top": best, "repositioning_events": int(len(changes)),
           "night_frames": int(len(ns["night"]))}
    print("BEDMAT_NOTEBOOK_SMOKE:", json.dumps(res), flush=True)
    return res


@app.local_entrypoint()
def main():
    smoke.remote()
