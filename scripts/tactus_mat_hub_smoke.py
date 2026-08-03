"""Release smoke for EximiusLabs/fusion-embedding-2-tactus-mat: pull the artifact from the
Hub at its tagged revision and rank real held-out PmatData windows against text queries."""
import os as _os

import modal

app = modal.App("tactus-mat-hub-smoke")
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

REPO = "EximiusLabs/fusion-embedding-2-tactus-mat"
REV = "v0.1-preview"
# One held-out session per query; positions per the dataset's own posture table.
CASES = [(1, "a person lying flat on their back"),
         (2, "a person lying on their right side"),
         (3, "a person lying on their left side"),
         (13, "a person curled up in the fetal position on their right side")]
QUERIES = ["a person lying flat on their back",
           "a person lying on their right side",
           "a person lying on their left side",
           "a person curled up in the fetal position on their right side",
           "an empty bed with nobody on it"]


@app.function(image=img, volumes={"/vol": volume}, secrets=[hf_secret], gpu="A10G",
              memory=32768, timeout=3600, env={"HF_HOME": "/vol/hf"})
def smoke() -> dict:
    import importlib.util
    import json
    import sys

    import numpy as np
    from huggingface_hub import hf_hub_download

    files = {f: hf_hub_download(REPO, f, revision=REV)
             for f in ("inference.py", "tactile.py", "model.safetensors", "config.json")}
    moddir = _os.path.dirname(files["inference.py"])
    sys.path.insert(0, moddir)
    spec = importlib.util.spec_from_file_location("hub_mat_inference", files["inference.py"])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    tm = m.TactusMatEmbedder.from_pretrained(REPO, revision=REV, device="cuda",
                                             load_text=True)

    z = np.load("/vol/bedmat/bedmat_expi.npz")
    press, subj, pos, fidx = z["pressure"], z["subject"], z["position"], z["frame_idx"]
    out, ok = [], True
    for p, truth in CASES:
        rows = np.where((subj == 13) & (pos == p))[0]        # held-out subject
        rows = rows[np.argsort(fidx[rows])]
        idx = rows[np.linspace(0, len(rows) - 1, 8).astype(int)]   # 8 time-spread frames
        window = press[idx].astype(np.float32)
        ranked = tm.rank(window, QUERIES)
        hit = ranked[0][0] == truth
        ok &= hit
        out.append({"position": int(p), "truth": truth, "top": ranked[0][0],
                    "score": round(float(ranked[0][1]), 4), "correct": bool(hit)})
        print(f"pos {p:2d} -> {ranked[0][0]!r} {ranked[0][1]:+.3f} "
              f"{'OK' if hit else 'MISS'}", flush=True)
    res = {"repo": REPO, "revision": REV, "all_correct": bool(ok), "cases": out}
    print("TACTUS_MAT_HUB_SMOKE:", json.dumps(res), flush=True)
    assert ok, "hub smoke failed: a held-out window did not rank its own posture first"
    return res


@app.local_entrypoint()
def main():
    smoke.remote()
