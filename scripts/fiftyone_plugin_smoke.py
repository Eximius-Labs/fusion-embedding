"""End-to-end test of the FiftyOne integration, with real FiftyOne and real data.

Builds a dataset from held-out PmatData windows, registers this repo as a zoo model
source, loads eximius/tactus-mat through FiftyOne's own zoo API, embeds the sensor field,
builds a similarity index, and then asks the question the whole integration exists for:
does a plain-English query sort the dataset correctly?

Assertions (a pass means the integration actually works, not that it merely ran):
  1. the model loads via foz.load_zoo_model and advertises can_embed_prompts
  2. every sample embeds to a unit vector of the expected dimension
  3. the index reports supports_prompts
  4. sort_by_similarity("a person lying on their left side") ranks a left-side window first
  5. the same holds for a right-side and a supine query, so it is not one lucky phrasing

Run (billable GPU, small):
    PYTHONUTF8=1 uv run --env-file .env modal run scripts/fiftyone_plugin_smoke.py
"""
import os as _os

import modal

app = modal.App("fiftyone-fusion-smoke")
img = (modal.Image.debian_slim(python_version="3.11")
       .apt_install("git", "curl")
       .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "scipy", "pillow",
                    "transformers>=4.57", "accelerate", "huggingface_hub", "safetensors",
                    extra_index_url="https://download.pytorch.org/whl/cu124")
       .pip_install("fiftyone", "fiftyone-brain")
       .add_local_python_source("fusion_embedding")
       .add_local_dir("fiftyone_fusion_embedding", "/root/fiftyone_fusion_embedding"))
volume = modal.Volume.from_name("fusion-data")
if _os.environ.get("HF_TOKEN"):
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": _os.environ["HF_TOKEN"],
                                        "HUGGING_FACE_HUB_TOKEN": _os.environ["HF_TOKEN"]})
else:
    hf_secret = modal.Secret.from_name("huggingface")

# (position id, the query that should retrieve it) from the dataset's own posture table
CASES = [(3, "a person lying on their left side"),
         (2, "a person lying on their right side"),
         (1, "a person lying flat on their back")]


@app.function(image=img, volumes={"/vol": volume}, secrets=[hf_secret], gpu="A10G",
              memory=32768, timeout=3600,
              env={"HF_HOME": "/vol/hf", "FIFTYONE_DATABASE_DIR": "/tmp/fodb"})
def smoke() -> dict:
    import json

    import numpy as np

    import fiftyone as fo
    import fiftyone.brain as fob
    import fiftyone.zoo as foz

    # The manifest requires the `fusion-embedding` package, which is correct for a user who
    # pip installs it. This container mounts the repo as a source instead, so the package
    # is importable but not visible to a pip requirement check. Confirm it imports, then
    # downgrade the check rather than weakening the manifest.
    import fusion_embedding  # noqa: F401
    fo.config.requirement_error_level = 1

    # ---- 1. install this repo as a zoo model source ----
    # A user runs foz.register_zoo_model_source("https://github.com/Eximius-Labs/..."),
    # which downloads the repo and copies it into the local model zoo. That downloader
    # only accepts GitHub or archive URLs, so here we stage the same on-disk result
    # directly and then exercise everything downstream through FiftyOne's own API.
    import shutil
    src = "/root/fiftyone_fusion_embedding"
    dest = _os.path.join(fo.config.model_zoo_dir, "Eximius-Labs",
                         "fiftyone-fusion-embedding")
    shutil.rmtree(dest, ignore_errors=True)
    _os.makedirs(_os.path.dirname(dest), exist_ok=True)
    shutil.copytree(src, dest)
    print("staged model source at", dest, flush=True)

    names = foz.list_zoo_models()
    ours = [n for n in names if n.startswith("eximius/")]
    print("models this source declares:", ours, flush=True)
    assert any("tactus-mat" in n for n in ours), (
        f"source not picked up; zoo sees {len(names)} models, none of ours")

    model = foz.load_zoo_model("eximius/tactus-mat")
    assert model.can_embed_prompts, "model must advertise prompt support"
    assert model.has_embeddings
    print("loaded eximius/tactus-mat via the zoo API", flush=True)

    # ---- 2. build a dataset of held-out sensor windows ----
    z = np.load("/vol/bedmat/bedmat_expi.npz")
    press, subj, pos, fidx = z["pressure"], z["subject"], z["position"], z["frame_idx"]
    meta = json.load(open("/vol/bedmat/bedmat_meta.json"))
    names_by_pos = {int(k): v for k, v in meta["position_names"].items()}

    _os.makedirs("/tmp/windows", exist_ok=True)
    samples = []
    for p in range(1, 18):                       # one window per posture, held-out subject
        rows = np.where((subj == 13) & (pos == p))[0]
        rows = rows[np.argsort(fidx[rows])]
        idx = rows[np.linspace(0, len(rows) - 1, 8).astype(int)]
        path = f"/tmp/windows/pos{p}.npy"
        np.save(path, press[idx].astype(np.float32))
        # FiftyOne needs a filepath per sample; the sensor window rides in a field, which
        # is exactly the gap this integration fills
        img_path = f"/tmp/windows/pos{p}.png"
        from PIL import Image
        Image.fromarray((press[idx].astype(np.float32).max(0) * 255)
                        .astype(np.uint8)).save(img_path)
        s = fo.Sample(filepath=img_path)
        s["pressure_window"] = path
        s["posture"] = names_by_pos[p]
        s["position_id"] = int(p)
        samples.append(s)

    ds = fo.Dataset("fusion_smoke", overwrite=True)
    ds.add_samples(samples)
    print(f"dataset: {len(ds)} samples", flush=True)

    # ---- 3. embed the sensor field ----
    embeddings, ids = [], []
    for sample in ds:
        vec = model.embed_sensor(np.load(sample["pressure_window"]))
        assert vec.shape == (2048,), vec.shape
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3, "expected a unit vector"
        sample["fusion_embedding"] = vec.tolist()
        sample.save()
        embeddings.append(vec)
        ids.append(sample.id)
    print(f"embedded {len(embeddings)} sensor windows", flush=True)

    # ---- 4. build the index, naming the model so prompts survive ----
    results = fob.compute_similarity(
        ds, embeddings=np.stack(embeddings), model="eximius/tactus-mat",
        brain_key="fusion_sim")
    print("supports_prompts:", results.supports_prompts, flush=True)
    assert results.supports_prompts, (
        "index cannot answer text queries; the model must be named as a string")

    # ---- 5. the actual question: does English sort the sensor data? ----
    checks = []
    for want_pos, query in CASES:
        view = ds.sort_by_similarity(query, k=3, brain_key="fusion_sim")
        top = view.first()
        ok = int(top["position_id"]) == want_pos
        checks.append({"query": query, "want": names_by_pos[want_pos],
                       "got": top["posture"], "correct": bool(ok)})
        print(f"{'OK  ' if ok else 'MISS'} {query!r} -> {top['posture']!r}", flush=True)

    n_ok = sum(c["correct"] for c in checks)
    res = {"models": [n for n in names], "supports_prompts": bool(results.supports_prompts),
           "queries_correct": f"{n_ok}/{len(checks)}", "checks": checks,
           "n_samples": len(ds)}
    print("FIFTYONE_SMOKE:", json.dumps(res), flush=True)
    assert n_ok == len(checks), f"only {n_ok}/{len(checks)} text queries sorted correctly"
    return res


@app.local_entrypoint()
def main():
    smoke.remote()
