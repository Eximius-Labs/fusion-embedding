"""Bed-mat posture head: PmatData Exp I (64x32 FSA mat) -> canonical FE2 text space.

The medical-vertical demo model: same recipe class as Tactus (small conv trunk + conv
frame fusion + projector into the frozen text readout; optional same-sensor MAE init),
trained on PhysioNet PmatData experiment I (ODC-By v1.0: weights are shippable with
attribution). Open-vocabulary: training targets are 17 posture phrases embedded by the
frozen base; eval additionally scores the SAME embeddings against 3 coarse-class phrases
(supine / right side / left side), so both granularities are text queries, no classifier.

Data: /vol/bedmat/bedmat_expi.npz from scripts/bedmat_ingest.py (frames already
clip(raw/1000,0,1); stored fp16). Sessions = one (subject, position) file (~90-160
frames at 1 Hz of a HELD static posture); a training sample is K frames drawn uniformly
from one session (time-spread window; consecutive frames are near-duplicates).

Split: subjects 1-10 train / 11-13 test (subject-held-out). MAE pool = train-subject
frames only. LOSO is a follow-up, not this script.

Stages (Modal, A10G):
    modal run scripts/bedmat_train.py --action mae            # ~6k steps, minutes
    modal run scripts/bedmat_train.py --action smoke          # 150-step pipeline check
    modal run scripts/bedmat_train.py --action train --steps 4000 \
        --init-from /vol/bedmat/bedmat_mae.pt

Outputs: /vol/bedmat/bedmat_head_<variant>.pt (+ unsuffixed latest), eval JSON printed
as BEDMAT_HELDOUT, and /vol/bedmat/bedmat_demo_dump.npz (per-frame test-subject sims
for the GIF / timeline / gallery artifacts).
"""
import os as _os

import modal

import fsr_train as T                       # lr_at, kway; same image recipe
from fsr_mae_pretrain import patch_mask     # shape-agnostic patch masking

app = modal.App("bedmat-train")
img = (modal.Image.debian_slim(python_version="3.11")
       .apt_install("git")
       .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy", "scipy", "pillow",
                    "scikit-learn",
                    "transformers>=4.57", "accelerate", "huggingface_hub", "safetensors",
                    extra_index_url="https://download.pytorch.org/whl/cu124")
       .add_local_python_source("fusion_embedding", "fsr_train", "fsr_mae_pretrain"))
volume = modal.Volume.from_name("fusion-data")
if _os.environ.get("HF_TOKEN"):
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": _os.environ["HF_TOKEN"],
                                        "HUGGING_FACE_HUB_TOKEN": _os.environ["HF_TOKEN"]})
else:
    hf_secret = modal.Secret.from_name("huggingface")

CACHE = "/vol/bedmat"
TRAIN_SUBJ = tuple(range(1, 11))
TEST_SUBJ = (11, 12, 13)
K = 8                                       # frames per window (time-spread draw)
TEMP = 0.07

# Bed-context prompt templates ({name} = position phrase from ingest meta).
TEMPLATES = [
    "a person {name}, recorded by a bed pressure mat",
    "pressure map of a person {name}",
    "someone {name} in bed",
    "{name}",
]
CLASS3_PROMPTS = {
    "supine": ["a person lying flat on their back",
               "pressure map of someone lying supine in bed",
               "sleeping on the back"],
    "right": ["a person lying on their right side",
              "pressure map of someone lying on the right side in bed",
              "sleeping on the right side"],
    "left": ["a person lying on their left side",
             "pressure map of someone lying on the left side in bed",
             "sleeping on the left side"],
}


def build_sessions(subject, position, frame_idx):
    """Group cache rows into sessions keyed (subject, position) -> sorted row indices."""
    import numpy as np
    sess = {}
    for i in range(len(subject)):
        sess.setdefault((int(subject[i]), int(position[i])), []).append(i)
    return {k: np.array(sorted(v, key=lambda j: int(frame_idx[j])), dtype=np.int64)
            for k, v in sess.items()}


def draw_window(rows, k, rng):
    """K frame rows drawn uniformly without replacement (with replacement if short)."""
    import numpy as np
    if len(rows) >= k:
        return rows[np.sort(rng.choice(len(rows), size=k, replace=False))]
    return rows[np.sort(rng.choice(len(rows), size=k, replace=True))]


@app.function(image=img, volumes={"/vol": volume}, secrets=[hf_secret], gpu="A10G",
              memory=32768, timeout=2 * 3600, env={"HF_HOME": "/vol/hf"})
def mae(steps: int = 6000, mask_ratio: float = 0.6, patch: int = 4) -> dict:
    """Same-sensor MAE on train-subject frames only (test subjects excluded)."""
    import json
    import time

    import numpy as np
    import torch
    import torch.nn as nn

    from fusion_embedding.tactile import TactileEncoder

    dev = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    z = np.load(f"{CACHE}/bedmat_expi.npz")
    keep = np.isin(z["subject"], TRAIN_SUBJ)
    frames = z["pressure"][keep].astype(np.float32)          # [N,64,32] in [0,1]
    print(f"MAE pool {frames.shape} (train subjects only)", flush=True)

    enc = TactileEncoder(temporal="conv", depth="resnet18", frames=1, flow=False).to(dev)
    trunk = enc.trunk
    with torch.no_grad():
        probe = trunk(torch.zeros(1, 1, 64, 32, device=dev))
    c_out, ph, pw = probe.shape[1], probe.shape[2], probe.shape[3]
    scale = 64 // ph
    assert 32 // pw == scale, (probe.shape, scale)
    print(f"trunk probe: {tuple(probe.shape)} upsample x{scale}", flush=True)
    dec = nn.Sequential(nn.Conv2d(c_out, 64, 3, padding=1), nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True), nn.Upsample(scale_factor=scale),
                        nn.Conv2d(64, 1, 3, padding=1)).to(dev)
    params = list(trunk.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    rng = np.random.RandomState(0)
    t0 = time.time()
    for step in range(steps):
        sel = rng.randint(0, len(frames), size=128)
        x = torch.from_numpy(frames[sel]).unsqueeze(1).to(dev)          # [B,1,64,32]
        xm, mask = patch_mask(x, patch=patch, ratio=mask_ratio, rng=rng)
        rec = dec(trunk(xm))                                            # [B,1,64,32]
        # per-patch normalized targets (He et al. Table 1d; collapse pins at ~1.0)
        B, _, H, W = x.shape
        tp = x.reshape(B, 1, H // patch, patch, W // patch, patch)
        mu = tp.mean(dim=(3, 5), keepdim=True)
        sd = tp.var(dim=(3, 5), keepdim=True).add(1e-6).sqrt()
        tgt = ((tp - mu) / sd).reshape(B, 1, H, W)
        rp = rec.reshape(B, 1, H // patch, patch, W // patch, patch)
        rp = rp.reshape(B, 1, H, W)
        loss = (((rp - tgt) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"mae step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s",
                  flush=True)
    out = f"{CACHE}/bedmat_mae.pt"
    torch.save({"trunk": trunk.state_dict(),
                "config": {"steps": steps, "mask_ratio": mask_ratio, "patch": patch,
                           "pool": "pmatdata_expi_train_subjects",
                           "n_frames": int(len(frames)), "norm_pix": True}}, out)
    volume.commit()
    final = float(loss.item())
    print(f"MAE_DONE {json.dumps({'final_loss': final, 'path': out})}", flush=True)
    assert final < 0.9, f"MAE loss {final} near 1.0 = mean-predictor collapse"
    return {"final_loss": final, "path": out}


@app.function(image=img, volumes={"/vol": volume}, secrets=[hf_secret], gpu="A10G",
              memory=32768, timeout=2 * 3600, env={"HF_HOME": "/vol/hf"})
def train(smoke: bool = False, steps: int = 4000, init_from: str = "",
          augment: bool = True, seed: int = 0) -> dict:
    import json
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    from fusion_embedding import UnifiedEmbedder
    from fusion_embedding.tactile import TactileEncoder, build_fsr_head, build_projector

    dev = "cuda"
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)

    z = np.load(f"{CACHE}/bedmat_expi.npz")
    meta = json.load(open(f"{CACHE}/bedmat_meta.json"))
    pressure = z["pressure"].astype(np.float32)              # [N,64,32] in [0,1]
    subject, position, frame_idx = z["subject"], z["position"], z["frame_idx"]
    pos_names = {int(k): v for k, v in meta["position_names"].items()}
    class3 = {int(k): v for k, v in meta["class3"].items()}
    n_pos = 17

    sessions = build_sessions(subject, position, frame_idx)
    tr_sess = {k: v for k, v in sessions.items() if k[0] in TRAIN_SUBJ}
    te_sess = {k: v for k, v in sessions.items() if k[0] in TEST_SUBJ}
    assert len(tr_sess) == len(TRAIN_SUBJ) * n_pos and len(te_sess) == 3 * n_pos
    print(f"sessions: train {len(tr_sess)} test {len(te_sess)} "
          f"frames {pressure.shape} {time.time()-t0:.0f}s", flush=True)

    # ---- canonical text targets: 17 positions (train) + 3 classes (eval) ----
    ue = UnifiedEmbedder.from_pretrained(T.FE2_REPO, device=dev, dtype=torch.bfloat16)

    def target_rows(prompt_lists):
        rows = []
        for plist in prompt_lists:
            e = torch.stack([ue.embed_text(s).float() for s in plist])
            rows.append(F.normalize(F.normalize(e, dim=-1).mean(dim=0), dim=-1))
        tt = F.normalize(torch.stack(rows), dim=-1)
        return F.normalize(tt - tt.mean(dim=0, keepdim=True), dim=-1)   # centered

    tt17 = target_rows([[t.format(name=pos_names[p]) for t in TEMPLATES]
                        for p in range(1, 18)]).to(dev)                  # [17,2048]
    tt3 = target_rows([CLASS3_PROMPTS[c] for c in ("supine", "right", "left")]).to(dev)
    cls_order = {"supine": 0, "right": 1, "left": 2}
    print(f"text targets built {time.time()-t0:.0f}s", flush=True)

    enc = TactileEncoder(temporal="conv", depth="resnet18", frames=K, flow=False).to(dev)
    if init_from:
        blob = torch.load(init_from, map_location="cpu")
        enc.trunk.load_state_dict(blob["trunk"], strict=True)
        print(f"trunk init from {init_from} (mae steps="
              f"{blob.get('config', {}).get('steps')})", flush=True)
    proj = build_projector().to(dev)
    head = build_fsr_head(enc, proj)
    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    scale = float(np.log(1.0 / TEMP))

    tr_keys = list(tr_sess.keys())
    if smoke:
        steps = 150
    bs = 128
    for step in range(steps):
        lr = T.lr_at(step, 3e-4, min(200, steps // 2), steps, "cosine")
        for g in opt.param_groups:
            g["lr"] = lr
        keys = [tr_keys[i] for i in rng.randint(0, len(tr_keys), size=bs)]
        rows = np.stack([draw_window(tr_sess[k], K, rng) for k in keys])   # [B,K]
        x = torch.from_numpy(pressure[rows]).to(dev)                       # [B,K,64,32]
        if augment:
            # light spatial jitter: random roll up to 2 taxels each axis + amp scale
            sh, sw = int(rng.randint(-2, 3)), int(rng.randint(-2, 3))
            x = torch.roll(x, shifts=(sh, sw), dims=(2, 3))
            x = x * float(rng.uniform(0.8, 1.2))
        y = torch.tensor([k[1] - 1 for k in keys], device=dev)
        p = F.normalize(head(x), dim=-1)
        loss = F.cross_entropy((p @ tt17.T) * np.exp(scale), y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)

    enc.eval(); proj.eval()

    # ---- eval on held-out subjects: window-level, 4-draw voting ----
    def embed_windows(sess_dict, draws=4):
        keys = sorted(sess_dict.keys())
        embs, labels = [], []
        with torch.no_grad():
            for k in keys:
                acc = None
                for d in range(draws):
                    r = np.random.RandomState(1234 + d)
                    w = np.stack([draw_window(sess_dict[k], K, r)
                                  for _ in range(8)])                     # 8 windows/sess
                    x = torch.from_numpy(pressure[w]).to(dev)
                    e = F.normalize(head(x), dim=-1)
                    acc = e if acc is None else acc + e
                embs.append(F.normalize(acc, dim=-1).cpu())
                labels.extend([k[1]] * 8)
        return torch.cat(embs), np.array(labels), keys

    te_p, te_pos_labels, te_keys = embed_windows(te_sess)
    y17 = torch.from_numpy(te_pos_labels - 1)
    with torch.no_grad():
        s17 = (te_p.to(dev) @ tt17.T).cpu()
        s3 = (te_p.to(dev) @ tt3.T).cpu()
    top1_17 = float((s17.argmax(1) == y17).float().mean())
    top3_17 = float((s17.topk(3, dim=1).indices == y17[:, None]).any(1).float().mean())
    y3 = torch.tensor([cls_order[class3[int(p)]] for p in te_pos_labels])
    top1_3 = float((s3.argmax(1) == y3).float().mean())
    per_pos = {pos_names[p]: round(float((s17.argmax(1) == (p - 1))[y17 == (p - 1)]
                                         .float().mean()), 3)
               for p in range(1, 18)}
    held = {"pos17_top1": round(top1_17, 4), "pos17_top3": round(top3_17, 4),
            "class3_top1": round(top1_3, 4),
            "chance17": round(1 / 17, 4), "chance3": round(1 / 3, 4),
            "n_test_windows": int(len(y17)), "test_subjects": list(TEST_SUBJ),
            "smoke": bool(smoke), "init_from": init_from or None, "seed": int(seed)}
    print(f"BEDMAT_HELDOUT: {json.dumps(held)}", flush=True)
    print(f"per-position: {json.dumps(per_pos)}", flush=True)

    # ---- demo dump: per-frame sims for ALL test-subject frames (GIF/timeline) ----
    te_rows = np.where(np.isin(subject, TEST_SUBJ))[0]
    sims17 = np.zeros((len(te_rows), 17), dtype=np.float16)
    sims3 = np.zeros((len(te_rows), 3), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, len(te_rows), 512):
            idx = te_rows[i:i + 512]
            x = torch.from_numpy(pressure[idx]).unsqueeze(1).to(dev)     # [B,1,64,32]
            e = F.normalize(head(x), dim=-1)
            sims17[i:i + 512] = (e @ tt17.T).cpu().numpy().astype(np.float16)
            sims3[i:i + 512] = (e @ tt3.T).cpu().numpy().astype(np.float16)
    # Seed-suffixed paths: two runs of one config must never clobber each other
    # (the same variant-path rule fsr_train uses; violating it cost a real result).
    tag = "smoke" if smoke else f"{'mae' if init_from else 'scratch'}_s{int(seed)}"
    np.savez_compressed(f"{CACHE}/bedmat_demo_dump_{tag}.npz",
                        rows=te_rows, sims17=sims17, sims3=sims3,
                        subject=subject[te_rows], position=position[te_rows],
                        frame_idx=frame_idx[te_rows])
    torch.save({"encoder": {k: v.half().cpu() for k, v in enc.state_dict().items()},
                "proj": {k: v.half().cpu() for k, v in proj.state_dict().items()},
                "eval": held, "per_position": per_pos,
                "config": {"K": K, "grid": [64, 32], "depth": "resnet18",
                           "temporal": "conv", "flow": False, "steps": steps,
                           "augment": bool(augment), "init_from": init_from or None,
                           "templates": TEMPLATES, "class3_prompts": CLASS3_PROMPTS,
                           "position_names": pos_names,
                           "normalization": meta["normalization"],
                           "dataset": "PhysioNet pmd/1.0.0 exp I (ODC-By v1.0)",
                           "fe2_source": T.FE2_REPO}},
               f"{CACHE}/bedmat_head_{tag}.pt")
    volume.commit()
    return held


@app.local_entrypoint()
def main(action: str = "smoke", steps: int = 4000, init_from: str = "",
         augment: bool = True, seed: int = 0, seeds: str = ""):
    if action == "mae":
        mae.remote(steps=min(steps, 6000) if steps != 4000 else 6000)
    elif action == "smoke":
        train.remote(smoke=True)
    elif action == "train":
        seed_list = [int(s) for s in seeds.split(",")] if seeds else [int(seed)]
        results = list(train.map(
            [False] * len(seed_list), [steps] * len(seed_list),
            [init_from] * len(seed_list), [augment] * len(seed_list), seed_list,
            kwargs={}, order_outputs=True))
        import json as _json
        tops = [r["pos17_top1"] for r in results]
        summary = {"seeds": seed_list, "top1s": tops,
                   "mean": round(sum(tops) / len(tops), 4),
                   "init_from": init_from or None}
        print(f"BEDMAT_SEED_SUMMARY: {_json.dumps(summary)}")
    else:
        raise SystemExit(f"unknown action {action}")
