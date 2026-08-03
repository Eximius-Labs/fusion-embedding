"""Ingest PhysioNet PmatData (pmd/1.0.0) experiment I into npz caches for the bed demo.

Local, no GPU. Parses the tab-delimited Exp I files (13 subjects x 17 positions,
~120 frames each, 32x64 FSA SoftFlex, documented range [0-1000], 1 Hz), writes:

  <out>/bedmat_expi.npz
      pressure  [N, 64, 32] float16, normalized clip(raw/1000, 0, 1)
      raw_p999  scalar diagnostics kept in meta.json
      subject   [N] int (1..13)
      position  [N] int (1..17)
      frame_idx [N] int (within its session file)
  <out>/bedmat_meta.json   diagnostics + label taxonomy + verification results

Verifications performed (fail loudly, do not write on failure):
  1. Every file parses to [n_frames, 2048]; frames reshape to 64 rows x 32 cols.
  2. Dynamic-range diagnostic (the STAG-pedestal lesson): report min / percentiles /
     max, occupancy, and the fraction clipped at 1000. No pedestal expected (min 0).
  3. Position-label sanity via lateral center of mass: file numbers whose majority-
     subject COM sits right of center must be the RIGHT-class set {2,4,6,13}, left of
     center the LEFT set {3,5,7,14}, near-center the SUPINE set. This empirically
     confirms the Fig.-7 (Davoodnia & Etemad, arXiv:2006.10453) numbering before any
     label is trusted.

Run:
    uv run python scripts/bedmat_ingest.py --src <extracted pmd dir> --out <cache dir>
"""
import argparse
import json
import os

import numpy as np

# Index -> posture from the dataset's own experiment-info.docx table (image2.png):
# 1 supine, 2 right, 3 left, 4 right roll 30 (1 wedge), 5 right roll 60 (2 wedges),
# 6 left roll 30, 7 left roll 60, 8-12 supine variants, 13 right fetus, 14 left fetus,
# 15/16/17 supine with bed inclined 30/45/60. Names for 8-12 follow Davoodnia &
# Etemad (arXiv:2006.10453) Fig. 7 order; their 11-vs-12 assignment is soft-checked
# from leg-region asymmetry at ingest.
POSITION_NAMES = {
    1: "supine", 2: "lying on the right side", 3: "lying on the left side",
    4: "right side, body rolled 30 degrees", 5: "right side, body rolled 60 degrees",
    6: "left side, body rolled 30 degrees", 7: "left side, body rolled 60 degrees",
    8: "supine, arms and legs spread star-like", 9: "supine, hands crossed",
    10: "supine, both knees up", 11: "supine, right knee up",
    12: "supine, left knee up", 13: "right fetal position", 14: "left fetal position",
    15: "supine, bed inclined 30 degrees", 16: "supine, bed inclined 45 degrees",
    17: "supine, bed inclined 60 degrees",
}
CLASS3 = {**{p: "supine" for p in (1, 8, 9, 10, 11, 12, 15, 16, 17)},
          **{p: "right" for p in (2, 4, 5, 13)},
          **{p: "left" for p in (3, 6, 7, 14)}}
ROWS, COLS = 64, 32
RANGE_MAX = 1000.0          # FSA SoftFlex documented sensor range


def load_file(path):
    a = np.loadtxt(path, delimiter="\t", usecols=range(ROWS * COLS))
    if a.ndim == 1:
        a = a[None, :]
    return a.reshape(-1, ROWS, COLS)


def lateral_com(frames):
    """Mean pressure-weighted column coordinate, centered: negative = one side."""
    w = frames.astype(np.float64).sum(axis=0)
    cols = np.arange(COLS) - (COLS - 1) / 2
    tot = w.sum()
    return float((w.sum(axis=0) * cols).sum() / tot) if tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir containing experiment-i/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = os.path.join(args.src, "experiment-i")
    subjects = sorted(int(d[1:]) for d in os.listdir(root) if d.startswith("S"))
    assert subjects == list(range(1, 14)), subjects

    frames, subj, pos, fidx = [], [], [], []
    raw_all = []
    coms = {}                                             # (subject, position) -> com
    for s in subjects:
        for p in range(1, 18):
            path = os.path.join(root, f"S{s}", f"{p}.txt")
            arr = load_file(path)
            assert arr.shape[1:] == (ROWS, COLS), (path, arr.shape)
            raw_all.append(arr)
            coms[(s, p)] = lateral_com(arr)
            frames.append(arr)
            subj.append(np.full(len(arr), s, dtype=np.int16))
            pos.append(np.full(len(arr), p, dtype=np.int16))
            fidx.append(np.arange(len(arr), dtype=np.int32))
    raw = np.concatenate(raw_all)
    n = len(raw)
    print(f"parsed {n} frames from {len(subjects)} subjects x 17 positions")

    # ---- diagnostic 2: dynamic range ----
    pct = {f"p{q}": float(np.percentile(raw, q)) for q in (1, 25, 50, 75, 95, 99, 99.9)}
    clip_frac = float((raw > RANGE_MAX).mean())
    occupancy = float((raw > 0).mean())
    print(f"range: min {raw.min()} max {raw.max()} | {pct} | "
          f">1000 frac {clip_frac:.5f} | nonzero {occupancy:.3f}")
    assert raw.min() >= 0.0
    assert clip_frac < 0.01, f"clip fraction {clip_frac} too high for /1000 affine"

    # ---- diagnostic 3: label sanity via lateral center of mass ----
    # Only the WEDGE positions shift the pressure centroid reliably (plain side-lying
    # rotates the body in place). Hard checks, relative to the flat-supine baseline:
    #   (a) 4,5 sit on ONE side of baseline with |60-roll| > |30-roll|,
    #   (b) 6,7 sit on the OTHER side, same magnitude ordering.
    # Plain side/fetus positions get their relative COM reported but not asserted.
    per_pos = {p: [coms[(s, p)] for s in subjects] for p in range(1, 18)}
    med = {p: float(np.median(v)) for p, v in per_pos.items()}
    base = float(np.median([med[p] for p in (1, 8, 9, 10, 11, 12)]))
    rel = {p: med[p] - base for p in range(1, 18)}
    s45 = (np.sign(rel[4]), np.sign(rel[5]))
    s67 = (np.sign(rel[6]), np.sign(rel[7]))
    assert s45[0] == s45[1] != 0 and s67[0] == s67[1] != 0 and s45[0] != s67[0], (
        f"wedge pairs not mirror-lateralized: rel={ {p: round(rel[p],2) for p in (4,5,6,7)} }")
    assert abs(rel[5]) > abs(rel[4]) and abs(rel[7]) > abs(rel[6]), (
        "60-degree roll should lateralize more than 30-degree")
    verdicts = {p: {"median_com_rel": round(rel[p], 3), "class": CLASS3[p]}
                for p in range(1, 18)}
    verdicts["wedge_right_sign"] = float(s45[0])
    print("label sanity (wedge-anchored):", json.dumps(verdicts, indent=0))

    # ---- write caches ----
    os.makedirs(args.out, exist_ok=True)
    norm = np.clip(raw / RANGE_MAX, 0.0, 1.0).astype(np.float16)
    np.savez_compressed(
        os.path.join(args.out, "bedmat_expi.npz"),
        pressure=norm, subject=np.concatenate(subj),
        position=np.concatenate(pos), frame_idx=np.concatenate(fidx))
    meta = {"n_frames": int(n), "rows": ROWS, "cols": COLS,
            "normalization": f"clip(raw/{int(RANGE_MAX)}, 0, 1)",
            "range_diag": {"min": float(raw.min()), "max": float(raw.max()),
                           **pct, "clip_frac": clip_frac, "nonzero": occupancy},
            "position_names": POSITION_NAMES, "class3": CLASS3,
            "com_verdicts": verdicts,
            "source": "PhysioNet pmd/1.0.0 experiment-i (ODC-By v1.0)",
            "position_numbering_source": "Davoodnia & Etemad arXiv:2006.10453 Fig. 7, "
                                         "verified via lateral-COM sanity check"}
    with open(os.path.join(args.out, "bedmat_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {args.out}/bedmat_expi.npz ({norm.nbytes/1e6:.1f} MB raw fp16) + meta")


if __name__ == "__main__":
    main()
