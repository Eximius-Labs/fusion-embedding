"""Build the small demo data file shipped with Tactus Mat (for the notebook).

Held-out subjects only (11, 12, 13). Writes:
  gallery      [P,8,64,32] float16   one 8-frame window per shown posture
  gallery_pos  [P]                   position id (1..17)
  night        [T,64,32]  float16    one composed sequence for temporal queries
  night_pos    [T]                   the recorded position of each night frame

Run:
    python scripts/bedmat_make_demo_npz.py --cache <bedmat cache> --out <file.npz>
"""
import argparse
import json
import os

import numpy as np

GALLERY_POSITIONS = [1, 2, 3, 13, 14, 10, 8, 16]
NIGHT_ORDER = [1, 3, 14, 1, 2, 13, 10, 1]
NIGHT_SUBJECT = 13
GALLERY_SUBJECT = 13
K = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(os.path.join(args.cache, "bedmat_expi.npz"))
    meta = json.load(open(os.path.join(args.cache, "bedmat_meta.json")))
    press, subj, pos, fidx = z["pressure"], z["subject"], z["position"], z["frame_idx"]

    gal, gal_pos = [], []
    for p in GALLERY_POSITIONS:
        rows = np.where((subj == GALLERY_SUBJECT) & (pos == p))[0]
        rows = rows[np.argsort(fidx[rows])]
        idx = rows[np.linspace(0, len(rows) - 1, K).astype(int)]   # time-spread window
        gal.append(press[idx])
        gal_pos.append(p)

    night, night_pos = [], []
    for p in NIGHT_ORDER:
        rows = np.where((subj == NIGHT_SUBJECT) & (pos == p))[0]
        rows = rows[np.argsort(fidx[rows])]
        night.append(press[rows])
        night_pos.extend([p] * len(rows))

    out = dict(gallery=np.stack(gal).astype(np.float16),
               gallery_pos=np.array(gal_pos, dtype=np.int16),
               night=np.concatenate(night).astype(np.float16),
               night_pos=np.array(night_pos, dtype=np.int16),
               position_names=np.array([meta["position_names"][str(i)]
                                        for i in range(1, 18)]))
    np.savez_compressed(args.out, **out)
    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({mb:.2f} MB): gallery {out['gallery'].shape}, "
          f"night {out['night'].shape}")
    assert mb < 5, "demo file should stay small enough to fetch in a notebook"


if __name__ == "__main__":
    main()
