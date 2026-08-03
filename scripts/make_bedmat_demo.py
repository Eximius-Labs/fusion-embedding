"""Render the bed-mat demo artifacts from bedmat_demo_dump_<tag>.npz.

Inputs (local):
    --cache  dir with bedmat_expi.npz + bedmat_meta.json (from bedmat_ingest)
    --dump   bedmat_demo_dump_<tag>.npz fetched from the volume
    --out    output dir

Outputs:
    bedmat_hero.gif        composed sequence: heatmap left, live ranked queries right
    bedmat_gallery.png     8 real held-out frames + top-3 ranked posture queries
    bedmat_timeline.png    composed-night 3-class timeline + "last repositioned" answer

The "night" is a DISCLOSED composition: real recordings from one held-out subject,
sessions concatenated in a fixed order; every frame and every score is real model
output on real mat data. Captions state this.
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

NAVY = "#1F3A5F"
TERRA = "#C96F4A"
CREAM = "#F5EFE6"
SAND = "#D9CDBA"
CLS_COLORS = {"supine": NAVY, "right": TERRA, "left": "#7A9E7E"}

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": NAVY,
                     "axes.edgecolor": NAVY})

# Composed-night session order (position indices): believable posture sequence.
NIGHT_ORDER = [1, 3, 14, 1, 2, 13, 10, 1]
CLS_NAMES = ("supine", "right", "left")
CLS_LABELS = {"supine": "lying on their back", "right": "on the right side",
              "left": "on the left side"}


def load(args):
    z = np.load(os.path.join(args.cache, "bedmat_expi.npz"))
    meta = json.load(open(os.path.join(args.cache, "bedmat_meta.json")))
    d = np.load(args.dump)
    pos_names = {int(k): v for k, v in meta["position_names"].items()}
    class3 = {int(k): v for k, v in meta["class3"].items()}
    return z, meta, d, pos_names, class3


def night_rows(d, subject_id):
    """Row indices (into dump arrays) for the composed night of one subject."""
    rows = []
    bounds = []
    for p in NIGHT_ORDER:
        m = np.where((d["subject"] == subject_id) & (d["position"] == p))[0]
        m = m[np.argsort(d["frame_idx"][m])]
        rows.append(m)
        bounds.append(len(m))
    return np.concatenate(rows), np.array(bounds)


def draw_frame(ax_hm, ax_tx, frame, sims3, sims17, pos_names, t_s):
    ax_hm.clear(); ax_tx.clear()
    ax_hm.imshow(frame, cmap="magma", vmin=0, vmax=0.6, aspect="auto")
    ax_hm.set_xticks([]); ax_hm.set_yticks([])
    ax_hm.set_title(f"bed pressure mat (64x32)   t={t_s}s", fontsize=9, color=NAVY)
    ax_tx.set_xlim(0, 1); ax_tx.set_ylim(0, 1); ax_tx.axis("off")
    order3 = np.argsort(-sims3)
    y = 0.92
    ax_tx.text(0, y, "text queries, ranked:", fontsize=9, color=NAVY, weight="bold")
    y -= 0.14
    track = 0.70                                   # label track width
    for j, c in enumerate(order3):
        s = float(sims3[c])
        fill = float(np.clip((s + 0.6) / 1.2, 0.02, 1.0)) * track
        ax_tx.barh(y, track, height=0.085, color="#EDE4D6", align="center", left=0.0)
        ax_tx.barh(y, fill, height=0.085, color=TERRA if j == 0 else SAND,
                   align="center", left=0.0)
        ax_tx.text(0.015, y, f'"a person {CLS_LABELS[CLS_NAMES[c]]}"',
                   fontsize=8.5, va="center",
                   color=CREAM if j == 0 else NAVY,
                   weight="bold" if j == 0 else "normal")
        ax_tx.text(0.99, y, f"{s:+.2f}", fontsize=7.5, va="center", ha="right",
                   color=NAVY)
        y -= 0.115
    y -= 0.06
    top17 = int(np.argmax(sims17))
    ax_tx.text(0, y, "best fine-grained match:", fontsize=8, color=NAVY)
    ax_tx.text(0, y - 0.1, f'"{pos_names[top17 + 1]}"', fontsize=8.5, color=TERRA,
               style="italic")


def make_hero_gif(args, z, d, pos_names, class3):
    subject_id = int(np.unique(d["subject"])[0])
    rows, bounds = night_rows(d, subject_id)
    step = max(1, len(rows) // 36)                      # ~36 animation frames
    sel = rows[::step]
    fig = plt.figure(figsize=(7.2, 3.4), dpi=110, facecolor=CREAM)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.05)
    ax_hm, ax_tx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    images = []
    from PIL import Image
    for i, r in enumerate(sel):
        frame = z["pressure"][d["rows"][r]].astype(np.float32)
        draw_frame(ax_hm, ax_tx, frame, d["sims3"][r].astype(np.float32),
                   d["sims17"][r].astype(np.float32), pos_names, int(i * step))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        images.append(Image.fromarray(buf))
    p = os.path.join(args.out, "bedmat_hero.gif")
    images[0].save(p, save_all=True, append_images=images[1:], duration=350, loop=0)
    plt.close(fig)
    print("wrote", p, f"({len(images)} frames)")


def make_gallery(args, z, d, pos_names, class3):
    # selection policy: per shown position, the held-out frame with the highest
    # correct-query margin (stated in caption)
    show = [1, 2, 3, 13, 14, 10, 16, 8]
    fig, axes = plt.subplots(2, 8, figsize=(13, 4.6), dpi=150,
                             gridspec_kw={"height_ratios": [2.2, 1]},
                             facecolor="white")
    for col, p in enumerate(show):
        m = np.where(d["position"] == p)[0]
        s17 = d["sims17"][m].astype(np.float32)
        margin = s17[:, p - 1] - np.max(
            np.delete(s17, p - 1, axis=1), axis=1)
        r = m[int(np.argmax(margin))]
        frame = z["pressure"][d["rows"][r]].astype(np.float32)
        ax = axes[0, col]
        ax.imshow(frame, cmap="magma", vmin=0, vmax=0.6, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'"{pos_names[p]}"', fontsize=6.5, color=NAVY, wrap=True)
        axb = axes[1, col]
        order = np.argsort(-s17[int(np.argmax(margin))])[:3]
        vals = s17[int(np.argmax(margin))][order]
        axb.barh(range(3)[::-1], np.clip(vals + 0.2, 0.02, None),
                 color=[TERRA if o == p - 1 else SAND for o in order])
        for j, (o, v) in enumerate(zip(order, vals)):
            axb.text(0.01, 2 - j, pos_names[o + 1][:26], fontsize=5.2, va="center",
                     color=NAVY)
        axb.set_xticks([]); axb.set_yticks([])
        for s in axb.spines.values():
            s.set_visible(False)
    fig.suptitle("Held-out subjects, real mat frames: top-3 ranked text queries "
                 "(orange = ground truth). Selection: highest-margin frame per posture.",
                 fontsize=8.5, color=NAVY)
    p = os.path.join(args.out, "bedmat_gallery.png")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def smooth_pred(pred, w=9):
    """Majority filter over a window of w frames."""
    out = pred.copy()
    for i in range(len(pred)):
        a, b = max(0, i - w // 2), min(len(pred), i + w // 2 + 1)
        vals, counts = np.unique(pred[a:b], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def make_timeline(args, z, d, pos_names, class3):
    subject_id = int(np.unique(d["subject"])[0])
    rows, bounds = night_rows(d, subject_id)
    pred3 = smooth_pred(d["sims3"][rows].astype(np.float32).argmax(1))
    true3 = np.array([CLS_NAMES.index(class3[int(p)])
                      for p in d["position"][rows]])
    t = np.arange(len(rows))                             # 1 Hz -> seconds
    agree = float((pred3 == true3).mean())
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 3.6), dpi=150,
                                  sharex=True, facecolor="white",
                                  gridspec_kw={"height_ratios": [1, 1],
                                               "hspace": 0.5,
                                               "top": 0.74, "bottom": 0.22})
    for axis, series, name in ((ax, pred3, "model\n(text queries)"),
                               (ax2, true3, "recorded\nposture")):
        for c in range(3):
            m = series == c
            axis.fill_between(t, 0, 1, where=m, step="mid",
                              color=CLS_COLORS[CLS_NAMES[c]], alpha=0.9)
        axis.set_yticks([]); axis.set_ylim(0, 1)
        axis.set_ylabel(name, fontsize=7.5, color=NAVY, rotation=0,
                        ha="right", va="center", labelpad=8)
        axis.set_xlim(0, t[-1])
    ax2.set_xlabel("time (s), 1 Hz mat frames", fontsize=7.5, color=NAVY, labelpad=2)
    # "when did the patient last reposition?" = last smoothed class change
    ch = np.where(np.diff(pred3) != 0)[0]
    if len(ch):
        last = int(ch[-1]) + 1
        for axis in (ax, ax2):
            axis.axvline(last, color=NAVY, lw=1.1, ls="--")
        ax.annotate(f'"when did they last move?"  ->  t = {last} s,'
                    f' {CLS_LABELS[CLS_NAMES[pred3[-1]]]} since',
                    xy=(last, 1.02), xycoords=("data", "axes fraction"),
                    xytext=(0.03, 1.34), textcoords="axes fraction",
                    fontsize=8, color=NAVY,
                    arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.9,
                                    connectionstyle="arc3,rad=-0.15"))
    import matplotlib.patches as mpatches
    fig.legend(handles=[mpatches.Patch(color=CLS_COLORS[c], label=CLS_LABELS[c])
                        for c in CLS_NAMES],
               loc="lower center", ncol=3, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.955, "A night on the mat, answered in language",
             ha="center", fontsize=10.5, color=NAVY, weight="bold")
    fig.text(0.5, 0.905,
             f"held-out subject, sessions concatenated; per-frame agreement with the "
             f"recorded posture: {agree:.0%}", ha="center", fontsize=7.5, color=NAVY)
    p = os.path.join(args.out, "bedmat_timeline.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p, f"agreement {agree:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    z, meta, d, pos_names, class3 = load(args)
    make_hero_gif(args, z, d, pos_names, class3)
    make_gallery(args, z, d, pos_names, class3)
    make_timeline(args, z, d, pos_names, class3)


if __name__ == "__main__":
    main()
