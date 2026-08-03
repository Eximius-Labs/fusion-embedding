"""Analysis figures for the expanded Tactus paper, from the fsr_eval_dump artifacts.

Inputs (fetch first):
    uv run --env-file .env modal volume get fusion-data \
        fusion_perception/tactus_eval_dump.json papers/tactus_workshop/analysis/
    uv run --env-file .env modal volume get fusion-data \
        fusion_perception/tactus_eval_dump.npz papers/tactus_workshop/analysis/

Outputs (same dir as this script):
    fig_confusion.pdf   confusion matrix, row-normalized, brand palette
    fig_ksweep.pdf      accuracy vs distinct frames per window

Self-checks: confusion row sums == 1 for nonempty rows; k-sweep monotone axis labels.
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(HERE, "analysis")

# Eximius brand palette
NAVY = "#1F3A5F"
TERRA = "#C96F4A"
CREAM = "#F5EFE6"
SAND = "#D9CDBA"

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.edgecolor": NAVY, "text.color": NAVY,
                     "axes.labelcolor": NAVY, "xtick.color": NAVY,
                     "ytick.color": NAVY})


def main():
    d = json.load(open(os.path.join(AN, "tactus_eval_dump.json")))
    z = np.load(os.path.join(AN, "tactus_eval_dump.npz"), allow_pickle=True)
    conf = z["confusion"].astype(np.float64)
    names = [str(n).replace("_", " ") for n in z["class_names"]]
    C = len(names)

    # ---- fig_confusion: row-normalized ----
    rows = conf.sum(axis=1, keepdims=True)
    norm = np.divide(conf, rows, out=np.zeros_like(conf), where=rows > 0)
    nonempty = (rows[:, 0] > 0)
    assert np.allclose(norm[nonempty].sum(axis=1), 1.0), "row normalization broken"
    cmap = LinearSegmentedColormap.from_list("brand", [CREAM, TERRA, NAVY])
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels(names, rotation=90, fontsize=5.2)
    ax.set_yticklabels(names, fontsize=5.2)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(C):
        for j in range(C):
            if norm[i, j] >= 0.08 and i != j:
                ax.text(j, i, f"{norm[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=4.2, color=NAVY if norm[i, j] < 0.5 else CREAM)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout()
    p1 = os.path.join(HERE, "fig_confusion.pdf")
    fig.savefig(p1, bbox_inches="tight"); plt.close(fig)

    # ---- fig_ksweep ----
    ks = sorted(int(k) for k in d["k_sweep"])
    top1 = [d["k_sweep"][str(k)]["top1"] for k in ks]
    top3 = [d["k_sweep"][str(k)]["top3"] for k in ks]
    fig, ax = plt.subplots(figsize=(3.2, 2.3), dpi=200)
    ax.plot(ks, top1, "o-", color=NAVY, label="top-1", lw=1.6, ms=4)
    ax.plot(ks, top3, "s--", color=TERRA, label="top-3", lw=1.6, ms=4)
    ax.axhline(1 / 27, color=SAND, lw=1, ls=":")
    ax.text(ks[0], 1 / 27 + 0.015, "chance", fontsize=6, ha="left", color=NAVY)
    for k, v in zip(ks, top1):
        ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points", xytext=(0, -11),
                    fontsize=6, ha="center")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("distinct frames per grasp window")
    ax.set_ylabel("27-way accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p2 = os.path.join(HERE, "fig_ksweep.pdf")
    fig.savefig(p2, bbox_inches="tight"); plt.close(fig)

    print(f"wrote {p1}\nwrote {p2}")
    print("top confusions:", d["confusion_pairs_pre_center"][:5])
    print("rho:", d["spearman_confusion_vs_textcos"])


if __name__ == "__main__":
    main()
