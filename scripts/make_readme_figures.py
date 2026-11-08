"""Summary figures for the README (reads the result tables, no GPU needed).

python scripts/make_readme_figures.py   ->  docs/figures/*.png
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

N_VAL = 5000

OTS = "Off-the-shelf (no training on our data)"

# display name, family
DOGCAT = {
    "M0_hog_svm": ("HOG + linear SVM", "Baselines"),
    "M1_simplecnn_noaug": ("Small CNN, no augmentation", "Baselines"),
    "M1_simplecnn": ("Small CNN", "Baselines"),
    "M2b_resnet50_svm": ("ResNet-50 features + SVM", "Frozen ImageNet features"),
    "M2c_resnet50_linear": ("ResNet-50 features + linear layer", "Frozen ImageNet features"),
    "M2a_resnet50_logreg": ("ResNet-50 features + log. regression", "Frozen ImageNet features"),
    "M3_resnet50_ft_noaug": ("ResNet-50 fine-tuned, no augmentation", "Fine-tuned"),
    "M3_resnet50_ft": ("ResNet-50 fine-tuned", "Fine-tuned"),
    "M4_convnext_tiny_ft": ("ConvNeXt-Tiny fine-tuned (our model)", "Final (submitted)"),
    "M5a_clip_vitl14_zeroshot": ("CLIP ViT-L/14 zero-shot", OTS),
    "M5b_clip_vitl14_logreg": ("CLIP ViT-L/14 features + log. regression", OTS),
}
FAMILY_COLOR = {
    "Baselines": MUTED,
    "Frozen ImageNet features": BLUE,
    "Fine-tuned": ORANGE,
    OTS: AQUA,
    "Final (submitted)": INK,
}

METHODS = {
    "ce": "Cross-entropy",
    "focal": "Focal loss",
    "resample": "Class-balanced sampling",
    "inv": "Inverse-frequency weights",
    "cb": "Class-balanced loss",
    "la": "Logit-adjusted loss",
}
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MINORITY = {"bird", "cat", "dog"}


def err_interval(n_wrong, n=N_VAL, reps=20000, seed=0):
    # percentile bootstrap over the val images (resampling correct/wrong flags)
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n, n_wrong / n, size=reps)
    return np.percentile(draws, [2.5, 97.5])


def dogcat_figure(root, out):
    df = pd.read_csv(root / "results/dogcat/comparison.csv")

    rows = []
    for _, r in df.iterrows():
        if r["model"] not in DOGCAT:
            continue
        name, fam = DOGCAT[r["model"]]
        seeds = [float(a) for a in str(r["val_acc_per_seed"]).split()]
        wrong = [round((1 - a) * N_VAL) for a in seeds]
        label = name + (f"  ({len(seeds)} seeds)" if len(seeds) > 1 else "")
        rows.append(dict(name=label, fam=fam, mean=np.mean(wrong), seeds=wrong,
                         ci=err_interval(round(np.mean(wrong)))))
    rows.sort(key=lambda d: (d["fam"] != "Final (submitted)", d["mean"]))  # list is drawn bottom-up: our model last row

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y = np.arange(len(rows))
    for i, d in enumerate(rows):
        c = FAMILY_COLOR[d["fam"]]
        lo, hi = np.maximum(d["ci"], 1.6)
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round", alpha=0.45, zorder=2)
        if len(d["seeds"]) > 1:
            ax.scatter(d["seeds"], [i] * len(d["seeds"]), s=22, facecolors="white",
                       edgecolors=c, linewidths=1.2, zorder=3)
        ax.scatter([d["mean"]], [i], s=70, color=c, edgecolors="white", linewidths=1.5, zorder=4)
        acc = 1 - d["mean"] / N_VAL
        right = max(hi, max(d["seeds"]))
        ax.text(right * 1.18, i, f"{d['mean']:.0f}  ({acc:.2%})", va="center", ha="left",
                fontsize=9, color=INK if d["fam"] == "Final (submitted)" else INK_2)

    ax.set_xscale("log")
    ax.set_xlim(1.5, 6000)
    ax.set_xticks([2, 5, 10, 30, 100, 300, 1000])
    ax.set_xticklabels(["2", "5", "10", "30", "100", "300", "1,000"])
    ax.set_yticks(y)
    ax.set_yticklabels([d["name"] for d in rows], color=INK)
    for t, d in zip(ax.get_yticklabels(), rows):
        if d["fam"] == "Final (submitted)":
            t.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("misclassified validation images (log scale)   \u2190 fewer is better")
    ax.set_title("Dogs vs. Cats: validation errors out of 5,000 images")

    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=f)
               for f, c in FAMILY_COLOR.items()]
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="white",
                          markeredgecolor=INK_2, markersize=5, label="single seed"))
    handles.append(Line2D([], [], color=INK_2, lw=2, alpha=0.45, label="95% bootstrap interval"))
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=9, labelcolor=INK_2)
    fig.savefig(out / "dogcat_val_errors.png")
    plt.close(fig)


def imbalance_figures(root, out):
    s = pd.read_csv(root / "results/cifar/imbalance/summary_test.csv")
    s = s[s["model"] == "resnet18"].set_index("method")
    pc = pd.read_csv(root / "results/cifar/imbalance/per_class_recall_test.csv")
    pc = pc[pc["model"] == "resnet18"].groupby("method")[CLASSES].mean()
    order = [m for m in METHODS if m in s.index]

    # change vs. cross-entropy: minority and majority recall
    others = [m for m in order if m != "ce"][::-1]
    ce = s.loc["ce"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    cols = [("minority_recall", "Minority classes (bird, cat, dog)"),
            ("majority_recall", "Majority classes (other 7)")]
    y = np.arange(len(others))
    for ax, (key, title) in zip(axes, cols):
        d = np.array([(s.loc[m][key + "_mean"] - ce[key + "_mean"]) * 100 for m in others])
        e = np.array([s.loc[m][key + "_std"] * 100 for m in others])
        colors = [BLUE if v >= 0 else ORANGE for v in d]
        ax.barh(y, d, height=0.62, color=colors, zorder=2)
        ax.errorbar(d, y, xerr=e, fmt="none", ecolor=INK_2, elinewidth=1, capsize=0, zorder=3)
        for yi, v, ei in zip(y, d, e):
            off = ei + 0.6
            ax.text(v + off if v >= 0 else v - off, yi, f"{v:+.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9, color=INK)
        ax.axvline(0, color=INK_2, lw=1, zorder=4)
        ax.set_xlim(-8, 16)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("recall change vs. cross-entropy (pp)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{METHODS[m]}  ({s.loc[m]['bal_acc_mean']:.1%})" for m in others],
                            color=INK)
    for t, m in zip(axes[0].get_yticklabels(), others):
        if m == "la":
            t.set_fontweight("bold")
    fig.suptitle("Imbalanced CIFAR-10 (ResNet-18, 3 seeds): recall change vs. plain cross-entropy",
                 x=0.0, ha="left", fontsize=11.5, fontweight="semibold", color=INK, y=1.11)
    fig.text(0.0, 1.0, f"In brackets: test balanced accuracy (cross-entropy: {ce['bal_acc_mean']:.1%}). "
             "Whiskers: std over seeds.", ha="left", fontsize=9, color=INK_2)
    fig.savefig(out / "imbalance_tradeoff.png")
    plt.close(fig)

    # per-class recall heatmap
    data = pc.loc[order, CLASSES].values
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    im = ax.imshow(data, cmap="Blues", vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.75 else INK)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([c + ("*" if c in MINORITY else "") for c in CLASSES], fontsize=8.5)
    for t, c in zip(ax.get_xticklabels(), CLASSES):
        if c in MINORITY:
            t.set_fontweight("bold")
            t.set_color(INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([METHODS[m] for m in order], color=INK)
    for t, m in zip(ax.get_yticklabels(), order):
        if m == "la":
            t.set_fontweight("bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(CLASSES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_2, length=0)
    cb.set_label("test recall", color=INK_2)
    ax.set_title("Per-class test recall (%) on imbalanced CIFAR-10, mean of 3 seeds   * = minority class",
                 fontsize=11)
    fig.savefig(out / "imbalance_per_class_recall.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    p.add_argument("--out", default="docs/figures")
    a = p.parse_args()
    t0 = time.time()
    out = a.root / a.out
    out.mkdir(parents=True, exist_ok=True)
    dogcat_figure(a.root, out)
    imbalance_figures(a.root, out)
    print(f"saved to {out}")
    print(f"total time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
