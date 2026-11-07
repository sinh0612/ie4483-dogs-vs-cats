"""Part g: per-class F1 on test, mean of 3 seeds, from the saved test_preds.csv (no model is run)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cifar import CIFAR_CLASSES
from src.utils import CIFAR_RESULTS, Timer

MODELS = [("G1_resnet18_lr0.2_erase", "ResNet-18 from scratch", "#2a78d6"),
          ("G2_convnext_tiny", "ConvNeXt-Tiny pretrained", "#eb6834")]
SEEDS = [42, 43, 44]


def main():
    timer = Timer()
    rows = []
    for g, name, _ in MODELS:
        f1 = []
        for s in SEEDS:
            d = pd.read_csv(os.path.join(CIFAR_RESULTS, "runs", f"{g}_s{s}", "test_preds.csv"))
            f1.append(f1_score(d.label, d.pred, labels=range(10), average=None))
        f1 = np.array(f1)
        for c, cls in enumerate(CIFAR_CLASSES):
            rows.append(dict(model=name, cls=cls, f1_mean=round(f1[:, c].mean(), 4), f1_std=round(f1[:, c].std(ddof=1), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CIFAR_RESULTS, "per_class_f1_seeds_test.csv"), index=False)
    t = df.pivot(index="cls", columns="model", values="f1_mean").sort_values(MODELS[0][1])
    print(t.to_string())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(t))
    ax.hlines(y, t[MODELS[0][1]], t[MODELS[1][1]], color="#c8c7c2", lw=2, zorder=1)
    for _, name, color in MODELS:
        ax.scatter(t[name], y, s=60, color=color, label=name, zorder=2, edgecolors="white", linewidths=1)
    ax.set_yticks(y, t.index)
    ax.set_xlabel("test F1 (mean of seeds 42/43/44)")
    ax.set_title("CIFAR-10 per-class F1, sorted by ResNet-18 (worst at the bottom)", pad=30)
    ax.set_xlim(0.85, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(CIFAR_RESULTS, "figures", "fig_cifar_per_class.png"), dpi=200)
    plt.close(fig)
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
