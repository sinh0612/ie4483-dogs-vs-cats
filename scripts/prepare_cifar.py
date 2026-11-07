"""Download CIFAR-10, make the 45k/5k class-balanced train/val split, write dataset_info.md and a sample figure."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cifar import CIFAR_CLASSES, load_cifar, split_train_val
from src.utils import CIFAR_RESULTS, Timer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=8, help="images per class in the sample figure")
    args = parser.parse_args()

    timer = Timer()
    os.makedirs(os.path.join(CIFAR_RESULTS, "figures"), exist_ok=True)
    X, y = load_cifar(True, download=True)
    Xt, yt = load_cifar(False, download=True)
    train_idx, val_idx = split_train_val(y, seed=args.seed)
    np.savetxt(os.path.join(CIFAR_RESULTS, "val_indices.txt"), val_idx, fmt="%d")

    fmt = lambda v: "(" + ", ".join(f"{x:.4f}" for x in v) + ")"
    counts = lambda labels: [int((labels == c).sum()) for c in range(10)]
    tr, va, te = counts(y[train_idx]), counts(y[val_idx]), counts(yt)
    lines = ["# CIFAR-10 dataset", "",
             "Source: torchvision.datasets.CIFAR10 (Krizhevsky, 2009), stored in data/cifar10.", "",
             f"- Image size: {X.shape[1]}x{X.shape[2]} RGB, uint8",
             f"- Original train set: {len(y)} images, official test set: {len(yt)} images",
             f"- Our split of the original train set (seed {args.seed}, 500 val images per class): "
             f"train {len(train_idx)}, val {len(val_idx)}",
             "- Val indices (into the original train set): results/cifar/val_indices.txt",
             "- The official test set is used only once, for the final evaluation.", "",
             "| id | class | train | val | test |", "|---|---|---|---|---|"]
    lines += [f"| {c} | {name} | {tr[c]} | {va[c]} | {te[c]} |" for c, name in enumerate(CIFAR_CLASSES)]
    lines += [f"| | total | {sum(tr)} | {sum(va)} | {sum(te)} |", "",
              "Per-channel mean / std used for normalisation in G1 (standard values over the 50k original "
              "train images) and the same statistics over our 45k train split only:",
              f"- 50k: mean {fmt(X.reshape(-1, 3).mean(0) / 255)}, std {fmt((X.reshape(-1, 3) / 255).std(0))}",
              f"- 45k: mean {fmt(X[train_idx].reshape(-1, 3).mean(0) / 255)}, "
              f"std {fmt((X[train_idx].reshape(-1, 3) / 255).std(0))}", "",
              "Sample images: figures/fig_cifar_samples.png"]
    with open(os.path.join(CIFAR_RESULTS, "dataset_info.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    rng = np.random.RandomState(args.seed)
    k = args.per_class
    fig, axes = plt.subplots(10, k, figsize=(k * 1.1, 10 * 1.15))
    for c, name in enumerate(CIFAR_CLASSES):
        pick = rng.choice(train_idx[y[train_idx] == c], k, replace=False)
        for j, i in enumerate(pick):
            ax = axes[c, j]
            ax.imshow(X[i])
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=10)
    fig.suptitle("CIFAR-10 training images (32x32)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(CIFAR_RESULTS, "figures", "fig_cifar_samples.png"), dpi=200)
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
