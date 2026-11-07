"""Part h: evaluate the imbalanced-CIFAR runs on the official test set (run once after the G2 runs are done).
--split val gives the same tables on the balanced val split (used for model selection and to check the script)."""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_cifar import RUNS, load_run, run_model
from src.cifar import CIFAR_CLASSES, load_cifar, split_train_val
from src.utils import CIFAR_RESULTS, Timer, write_csv

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
METHODS = ["ce", "cb", "inv", "resample", "la", "focal"]
LABELS = {"ce": "B0 CE", "cb": "H1 class-balanced (beta 0.999)", "inv": "H1 inverse frequency",
          "resample": "H2 re-sampling", "la": "H3 logit adjustment", "focal": "H4 focal (gamma 2)"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed categorical order
MINORITY = [CIFAR_CLASSES.index(c) for c in ("bird", "cat", "dog")]
MAJORITY = [c for c in range(10) if c not in MINORITY]


def metrics(labels, pred):
    rec = recall_score(labels, pred, labels=range(10), average=None, zero_division=0)
    return dict(acc=(pred == labels).mean(), bal_acc=rec.mean(),
                macro_f1=f1_score(labels, pred, labels=range(10), average="macro", zero_division=0),
                minority_recall=rec[MINORITY].mean(), majority_recall=rec[MAJORITY].mean()), rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--out", default=os.path.join(CIFAR_RESULTS, "imbalance"))
    args = parser.parse_args()

    timer = Timer()
    S = args.split
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    if S == "test":
        images, labels = load_cifar(False)
    else:
        X, y = load_cifar(True)
        _, val_idx = split_train_val(y)
        images, labels = X[val_idx], y[val_idx]

    rows, recalls, preds = [], {}, {}
    for f in sorted(glob.glob(os.path.join(RUNS, "H_*", "run_info.json"))):
        run = os.path.basename(os.path.dirname(f))
        info, _ = load_run(run)
        probs, _ = run_model(run, info, images, labels, "cuda")
        pred = probs.argmax(1)
        m, rec = metrics(labels, pred)
        model = "resnet18" if info["model"] == "resnet18_cifar" else "convnext_tiny"
        rows.append(dict(run=run, model=model, method=info["method"], seed=info["seed"],
                         **{k: round(float(v), 5) for k, v in m.items()},
                         train_minutes=round(info["train_seconds"] / 60, 1)))
        recalls[(model, info["method"], info["seed"])] = rec
        preds[(model, info["method"], info["seed"])] = pred
        print(f"{run}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"runs_{S}.csv"), index=False)

    # mean / std over seeds, change vs CE baseline of the same model
    keys = ["acc", "bal_acc", "macro_f1", "minority_recall", "majority_recall"]
    summary = []
    for (model, method), d in df.groupby(["model", "method"], sort=False):
        r = dict(model=model, method=method, seeds=" ".join(map(str, sorted(d.seed))))
        for k in keys:
            r[f"{k}_mean"] = round(d[k].mean(), 5)
            r[f"{k}_std"] = round(d[k].std(ddof=1), 5) if len(d) > 1 else None
        summary.append(r)
    summary = pd.DataFrame(summary)
    for k in ["minority_recall", "majority_recall", "bal_acc"]:
        base = summary[summary.method == "ce"].set_index("model")[f"{k}_mean"]
        summary[f"{k}_vs_ce"] = (summary[f"{k}_mean"] - summary.model.map(base)).round(5)
    order = {m: i for i, m in enumerate(METHODS)}
    summary = summary.sort_values(["model", "method"], key=lambda c: c.map(order) if c.name == "method" else c)
    summary.to_csv(os.path.join(args.out, f"summary_{S}.csv"), index=False)
    print(summary.to_string(index=False))

    rec_rows = []
    for (model, method, seed), rec in recalls.items():
        rec_rows.append(dict(model=model, method=method, seed=seed, **{c: round(float(v), 4)
                                                                     for c, v in zip(CIFAR_CLASSES, rec)}))
    write_csv(os.path.join(args.out, f"per_class_recall_{S}.csv"), rec_rows)

    # best ResNet-18 method = highest mean balanced val acc (val is balanced, so val acc = balanced acc)
    sel = pd.read_csv(os.path.join(args.out, "summary_val.csv")) if S == "test" else summary
    sel = sel[(sel.model == "resnet18") & (sel.method != "ce")]
    best = sel.sort_values("bal_acc_mean", ascending=False).method.iloc[0]
    print(f"best method on val: {best}")

    # per-class recall, ResNet-18, mean over seeds with std error bars
    fig, ax = plt.subplots(figsize=(15, 5.5))
    width = 0.8 / len(METHODS)
    for c in MINORITY:
        ax.axvspan(c - 0.45, c + 0.45, color="#f0efec", zorder=0)
    for i, m in enumerate(METHODS):
        r = np.array([rec for (mo, me, _), rec in recalls.items() if mo == "resnet18" and me == m])
        if not len(r):
            continue
        ax.bar(np.arange(10) - 0.4 + width * (i + 0.5), r.mean(0), width * 0.9, yerr=r.std(0, ddof=1),
               color=COLORS[i], label=LABELS[m], error_kw=dict(lw=1, ecolor="#555"), zorder=2)
    ax.set_xticks(range(10), [c + (" *" if i in MINORITY else "") for i, c in enumerate(CIFAR_CLASSES)])
    ax.set(ylabel=f"{S} recall", ylim=(0, 1.14), yticks=np.arange(0, 1.01, 0.2),
           title=f"Per-class {S} recall, ResNet-18 on imbalanced CIFAR-10 (mean of seeds 42/43/44, "
                 f"* = minority class, 225 train images)")
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.legend(fontsize=9, ncol=6, loc="upper center")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"fig_imb_per_class_recall_{S}.png"), dpi=200)
    plt.close(fig)

    # confusion matrices: CE vs best method (seed 42), plus the ConvNeXt pair if present
    pairs = [("resnet18", "ce"), ("resnet18", best)]
    pairs += [p for p in [("convnext_tiny", "ce"), ("convnext_tiny", best)] if (p[0], p[1], 42) in preds]
    ncols = 2
    nrows = len(pairs) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 7.5 * nrows), squeeze=False)
    for ax, (model, method) in zip(axes.flat, pairs):
        pred = preds[(model, method, 42)]
        cm = confusion_matrix(labels, pred, labels=range(10), normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > 0.5 else "black")
        names = [c + (" *" if i in MINORITY else "") for i, c in enumerate(CIFAR_CLASSES)]
        ax.set_xticks(range(10), names, rotation=45, ha="right")
        ax.set_yticks(range(10), names)
        m, _ = metrics(labels, pred)
        ax.set(xlabel="predicted", ylabel="true label",
               title=f"{model}, {LABELS[method]}, s42\n{S} acc {m['acc']:.4f}, balanced acc {m['bal_acc']:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"fig_imb_confusion_{S}.png"), dpi=200)
    plt.close(fig)

    # train class distribution (from any run_info)
    counts = json.load(open(glob.glob(os.path.join(RUNS, "H_*", "run_info.json"))[0]))["train_counts"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(10)
    ax.bar(x - 0.2, counts, 0.38, color=COLORS[0], label="train (imbalanced)", zorder=2)
    ax.bar(x + 0.2, [500] * 10, 0.38, color=COLORS[1], label="val (balanced)", zorder=2)
    for i, n in enumerate(counts):
        if i in MINORITY:
            ax.text(i - 0.2, n + 60, str(n), ha="center", fontsize=9)
    ax.set_xticks(x, CIFAR_CLASSES, rotation=30, ha="right")
    ax.set(ylabel="images per class", ylim=(0, 5400),
           title=f"Part h data: {sum(counts)} train images, bird/cat/dog kept at 5% (ratio 20:1); test 1000/class")
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.legend(fontsize=10, ncol=2, loc="upper center")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_imb_distribution.png"), dpi=200)
    plt.close(fig)
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
