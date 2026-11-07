"""Evaluate the chosen CIFAR-10 models (all seeds) on the official test set, run ONCE after model selection.
--split val runs the same code on our val split (used only to check the script before the test run).
Usage: python scripts/eval_cifar.py --groups G1_resnet18_lr0.1_erase G2_convnext_tiny"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cifar import CIFAR_CLASSES, CifarDataset, get_cifar_transform, load_cifar, split_train_val
from src.data import loader_for
from src.engine import predict
from src.models import build_model
from src.utils import CIFAR_RESULTS, PROJECT_ROOT, Timer, write_csv

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
RUNS = os.path.join(CIFAR_RESULTS, "runs")
DISPLAY = {"G1_resnet18_lr0.2_erase": "ResNet-18 from scratch", "G2_convnext_tiny": "ConvNeXt-Tiny pretrained"}


def load_run(name):
    with open(os.path.join(RUNS, name, "run_info.json")) as f:
        info = json.load(f)
    return info, pd.read_csv(os.path.join(RUNS, name, "history.csv"))


def run_model(name, info, images, labels, device):
    model = build_model(info["model"], num_classes=10, pretrained=False, resize_to=info["img_size"] or None)
    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "cifar", f"{name}.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model = model.to(device).to(memory_format=torch.channels_last)
    tf = get_cifar_transform(False, imagenet_norm=info["model"] == "convnext_tiny")
    probs, _, loss = predict(model, loader_for(CifarDataset(images, labels, tf), 500, False, num_workers=8), device)
    return probs, loss


def bootstrap_ci(correct, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    accs = correct[rng.randint(0, len(correct), (n_boot, len(correct)))].mean(1)
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def image_grid(images, titles, path, suptitle, ncols=8):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 2.2 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, t in zip(axes.flat, images, titles):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(t, fontsize=8)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(main_preds, labels, S, seed, fig_dir):
    # row-normalised confusion matrices
    fig, axes = plt.subplots(1, len(main_preds), figsize=(8.5 * len(main_preds), 7.5), squeeze=False)
    for ax, (g, (pred, _)) in zip(axes[0], main_preds.items()):
        cm = confusion_matrix(labels, pred, labels=range(10), normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(10), CIFAR_CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(10), CIFAR_CLASSES)
        ax.set(xlabel="predicted", ylabel="true label",
               title=f"{DISPLAY.get(g, g)}, seed {seed} ({S} acc {(pred == labels).mean():.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"fig_cifar_confusion_{S}.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", required=True, help="run names without the _s<seed> suffix")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--out", default=CIFAR_RESULTS)
    parser.add_argument("--replot", action="store_true", help="redraw the confusion figure from saved preds")
    args = parser.parse_args()

    timer = Timer()
    device = "cuda"
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    if args.replot:
        main_preds = {}
        for g in args.groups:
            d = pd.read_csv(os.path.join(args.out, "runs", f"{g}_s{args.seeds[0]}", f"{args.split}_preds.csv"))
            main_preds[g] = (d.pred.values, None)
        plot_confusion(main_preds, d.label.values, args.split, args.seeds[0], fig_dir)
        print(f"Total time: {timer}")
        return
    if args.split == "test":
        images, labels = load_cifar(False)
    else:
        X, y = load_cifar(True)
        _, val_idx = split_train_val(y)
        images, labels = X[val_idx], y[val_idx]
    names = CIFAR_CLASSES
    S = args.split

    rows, main_preds, curves = [], {}, {}
    for g in args.groups:
        accs, val_accs, times = [], [], []
        for s in args.seeds:
            run = f"{g}_s{s}"
            if not os.path.exists(os.path.join(RUNS, run, "run_info.json")):
                print(f"missing {run}, skipped")
                continue
            info, hist = load_run(run)
            t0 = Timer()
            probs, loss = run_model(run, info, images, labels, device)
            pred = probs.argmax(1)
            acc = float((pred == labels).mean())
            print(f"{run}: {S} acc {acc:.4f} loss {loss:.4f} ({t0})", flush=True)
            write_csv(os.path.join(args.out, "runs", run, f"{S}_preds.csv"),
                      [dict(index=i, label=int(l), pred=int(p), prob=round(float(q.max()), 5))
                       for i, (l, p, q) in enumerate(zip(labels, pred, probs))])
            accs.append(acc)
            val_accs.append(info["best_val_acc"])
            times.append(info["train_seconds"])
            if s == args.seeds[0]:
                main_preds[g] = (pred, probs)
                curves[g] = hist
                params, size = info["params_total"], info["input_size"]
        lo, hi = bootstrap_ci((main_preds[g][0] == labels).astype(float))
        rows.append(dict(group=g, seeds=" ".join(str(s) for s in args.seeds[:len(accs)]), params=params,
                         input_size=size, sel_val_acc_mean=round(np.mean(val_accs), 5),
                         sel_val_acc_std=round(np.std(val_accs, ddof=1), 5) if len(accs) > 1 else None,
                         **{f"{S}_acc_mean": round(np.mean(accs), 5),
                            f"{S}_acc_std": round(np.std(accs, ddof=1), 5) if len(accs) > 1 else None,
                            f"{S}_acc_per_seed": " ".join(f"{a:.4f}" for a in accs),
                            f"{S}_ci95_seed{args.seeds[0]}": f"{lo:.4f}-{hi:.4f}"},
                         train_minutes_mean=round(np.mean(times) / 60, 1)))
    write_csv(os.path.join(args.out, f"results_{S}.csv"), rows)
    print(pd.DataFrame(rows).to_string())

    # per-class metrics and confused pairs (first seed of each group)
    per_class, pairs = [], []
    for g, (pred, _) in main_preds.items():
        prec, rec, f1, sup = precision_recall_fscore_support(labels, pred, labels=range(10), zero_division=0)
        for c in range(10):
            per_class.append(dict(group=g, cls=names[c], precision=round(prec[c], 4), recall=round(rec[c], 4),
                                  f1=round(f1[c], 4), support=int(sup[c])))
        cm = confusion_matrix(labels, pred, labels=range(10))
        for i in range(10):
            for j in range(i + 1, 10):
                pairs.append(dict(group=g, pair=f"{names[i]}<->{names[j]}", total=int(cm[i, j] + cm[j, i]),
                                  a_as_b=int(cm[i, j]), b_as_a=int(cm[j, i])))
    write_csv(os.path.join(args.out, f"per_class_{S}.csv"), per_class)  # recall = per-class accuracy
    pairs = pd.DataFrame(pairs).sort_values(["group", "total"], ascending=[True, False])
    pairs.groupby("group").head(10).to_csv(os.path.join(args.out, f"confused_pairs_{S}.csv"), index=False)
    print(pd.DataFrame(per_class).pivot(index="cls", columns="group", values="f1").to_string())
    print(pairs.groupby("group").head(5).to_string(index=False))

    plot_confusion(main_preds, labels, S, args.seeds[0], fig_dir)

    # most confident mistakes, and cat<->dog mistakes
    cat, dog = names.index("cat"), names.index("dog")
    for g, (pred, probs) in main_preds.items():
        wrong = np.where(pred != labels)[0]
        wrong = wrong[np.argsort(-probs[wrong].max(1))][:32]
        image_grid(images[wrong], [f"#{i} {names[labels[i]]}->{names[pred[i]]}\np={probs[i].max():.2f}" for i in wrong],
                   os.path.join(fig_dir, f"fig_cifar_errors_{g}_{S}.png"),
                   f"{g}_s{args.seeds[0]}: most confident {S} mistakes (true->pred)")
        cd = np.where(((labels == cat) & (pred == dog)) | ((labels == dog) & (pred == cat)))[0]
        cd = cd[np.argsort(-probs[cd].max(1))][:24]
        image_grid(images[cd], [f"#{i} {names[labels[i]]}->{names[pred[i]]}\np={probs[i].max():.2f}" for i in cd],
                   os.path.join(fig_dir, f"fig_cifar_catdog_{g}_{S}.png"),
                   f"{g}_s{args.seeds[0]}: cat<->dog {S} mistakes, most confident first")

    # learning curves (train/val history, does not depend on the split)
    fig, axes = plt.subplots(len(curves), 2, figsize=(13, 4.3 * len(curves)), squeeze=False)
    for row, g in zip(axes, curves):
        for s in args.seeds:
            if not os.path.exists(os.path.join(RUNS, f"{g}_s{s}", "history.csv")):
                continue
            h = load_run(f"{g}_s{s}")[1]
            line, = row[0].plot(h.epoch, h.train_loss, "--", alpha=0.8)
            row[0].plot(h.epoch, h.val_loss, "-", color=line.get_color(), label=f"seed {s}")
            row[1].plot(h.epoch, h.train_acc, "--", color=line.get_color(), alpha=0.8)
            row[1].plot(h.epoch, h.val_acc, "-", color=line.get_color(), label=f"seed {s}")
        row[0].set(xlabel="epoch", ylabel="cross-entropy loss", title=f"{g}: loss (dashed = train, solid = val)")
        row[1].set(xlabel="epoch", ylabel="accuracy", title=f"{g}: accuracy (dashed = train, solid = val)")
        for ax in row:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_cifar_curves.png"), dpi=200)
    plt.close(fig)
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
