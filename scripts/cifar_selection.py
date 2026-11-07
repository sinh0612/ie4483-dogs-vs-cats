"""Val-only summary of the part g CIFAR-10 runs (lr sweeps, augmentation variants, seeds) and a figure of the G1 variants."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import CIFAR_RESULTS, Timer, write_csv

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})


def main():
    timer = Timer()
    rows = []
    for f in sorted(glob.glob(os.path.join(CIFAR_RESULTS, "runs", "*", "run_info.json"))):
        if os.path.basename(os.path.dirname(f)).startswith("H_"):  # part h runs: eval_imbalance.py
            continue
        d = json.load(open(f))
        rows.append(dict(run=d["run_name"], model=d["model"], optimizer=d["optimizer"], lr=d["lr"],
                         epochs=d["epochs"], n_train=d["n_train"], erase=d["erase"], no_aug=d["no_aug"],
                         seed=d["seed"], best_epoch=d["best_epoch"], best_val_acc=d["best_val_acc"],
                         best_val_loss=d["best_val_loss"], last_val_acc=d["last_val_acc"],
                         train_acc_at_best=d["train_acc_at_best"], train_minutes=round(d["train_seconds"] / 60, 1)))
    write_csv(os.path.join(CIFAR_RESULTS, "selection_val.csv"), rows)
    df = pd.DataFrame(rows)
    print(df.drop(columns=["model", "optimizer"]).to_string(index=False))

    # configs with several seeds: mean / std of best val acc
    df["config"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    groups = df.groupby("config").filter(lambda d: len(d) > 1).groupby("config")
    seeds = groups.agg(seeds=("seed", lambda x: " ".join(map(str, sorted(x)))),
                       val_acc_mean=("best_val_acc", "mean"), val_acc_std=("best_val_acc", "std"),
                       train_minutes_mean=("train_minutes", "mean")).round(5).reset_index()
    seeds.to_csv(os.path.join(CIFAR_RESULTS, "selection_val_seeds.csv"), index=False)
    print(seeds.to_string(index=False))

    g1 = [r["run"] for r in rows if r["run"].startswith("G1_") and r["seed"] == 42]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for run in g1:
        h = pd.read_csv(os.path.join(CIFAR_RESULTS, "runs", run, "history.csv"))
        label = run.replace("G1_resnet18_", "").replace("_s42", "")
        line, = axes[0].plot(h.epoch, h.val_acc, label=label)
        axes[1].plot(h.epoch, h.train_acc - h.val_acc, color=line.get_color(), label=label)
    axes[0].set(xlabel="epoch", ylabel="val accuracy", title="G1 ResNet-18 variants (seed 42): val accuracy",
                ylim=(0.8, 0.97))
    axes[1].set(xlabel="epoch", ylabel="train acc - val acc", title="Train-val gap (train acc measured with aug)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.join(CIFAR_RESULTS, "figures"), exist_ok=True)
    plt.savefig(os.path.join(CIFAR_RESULTS, "figures", "fig_cifar_g1_variants.png"), dpi=200)
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
