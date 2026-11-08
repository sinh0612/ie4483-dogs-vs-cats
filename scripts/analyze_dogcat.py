"""Collect all dogs-vs-cats runs: comparison table, bootstrap CIs, McNemar tests, TTA/ensemble selection on val,
submission.csv and figures."""
import glob
import json
import os
import re
import shutil
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import CACHE_ROOT, DOGCAT_RESULTS, PROJECT_ROOT, Timer

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
RUNS = os.path.join(DOGCAT_RESULTS, "runs")
FIG = os.path.join(DOGCAT_RESULTS, "figures")

# group name -> short description (order = comparison chain)
GROUPS = {
    "M0_hog_svm": "HOG (gray 128x128) + LinearSVC",
    "M1_simplecnn_noaug": "SimpleCNN from scratch, no augmentation",
    "M1_simplecnn": "SimpleCNN from scratch",
    "M2b_resnet50_svm": "ResNet-50 frozen features + LinearSVC",
    "M2a_resnet50_logreg": "ResNet-50 frozen features + LogisticRegression",
    "M2c_resnet50_linear": "ResNet-50 frozen features + linear layer (PyTorch)",
    "M3_resnet50_ft_noaug": "ResNet-50 full fine-tune, no augmentation",
    "M3_resnet50_ft": "ResNet-50 full fine-tune",
    "M4_convnext_tiny_ft": "ConvNeXt-Tiny full fine-tune",
    "M5a_clip_vitl14_zeroshot": "CLIP ViT-L/14 zero-shot (no training)",
    "M5b_clip_vitl14_logreg": "CLIP ViT-L/14 frozen features + LogisticRegression",
}
# models whose prob_dog is a real probability (SVM scores are not), usable for ensembles
FINAL = "M4_convnext_tiny_ft"
PROB_OK = {"M1_simplecnn", "M1_simplecnn_noaug", "M2a_resnet50_logreg", "M2c_resnet50_linear",
           "M3_resnet50_ft", "M3_resnet50_ft_noaug", "M4_convnext_tiny_ft", "M5a_clip_vitl14_zeroshot",
           "M5b_clip_vitl14_logreg"}
# (change, from, to) for question f
CHAIN = [
    ("hand-crafted HOG+SVM -> CNN learned from scratch", "M0_hog_svm", "M1_simplecnn"),
    ("augmentation on (scratch CNN)", "M1_simplecnn_noaug", "M1_simplecnn"),
    ("scratch CNN -> frozen ImageNet ResNet-50 + linear layer", "M1_simplecnn", "M2c_resnet50_linear"),
    ("frozen features: SVM -> logistic regression", "M2b_resnet50_svm", "M2a_resnet50_logreg"),
    ("frozen features: logistic regression -> linear layer (PyTorch)", "M2a_resnet50_logreg", "M2c_resnet50_linear"),
    ("frozen ResNet-50 -> full fine-tuning", "M2c_resnet50_linear", "M3_resnet50_ft"),
    ("augmentation on (fine-tuned ResNet-50)", "M3_resnet50_ft_noaug", "M3_resnet50_ft"),
    ("ResNet-50 -> ConvNeXt-Tiny (both fine-tuned)", "M3_resnet50_ft", "M4_convnext_tiny_ft"),
    ("CLIP zero-shot -> CLIP linear probe", "M5a_clip_vitl14_zeroshot", "M5b_clip_vitl14_logreg"),
    ("fine-tuned ConvNeXt-Tiny -> CLIP ViT-L/14 linear probe", "M4_convnext_tiny_ft", "M5b_clip_vitl14_logreg"),
]


def bootstrap_ci(correct, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(1)
    return np.percentile(accs, [2.5, 97.5])


def mcnemar(correct_a, correct_b):
    # exact McNemar: binomial test on discordant pairs
    n01 = int((correct_a & ~correct_b).sum())  # A right, B wrong
    n10 = int((~correct_a & correct_b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def nll(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def load_runs():
    runs = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        name = os.path.basename(d)
        if name.startswith("sweep_") or not os.path.exists(os.path.join(d, "val_preds.csv")):
            continue
        m = re.match(r"(.*)_s(\d+)$", name)
        group, seed = (m.group(1), int(m.group(2))) if m else (name, 42)
        info = json.load(open(os.path.join(d, "run_info.json")))
        runs.setdefault(group, {})[seed] = dict(
            info=info, val=pd.read_csv(os.path.join(d, "val_preds.csv")),
            test=pd.read_csv(os.path.join(d, "test_preds.csv")),
            history=pd.read_csv(os.path.join(d, "history.csv")) if os.path.exists(os.path.join(d, "history.csv")) else None)
    return runs


def comparison_table(runs):
    rows = []
    for g in [g for g in GROUPS if g in runs]:
        seeds = sorted(runs[g])
        accs = [((runs[g][s]["val"].prob_dog > 0.5) == runs[g][s]["val"].label).mean() for s in seeds]
        r42 = runs[g][42]
        correct = ((r42["val"].prob_dog > 0.5) == r42["val"].label).values
        lo, hi = bootstrap_ci(correct)
        info = r42["info"]
        train_acc = info.get("train_acc_at_best")
        rows.append(dict(
            model=g, description=GROUPS[g],
            params_total=info.get("params_total", info.get("backbone_params")),
            params_trainable=info.get("params_trainable"),
            input_size=info.get("input_size"), epochs=info.get("epochs"), lr=info.get("lr"),
            C=info.get("C"), best_epoch=info.get("best_epoch"),
            feature_seconds=info.get("extract_seconds", info.get("feature_seconds")),
            train_seconds_mean=round(np.mean([runs[g][s]["info"].get("train_seconds") or 0 for s in seeds]), 1),
            n_train=info.get("n_train"), seeds=" ".join(map(str, seeds)), n_seeds=len(seeds),
            val_acc_mean=round(float(np.mean(accs)), 5),
            val_acc_std=round(float(np.std(accs, ddof=1)), 5) if len(seeds) > 1 else None,
            val_acc_per_seed=" ".join(f"{a:.4f}" for a in accs),
            val_acc_s42=round(float(correct.mean()), 5),
            ci95_low_s42=round(float(lo), 5), ci95_high_s42=round(float(hi), 5),
            n_wrong_s42=int((~correct).sum()),
            train_acc_s42=train_acc,
            train_val_gap_s42=round(train_acc - correct.mean(), 5) if train_acc is not None else None,
        ))
    return pd.DataFrame(rows)


def correct_vec(runs, g, col="prob_dog"):
    v = runs[g][42]["val"]
    return ((v[col] > 0.5) == v.label).values


def main():
    timer = Timer()
    os.makedirs(FIG, exist_ok=True)
    runs = load_runs()
    print("groups:", {g: sorted(r) for g, r in runs.items()})
    for g in runs:  # all runs must use the same val images in the same order
        assert (runs[g][42]["val"].path.values == runs["M0_hog_svm"][42]["val"].path.values).all(), g
    labels = runs["M0_hog_svm"][42]["val"].label.values
    val_paths = runs["M0_hog_svm"][42]["val"].path.values

    comp = comparison_table(runs)
    comp.to_csv(os.path.join(DOGCAT_RESULTS, "comparison.csv"), index=False)
    print(comp[["model", "n_seeds", "val_acc_mean", "val_acc_std", "ci95_low_s42", "ci95_high_s42", "n_wrong_s42"]].to_string(index=False))

    # McNemar: chain steps + all pairs among the top 5 (seed 42 runs)
    top = comp.sort_values("val_acc_mean", ascending=False).model.tolist()[:5]
    pairs = [(a, b, c) for c, a, b in CHAIN if a in runs and b in runs]
    pairs += [(a, b, "top-5 pair") for a, b in combinations(top, 2)]
    mc_rows = []
    acc_of = dict(zip(comp.model, comp.val_acc_s42))
    for a, b, why in pairs:
        n01, n10, p = mcnemar(correct_vec(runs, a), correct_vec(runs, b))
        mc_rows.append(dict(comparison=why, model_a=a, model_b=b, acc_a_s42=acc_of[a], acc_b_s42=acc_of[b],
                            delta_b_minus_a=round(acc_of[b] - acc_of[a], 5),
                            a_right_b_wrong=n01, a_wrong_b_right=n10, p_value=float(f"{p:.4g}"),
                            significant_0_05=p < 0.05))
    mc = pd.DataFrame(mc_rows)
    mc.to_csv(os.path.join(DOGCAT_RESULTS, "mcnemar.csv"), index=False)
    print(mc[["comparison", "model_a", "model_b", "delta_b_minus_a", "p_value"]].to_string(index=False))

    # final model: ConvNeXt-Tiny seed 42, trained on our data. Flip TTA on/off is chosen on its own val
    # predictions (higher val acc, ties -> lower val NLL)
    v = runs[FINAL][42]["val"]
    opts = [(tta, float(((v[c] > 0.5) == labels).mean()), nll(v[c].values, labels))
            for tta, c in ((False, "prob_dog"), (True, "prob_dog_tta"))]
    tta, final_acc, final_nll = sorted(opts, key=lambda o: (-o[1], o[2]))[0]
    col = "prob_dog_tta" if tta else "prob_dog"
    p_val_final = v[col].values
    p_test_final = runs[FINAL][42]["test"][col].values
    print(f"FINAL: {FINAL} s42 tta={tta} val_acc={final_acc} (options: {opts})")

    # side analysis only: ensembles of the top 3 models. Not used for the submission, because the best ones
    # include CLIP, which was never trained on our data.
    cands = [g for g in comp.sort_values("val_acc_mean", ascending=False).model if g in PROB_OK][:3]
    configs = []
    for k in (1, 2, 3):
        for members in combinations(cands, k):
            for t in (False, True):
                c = "prob_dog_tta" if t else "prob_dog"
                p = np.mean([runs[g][42]["val"][c].values for g in members], axis=0)
                configs.append(dict(members="+".join(members), n_models=k, tta=t,
                                    val_acc=round(float(((p > 0.5) == labels).mean()), 5),
                                    val_nll=round(nll(p, labels), 5), n_wrong=int(((p > 0.5) != labels).sum())))
    sel = pd.DataFrame(configs).sort_values(["val_acc", "val_nll", "n_models"], ascending=[False, True, True])
    sel.to_csv(os.path.join(DOGCAT_RESULTS, "final_selection.csv"), index=False)
    ens = sel.iloc[0]
    c = "prob_dog_tta" if ens.tta else "prob_dog"
    p_ens = np.mean([runs[g][42]["val"][c].values for g in ens.members.split("+")], axis=0)
    n01, n10, p_mc = mcnemar((p_val_final > 0.5) == labels, (p_ens > 0.5) == labels)
    print(f"best ensemble (not used): {ens.members} tta={ens.tta} val_acc={ens.val_acc}, McNemar vs final p={p_mc:.4g}")
    with open(os.path.join(DOGCAT_RESULTS, "final_config.json"), "w") as f:
        json.dump(dict(members=[FINAL], seed=42, tta=bool(tta), val_acc=round(final_acc, 5),
                       val_nll=round(final_nll, 5), n_wrong=int(((p_val_final > 0.5) != labels).sum()),
                       rule="final = ConvNeXt-Tiny seed 42 (our trained model); TTA chosen on val acc, ties -> lower NLL",
                       ensemble_not_used=dict(members=ens.members.split("+"), tta=bool(ens.tta),
                                              val_acc=float(ens.val_acc), n_wrong=int(ens.n_wrong),
                                              ensemble_right_final_wrong=n10, ensemble_wrong_final_right=n01,
                                              mcnemar_p=float(f"{p_mc:.4g}"),
                                              why="contains CLIP, which was not trained on our data")), f, indent=2)

    # test probabilities + submission
    test_ids = runs["M0_hog_svm"][42]["test"].id.values
    tp = pd.DataFrame({"id": test_ids})
    for g in [g for g in GROUPS if g in runs]:
        assert (runs[g][42]["test"].id.values == test_ids).all()
        tp[g] = runs[g][42]["test"].prob_dog.round(6)
    tp["final"] = np.round(p_test_final, 6)
    tp.to_csv(os.path.join(DOGCAT_RESULTS, "test_probs.csv"), index=False)
    sample = pd.read_csv(os.path.join(PROJECT_ROOT, "sampleSubmission.csv"))
    pred = dict(zip(test_ids, (p_test_final > 0.5).astype(int)))
    sub = pd.DataFrame({"id": sample.id, "label": [pred[i] for i in sample.id]})
    assert len(sub) == 500 and set(sub.label) <= {0, 1} and (sub.id.values == sample.id.values).all()
    sub.to_csv(os.path.join(DOGCAT_RESULTS, "submission.csv"), index=False)
    shutil.copy(os.path.join(DOGCAT_RESULTS, "submission.csv"), os.path.join(PROJECT_ROOT, "submission.csv"))
    print(f"submission: {len(sub)} rows, dogs={sub.label.sum()}, cats={(sub.label == 0).sum()}")

    make_figures(runs, comp, labels, val_paths, p_val_final, tp)
    print(f"Total time: {timer}")


def make_figures(runs, comp, labels, val_paths, p_val_final, tp):
    # learning curves: rows = loss / accuracy, one column per model family; dashed = train, solid = val
    panels = [("SimpleCNN from scratch", [("M1_simplecnn", 42, "with augmentation"),
                                          ("M1_simplecnn_noaug", 42, "no augmentation")]),
              ("ResNet-50 fine-tuned", [("M3_resnet50_ft", 42, "with augmentation"),
                                        ("M3_resnet50_ft_noaug", 42, "no augmentation")]),
              ("ConvNeXt-Tiny fine-tuned", [("M4_convnext_tiny_ft", s, f"seed {s}") for s in (42, 43, 44)])]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for col, (title, items) in enumerate(panels):
        items = [it for it in items if it[0] in runs and it[1] in runs[it[0]]]
        for (g, s, label), c in zip(items, colors):
            h = runs[g][s]["history"]
            axes[0, col].plot(h.epoch, h.train_loss, "--", color=c, lw=1.5)
            axes[0, col].plot(h.epoch, h.val_loss, "-", color=c, lw=2, label=label)
            axes[1, col].plot(h.epoch, h.train_acc, "--", color=c, lw=1.5)
            axes[1, col].plot(h.epoch, h.val_acc, "-", color=c, lw=2, label=label)
        axes[0, col].axhline(0.199, color="#8a8984", lw=0.8)
        axes[0, col].text(axes[0, col].get_xlim()[0] + 0.3, 0.192, "train-loss floor with label smoothing 0.1",
                          ha="left", va="top", fontsize=8.5, color="#52514e")
        axes[0, col].set_title(title)
        axes[0, col].legend(fontsize=9, title="solid = val, dashed = train", title_fontsize=8.5,
                            loc="upper right", frameon=False)
        axes[0, col].set(ylabel="cross-entropy loss" if col == 0 else None)
        axes[1, col].set(xlabel="epoch", ylabel="accuracy" if col == 0 else None)
        for ax in axes[:, col]:
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0, 0].set_ylim(0, None)
    axes[1, 0].set_ylim(None, 1.0)
    for col in (1, 2):
        axes[0, col].set_ylim(0, 0.45)
        axes[1, col].set_ylim(0.90, 1.002)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_curves.png"), dpi=200)
    plt.close(fig)

    # zoomed curves for the fine-tuned models
    ft = [g for g in ["M3_resnet50_ft", "M3_resnet50_ft_noaug", "M4_convnext_tiny_ft"] if g in runs]
    fig, axes = plt.subplots(1, len(ft), figsize=(5.5 * len(ft), 4.5), squeeze=False)
    for ax, g in zip(axes[0], ft):
        for s, run in sorted(runs[g].items()):
            h = run["history"]
            ax.plot(h.epoch, h.val_acc, marker="o", ms=3, label=f"val, seed {s}")
            ax.plot(h.epoch, h.train_acc, "--", alpha=0.6, label=f"train, seed {s}")
        ax.set(xlabel="epoch", ylabel="accuracy", title=g)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_curves_finetune.png"), dpi=200)
    plt.close(fig)

    # bar chart: mean val acc, error bar = 95% bootstrap CI (seed 42), dots = individual seeds
    c = comp.reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1, 1]})
    for ax, zoom in zip(axes, [False, True]):
        d = c[c.val_acc_mean > 0.97] if zoom else c
        y = np.arange(len(d))
        ax.barh(y, d.val_acc_mean, color=["#4C72B0" if g in PROB_OK else "#8C8C8C" for g in d.model], alpha=0.8)
        err = np.vstack([d.val_acc_s42 - d.ci95_low_s42, d.ci95_high_s42 - d.val_acc_s42])
        ax.errorbar(d.val_acc_s42, y, xerr=err, fmt="none", ecolor="black", capsize=4)
        for yi, accs in zip(y, d.val_acc_per_seed):
            vals = [float(a) for a in accs.split()]
            ax.scatter(vals, [yi] * len(vals), color="red", s=14, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(d.model, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("validation accuracy (5000 images)")
        if zoom:
            ax.set_xlim(d.ci95_low_s42.min() - 0.001, 1.0015)
            ax.set_title("Zoom: models above 97%")
        else:
            ax.set_xlim(0.5, 1.0)
            ax.set_title("All models")
        for yi, (m, sd, hi) in enumerate(zip(d.val_acc_mean, d.val_acc_std, d.ci95_high_s42)):
            txt = f"{m:.4f}" + (f" ± {sd:.4f}" if pd.notna(sd) else "")
            if zoom:  # right of the CI so it does not cover the bars
                ax.text(hi + 0.0002, yi, txt, va="center", fontsize=9)
            else:
                ax.text(0.52, yi, txt, va="center", fontsize=9, color="white")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Bars = mean val acc over seeds; black = 95% bootstrap CI (seed 42); red dots = each seed", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_model_comparison.png"), dpi=200)
    plt.close(fig)

    # lr sweep
    sweep = []
    for f in glob.glob(os.path.join(RUNS, "sweep_*", "run_info.json")):
        i = json.load(open(f))
        sweep.append(dict(model=i["model"], lr=i["lr"], val_acc=i["best_val_acc"], val_loss=i["best_val_loss"]))
    if sweep:
        sw = pd.DataFrame(sweep).sort_values(["model", "lr"])
        sw.to_csv(os.path.join(DOGCAT_RESULTS, "lr_sweep.csv"), index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for m, d in sw.groupby("model"):
            axes[0].plot(d.lr, d.val_acc, marker="o", label=m)
            axes[1].plot(d.lr, d.val_loss, marker="o", label=m)
            b = d.sort_values(["val_acc", "val_loss"], ascending=[False, True]).iloc[0]
            axes[0].scatter([b.lr], [b.val_acc], s=150, facecolors="none", edgecolors="red", linewidths=2)
        axes[0].set(xscale="log", xlabel="peak learning rate (AdamW)", ylabel="best val accuracy",
                    title="LR sweep, 3 epochs each (red circle = chosen)")
        axes[1].set(xscale="log", xlabel="peak learning rate (AdamW)", ylabel="val loss at best epoch",
                    title="LR sweep: val loss")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "fig_lr_sweep.png"), dpi=200)
        plt.close(fig)

    # confusion matrix of the final model on val
    pred = (p_val_final > 0.5).astype(int)
    cm = confusion_matrix(labels, pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["cat (0)", "dog (1)"])
    ax.set_yticks([0, 1], ["cat (0)", "dog (1)"])
    ax.set(xlabel="predicted", ylabel="true label",
           title=f"ConvNeXt-Tiny (seed 42) on val (acc {(pred == labels).mean():.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_confusion_final.png"), dpi=200)
    plt.close(fig)

    # misclassified val images of the final model
    wrong = np.nonzero(pred != labels)[0]
    wrong = wrong[np.argsort(-np.abs(p_val_final[wrong] - 0.5))]  # most confident mistakes first
    pd.DataFrame({"path": val_paths[wrong], "label": labels[wrong], "prob_dog_final": p_val_final[wrong].round(4)}) \
        .to_csv(os.path.join(DOGCAT_RESULTS, "val_errors_final.csv"), index=False)
    image_grid([val_paths[i] for i in wrong],
               [f"{os.path.basename(val_paths[i])}\ntrue {'dog' if labels[i] else 'cat'}, P(dog)={p_val_final[i]:.3f}"
                for i in wrong], os.path.join(FIG, "fig_val_errors_final.png"),
               "Validation images misclassified by ConvNeXt-Tiny (seed 42, flip TTA)")

    # test: disagreement between strong models and low confidence of the final model
    strong = [g for g in ["M2a_resnet50_logreg", "M3_resnet50_ft", "M4_convnext_tiny_ft",
                          "M5a_clip_vitl14_zeroshot", "M5b_clip_vitl14_logreg"] if g in tp]
    votes = (tp[strong] > 0.5).sum(1)
    tp = tp.assign(n_dog_votes=votes, disagree=(votes > 0) & (votes < len(strong)),
                   confidence=(tp.final - 0.5).abs() * 2)
    tp.sort_values("confidence").to_csv(os.path.join(DOGCAT_RESULTS, "test_uncertain.csv"), index=False)
    hard = tp[tp.disagree | (tp.confidence < 0.8)].sort_values("confidence")
    easy = tp.sort_values("confidence", ascending=False).head(6)
    for d, name, title in [(hard, "fig_test_uncertain.png",
                            f"Test images where models disagree ({len(strong)} strong models) or ConvNeXt-Tiny confidence is low"),
                           (easy, "fig_test_confident.png", "Test images with the highest ConvNeXt-Tiny confidence")]:
        image_grid([f"test/{i}.jpg" for i in d.id],
                   [f"test {i}: P(dog)={p:.3f}\ndog votes {v}/{len(strong)}" for i, p, v in zip(d.id, d.final, d.n_dog_votes)],
                   os.path.join(FIG, name), title)


def image_grid(paths, titles, out, suptitle, ncols=6):
    if not paths:
        return
    nrows = (len(paths) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, p, t in zip(axes.flat, paths, titles):
        ax.imshow(Image.open(os.path.join(CACHE_ROOT, p)).convert("RGB"))
        ax.set_title(t, fontsize=9)
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
