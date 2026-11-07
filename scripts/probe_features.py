"""Classifiers on frozen features: logistic regression, linear SVM (sklearn) and a linear layer trained in PyTorch.
C / epoch are chosen on val."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_items
from src.engine import warmup_cosine
from src.utils import DOGCAT_RESULTS, FEATURE_DIR, Timer, set_seed, write_csv

LOGREG_C = [0.001, 0.01, 0.1, 1.0]
SVM_C = [0.0001, 0.001, 0.01, 0.1]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def save_run(run_dir, val_items, p_val, p_val_flip, test_items, p_test, p_test_flip, info, history):
    write_csv(os.path.join(run_dir, "history.csv"), history)
    write_csv(os.path.join(run_dir, "val_preds.csv"),
              [dict(path=it[0], label=it[1], prob_dog=round(float(a), 6), prob_dog_tta=round(float((a + b) / 2), 6))
               for it, a, b in zip(val_items, p_val, p_val_flip)])
    write_csv(os.path.join(run_dir, "test_preds.csv"),
              [dict(id=it[1], prob_dog=round(float(a), 6), prob_dog_tta=round(float((a + b) / 2), 6))
               for it, a, b in zip(test_items, p_test, p_test_flip)])
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"  -> {run_dir}: val acc {info['best_val_acc']}")


def sweep_sklearn(make_clf, Cs, X, y, Xv, yv):
    history, best = [], None
    for C in Cs:
        t0 = time.time()
        clf = make_clf(C).fit(X, y)
        row = dict(C=C, train_acc=round(clf.score(X, y), 5), val_acc=round(clf.score(Xv, yv), 5),
                   seconds=round(time.time() - t0, 1))
        history.append(row)
        print("  ", row, flush=True)
        # ties go to the smaller C (stronger regularisation)
        if best is None or row["val_acc"] > best[0]["val_acc"]:
            best = (row, clf)
    return history, best


def train_linear(X, y, Xv, yv, seed, epochs=30, lr=1e-3, wd=0.05, batch_size=256):
    set_seed(seed)
    X, y = torch.tensor(X).cuda(), torch.tensor(y).cuda()
    Xv, yv = torch.tensor(Xv).cuda(), torch.tensor(yv).cuda()
    model = nn.Linear(X.shape[1], 2).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = (len(X) + batch_size - 1) // batch_size
    sched = warmup_cosine(opt, steps, epochs * steps)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    g = torch.Generator(device="cuda").manual_seed(seed)
    history, best = [], None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(X), device="cuda", generator=g)
        tot, correct = 0.0, 0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            out = model(X[idx])
            loss = crit(out, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            correct += (out.argmax(1) == y[idx]).sum().item()
        model.eval()
        with torch.no_grad():
            out = model(Xv)
            val_loss = nn.functional.cross_entropy(out, yv).item()
            val_acc = (out.argmax(1) == yv).float().mean().item()
        row = dict(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=round(tot / len(X), 5),
                   train_acc=round(correct / len(X), 5), val_loss=round(val_loss, 5), val_acc=round(val_acc, 5),
                   seconds=round(time.time() - t0, 2))
        history.append(row)
        if best is None or (val_acc, -val_loss) > (best[0]["val_acc"], -best[0]["val_loss"]):
            best = (row, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return history, best[0], model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="e.g. resnet50, clip_vitl14")
    parser.add_argument("--methods", default="logreg,svm,linear")
    parser.add_argument("--run-names", default=None, help="comma list matching --methods (default M2a/M2b/M2c_...)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timer = Timer()
    set_seed(args.seed)
    load = lambda s: np.load(os.path.join(FEATURE_DIR, f"{args.features}_{s}.npy"))
    train_items, val_items, test_items = get_items("train"), get_items("val"), get_items("test")
    y = np.array([it[1] for it in train_items])
    yv = np.array([it[1] for it in val_items])

    scaler = StandardScaler().fit(load("train"))  # fit on train only
    X, Xv, Xvf = scaler.transform(load("train")), scaler.transform(load("val")), scaler.transform(load("val_flip"))
    Xt, Xtf = scaler.transform(load("test")), scaler.transform(load("test_flip"))
    X, Xv, Xvf, Xt, Xtf = [a.astype(np.float32) for a in (X, Xv, Xvf, Xt, Xtf)]
    with open(os.path.join(FEATURE_DIR, f"{args.features}_info.json")) as f:
        feat_info = json.load(f)
    print(f"features {args.features}: train {X.shape}, val {Xv.shape}, test {Xt.shape}")

    base = dict(features=args.features, backbone_weights=feat_info["weights"], seed=args.seed,
                n_train=len(y), n_val=len(yv), input_size="3x224x224", feature_dim=X.shape[1],
                backbone_params=feat_info["params_total"], extract_seconds=feat_info["extract_seconds"],
                preprocessing="StandardScaler fitted on train features")
    runs = os.path.join(DOGCAT_RESULTS, "runs")

    methods = args.methods.split(",")
    default_names = {"logreg": "M2a", "svm": "M2b", "linear": "M2c"}
    names = args.run_names.split(",") if args.run_names else [f"{default_names[m]}_{args.features}_{m}" for m in methods]
    for method, run_name in zip(methods, names):
        print(f"[{method}]", flush=True)
        t0 = time.time()
        if method in ("logreg", "svm"):
            if method == "logreg":
                make = lambda C: LogisticRegression(C=C, max_iter=3000)
                hist, (row, clf) = sweep_sklearn(make, LOGREG_C, X, y, Xv, yv)
                prob = lambda A: clf.predict_proba(A)[:, 1]
                note = "sklearn LogisticRegression (lbfgs, L2), C chosen on val"
            else:
                make = lambda C: LinearSVC(C=C, dual=False, max_iter=5000)
                hist, (row, clf) = sweep_sklearn(make, SVM_C, X, y, Xv, yv)
                prob = lambda A: sigmoid(clf.decision_function(A))  # uncalibrated score squashed to (0,1)
                note = "sklearn LinearSVC (squared hinge, L2), C chosen on val; prob = sigmoid(decision score)"
            info = dict(base, model=f"{args.features} frozen + {method}", method=note, C=row["C"],
                        params_trainable=X.shape[1] + 1, epochs=None, lr=None,
                        best_val_acc=row["val_acc"], train_acc_at_best=row["train_acc"],
                        train_seconds=round(sum(h["seconds"] for h in hist), 1),
                        final_fit_seconds=row["seconds"])
        else:
            hist, row, model = train_linear(X, y, Xv, yv, args.seed)
            with torch.no_grad():
                prob = lambda A: model(torch.tensor(A).cuda()).softmax(1)[:, 1].cpu().numpy()
            note = "nn.Linear trained with AdamW (lr 1e-3, wd 0.05), batch 256, 30 epochs, warmup+cosine, label smoothing 0.1; best epoch on val"
            info = dict(base, model=f"{args.features} frozen + linear layer (PyTorch)", method=note,
                        params_trainable=X.shape[1] * 2 + 2, epochs=30, lr=1e-3, best_epoch=row["epoch"],
                        best_val_acc=row["val_acc"], best_val_loss=row["val_loss"],
                        train_acc_at_best=row["train_acc"], train_seconds=round(time.time() - t0, 1))
        save_run(os.path.join(runs, run_name), val_items, prob(Xv), prob(Xvf),
                 test_items, prob(Xt), prob(Xtf), info, hist)

    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
