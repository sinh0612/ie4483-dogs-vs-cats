"""M0 baseline: HOG features on 128x128 grayscale images + LinearSVC (C chosen on val)."""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_items
from src.utils import CACHE_ROOT, DOGCAT_RESULTS, FEATURE_DIR, Timer, set_seed, write_csv

SIZE = 128
HOG_ARGS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
SVM_C = [0.00001, 0.00003, 0.0001, 0.0003, 0.001]


def load_gray(path, flip=False):
    # short side -> 128, center crop 128x128, grayscale
    im = Image.open(os.path.join(CACHE_ROOT, path)).convert("L")
    w, h = im.size
    s = SIZE / min(w, h)
    im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR)
    w, h = im.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    im = im.crop((left, top, left + SIZE, top + SIZE))
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return np.asarray(im, dtype=np.float32) / 255.0


def hog_one(args):
    path, flip = args
    return hog(load_gray(path, flip), **HOG_ARGS).astype(np.float32)


def get_hog(items, name, workers, flip=False):
    out = os.path.join(FEATURE_DIR, f"hog_{name}.npy")
    if os.path.exists(out):
        return np.load(out)
    with Pool(workers) as pool:
        f = np.stack(pool.map(hog_one, [(it[0], flip) for it in items], chunksize=64))
    np.save(out, f)
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timer = Timer()
    set_seed(args.seed)
    os.makedirs(FEATURE_DIR, exist_ok=True)
    train_items, val_items, test_items = get_items("train"), get_items("val"), get_items("test")
    X = get_hog(train_items, "train", args.workers)
    Xv, Xvf = get_hog(val_items, "val", args.workers), get_hog(val_items, "val_flip", args.workers, flip=True)
    Xt, Xtf = get_hog(test_items, "test", args.workers), get_hog(test_items, "test_flip", args.workers, flip=True)
    feat_seconds = timer.elapsed()
    print(f"HOG features: {X.shape} ({timer})", flush=True)

    y = np.array([it[1] for it in train_items])
    yv = np.array([it[1] for it in val_items])
    scaler = StandardScaler().fit(X)
    X, Xv, Xvf, Xt, Xtf = [scaler.transform(a) for a in (X, Xv, Xvf, Xt, Xtf)]

    history, best = [], None
    for C in SVM_C:
        t0 = time.time()
        clf = LinearSVC(C=C, dual=False, max_iter=5000).fit(X, y)
        row = dict(C=C, train_acc=round(clf.score(X, y), 5), val_acc=round(clf.score(Xv, yv), 5),
                   seconds=round(time.time() - t0, 1))
        history.append(row)
        print("  ", row, flush=True)
        if best is None or row["val_acc"] > best[0]["val_acc"]:
            best = (row, clf)
    row, clf = best

    prob = lambda A: 1 / (1 + np.exp(-clf.decision_function(A)))
    run_dir = os.path.join(DOGCAT_RESULTS, "runs", "M0_hog_svm")
    write_csv(os.path.join(run_dir, "history.csv"), history)
    pv, pvf, pt, ptf = prob(Xv), prob(Xvf), prob(Xt), prob(Xtf)
    write_csv(os.path.join(run_dir, "val_preds.csv"),
              [dict(path=it[0], label=it[1], prob_dog=round(float(a), 6), prob_dog_tta=round(float((a + b) / 2), 6))
               for it, a, b in zip(val_items, pv, pvf)])
    write_csv(os.path.join(run_dir, "test_preds.csv"),
              [dict(id=it[1], prob_dog=round(float(a), 6), prob_dog_tta=round(float((a + b) / 2), 6))
               for it, a, b in zip(test_items, pt, ptf)])
    info = dict(model="HOG + LinearSVC", hog=dict(HOG_ARGS, image="grayscale 128x128 (resize short side + center crop)"),
                feature_dim=X.shape[1], C=row["C"], seed=args.seed, n_train=len(y), n_val=len(yv),
                input_size="1x128x128", params_total=X.shape[1] + 1, params_trainable=X.shape[1] + 1,
                epochs=None, lr=None, best_val_acc=row["val_acc"], train_acc_at_best=row["train_acc"],
                feature_seconds=round(feat_seconds, 1),
                train_seconds=round(sum(h["seconds"] for h in history), 1), final_fit_seconds=row["seconds"],
                note="prob = sigmoid(SVM decision score), not calibrated")
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"best C={row['C']} val acc {row['val_acc']}")
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
