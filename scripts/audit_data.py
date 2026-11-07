"""Check the dataset: counts, broken files, image sizes, test ids, and near-duplicates across splits."""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_ROOT, PROJECT_ROOT, SPLIT_DIRS, format_time, list_images, set_seed


def read_image(rel_path):
    path = os.path.join(DATA_ROOT, rel_path)
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            h64 = int(str(imagehash.phash(im)), 16)
        return rel_path, w, h, h64, None
    except Exception as e:
        return rel_path, None, None, None, repr(e)


def hamming_pairs(hash_a, hash_b, thresh, chunk=2000):
    # all (i, j, dist) with popcount(a_i xor b_j) <= thresh
    out = []
    for start in range(0, len(hash_a), chunk):
        a = hash_a[start:start + chunk, None]
        dist = np.bitwise_count(a ^ hash_b[None, :])
        ii, jj = np.nonzero(dist <= thresh)
        out.extend(zip(ii + start, jj, dist[ii, jj]))
    return out


def save_grid(pairs_df, out_path, n=8, seed=42):
    sample = pairs_df.sample(n=min(n, len(pairs_df)), random_state=seed).sort_values("distance")
    fig, axes = plt.subplots(len(sample), 2, figsize=(6, 2.6 * len(sample)))
    axes = np.atleast_2d(axes)
    for row, (_, r) in zip(axes, sample.iterrows()):
        for ax, f in zip(row, [r.file_a, r.file_b]):
            ax.imshow(Image.open(os.path.join(DATA_ROOT, f)).convert("RGB"))
            ax.set_title(f, fontsize=8)
            ax.axis("off")
        row[0].text(-0.05, 0.5, f"d={r.distance}", transform=row[0].transAxes,
                    ha="right", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results", "data_audit"))
    parser.add_argument("--thresh", type=int, default=5, help="max phash hamming distance")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    files = []
    print("Image counts:")
    counts = {}
    for d in SPLIT_DIRS:
        names = list_images(os.path.join(DATA_ROOT, d))
        counts[d] = len(names)
        files += [f"{d}/{n}" for n in names]
        print(f"  {d:10s} {len(names)}")

    print(f"Reading {len(files)} images with {args.workers} workers...")
    with Pool(args.workers) as pool:
        rows = pool.map(read_image, files, chunksize=64)
    df = pd.DataFrame(rows, columns=["file", "width", "height", "phash", "error"])
    df["split"] = df.file.str.split("/").str[0]
    df["label"] = df.file.apply(lambda f: f.split("/")[1] if f.startswith(("train", "val")) else "")

    broken = df[df.error.notna()]
    print(f"Broken images: {len(broken)}")
    for _, r in broken.iterrows():
        print(f"  {r.file}: {r.error}")
    df = df[df.error.isna()].copy()

    size_rows = []
    for split, g in df.groupby("split", sort=False):
        size_rows.append({
            "split": split, "n": len(g),
            "w_min": g.width.min(), "w_median": g.width.median(), "w_max": g.width.max(),
            "h_min": g.height.min(), "h_median": g.height.median(), "h_max": g.height.max(),
        })
    sizes = pd.DataFrame(size_rows)
    print("Image sizes:")
    print(sizes.to_string(index=False))

    # test file names should be exactly 1..500 from the sample submission
    sample_ids = set(pd.read_csv(os.path.join(PROJECT_ROOT, "sampleSubmission.csv")).id)
    test_ids = set(int(os.path.splitext(n)[0]) for n in list_images(os.path.join(DATA_ROOT, "test")))
    missing, extra = sorted(sample_ids - test_ids), sorted(test_ids - sample_ids)
    ids_ok = not missing and not extra
    print(f"Test ids match sampleSubmission: {ids_ok} (missing={missing[:10]}, extra={extra[:10]})")

    hashes = {s: (g.file.values, np.array(g.phash.values, dtype=np.uint64)) for s, g in df.groupby("split")}
    pair_rows = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        fa, ha = hashes[a]
        fb, hb = hashes[b]
        found = hamming_pairs(ha, hb, args.thresh)
        print(f"Near-duplicate pairs {a}-{b} (d<={args.thresh}): {len(found)}")
        for i, j, d in found:
            pair_rows.append({"split_a": a, "file_a": fa[i], "split_b": b, "file_b": fb[j], "distance": int(d)})
    pairs = pd.DataFrame(pair_rows, columns=["split_a", "file_a", "split_b", "file_b", "distance"])
    label_of = dict(zip(df.file, df.label))
    pairs["label_a"] = pairs.file_a.map(label_of)
    pairs["label_b"] = pairs.file_b.map(label_of)
    pairs.to_csv(os.path.join(args.out, "duplicate_pairs.csv"), index=False)

    if len(pairs):
        save_grid(pairs, os.path.join(args.out, "duplicate_examples.png"), seed=args.seed)

    excluded = sorted(set(pairs[pairs.split_a == "train"].file_a))
    with open(os.path.join(args.out, "excluded_train.txt"), "w") as f:
        f.write("".join(e + "\n" for e in excluded))
    n_train = (df.split == "train").sum()
    remaining = df[(df.split == "train") & ~df.file.isin(excluded)]
    print(f"Train images excluded: {len(excluded)}, remaining: {len(remaining)} "
          f"(cat={(remaining.label == 'cat').sum()}, dog={(remaining.label == 'dog').sum()})")

    tv = pairs[pairs.split_b == "val"]
    label_conflicts = int((tv.label_a != tv.label_b).sum())
    by_split = pairs.groupby(["split_a", "split_b"]).size()

    lines = [
        "# Data audit",
        "",
        "## Image counts",
        "",
        "| folder | images |",
        "|---|---|",
        *[f"| {d} | {n} |" for d, n in counts.items()],
        "",
        f"Broken images (PIL failed to open/decode): {len(broken)}",
        "",
        "## Image sizes (pixels)",
        "",
        "| split | n | width min/median/max | height min/median/max |",
        "|---|---|---|---|",
        *[f"| {r.split} | {r.n} | {r.w_min} / {r.w_median:g} / {r.w_max} | {r.h_min} / {r.h_median:g} / {r.h_max} |"
          for r in sizes.itertuples()],
        "",
        f"Test file names match ids 1..{len(sample_ids)} in sampleSubmission.csv: {'yes' if ids_ok else 'no'}",
        "",
        "## Near-duplicates across splits",
        "",
        f"Perceptual hash (imagehash.phash, 64-bit), pair counted if Hamming distance <= {args.thresh}.",
        "",
        "| pair | count |",
        "|---|---|",
        *[f"| {a}-{b} | {int(by_split.get((a, b), 0))} |" for a, b in [("train", "val"), ("train", "test"), ("val", "test")]],
        "",
        f"Train-val pairs with different labels: {label_conflicts}",
        "",
        f"Train images excluded (duplicate of a val or test image): {len(excluded)}",
        f"Train images remaining: {len(remaining)} of {n_train} "
        f"(cat {(remaining.label == 'cat').sum()}, dog {(remaining.label == 'dog').sum()})",
        "",
        "Files: `duplicate_pairs.csv`, `excluded_train.txt`, `duplicate_examples.png`. Original files are not deleted.",
        "",
    ]
    with open(os.path.join(args.out, "audit.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote results to {args.out}")
    print(f"Total time: {format_time(time.time() - t0)}")


if __name__ == "__main__":
    main()
