"""Part e: val images the scratch CNN gets confidently wrong but the final model gets right.
Prints the candidate list; the 6 images in the figure were picked by eye from the top 30."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import CACHE_ROOT, DOGCAT_RESULTS, Timer

RUNS = os.path.join(DOGCAT_RESULTS, "runs")
# (image, type of difficulty), chosen after looking at the top 30 candidates
PICKED = [
    ("val/cat/cat.9897.jpg", "behind a wire fence"),
    ("val/dog/dog.3074.jpg", "small animal, far away"),
    ("val/cat/cat.10855.jpg", "on a stuffed toy face"),
    ("val/cat/cat.3789.jpg", "dark photo, black cat"),
    ("val/dog/dog.7588.jpg", "odd pose, face hidden"),
    ("val/cat/cat.2893.jpg", "cluttered scene, people"),
]


def load(run):
    return pd.read_csv(os.path.join(RUNS, run, "val_preds.csv")).set_index("path")


def main():
    timer = Timer()
    cfg = json.load(open(os.path.join(DOGCAT_RESULTS, "final_config.json")))
    col = "prob_dog_tta" if cfg["tta"] else "prob_dog"
    final = np.mean([load(m if m.startswith("M5") else f"{m}_s42")[col] for m in cfg["members"]], axis=0)
    cnn, hog = load("M1_simplecnn_s42"), load("M0_hog_svm")
    d = pd.DataFrame(dict(label=cnn.label, cnn=cnn[col], hog=hog[col], final=final), index=cnn.index)
    # confidence of the wrong answer (> 0.5 means the model is wrong)
    d["cnn_wrong"] = np.where(d.label == 1, 1 - d.cnn, d.cnn)
    d["hog_wrong"] = np.where(d.label == 1, 1 - d.hog, d.hog)
    right = (d.final > 0.5) == (d.label == 1)
    print(f"final right, CNN wrong: {(right & (d.cnn_wrong > 0.5)).sum()}, "
          f"HOG wrong: {(right & (d.hog_wrong > 0.5)).sum()}")
    cand = d[right & (d.cnn_wrong > 0.5)].sort_values("cnn_wrong", ascending=False)
    print(cand.head(30)[["label", "cnn", "hog", "final"]].round(3).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    for ax, (path, kind) in zip(axes.flat, PICKED):
        r = d.loc[path]
        ax.imshow(Image.open(os.path.join(CACHE_ROOT, path)).convert("RGB"))
        ax.set_title(f"{kind}\ntrue: {'dog' if r.label else 'cat'}   P(dog): SimpleCNN {r.cnn:.2f}, "
                     f"final {r.final:.2f}", fontsize=10.5)
        ax.axis("off")
    fig.suptitle("Val images the scratch CNN gets confidently wrong and the final model gets right",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(DOGCAT_RESULTS, "figures", "fig_hard_cases.png"), dpi=200)
    plt.close(fig)
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
