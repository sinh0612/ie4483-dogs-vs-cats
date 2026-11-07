"""Figures for question a): sample training images and augmentation examples."""
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import CLASSES, get_items
from src.utils import CACHE_ROOT, DATA_ROOT, DOGCAT_RESULTS, Timer, set_seed

plt.rcParams.update({"font.size": 12})


def main():
    timer = Timer()
    set_seed(42)
    out = os.path.join(DOGCAT_RESULTS, "figures")
    os.makedirs(out, exist_ok=True)
    items = get_items("train")
    rng = random.Random(42)

    # sample images: original files so the real sizes show up in titles
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6))
    for row, label in zip(axes, [0, 1]):
        picks = rng.sample([it for it in items if it[1] == label], 6)
        for ax, (path, _) in zip(row, picks):
            im = Image.open(os.path.join(DATA_ROOT, path))
            ax.imshow(im)
            ax.set_title(f"{CLASSES[label]} ({im.width}x{im.height})", fontsize=11)
            ax.axis("off")
    fig.suptitle("Sample training images (label, original size)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_train_samples.png"), dpi=200)
    plt.close(fig)

    # augmentation: same pipeline as training, without normalisation
    aug = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                     T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.2)])
    evaltf = T.Compose([T.Resize(256), T.CenterCrop(224)])
    path = "train/dog/dog.1004.jpg" if ("train/dog/dog.1004.jpg", 1) in items else items[-1][0]
    img = Image.open(os.path.join(CACHE_ROOT, path)).convert("RGB")
    torch.manual_seed(42)
    fig, axes = plt.subplots(1, 7, figsize=(17, 3.2))
    axes[0].imshow(evaltf(img))
    axes[0].set_title("original\n(resize + center crop)", fontsize=11)
    for i, ax in enumerate(axes[1:], 1):
        ax.imshow(aug(img))
        ax.set_title(f"augmented #{i}", fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Training augmentation: RandomResizedCrop(224, scale 0.7-1.0), horizontal flip, "
                 "rotation +-15 deg, ColorJitter(0.2, 0.2, 0.2)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_augmentation.png"), dpi=200)
    plt.close(fig)
    print(f"augmentation example: {path}")
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
