"""Grad-CAM (own implementation with hooks) for the fine-tuned ResNet-50 and ConvNeXt-Tiny (seed 42 checkpoints).
Usage: python scripts/gradcam.py --images test/12.jpg val/cat/cat.1.jpg --out fig_gradcam_test.png"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import IMAGENET_MEAN, IMAGENET_STD, get_transform
from src.models import build_model, gradcam_layer
from src.utils import CACHE_ROOT, DOGCAT_RESULTS, PROJECT_ROOT, Timer

MODELS = [("resnet50", "M3_resnet50_ft_s42"), ("convnext_tiny", "M4_convnext_tiny_ft_s42")]


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = self.grads = None
        layer.register_forward_hook(self._save_act)

    def _save_act(self, module, inp, out):
        self.acts = out
        if out.requires_grad:  # skip during plain no_grad forward passes
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target):
        """Heatmap (HxW, 0..1) for class `target`, plus softmax probabilities."""
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, target].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)  # GAP of gradients per channel
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), logits.softmax(1)[0].detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True, help="paths relative to data/cache256")
    parser.add_argument("--out", required=True, help="file name inside results/dogcat/figures")
    parser.add_argument("--title", default="Grad-CAM (class = predicted label)")
    args = parser.parse_args()

    timer = Timer()
    tf = get_transform(False)
    cams = {}
    for arch, run in MODELS:
        model = build_model(arch, pretrained=False)
        model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "checkpoints", "dogcat", f"{run}.pt"),
                                         map_location="cpu", weights_only=True))
        model = model.cuda().eval()
        gc = GradCAM(model, gradcam_layer(model, arch))
        for p in args.images:
            x = tf(Image.open(os.path.join(CACHE_ROOT, p)).convert("RGB")).unsqueeze(0).cuda().requires_grad_(True)
            with torch.no_grad():
                pred = int(model(x).argmax(1))
            cams[(arch, p)] = (*gc(x, pred), pred)

    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    fig, axes = plt.subplots(len(args.images), 3, figsize=(10, 3.4 * len(args.images)), squeeze=False)
    for row, p in zip(axes, args.images):
        img = tf(Image.open(os.path.join(CACHE_ROOT, p)).convert("RGB")).permute(1, 2, 0).numpy() * std + mean
        img = img.clip(0, 1)
        row[0].imshow(img)
        row[0].set_title(p, fontsize=10)
        for ax, (arch, run) in zip(row[1:], MODELS):
            cam, prob, pred = cams[(arch, p)]
            ax.imshow(img)
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.set_title(f"{arch}: pred {'dog' if pred else 'cat'}, P(dog)={prob[1]:.3f}", fontsize=10)
        for ax in row:
            ax.axis("off")
    fig.suptitle(args.title, fontsize=13)
    plt.tight_layout()
    out = os.path.join(DOGCAT_RESULTS, "figures", args.out)
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
