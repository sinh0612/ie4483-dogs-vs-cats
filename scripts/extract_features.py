"""Extract frozen features (ResNet-50 or CLIP) for train/val/test. Val/test also get features of the flipped image (for TTA).
For CLIP this also runs zero-shot classification with text prompts."""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_items, get_transform, make_loader
from src.models import count_params
from src.utils import DOGCAT_RESULTS, FEATURE_DIR, Timer, set_seed, write_csv

PROMPTS = ["a photo of a cat.", "a photo of a dog."]


@torch.no_grad()
def extract(encode, loader, flip=False):
    feats = []
    for x, _ in loader:
        x = x.cuda(non_blocking=True)
        if flip:
            x = torch.flip(x, dims=[3])
        with torch.autocast("cuda", dtype=torch.float16):
            feats.append(encode(x).float().cpu())
    return torch.cat(feats).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["resnet50", "clip_vitl14", "clip_vitb16"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    timer = Timer()
    set_seed(42)
    os.makedirs(FEATURE_DIR, exist_ok=True)

    if args.backbone == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Identity()
        model = model.cuda().eval()
        encode = model
        transform = get_transform(False)
        info = dict(weights="torchvision IMAGENET1K_V2", feature_dim=2048)
    else:
        import open_clip
        arch = {"clip_vitl14": "ViT-L-14", "clip_vitb16": "ViT-B-16"}[args.backbone]
        model, _, transform = open_clip.create_model_and_transforms(arch, pretrained="openai")
        model = model.cuda().eval()
        encode = model.encode_image
        info = dict(weights=f"open_clip {arch} pretrained=openai", feature_dim=model.visual.output_dim)
    total, _ = count_params(model)
    info["params_total"] = total
    print(f"{args.backbone}: {total:,} params")

    splits = {s: get_items(s) for s in ["train", "val", "test"]}
    for split, items in splits.items():
        if os.path.exists(os.path.join(FEATURE_DIR, f"{args.backbone}_{split}.npy")) and \
                (split == "train" or os.path.exists(os.path.join(FEATURE_DIR, f"{args.backbone}_{split}_flip.npy"))):
            print(f"{split}: features already there, skipping")
            continue
        loader = make_loader(items, transform, args.batch_size, shuffle=False, num_workers=args.workers)
        f = extract(encode, loader)
        np.save(os.path.join(FEATURE_DIR, f"{args.backbone}_{split}.npy"), f)
        if split != "train":
            np.save(os.path.join(FEATURE_DIR, f"{args.backbone}_{split}_flip.npy"), extract(encode, loader, flip=True))
        print(f"{split}: {f.shape}  ({timer})", flush=True)
    info["extract_seconds"] = round(timer.elapsed(), 1)

    if args.backbone.startswith("clip"):
        # zero-shot: cosine similarity with the two prompt embeddings, softmax with CLIP's logit scale
        tokenizer = open_clip.get_tokenizer(arch)
        with torch.no_grad():
            text = model.encode_text(tokenizer(PROMPTS).cuda()).float()
        text = text / text.norm(dim=-1, keepdim=True)
        scale = model.logit_scale.exp().item()

        def zs_prob(name):
            f = torch.from_numpy(np.load(os.path.join(FEATURE_DIR, f"{args.backbone}_{name}.npy"))).cuda()
            f = f / f.norm(dim=-1, keepdim=True)
            return (scale * f @ text.T).softmax(1)[:, 1].cpu().numpy()

        run_dir = os.path.join(DOGCAT_RESULTS, "runs", f"M5a_{args.backbone}_zeroshot")
        p, pf = zs_prob("val"), zs_prob("val_flip")
        p_tta = (p + pf) / 2
        write_csv(os.path.join(run_dir, "val_preds.csv"),
                  [dict(path=it[0], label=it[1], prob_dog=round(float(a), 6), prob_dog_tta=round(float(b), 6))
                   for it, a, b in zip(splits["val"], p, p_tta)])
        tp, tpf = zs_prob("test"), zs_prob("test_flip")
        write_csv(os.path.join(run_dir, "test_preds.csv"),
                  [dict(id=it[1], prob_dog=round(float(a), 6), prob_dog_tta=round(float((a + b) / 2), 6))
                   for it, a, b in zip(splits["test"], tp, tpf)])
        val_labels = np.array([it[1] for it in splits["val"]])
        acc = float(((p > 0.5) == val_labels).mean())
        print(f"zero-shot val acc: {acc:.4f}")
        with open(os.path.join(run_dir, "run_info.json"), "w") as f:
            json.dump(dict(model=f"{args.backbone} zero-shot", prompts=PROMPTS, logit_scale=scale, seed=42,
                           params_trainable=0, n_train=0, n_val=len(val_labels),
                           input_size="3x224x224", epochs=0, lr=None, best_val_acc=round(acc, 5),
                           train_acc_at_best=None, train_seconds=0.0, **info), f, indent=2)

    with open(os.path.join(FEATURE_DIR, f"{args.backbone}_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
