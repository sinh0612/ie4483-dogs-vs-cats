"""Architecture diagrams (matplotlib) + output shape / parameter tables (arch_<name>.txt).
Usage: python scripts/plot_arch.py simplecnn final [resnet50 convnext_tiny clip_vitl14]"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import build_model
from src.utils import DOGCAT_RESULTS, Timer

FIG = os.path.join(DOGCAT_RESULTS, "figures")


def shape_str(t):
    return "x".join(str(s) for s in t.shape[1:])


def shape_table(model, x, depth):
    """Output shape and params of every module up to `depth` levels deep (hooks)."""
    rows, hooks = [], []
    for name, m in model.named_modules():
        if not name or name.count(".") >= depth:
            continue
        own = sum(p.numel() for p in m.parameters())

        def hook(mod, inp, out, name=name, own=own):
            out = out[0] if isinstance(out, (tuple, list)) else out
            rows.append((name, type(mod).__name__, shape_str(out) if torch.is_tensor(out) else "-", own))
        hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        y = model(x)
    for h in hooks:
        h.remove()
    return rows, y


def write_table(path, title, rows, x, y, model, extra=""):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [title, "", f"input: {shape_str(x)}   output: {shape_str(y)}", "",
             f"{'layer':42s} {'type':24s} {'output shape':>16s} {'params':>12s}", "-" * 98]
    for name, typ, shp, n in rows:
        indent = "  " * name.count(".")
        lines.append(f"{(indent + name):42s} {typ:24s} {shp:>16s} {n:>12,}")
    lines += ["-" * 98, f"total params: {total:,}   trainable: {trainable:,}", extra]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def stage_shapes(model, x, names):
    out, hooks = {}, []
    mods = dict(model.named_modules())
    for n in names:
        hooks.append(mods[n].register_forward_hook(lambda m, i, o, n=n: out.__setitem__(n, o[0] if isinstance(o, tuple) else o)))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return [shape_str(out[n]) for n in names]


def draw_chain(ax, y, boxes, colors, width=1.55, gap=0.35, height=1.0):
    """boxes: list of (title, detail, shape below). Returns x position of right edge."""
    x = 0
    for i, ((title, detail, shp), c) in enumerate(zip(boxes, colors)):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.03", fc=c, ec="black", lw=1))
        ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x + width / 2, y + height * 0.3, detail, ha="center", va="center", fontsize=7.8)
        ax.text(x + width / 2, y - 0.22, shp, ha="center", va="center", fontsize=8.5, color="#333333")
        if i:
            ax.add_patch(FancyArrowPatch((x - gap, y + height / 2), (x, y + height / 2), arrowstyle="-|>",
                                         mutation_scale=12, lw=1))
        x += width + gap
    return x - gap


def save_diagram(rows, title, path, merge=None):
    """rows: list of (row label, boxes, colors). merge: (title, detail, shape) box that averages all rows."""
    width_units = max(len(b) for _, b, _ in rows) * 1.9 + (2.5 if merge else 0)
    fig, ax = plt.subplots(figsize=(min(26, width_units * 0.95), 1.9 * len(rows) + 0.8))
    ends, ys = [], []
    for k, (label, boxes, colors) in enumerate(rows):
        y = (len(rows) - 1 - k) * 1.9
        ends.append(draw_chain(ax, y, boxes, colors))
        ys.append(y)
        if label:
            ax.text(-0.15, y + 0.5, label, ha="right", va="center", fontsize=10, weight="bold")
    xmax = max(ends)
    if merge:
        mx, my = xmax + 0.7, (ys[0] + ys[-1]) / 2
        ax.add_patch(FancyBboxPatch((mx, my), 1.7, 1.0, boxstyle="round,pad=0.03", fc="#F6D6A8", ec="black"))
        ax.text(mx + 0.85, my + 0.68, merge[0], ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(mx + 0.85, my + 0.3, merge[1], ha="center", va="center", fontsize=7.8)
        ax.text(mx + 0.85, my - 0.22, merge[2], ha="center", va="center", fontsize=8.5)
        for e, y in zip(ends, ys):
            ax.add_patch(FancyArrowPatch((e, y + 0.5), (mx, my + 0.5), arrowstyle="-|>", mutation_scale=12))
        xmax = mx + 1.7
    ax.set_xlim(-2.2 if any(r[0] for r in rows) else -0.3, xmax + 0.3)
    ax.set_ylim(-0.7, (len(rows) - 1) * 1.9 + 1.4)
    ax.axis("off")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


C_IN, C_CONV, C_POOL, C_HEAD = "#DDDDDD", "#AFCBEA", "#C9E4C5", "#F4B6B6"


def simplecnn_boxes():
    model = build_model("simplecnn").eval()
    x = torch.zeros(1, 3, 224, 224)
    names = [f"features.{i}" for i in range(5)] + ["pool", "fc"]
    shp = stage_shapes(model, x, names)
    boxes = [("Input", "RGB image\nImageNet norm.", "3x224x224")]
    for i, c in enumerate([32, 64, 128, 256, 512]):
        boxes.append((f"Block {i + 1}", f"Conv3x3({c})\nBN-ReLU-MaxPool2", shp[i]))
    boxes += [("GAP", "global avg pool", shp[5]), ("Dropout+FC", "p=0.3, Linear\n512->2", shp[6]),
              ("Softmax", "P(cat), P(dog)", "2")]
    colors = [C_IN] + [C_CONV] * 5 + [C_POOL, C_HEAD, C_HEAD]
    rows, y = shape_table(model, x, depth=3)
    write_table(os.path.join(DOGCAT_RESULTS, "arch_simplecnn.txt"), "SimpleCNN (trained from scratch)", rows, x, y, model,
                "loss: CrossEntropy with label smoothing 0.1")
    return boxes, colors


def resnet50_boxes():
    model = build_model("resnet50", pretrained=False).eval()
    x = torch.zeros(1, 3, 224, 224)
    names = ["maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    shp = stage_shapes(model, x, names)
    boxes = [("Input", "RGB, ImageNet norm.", "3x224x224"),
             ("Stem", "Conv7x7/2(64)\nBN-ReLU-MaxPool", shp[0]),
             ("Stage 1", "3 bottlenecks", shp[1]), ("Stage 2", "4 bottlenecks", shp[2]),
             ("Stage 3", "6 bottlenecks", shp[3]), ("Stage 4", "3 bottlenecks", shp[4]),
             ("GAP", "global avg pool", shp[5].replace("x1x1", "")),
             ("FC (new)", "Linear 2048->2", shp[6]), ("Softmax", "P(dog)", "2")]
    colors = [C_IN] + [C_CONV] * 5 + [C_POOL, C_HEAD, C_HEAD]
    rows, y = shape_table(model, x, depth=2)
    write_table(os.path.join(DOGCAT_RESULTS, "arch_resnet50.txt"), "ResNet-50 (ImageNet IMAGENET1K_V2), fc replaced by Linear(2048, 2), all layers fine-tuned",
                rows, x, y, model, "loss: CrossEntropy with label smoothing 0.1")
    return boxes, colors


def convnext_boxes():
    model = build_model("convnext_tiny", pretrained=False).eval()
    x = torch.zeros(1, 3, 224, 224)
    names = ["features.0", "features.1", "features.3", "features.5", "features.7", "avgpool", "classifier"]
    shp = stage_shapes(model, x, names)
    boxes = [("Input", "RGB, ImageNet norm.", "3x224x224"),
             ("Stem", "Conv4x4/4(96)\n+ LayerNorm", shp[0]),
             ("Stage 1", "3 ConvNeXt blocks", shp[1]), ("Stage 2", "downsample\n+ 3 blocks", shp[2]),
             ("Stage 3", "downsample\n+ 9 blocks", shp[3]), ("Stage 4", "downsample\n+ 3 blocks", shp[4]),
             ("GAP + LN", "avg pool,\nLayerNorm", "768"),
             ("FC (new)", "Linear 768->2", shp[6]), ("Softmax", "P(dog)", "2")]
    colors = [C_IN] + [C_CONV] * 5 + [C_POOL, C_HEAD, C_HEAD]
    rows, y = shape_table(model, x, depth=2)
    write_table(os.path.join(DOGCAT_RESULTS, "arch_convnext_tiny.txt"),
                "ConvNeXt-Tiny (ImageNet IMAGENET1K_V1), last Linear replaced by Linear(768, 2), all layers fine-tuned",
                rows, x, y, model, "ConvNeXt block: depthwise Conv7x7 -> LayerNorm -> Linear(4C) -> GELU -> Linear(C), residual\n"
                "loss: CrossEntropy with label smoothing 0.1")
    return boxes, colors


def clip_boxes():
    import open_clip
    model = open_clip.create_model("ViT-L-14", pretrained=None).visual.eval()
    x = torch.zeros(1, 3, 224, 224)
    rows, y = shape_table(model, x, depth=2)
    write_table(os.path.join(DOGCAT_RESULTS, "arch_clip_vitl14.txt"),
                "CLIP ViT-L/14 image encoder (open_clip, pretrained=openai), frozen; features -> StandardScaler -> LogisticRegression",
                rows, x, y, model, "classifier: sklearn LogisticRegression on 768-d features (768 weights + 1 bias)")
    boxes = [("Input", "RGB, CLIP norm.", "3x224x224"),
             ("Patch embed", "Conv14x14/14\n(1024)", "256 patches x1024"),
             ("+CLS, +pos", "LayerNorm", "257x1024"),
             ("Transformer", "24 blocks, 16 heads\n(frozen)", "257x1024"),
             ("CLS token", "LayerNorm +\nproj 1024->768", "768"),
             ("Standardise", "train mean/std", "768"),
             ("LogReg", "768->1, sigmoid", "P(dog)")]
    colors = [C_IN, C_CONV, C_CONV, C_CONV, C_POOL, C_HEAD, C_HEAD]
    return boxes, colors


def clip_zeroshot_boxes():
    boxes, colors = clip_boxes()
    boxes = boxes[:5] + [("L2 normalise", "unit-length\nimage embedding", "768"),
                         ("Cosine sim.", "with text emb. of\n'a photo of a cat/dog.'", "2"),
                         ("x100, softmax", "CLIP logit scale\n(no training)", "P(dog)")]
    return boxes, colors[:5] + [C_POOL, C_HEAD, C_HEAD]


BUILDERS = {"simplecnn": simplecnn_boxes, "resnet50": resnet50_boxes, "convnext_tiny": convnext_boxes,
            "clip_vitl14": clip_boxes, "clip_zeroshot": clip_zeroshot_boxes}
MEMBER_ARCH = {"M3_resnet50_ft": "resnet50", "M4_convnext_tiny_ft": "convnext_tiny",
               "M5b_clip_vitl14_logreg": "clip_vitl14", "M5a_clip_vitl14_zeroshot": "clip_zeroshot",
               "M1_simplecnn": "simplecnn"}
DISPLAY = {"M3_resnet50_ft": "ResNet-50", "M4_convnext_tiny_ft": "ConvNeXt-Tiny",
           "M5b_clip_vitl14_logreg": "CLIP + log. regression", "M5a_clip_vitl14_zeroshot": "CLIP zero-shot",
           "M1_simplecnn": "SimpleCNN"}
FULL_NAME = {"M5a_clip_vitl14_zeroshot": "CLIP ViT-L/14 zero-shot",
             "M5b_clip_vitl14_logreg": "CLIP ViT-L/14 + logistic regression"}


def main():
    timer = Timer()
    os.makedirs(FIG, exist_ok=True)
    targets = sys.argv[1:] or ["simplecnn", "final"]
    if "simplecnn" in targets:
        boxes, colors = simplecnn_boxes()
        save_diagram([("", boxes, colors)], "SimpleCNN (from scratch), 1.57M parameters",
                     os.path.join(FIG, "fig_arch_simplecnn.png"))
    if "final" in targets:
        cfg = json.load(open(os.path.join(DOGCAT_RESULTS, "final_config.json")))
        rows = []
        for m in cfg["members"]:
            boxes, colors = BUILDERS[MEMBER_ARCH[m]]()
            if cfg["tta"]:
                boxes[0] = (boxes[0][0], "image + its\nhorizontal flip", boxes[0][2])
            rows.append((DISPLAY[m], boxes, colors))
        tta = ", with flip TTA" if cfg["tta"] else ""
        merge = ("Average", "mean of P(dog)\nlabel = P > 0.5", "dog=1 / cat=0") if len(rows) > 1 else None
        save_diagram(rows, f"Final model: {' + '.join(FULL_NAME.get(m, DISPLAY[m]) for m in cfg['members'])}{tta}", os.path.join(FIG, "fig_arch_final.png"), merge)
    for t in ("resnet50", "convnext_tiny", "clip_vitl14"):  # shape tables only
        if t in targets:
            BUILDERS[t]()
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
