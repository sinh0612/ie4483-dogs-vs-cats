"""Train on CIFAR-10 (45k train / 5k val split), keep the best epoch by val acc, save val predictions.
The official test set is not touched here (see eval_cifar.py)."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cifar import CifarDataset, get_cifar_transform, load_cifar, split_train_val
from src.data import loader_for
from src.engine import fit, predict
from src.models import build_model, count_params
from src.utils import CIFAR_RESULTS, PROJECT_ROOT, Timer, set_seed, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet18_cifar", "convnext_tiny"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--erase", action="store_true", help="add RandomErasing (Cutout-like)")
    parser.add_argument("--no-aug", action="store_true", help="no crop / flip / erasing")
    parser.add_argument("--img-size", type=int, default=0, help="upsample 32x32 to this size on the GPU (G2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N train images (and N val)")
    args = parser.parse_args()

    timer = Timer()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    run_dir = os.path.join(CIFAR_RESULTS, "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    X, y = load_cifar(True)
    train_idx, val_idx = split_train_val(y)  # split seed is fixed (42), independent of the training seed
    if args.limit:
        train_idx, val_idx = train_idx[:args.limit], val_idx[:args.limit]
    print(f"train {len(train_idx)}  val {len(val_idx)}")

    pretrained = args.model == "convnext_tiny"
    train_tf = get_cifar_transform(True, aug=not args.no_aug, erase=args.erase and not args.no_aug,
                                   imagenet_norm=pretrained)
    eval_tf = get_cifar_transform(False, imagenet_norm=pretrained)
    train_loader = loader_for(CifarDataset(X[train_idx], y[train_idx], train_tf), args.batch_size,
                              shuffle=True, seed=args.seed, num_workers=args.workers)
    val_loader = loader_for(CifarDataset(X[val_idx], y[val_idx], eval_tf), 500, shuffle=False,
                            num_workers=args.workers)

    model = build_model(args.model, num_classes=10, pretrained=pretrained, resize_to=args.img_size or None)
    model = model.to(device).to(memory_format=torch.channels_last)
    total, trainable = count_params(model)
    print(f"{args.model}: {total:,} params ({trainable:,} trainable)")

    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "cifar", f"{args.run_name}.pt")
    history, best = fit(model, train_loader, val_loader, args.epochs, args.lr, args.wd, device, run_dir,
                        label_smoothing=args.label_smoothing, warmup_epochs=args.warmup_epochs,
                        ckpt_path=ckpt, optimizer=args.optimizer)
    train_time = timer.elapsed()
    print(f"best epoch {best['epoch']}: val_acc={best['val_acc']} val_loss={best['val_loss']}")

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    p, labels, _ = predict(model, val_loader, device)
    write_csv(os.path.join(run_dir, "val_preds.csv"),
              [dict(index=int(i), label=int(l), pred=int(q.argmax()), prob=round(float(q.max()), 6))
               for i, l, q in zip(val_idx, labels, p)])

    info = dict(vars(args), n_train=len(train_idx), n_val=len(val_idx), params_total=total,
                params_trainable=trainable, input_size=f"3x{args.img_size or 32}x{args.img_size or 32}",
                best_epoch=best["epoch"], best_val_acc=best["val_acc"], best_val_loss=best["val_loss"],
                train_acc_at_best=next(h["train_acc"] for h in history if h["epoch"] == best["epoch"]),
                last_val_acc=history[-1]["val_acc"], train_seconds=round(train_time, 1),
                gpu=torch.cuda.get_device_name(0))
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
