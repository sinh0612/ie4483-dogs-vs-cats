"""Train a CNN on dogs vs cats, keep the best epoch by val acc, then predict val and test (with and without flip TTA)."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_items, get_transform, make_loader, subsample
from src.engine import fit, predict
from src.models import build_model, count_params
from src.utils import DOGCAT_RESULTS, PROJECT_ROOT, Timer, set_seed, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["simplecnn", "resnet50", "convnext_tiny"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="use only N train / N val images (smoke test)")
    parser.add_argument("--sweep", action="store_true", help="no checkpoint, no test predictions")
    parser.add_argument("--out", default=os.path.join(DOGCAT_RESULTS, "runs"))
    args = parser.parse_args()

    timer = Timer()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    run_dir = os.path.join(args.out, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_items, val_items, test_items = get_items("train"), get_items("val"), get_items("test")
    if args.limit:
        train_items = subsample(train_items, args.limit, args.seed)
        val_items = subsample(val_items, args.limit, args.seed)
    print(f"train {len(train_items)}  val {len(val_items)}  test {len(test_items)}")

    train_loader = make_loader(train_items, get_transform(True, aug=not args.no_aug), args.batch_size,
                               shuffle=True, seed=args.seed, num_workers=args.workers)
    eval_tf = get_transform(False)
    val_loader = make_loader(val_items, eval_tf, 128, shuffle=False, num_workers=args.workers)

    model = build_model(args.model).to(device).to(memory_format=torch.channels_last)
    total, trainable = count_params(model)
    print(f"{args.model}: {total:,} params ({trainable:,} trainable)")

    ckpt = None if args.sweep else os.path.join(PROJECT_ROOT, "checkpoints", "dogcat", f"{args.run_name}.pt")
    history, best = fit(model, train_loader, val_loader, args.epochs, args.lr, args.wd, device, run_dir,
                        label_smoothing=args.label_smoothing, ckpt_path=ckpt)
    train_time = timer.elapsed()
    print(f"best epoch {best['epoch']}: val_acc={best['val_acc']} val_loss={best['val_loss']}")

    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        p, y, _ = predict(model, val_loader, device)
        p_tta, _, _ = predict(model, val_loader, device, tta=True)
        write_csv(os.path.join(run_dir, "val_preds.csv"),
                  [dict(path=it[0], label=int(l), prob_dog=round(float(a), 6), prob_dog_tta=round(float(b), 6))
                   for it, l, a, b in zip(val_items, y, p, p_tta)])
        test_loader = make_loader(test_items, eval_tf, 128, shuffle=False, num_workers=4)
        p, ids, _ = predict(model, test_loader, device, with_loss=False)
        p_tta, _, _ = predict(model, test_loader, device, tta=True, with_loss=False)
        write_csv(os.path.join(run_dir, "test_preds.csv"),
                  [dict(id=int(i), prob_dog=round(float(a), 6), prob_dog_tta=round(float(b), 6))
                   for i, a, b in zip(ids, p, p_tta)])

    info = dict(vars(args), n_train=len(train_items), n_val=len(val_items), params_total=total,
                params_trainable=trainable, input_size="3x224x224", best_epoch=best["epoch"],
                best_val_acc=best["val_acc"], best_val_loss=best["val_loss"],
                train_acc_at_best=next(h["train_acc"] for h in history if h["epoch"] == best["epoch"]),
                train_seconds=round(train_time, 1), gpu=torch.cuda.get_device_name(0))
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Total time: {timer}")


if __name__ == "__main__":
    main()
