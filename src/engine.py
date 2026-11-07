import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .utils import write_csv


def warmup_cosine(optimizer, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device, tta=False, with_loss=True):
    """Returns (class probabilities N x C, labels, mean cross-entropy loss). tta = average with horizontal flip.
    For the dogs vs cats test set the "labels" are image ids, so pass with_loss=False."""
    model.eval()
    probs, labels, losses = [], [], []
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x).float()
            p = logits.softmax(1)
            if tta:
                p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        probs.append(p.cpu())
        labels.append(y)
        if with_loss:
            losses.append(ce(torch.log(p.clamp_min(1e-7)), y.to(device)).item())
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return probs, labels, (sum(losses) / len(labels) if with_loss else None)


def fit(model, train_loader, val_loader, epochs, lr, weight_decay, device, run_dir,
        label_smoothing=0.1, warmup_epochs=1, ckpt_path=None, optimizer="adamw"):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = len(train_loader)
    scheduler = warmup_cosine(optimizer, warmup_epochs * steps, epochs * steps)
    scaler = torch.amp.GradScaler("cuda")

    history, best = [], None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]  # lr at end of epoch
        probs, labels, val_loss = predict(model, val_loader, device)
        val_acc = float((probs.argmax(1) == labels).mean())
        row = dict(epoch=epoch, lr=lr_now, train_loss=round(train_loss, 5), train_acc=round(train_acc, 5),
                   val_loss=round(val_loss, 5), val_acc=round(val_acc, 5), seconds=round(time.time() - t0, 1))
        history.append(row)
        write_csv(os.path.join(run_dir, "history.csv"), history)
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

        # best by val acc, ties broken by lower val loss
        if best is None or (val_acc, -val_loss) > (best["val_acc"], -best["val_loss"]):
            best = dict(row)
            if ckpt_path:
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
    return history, best
