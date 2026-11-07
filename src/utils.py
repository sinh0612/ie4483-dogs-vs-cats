import csv
import os
import random
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "dataset")
CACHE_ROOT = os.path.join(PROJECT_ROOT, "data", "cache256")
FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "features")
DOGCAT_RESULTS = os.path.join(PROJECT_ROOT, "results", "dogcat")
CIFAR_ROOT = os.path.join(PROJECT_ROOT, "data", "cifar10")
CIFAR_RESULTS = os.path.join(PROJECT_ROOT, "results", "cifar")
SPLIT_DIRS = ["train/cat", "train/dog", "val/cat", "val/dog", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def seed_worker(worker_id):
    import torch
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def format_time(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0

    def __str__(self):
        return format_time(self.elapsed())


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = list(rows)
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
