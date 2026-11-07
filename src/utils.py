import os
import random

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "dataset")
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


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def format_time(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"
