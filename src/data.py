import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .utils import CACHE_ROOT, PROJECT_ROOT, list_images, seed_worker

CLASSES = ["cat", "dog"]  # cat=0, dog=1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXCLUDED_FILE = os.path.join(PROJECT_ROOT, "results", "data_audit", "excluded_train.txt")


def load_excluded():
    with open(EXCLUDED_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def get_items(split, exclude_dups=True):
    """List of (relative path, label). Test labels are the image ids instead."""
    if split == "test":
        names = list_images(os.path.join(CACHE_ROOT, "test"))
        names = sorted(names, key=lambda n: int(os.path.splitext(n)[0]))
        return [(f"test/{n}", int(os.path.splitext(n)[0])) for n in names]
    excluded = load_excluded() if (split == "train" and exclude_dups) else set()
    items = []
    for label, c in enumerate(CLASSES):
        for n in list_images(os.path.join(CACHE_ROOT, split, c)):
            rel = f"{split}/{c}/{n}"
            if rel not in excluded:
                items.append((rel, label))
    return items


def get_transform(train, aug=True, size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if train and aug:
        tf = [
            T.RandomResizedCrop(size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(0.2, 0.2, 0.2),
        ]
    else:
        tf = [T.Resize(256), T.CenterCrop(size)]
    return T.Compose(tf + [T.ToTensor(), T.Normalize(mean, std)])


class DogCatDataset(Dataset):
    def __init__(self, items, transform, root=CACHE_ROOT):
        self.items = items
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        return self.transform(img), label


def subsample(items, n, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(items), generator=g)[:n].tolist()
    return [items[i] for i in sorted(idx)]


def make_loader(items, transform, batch_size, shuffle, seed=42, num_workers=12):
    return loader_for(DogCatDataset(items, transform), batch_size, shuffle, seed, num_workers)


def loader_for(dataset, batch_size, shuffle, seed=42, num_workers=12, sampler=None):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=num_workers, pin_memory=True, drop_last=False,
                      worker_init_fn=seed_worker, generator=g, persistent_workers=num_workers > 0)
