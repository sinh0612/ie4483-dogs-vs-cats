import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from .data import IMAGENET_MEAN, IMAGENET_STD
from .utils import CIFAR_ROOT

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# per-channel mean/std of the 50k original training images
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar(train, download=False):
    ds = CIFAR10(CIFAR_ROOT, train=train, download=download)
    return ds.data, np.array(ds.targets)


def split_train_val(targets, val_per_class=500, seed=42):
    """Class-balanced split of the original training set. Returns sorted (train_idx, val_idx)."""
    rng = np.random.RandomState(seed)
    val_idx = []
    for c in range(len(CIFAR_CLASSES)):
        idx = np.where(targets == c)[0]
        val_idx.extend(rng.choice(idx, val_per_class, replace=False))
    val_idx = np.sort(np.array(val_idx))
    train_idx = np.setdiff1d(np.arange(len(targets)), val_idx)
    return train_idx, val_idx


def get_cifar_transform(train, aug=True, erase=False, imagenet_norm=False):
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if imagenet_norm else (CIFAR_MEAN, CIFAR_STD)
    tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if (train and aug) else []
    tf += [T.ToTensor(), T.Normalize(mean, std)]
    if train and erase:
        tf.append(T.RandomErasing(p=0.5, scale=(0.02, 0.2), value=0))  # Cutout-like, after normalisation
    return T.Compose(tf)


class CifarDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images  # uint8 N x 32 x 32 x 3
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), int(self.labels[i])
