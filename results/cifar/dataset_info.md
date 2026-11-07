# CIFAR-10 dataset

Source: torchvision.datasets.CIFAR10 (Krizhevsky, 2009), stored in data/cifar10.

- Image size: 32x32 RGB, uint8
- Original train set: 50000 images, official test set: 10000 images
- Our split of the original train set (seed 42, 500 val images per class): train 45000, val 5000
- Val indices (into the original train set): results/cifar/val_indices.txt
- The official test set is used only once, for the final evaluation.

| id | class | train | val | test |
|---|---|---|---|---|
| 0 | airplane | 4500 | 500 | 1000 |
| 1 | automobile | 4500 | 500 | 1000 |
| 2 | bird | 4500 | 500 | 1000 |
| 3 | cat | 4500 | 500 | 1000 |
| 4 | deer | 4500 | 500 | 1000 |
| 5 | dog | 4500 | 500 | 1000 |
| 6 | frog | 4500 | 500 | 1000 |
| 7 | horse | 4500 | 500 | 1000 |
| 8 | ship | 4500 | 500 | 1000 |
| 9 | truck | 4500 | 500 | 1000 |
| | total | 45000 | 5000 | 10000 |

Per-channel mean / std used for normalisation in G1 (standard values over the 50k original train images) and the same statistics over our 45k train split only:
- 50k: mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616)
- 45k: mean (0.4913, 0.4821, 0.4465), std (0.2470, 0.2434, 0.2615)

Sample images: figures/fig_cifar_samples.png
