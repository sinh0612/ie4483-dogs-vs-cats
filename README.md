# IE4483 Mini Project (Option 2): Dogs vs. Cats

Binary image classification (dog = 1, cat = 0) for the IE4483 Artificial Intelligence and Data Mining
mini project. The dataset is a subset of the Kaggle Dogs vs. Cats competition: 10000 images per class
for training, 2500 per class for validation, and 500 unlabeled test images. The project also covers
CIFAR-10 classification, including a class-imbalanced setting.

## Layout

```
data/          dataset and resized cache (not in git)
src/           shared code
scripts/       runnable scripts
results/       CSVs and figures
report/        report files
```

## Setup

Python 3.10, conda.

```bash
conda create -n ie4483 python=3.10 -y
conda activate ie4483
# install PyTorch separately, matching your CUDA version. We used torch 2.5.1 + torchvision 0.20.1 (CUDA 12.1):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Put the dataset at `data/dataset/{train/{cat,dog}, val/{cat,dog}, test/}`.

## Data preparation

```bash
python scripts/audit_data.py    # counts, broken files, sizes, near-duplicates -> results/data_audit/
python scripts/cache_images.py  # resized copies (short side 256) -> data/cache256/
```

`results/data_audit/excluded_train.txt` lists training images that are near-duplicates of a val or test
image. They are left out of training.

## How to run

TBD

## Results

TBD
