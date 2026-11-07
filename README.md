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

## How to run (dogs vs cats)

Run from the repo root with the `ie4483` env. Pick a free GPU with `CUDA_VISIBLE_DEVICES`.

```bash
python scripts/audit_data.py
python scripts/cache_images.py
python scripts/make_data_figures.py

# three queues, can run in parallel on different GPUs (about 55 min on RTX 3090s)
CUDA_VISIBLE_DEVICES=1 bash scripts/run_dogcat.sh resnet     # lr sweep + ResNet-50 fine-tune (3 seeds) + no-aug run
CUDA_VISIBLE_DEVICES=2 bash scripts/run_dogcat.sh convnext   # SimpleCNN (+ no-aug), ConvNeXt-Tiny sweep + 3 seeds
CUDA_VISIBLE_DEVICES=3 bash scripts/run_dogcat.sh features   # frozen ResNet-50 / CLIP features, zero-shot, linear probes
python scripts/hog_svm.py                                    # HOG + LinearSVC baseline (CPU)

python scripts/analyze_dogcat.py   # tables, McNemar, TTA/ensemble selection on val, submission.csv, figures
python scripts/plot_arch.py simplecnn final
python scripts/gradcam.py --images test/374.jpg test/464.jpg --out fig_gradcam_test.png
```

A single run: `python scripts/train_dogcat.py --model convnext_tiny --run-name my_run --lr 3e-5 --epochs 10 --seed 42`.
Each run writes `results/dogcat/runs/<run_name>/` (`history.csv`, `val_preds.csv`, `test_preds.csv`, `run_info.json`).
Checkpoints go to `checkpoints/` and features to `data/features/` (both not in git).

## Results (dogs vs cats, 5000 validation images)

| model | val acc |
|---|---|
| HOG + LinearSVC | 0.7344 |
| SimpleCNN from scratch | 0.9522 |
| ResNet-50 frozen + logistic regression | 0.9956 |
| ResNet-50 fine-tuned (seeds 42/43/44) | 0.99453 ± 0.00031 |
| ConvNeXt-Tiny fine-tuned (seeds 42/43/44) | 0.99640 ± 0.00087 |
| CLIP ViT-L/14 zero-shot | 0.9984 |
| CLIP ViT-L/14 + logistic regression | 0.9986 |
| Final: CLIP zero-shot + ConvNeXt-Tiny, flip TTA | 0.9990 |

The final config was picked on val, so its val accuracy is slightly optimistic. It is not significantly better than CLIP + logistic regression (McNemar p = 0.69).
Details: `results/dogcat/summary.md`. Predictions: `submission.csv`.

## How to run (CIFAR-10, part g)

```bash
python scripts/prepare_cifar.py                            # download to data/cifar10, 45k/5k split, dataset_info.md
CUDA_VISIBLE_DEVICES=1 bash scripts/run_cifar.sh g1        # ResNet-18 from scratch: lr sweep, erasing / no-aug, 3 seeds (~40 min)
CUDA_VISIBLE_DEVICES=2 bash scripts/run_cifar.sh g2sweep   # ConvNeXt-Tiny lr sweep on 15k images (~11 min)
CUDA_VISIBLE_DEVICES=2 bash scripts/run_cifar.sh g2 42     # ConvNeXt-Tiny 8 epochs at 224 (~18 min per seed), also 43, 44
python scripts/cifar_selection.py                          # val-only tables and G1 variant figure
python scripts/eval_cifar.py --groups G1_resnet18_lr0.2_erase G2_convnext_tiny   # official test set, run once
```

| model (CIFAR-10, seeds 42/43/44) | val acc | test acc |
|---|---|---|
| ResNet-18 (CIFAR stem) from scratch, crop + flip + RandomErasing | 0.94560 ± 0.00250 | 0.9428 ± 0.0021 |
| ConvNeXt-Tiny ImageNet-pretrained, 32 -> 224, fine-tuned | 0.98367 ± 0.00031 | 0.9819 ± 0.0015 |

Details: `results/cifar/summary_g.md`.
