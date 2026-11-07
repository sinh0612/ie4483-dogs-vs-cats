# Part g: CIFAR-10

All numbers below are copied from the files in `results/cifar/` (seeds stated). Test numbers come from
`logs/cifar_eval_test.log`, a single run of `scripts/eval_cifar.py` after all choices had been made on val.

## 1. Dataset and task

CIFAR-10 (Krizhevsky, 2009), downloaded with `torchvision.datasets.CIFAR10` (archive md5 checked).
10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Images are 32x32 RGB.

| split | images | per class | source |
|---|---|---|---|
| train | 45,000 | 4,500 | original train set minus val |
| val | 5,000 | 500 | class-balanced random split of the original train set, seed 42 (`val_indices.txt`) |
| test | 10,000 | 1,000 | official test set, used once at the end |

Details and per-class counts are in `dataset_info.md`. Sample images are in `figures/fig_cifar_samples.png`.

The task is 10-class classification, compared with 2 classes in dogs vs cats.
- **Images:** much smaller (32x32 instead of a median of about 448x374) and much less detailed.
- **Classes:** several pairs are visually close (cat/dog, automobile/truck, deer/horse).
- **Dogs vs cats as a subset:** cat and dog are 2 of the 10 classes, so part of the earlier problem appears again inside this one.

## 2. Pipeline

### Shared code
The code reuses `src/`:
- `src/data.py`: `loader_for`, a seeded DataLoader now shared with dogs vs cats.
- `src/engine.py`:
  - `fit` gained an SGD option.
  - `predict` now returns the full N x C probability matrix.
  - Accuracy is computed with argmax, so the same code works for 2 or 10 classes.
- `src/models.py`: `resnet18_cifar` and a `ResizeInput` wrapper.
- `src/cifar.py`: data loading, the split and the transforms.

### Scripts

| step | command |
|---|---|
| data | `python scripts/prepare_cifar.py` (download, split, `dataset_info.md`, sample figure) |
| training | `bash scripts/run_cifar.sh g1`, `bash scripts/run_cifar.sh g2sweep`, `bash scripts/run_cifar.sh g2 <seed>` (set `CUDA_VISIBLE_DEVICES`) |
| val summary | `python scripts/cifar_selection.py` (`selection_val.csv`, `selection_val_seeds.csv`, `fig_cifar_g1_variants.png`) |
| test (once) | `python scripts/eval_cifar.py --groups G1_resnet18_lr0.2_erase G2_convnext_tiny` |

`eval_cifar.py --split val` runs the same analysis on val. This was used to check the script before the single test run. One bug, a duplicate column name, was found and fixed this way without touching the test set.

### Two approaches

**G1: ResNet-18 from scratch, adapted to 32x32**
- The stem is changed to a 3x3 stride-1 conv1 and the maxpool is removed. The standard ImageNet stem (7x7 stride 2 conv plus maxpool) would shrink a 32x32 image to 8x8 before the first residual stage. With the change, layer4 still works on 4x4 maps.
- 11,173,962 parameters.

**G2: ConvNeXt-Tiny pretrained on ImageNet-1k, fine-tuned**
- Same backbone and weights as M4 in dogs vs cats; the head is replaced by a 10-class Linear layer.
- Inputs are upsampled from 32 to 224 with bilinear interpolation on the GPU (`ResizeInput`), so the pretrained filters see objects at roughly the scale they were trained on.
- Because the upsampling is done on the GPU, data loading stays cheap: augmentation is applied on the 32x32 image.
- 27,827,818 parameters.

### Changes compared with dogs vs cats

| | dogs vs cats (M4) | G1 ResNet-18 | G2 ConvNeXt-Tiny |
|---|---|---|---|
| output | 2-way Linear | 10-way Linear | 10-way Linear |
| loss | softmax CE, label smoothing 0.1 | softmax CE (10 classes), no smoothing | softmax CE (10 classes), label smoothing 0.1 |
| input | 224 crop from short-side-256 JPEG | 32x32 native | 32x32 upsampled to 224 |
| augmentation | RandomResizedCrop(224, 0.7-1), flip, rotation 15, ColorJitter 0.2 | RandomCrop(32, pad 4), flip, RandomErasing(p 0.5, scale 0.02-0.2, value 0) | RandomCrop(32, pad 4), flip |
| normalisation | ImageNet mean/std | CIFAR-10 mean/std (0.4914, 0.4822, 0.4465) / (0.2470, 0.2435, 0.2616) | ImageNet mean/std (pretrained weights) |
| optimiser | AdamW, wd 0.05 | SGD, momentum 0.9, Nesterov, wd 5e-4 | AdamW, wd 0.05 |
| schedule | 1 warmup epoch + cosine | 2 warmup epochs + cosine | 1 warmup epoch + cosine |
| lr | 3e-5 | 0.2 (swept) | 3e-4 (swept) |
| epochs / batch | 10 / 64 | 40 / 256 | 8 / 128 |
| AMP, channels_last | yes | yes | yes |

**Why the augmentation changed.**
- RandomResizedCrop, rotation and colour jitter were tuned for large photos.
- On 32x32 images, the standard CIFAR augmentation is pad-4 random crop plus flip, which shifts the image by up to 4 px without losing resolution.
- RandomErasing (a Cutout-like augmentation) was kept only because it improved val accuracy (see section 3).

**Model selection.** The best epoch is kept by val accuracy, with ties broken by val loss, exactly as for dogs vs cats.

**Normalisation statistics.** The G1 mean/std are the standard values over all 50k original train images. Computed over our 45k train split only, they differ by at most 1e-4 (see `dataset_info.md`), so the effect is negligible.

## 3. Hyperparameter choice (val only, `selection_val.csv`)

### G1 learning rate (seed 42, no erasing, 40 epochs)

| lr | best val acc | val loss |
|---|---|---|
| 0.05 | 0.9292 | 0.2836 |
| 0.1 | 0.9384 | 0.2246 |
| **0.2** | **0.9426** | 0.2106 |
| 0.4 | 0.9390 | 0.2152 |

- 0.2 was at the upper edge of the first grid {0.05, 0.1, 0.2}, so 0.4 was added. 0.4 is worse, so 0.2 is an interior optimum.
- 0.2 for batch 256 matches the usual 0.1 for batch 128 under the linear scaling rule (Goyal et al., 2017).

### G1 augmentation variants at lr 0.2

| variant | seeds | val acc (mean ± std) | train acc at best epoch (s42) |
|---|---|---|---|
| no augmentation | 42 | 0.8888 | 1.0000 |
| crop + flip | 42 / 43 / 44 | 0.94180 ± 0.00069 (0.9426 / 0.9414 / 0.9414) | 0.9980 |
| **crop + flip + RandomErasing** | 42 / 43 / 44 | **0.94560 ± 0.00250** (0.9464 / 0.9428 / 0.9476) | 0.9854 |

- **The choice was made with seed 42 only:** erasing scored 0.9464 against 0.9426 without it.
- **Seeds 43 and 44 of the no-erasing config were run afterwards, only to check that choice.** Erasing is better by 0.38 points on average. It wins for seeds 42 and 44 and essentially ties for seed 43 (0.9428 vs 0.9414).
- **Without augmentation the model memorises the train set:** train accuracy reaches 1.0000 and val drops by 5.4 points.
- `fig_cifar_g1_variants.png` shows the val curves and the train-val gap for all variants (train accuracy is measured on augmented images).

### G2 learning rate (3-epoch sweep on the first 15,000 train images, seed 42; full val set)

| lr | val acc | val loss |
|---|---|---|
| 1e-3 | 0.9604 | 0.1931 |
| **3e-4** | **0.9730** | 0.1626 |
| 1e-4 | 0.9728 | 0.1669 |
| 3e-5 | 0.9604 | 0.2165 |

- The first grid was {3e-4, 1e-4, 3e-5}. 1e-3 was added because 3e-4 was at the edge of that grid.
- 3e-4 and 1e-4 differ by only 1 val image. 3e-4 was chosen by the rule (val acc, then val loss).
- This is 10 times higher than the 3e-5 chosen for dogs vs cats. A plausible reason is that upsampled 32x32 images are further from ImageNet photos, so the features need to move more. Ten classes also give a harder task than two.

### Other choices (not swept)
- **G1:** 40 epochs, batch 256, wd 5e-4 and the SGD settings are the standard CIFAR ResNet recipe, as given in the task plan.
- **G2:** wd 0.05, label smoothing 0.1 and warmup are kept from dogs vs cats.
- **G2 epochs:** 8. The val curve is flat after epoch 5–6 (`fig_cifar_curves.png`), so more epochs would not help much.

## 4. Results

### Test results (official 10,000-image test set, evaluated once)

| model | params | input | train time per run (RTX 3090) | val acc (mean ± std) | **test acc (mean ± std)** | test acc per seed (42 / 43 / 44) | 95% bootstrap CI, seed 42 |
|---|---|---|---|---|---|---|---|
| G1 ResNet-18 from scratch, erasing | 11.2M | 32x32 | 5.3 min | 0.94560 ± 0.00250 | **0.9428 ± 0.0021** | 0.9418 / 0.9414 / 0.9452 | 0.9372–0.9463 |
| G2 ConvNeXt-Tiny fine-tuned | 27.8M | 224x224 | 19.7 min | 0.98367 ± 0.00031 | **0.9819 ± 0.0015** | 0.9836 / 0.9814 / 0.9807 | 0.9810–0.9860 |

Notes on the table:
- **Spread:** std is the sample std over the 3 seeds. The seed changes the weight initialisation (G1 only), the head initialisation, the data order and the augmentation. The split is fixed.
- **Train time:**
  - G1 seeds took 5.3 min each.
  - G2 seed 42 took 17.5 min and seed 44 took 18.2 min.
  - G2 seed 43 took 23.4 min because it shared GPU3 with the extra lr 1e-3 sweep run for part of the time.
- **Val vs test:** test accuracy is 0.2–0.3 points below val for both models. The val set is only 5,000 images, the gap is within the seed std and the val-test gap is the same order, so this is consistent with normal noise. It is not a sign of overfitting to val.

### Per-class test metrics (seed 42, `per_class_test.csv`)

Recall equals the per-class accuracy.

| class | G1 precision | G1 recall | G1 F1 | G2 precision | G2 recall | G2 F1 |
|---|---|---|---|---|---|---|
| airplane | 0.9389 | 0.953 | 0.9459 | 0.992 | 0.992 | 0.992 |
| automobile | 0.9719 | 0.969 | 0.9705 | 0.9812 | 0.990 | 0.9856 |
| bird | 0.9253 | 0.929 | 0.9271 | 0.986 | 0.985 | 0.9855 |
| **cat** | 0.8695 | 0.886 | **0.8777** | 0.9621 | 0.965 | **0.9636** |
| deer | 0.9371 | 0.954 | 0.9455 | 0.9879 | 0.982 | 0.985 |
| **dog** | 0.9084 | 0.893 | **0.9007** | 0.965 | 0.964 | **0.9645** |
| frog | 0.9646 | 0.954 | 0.9593 | 0.9881 | 0.997 | 0.9925 |
| horse | 0.9786 | 0.959 | 0.9687 | 0.996 | 0.991 | 0.9935 |
| ship | 0.9745 | 0.955 | 0.9646 | 0.991 | 0.989 | 0.990 |
| truck | 0.9517 | 0.966 | 0.9588 | 0.9869 | 0.981 | 0.984 |

### Most confused class pairs on test (seed 42, both directions summed, `confused_pairs_test.csv`)

| rank | G1 (582 errors) | G2 (164 errors) |
|---|---|---|
| 1 | cat <-> dog: 123 (54 cat->dog, 69 dog->cat) | cat <-> dog: 55 (25 cat->dog, 30 dog->cat) |
| 2 | automobile <-> truck: 45 | automobile <-> truck: 23 |
| 3 | airplane <-> ship: 36 | bird <-> deer: 8 |
| 4 | bird <-> cat: 30 | airplane <-> ship: 7 |
| 5 | cat <-> frog: 29 | bird <-> frog: 7 |

### Figures
- `fig_cifar_confusion_test.png`: 10x10 confusion matrices, row-normalised.
- `fig_cifar_errors_<model>_test.png`: the 32 most confident mistakes of each model.
- `fig_cifar_catdog_<model>_test.png`: cat <-> dog mistakes.
- `fig_cifar_curves.png`: train/val loss and accuracy for all seeds.

## 5. Observations

### 1. Pretraining matters even more here than in dogs vs cats
- **Test accuracy:** G2 reaches 98.19% against 94.28% for G1, and its error rate is about 3 times lower (1.8% vs 5.7%).
- **Upsampling:** G2 sees 32x32 images upsampled to 224, so it gets no new detail. Its advantage comes from features learned on 1.28M ImageNet images, while G1 learns from 45k tiny images only.
- **Cost:**
  - G2 has 2.5x the parameters and runs at 224x224, so each epoch costs far more compute.
  - In wall-clock time, 8 G2 epochs took about 3.7x as long as 40 G1 epochs.
- **Consistency with dogs vs cats:** pretrained models also won clearly there (SimpleCNN 95.22% vs fine-tuned ConvNeXt 99.64% on val).

### 2. Augmentation matters much more when training from scratch on small data
- **G1:** crop + flip adds 5.4 val points over no augmentation (0.8888 to 0.9426). RandomErasing adds another 0.38 points on average over 3 seeds.
- **Dogs vs cats:** augmentation gave +0.82 points for the scratch CNN and no measurable gain for the fine-tuned ResNet-50.
- **The no-augmentation G1 run:** train accuracy reaches 1.0000 while val stays at 0.8888. With only 4,500 images per class, a from-scratch network memorises the training set unless augmentation keeps changing the inputs.

### 3. Cat <-> dog is the hardest pair for both models, which links back to dogs vs cats
- **Share of errors:** cat <-> dog is the most confused pair for both models. It accounts for 21% of G1's test errors (123/582) and 34% of G2's (55/164).
- **Per-class F1:** cat and dog have the lowest F1 of all classes for both models (G1 0.878 / 0.901, G2 0.964 / 0.965).
- **Comparison with the binary task:** the same ConvNeXt-Tiny backbone separated cats from dogs with 99.64% val accuracy on full-size photos (dogs vs cats, M4). On CIFAR-10, its recall on these two classes is only about 96.5%.
- **Two reasons:**
  - **Resolution:** at 32x32 the cues that separate the two species (face shape, ears, nose, fur texture) are only a few pixels in size. Upsampling to 224 cannot recover them.
  - **Shared visual setting:** both are furry pets, often photographed indoors, lying down or seen from behind. The G2 mistakes in `fig_cifar_catdog_G2_convnext_tiny_test.png` fall into a few groups:
    - Animals lying down or curled up, seen from behind or with the head hidden (#8296, #4152, #8281, #1180).
    - Animals that are tiny or blurred in the frame (#7853, #4740).
    - One image with two animals (#7002).
    - A few that, to us, look more like the predicted class than the label (#4275, #2005).
  Most of them are hard to label even for a person at this resolution.
- **Takeaway:** dogs vs cats is "easy" for pretrained models mainly because of image resolution, not because the two classes are easy to tell apart.

### 4. Automobile <-> truck is the second-hardest pair
- This holds for both models.
- **Pickups and SUVs:** in the most confident G2 mistakes, many of these images are pickups and SUVs that sit between the two class definitions. CIFAR-10's "truck" is meant for big trucks only.
- **Other errors:** the remaining G2 errors include frogs and birds on cluttered or unusual backgrounds, and a few images that look ambiguous or possibly mislabelled at 32x32. One example is test #3753, labelled deer, which looks like a cat lying down. We did not relabel anything; this is only a visual remark.

### 5. Stability
- G2 is more stable across seeds on val (std 0.0003) than G1 (std 0.0025). Only its head is newly initialised, while G1 starts from random weights everywhere.
- On test, both have a std of about 0.15–0.2 points.

### 6. Limitations
- **G2 lr sweep:** done on a 15k subset for 3 epochs, not on the full schedule.
- **Other G2 hyperparameters:** epochs, weight decay and label smoothing were not swept.
- **Small differences:** differences below about 0.5 points (for example erasing vs no erasing) are close to the seed noise of a single run, which is why 3 seeds were used for the main configs.

## References
- A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", tech. report, 2009 (CIFAR-10).
- K. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016 (ResNet, CIFAR stem).
- Z. Liu et al., "A ConvNet for the 2020s", CVPR 2022 (ConvNeXt).
- T. DeVries and G. Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout", 2017.
- Z. Zhong et al., "Random Erasing Data Augmentation", AAAI 2020.
- P. Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", 2017 (linear lr scaling).
- I. Loshchilov and F. Hutter, "SGDR" (ICLR 2017) and "Decoupled Weight Decay Regularization" (ICLR 2019).
- C. Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016 (label smoothing).
- PyTorch / torchvision (ResNet-18 and ConvNeXt-Tiny ImageNet weights).
