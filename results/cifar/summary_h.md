# Part h: CIFAR-10 with a few under-labelled classes

All numbers are copied from `results/cifar/imbalance/`.
- Test numbers come from `logs/imb_eval_test.log`: a single run of `scripts/eval_imbalance.py --split test` after every run had finished and the best method had been chosen on val.
- ResNet-18 results are the mean ± sample std over seeds 42, 43 and 44. ConvNeXt results use seed 42 only.

## 1. Setup

**Training set.** We start from the 45,000-image train split of part g.
- **Minority classes:** bird, cat and dog keep only 5% of their images, i.e. 225 per class. The subset is drawn with split seed 42 (`src/cifar.py: make_imbalanced`) and is the same for every run and every training seed.
- **Majority classes:** the other 7 classes keep all 4,500 images.
- **Result:** 32,175 train images with an imbalance ratio of 4500 / 225 = 20 : 1. See `figures/fig_imb_distribution.png`.

**Val set (5,000, 500 per class).** Unchanged from part g and still balanced.
- It is used for model selection: the best epoch per run, and the best method for the ConvNeXt experiment.
- Because it is balanced, val accuracy equals balanced accuracy, so selection does not favour the majority classes.

**Test set.** The official 10,000 test images, unchanged and balanced (1,000 per class). Test accuracy therefore equals balanced accuracy.

**Why bird, cat and dog.** Cat and dog were already the hardest classes in part g (lowest F1 for both models). Making them rare is the worst case and connects to the dogs vs cats task.

**Models**
- **ResNet-18 (G1 recipe from part g):**
  - CIFAR stem, trained from scratch, SGD (momentum 0.9, Nesterov), lr 0.2, wd 5e-4, 2 warmup epochs + cosine schedule, 40 epochs, batch 256.
  - Augmentation: RandomCrop(32, pad 4) + flip + RandomErasing. No label smoothing. AMP.
  - Every method uses exactly these settings. Only the loss or the sampler changes.
- **ConvNeXt-Tiny (G2 recipe from part g):**
  - ImageNet-pretrained, 32x32 inputs upsampled to 224, AdamW lr 3e-4, wd 0.05, 1 warmup epoch + cosine, 8 epochs, batch 128, label smoothing 0.1.
  - Run with B0 and with the best ResNet-18 method (chosen on val), seed 42.

**Metrics.** All are computed on test from the argmax prediction.
- Overall accuracy.
- Balanced accuracy, i.e. the mean per-class recall (macro recall).
- Macro-F1.
- Mean recall of the 3 minority classes.
- Mean recall of the 7 majority classes.

## 2. Methods

Notation:
- n_c is the number of train images of class c.
- π_c = n_c / N is the train class prior.
- z is the logit vector and p = softmax(z).

**B0 – cross-entropy (baseline)**
- Loss: L = −log p_y.
- Every image counts equally, so the 7 majority classes provide 97.9% of the gradient signal. The model learns a bias towards them.

**H1 – re-weighting**
- Weighted cross-entropy: L = −w_y log p_y, with the weights normalised to sum to 10.
- **Class-balanced loss** (Cui et al., 2019): w_c ∝ (1 − β) / (1 − β^{n_c}) with β = 0.999.
  - The term (1 − β^n) / (1 − β) is the "effective number" of samples: extra images of a class add less and less new information because of overlap and augmentation.
  - Here this gives a weight of 2.26 for the minority classes and 0.46 for the majority classes, a ratio of 4.9 : 1.
- **Inverse frequency** (quick comparison): w_c ∝ 1 / n_c.
  - Weights are 2.99 vs 0.149, a ratio of 20 : 1. This fully cancels the imbalance in the expected gradient.

**H2 – re-sampling**
- A `WeightedRandomSampler` draws train images with probability ∝ 1 / n_{y_i}, with replacement. Each epoch has the same number of samples (32,175), so every class appears equally often in expectation and each batch is balanced on average.
- The loss is plain CE and augmentation is unchanged.
- Each minority image is drawn about 14 times per epoch, i.e. roughly 570 times over 40 epochs.

**H3 – logit-adjusted loss** (Menon et al., 2021, τ = 1)
- Loss: L = −log softmax(z + τ log π)_y during training. Prediction uses the raw logits z.
- Adding log π_c during training means the network does not need to learn the prior itself.
  - The logits z then estimate p(y|x) / π_y, which is what a balanced test set needs.
  - Equivalently, the loss asks for a larger margin for rare classes: log 20 ≈ 3.0 nats more for a minority class against a majority class.
- Menon et al. show this is Fisher-consistent for balanced error. Unlike re-weighting, it does not scale the gradients of individual samples.

**H4 – focal loss** (Lin et al., 2017, γ = 2)
- Loss: L = −(1 − p_y)^γ log p_y, with no class-dependent α term.
- It down-weights well-classified examples and focuses on hard ones. It is not class-aware: rare classes are helped only if their examples stay "hard".

## 3. Results on the official test set (10,000 images)

### ResNet-18, mean ± std over seeds 42/43/44 (`imbalance/summary_test.csv`)

| method | test acc = balanced acc | macro-F1 | minority recall (bird, cat, dog) | majority recall (7 classes) | minority Δ vs B0 | majority Δ vs B0 |
|---|---|---|---|---|---|---|
| B0 CE | 0.8313 ± 0.0018 | 0.8195 ± 0.0013 | 0.5281 ± 0.0186 | 0.9613 ± 0.0100 | – | – |
| H1 class-balanced (β 0.999) | 0.8364 ± 0.0035 | 0.8274 ± 0.0057 | 0.5625 ± 0.0274 | 0.9538 ± 0.0070 | +0.0343 | −0.0075 |
| H1 inverse frequency | 0.8146 ± 0.0034 | 0.8082 ± 0.0046 | 0.5814 ± 0.0177 | 0.9145 ± 0.0030 | +0.0533 | −0.0468 |
| H2 re-sampling | 0.8128 ± 0.0047 | 0.8022 ± 0.0086 | 0.5380 ± 0.0638 | 0.9306 ± 0.0276 | +0.0099 | −0.0307 |
| **H3 logit adjustment (τ 1)** | **0.8638 ± 0.0015** | **0.8590 ± 0.0018** | **0.6482 ± 0.0071** | 0.9561 ± 0.0032 | **+0.1201** | −0.0051 |
| H4 focal (γ 2) | 0.8269 ± 0.0016 | 0.8144 ± 0.0017 | 0.5122 ± 0.0068 | 0.9617 ± 0.0026 | −0.0159 | +0.0004 |

### Per-class test recall of the minority classes, mean over 3 seeds (`imbalance/per_class_recall_test.csv`)

| method | bird | cat | dog |
|---|---|---|---|
| B0 CE | 0.538 | 0.476 | 0.570 |
| H1 class-balanced | 0.537 | 0.533 | 0.618 |
| H1 inverse frequency | 0.561 | 0.537 | 0.647 |
| H2 re-sampling | 0.466 | 0.501 | 0.646 |
| H3 logit adjustment | 0.658 | 0.578 | 0.709 |
| H4 focal | 0.484 | 0.476 | 0.577 |
| reference: part g, balanced data (G1, seed 42) | 0.929 | 0.886 | 0.893 |

### Model selection on val

The val ranking by mean balanced accuracy is identical to the test ranking (`imbalance/summary_val.csv`):

| | LA | CB | B0 | focal | inv | resample |
|---|---|---|---|---|---|---|
| val balanced acc | 0.8705 | 0.8454 | 0.8355 | 0.8323 | 0.8211 | 0.8211 |

H3 was therefore chosen for the ConvNeXt experiment before the test set was used.

### ConvNeXt-Tiny pretrained, seed 42

| method | test acc = balanced acc | macro-F1 | minority recall | majority recall | bird / cat / dog recall |
|---|---|---|---|---|---|
| B0 CE | 0.9514 | 0.9505 | 0.8633 | 0.9891 | 0.892 / 0.806 / 0.892 |
| H3 logit adjustment | **0.9668** | **0.9667** | **0.9227** | 0.9857 | 0.941 / 0.898 / 0.929 |
| reference: part g, balanced data (seed 42) | 0.9836 | | 0.9713 | | 0.985 / 0.965 / 0.964 |

### Training time and figures
- **Train time per run (RTX 3090):** ResNet-18 3.4–4.0 min; ConvNeXt 13.3 min (B0) and 12.8 min (H3).
- **Figures:**
  - `imbalance/figures/fig_imb_per_class_recall_test.png`: per-class recall for every method.
  - `imbalance/figures/fig_imb_confusion_test.png`: B0 vs H3, ResNet-18 and ConvNeXt, seed 42, row-normalised.
  - `imbalance/figures/fig_imb_distribution.png`: class distribution.

## 4. Analysis

### Effect of the imbalance (B0)
- **Majority classes are almost unaffected.** Their recall is 0.961.
- **Minority classes collapse.** Recall falls to 0.528, against about 0.90 for the same classes with balanced data in part g.
- **Where the minority images go:** the confusion matrix (ResNet-18 B0, seed 42) shows they are pushed into majority classes that share their appearance or background:
  - **bird** → deer 0.15, frog 0.12, airplane 0.10 (birds on grass, or against the sky);
  - **cat** → frog 0.13, deer 0.12, dog 0.08, horse 0.08;
  - **dog** → cat 0.12, horse 0.12, deer 0.11.
- **Why:** with a 20 : 1 prior, the network's decision boundary moves towards the rare classes. An ambiguous image is cheaper to call "deer" than "bird".

### Trade-off between minority and majority recall

| method | minority gain (points) | majority cost (points) | net balanced accuracy vs B0 (points) |
|---|---|---|---|
| H3 logit adjustment | +12.0 | −0.5 | +3.2 |
| H1 class-balanced | +3.4 | −0.75 | +0.5 |
| H1 inverse frequency | +5.3 | −4.7 | −1.7 |
| H2 re-sampling | +1.0 | −3.1 (seed std 2.8) | −1.9 |
| H4 focal | −1.6 | 0 | −0.4 |

- **H3:** gains about 23 minority points for every majority point lost, so it is the only method that moves clearly towards the balanced optimum.
- **H1 class-balanced:** a smaller but positive trade.
- **H1 inverse frequency:** the majority classes lose almost as much as the minority gains. Balanced accuracy goes down, and overall accuracy goes down too because the test set is balanced.
- **H2 re-sampling:** a small and unstable minority gain at a large majority cost. Seed 44 reaches 0.611 minority recall but drops majority recall to 0.899; seeds 42 and 43 gain almost nothing.
- **H4 focal:** changes nothing for the minority classes.

### Why logit adjustment works best here
1. **It targets the actual problem: a shifted decision rule, not missing features.**
   - Adding log π during training makes the logits estimate p(y|x) / π_y, which is the Bayes-optimal rule for a balanced test set.
   - The rare classes get a margin of log 20 ≈ 3 nats, without changing how much each sample contributes to learning the features.
2. **It does not inflate the gradients of a few samples.**
   - With inverse-frequency weights, each of the 225 minority images gets about 20 times the gradient of a majority image. With re-sampling, it is seen about 570 times.
   - Both push the network to memorise those 225 images. Re-sampling reaches 95–98% train accuracy on the resampled stream and has its best val epoch earlier (24–31) than the other methods (31–40).
   - The boundary then moves into the majority classes: inverse-frequency majority recall drops from 0.961 to 0.915. The results also become seed-dependent: re-sampling has a minority recall std of 0.064, against 0.007 for LA.
3. **The milder class-balanced weights (4.9 : 1)** avoid most of this over-correction. That is why they help a little, while full inverse-frequency weights hurt.
4. **Focal loss is not class-aware.**
   - With strong augmentation and 40 epochs, the network fits the 225 minority train images well (train accuracy about 0.98), so they are no longer "hard" during training. Focal loss then gives them no extra weight.
   - It ends slightly below CE on the minority classes (−1.6 points). Lin et al. designed it for foreground/background imbalance in detection, where most of the easy examples belong to the frequent class.
5. **LA is also the most stable method** across seeds: balanced accuracy std 0.0015, minority recall std 0.0071.

### What is left after LA
- **ResNet-18 + LA still reaches only 0.648 minority recall,** far from 0.90 with balanced data.
- **Re-balancing cannot create missing information.** All these methods change how the classifier uses the features, but 225 images are too few to learn good bird, cat and dog features from scratch.
- **Errors shift between the minority classes.** With LA, part of the remaining error moves into confusion between the two rare classes: on ResNet-18 seed 42, cat → dog rises from 0.08 to 0.19, since both now get the same prior boost and are visually similar (see part g).

### Pretraining (ConvNeXt-Tiny, seed 42)
- **Pretraining is the biggest single factor.** It raises minority recall from 0.528 (ResNet-18 B0, 3-seed mean) to 0.863 (ConvNeXt B0), with no imbalance method at all.
- **Why:** ImageNet features for birds, cats and dogs already exist, so 225 examples are enough to adapt them.
- **LA still helps on top of pretraining:**
  - Minority recall rises by +5.9 points (0.863 → 0.923) and majority recall falls by −0.3 points (0.989 → 0.986).
  - Balanced accuracy rises by +1.5 points (0.9514 → 0.9668).
  - Cat benefits most (0.806 → 0.898).
- **Gap to balanced data:** the remaining gap to part g (0.9836) is 1.7 points.
- **Caveat:** this comparison uses a single seed. The part g seed std for ConvNeXt is 0.0015 on test, so a 1.5-point gain is about 10 std and unlikely to be noise. A second seed would still make it more solid.

### Recommendation
- Use a pretrained backbone when possible, and train it with the logit-adjusted loss.
- LA is a one-line change. It needs no extra hyperparameter search (τ = 1 is the theoretically motivated value) and no change to the data pipeline.
- **If re-weighting is preferred:** use the class-balanced (effective number) weights rather than raw inverse frequency.
- **Avoid aggressive re-sampling** when the minority classes have only a few hundred images.

## 5. Limitations
- **One imbalance setting only:** 3 classes at 5%. Other ratios or long-tailed (exponential) profiles were not tested.
- **No per-method tuning:** β, τ and γ were fixed at the values from the original papers, and the lr / epochs were fixed to the part g recipe. A method might do better with its own lr; for example, inverse frequency effectively raises the step size for minority samples.
- **ConvNeXt:** a single seed per method. Its runs keep the G2 label smoothing of 0.1 for both B0 and LA, so they are comparable with each other but not directly with the ResNet-18 runs, which use no smoothing.
- **Shared selection rule:** the best epoch is chosen on the balanced val set for every method, including B0. This already gives B0 a small advantage over simply taking the last epoch.

## References
- Y. Cui, M. Jia, T.-Y. Lin, Y. Song, S. Belongie, "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
- A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, S. Kumar, "Long-tail Learning via Logit Adjustment", ICLR 2021.
- T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, "Focal Loss for Dense Object Detection", ICCV 2017.
- M. Buda, A. Maki, M. A. Mazurowski, "A systematic study of the class imbalance problem in convolutional neural networks", Neural Networks 106, 2018 (re-sampling).
- H. He and E. A. Garcia, "Learning from Imbalanced Data", IEEE TKDE 21(9), 2009 (re-weighting and re-sampling background).
- A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009 (CIFAR-10).
- See `summary_g.md` for the ResNet-18 / ConvNeXt / augmentation references.
