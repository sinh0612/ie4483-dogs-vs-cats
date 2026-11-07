# Dogs vs Cats: results summary

All numbers come from the files in `results/dogcat/` and were measured on 7 Nov 2026. Accuracy is on the
5000-image validation set unless stated otherwise. Labels: cat = 0, dog = 1. Default seed is 42. M3 and M4
were also run with seeds 43 and 44.

## a) Data and preprocessing

| split | used | notes |
|---|---|---|
| train | 19981 (cat 9987, dog 9994) | 20000 given. 19 images removed because they are near-duplicates of val/test images (pHash Hamming <= 5, see `results/data_audit/`) |
| val | 5000 (2500 / 2500) | all, used only for validation / model selection |
| test | 500 | unlabeled, used only for the final prediction |

- Image sizes vary: median about 448x374, smallest 37x32, largest 1050x768. No images are broken.
- Offline resize: shorter side resized to 256 (aspect ratio kept, bicubic, JPEG q95) and stored in `data/cache256/` (`scripts/cache_images.py`).
- Train transform: RandomResizedCrop(224, scale 0.7-1.0), RandomHorizontalFlip, RandomRotation(15), ColorJitter(0.2, 0.2, 0.2), ImageNet mean/std normalisation.
  Rotation leaves black corners (see `figures/fig_augmentation.png`).
- Val/test transform: Resize(256) + CenterCrop(224) + normalisation. The CLIP models use CLIP's own preprocessing instead (Resize 224 bicubic, CenterCrop 224, CLIP mean/std).
- HOG baseline: grayscale, shorter side resized to 128, center crop 128x128.
- Figures: `figures/fig_train_samples.png`, `figures/fig_augmentation.png`.

## b) Models

| id | model | trainable / total params | input |
|---|---|---|---|
| M0 | HOG (9 orient., 8x8 cells, 2x2 blocks, L2-Hys, 8100-d) + StandardScaler + LinearSVC | 8101 / 8101 | 1x128x128 |
| M1 | SimpleCNN from scratch: 5x [Conv3x3-BN-ReLU-MaxPool2] (32-64-128-256-512 ch), GAP, Dropout 0.3, Linear 512->2. Kaiming init | 1,570,594 | 3x224x224 |
| M2a/b/c | ResNet-50 IMAGENET1K_V2, frozen, 2048-d pooled features -> (a) LogisticRegression, (b) LinearSVC, (c) nn.Linear(2048,2) | 2049 / 2049 / 4098 trainable (backbone 23,508,032 frozen) | 3x224x224 |
| M3 | ResNet-50 IMAGENET1K_V2, fc replaced by Linear(2048,2), all layers fine-tuned | 23,512,130 | 3x224x224 |
| M4 | ConvNeXt-Tiny IMAGENET1K_V1, last Linear replaced by Linear(768,2), all layers fine-tuned | 27,821,666 | 3x224x224 |
| M5a | CLIP ViT-L/14 (open_clip, pretrained "openai") zero-shot, prompts "a photo of a cat." / "a photo of a dog.", softmax(100 * cosine). No training | 0 / 427,616,513 | 3x224x224 |
| M5b | Same CLIP image encoder, frozen 768-d features -> StandardScaler -> LogisticRegression | 769 trainable | 3x224x224 |

ViT-L/14 fitted in memory without problems. Feature extraction for train + val + test, plus flipped val/test, took 208.6 s on one RTX 3090, so ViT-B/16 was not needed.

Architecture figures and tables:
- `figures/fig_arch_simplecnn.png` and `arch_simplecnn.txt` (output shape and params per layer).
- `figures/fig_arch_final.png` (final ensemble).
- `arch_convnext_tiny.txt`, `arch_clip_vitl14.txt`, `arch_resnet50.txt`.

Tensor shapes:
- SimpleCNN: 3x224x224 -> 32x112x112 -> 64x56x56 -> 128x28x28 -> 256x14x14 -> 512x7x7 -> GAP 512 -> 2.
- ConvNeXt-Tiny: 3x224x224 -> 96x56x56 -> 96x56x56 -> 192x28x28 -> 384x14x14 -> 768x7x7 -> 768 -> 2.
- ResNet-50: 3x224x224 -> 64x56x56 -> 256x56x56 -> 512x28x28 -> 1024x14x14 -> 2048x7x7 -> 2048 -> 2.

Training setup for M1, M3 and M4 (`src/engine.py`, `scripts/train_dogcat.py`):
- Loss: cross-entropy with label smoothing 0.1. Val loss is reported without smoothing. With smoothing 0.1 and 2 classes the train loss cannot go below about 0.199, which is why train loss in `fig_curves.png` flattens at about 0.2.
- Optimizer: AdamW, weight decay 0.05, batch 64. Mixed precision (fp16 autocast + GradScaler), channels_last.
- LR schedule: per-step linear warmup for 1 epoch, then cosine decay to 0.
- Epochs: 20 for M1, 10 for M3/M4. The checkpoint kept is the epoch with the best val acc (ties go to lower val loss).
- Peak LR: M1 1e-3, M3 1e-4, M4 3e-5. See c).
- M2c: AdamW lr 1e-3, wd 0.05, batch 256, 30 epochs, same warmup + cosine schedule and label smoothing, best epoch on val (epoch 13).
- Reproducibility: random/numpy/torch seeds and the DataLoader generator are seeded. cudnn.benchmark is on, so runs are not bit-exact. Seed variance is measured instead (M3, M4: seeds 42/43/44).
- M0, M2a, M2b and M5b: StandardScaler fitted on train features only; deterministic solvers (lbfgs, liblinear primal).
- M5a has no randomness. Running M5a/M5b with other seeds would give identical results, so the std over seeds is 0 by construction and they were not re-run.

How to run: see the Dogs vs Cats section of `README.md` (`scripts/run_dogcat.sh`, then `scripts/analyze_dogcat.py`).

## c) How parameters were chosen (all on val, never on test)

Learning rate sweep (`lr_sweep.csv`, `figures/fig_lr_sweep.png`): 3 epochs per value (1 warmup + cosine), best val acc of the 3 epochs.

| lr | ResNet-50 val acc / loss | ConvNeXt-Tiny val acc / loss |
|---|---|---|
| 1e-3 | 0.9844 / 0.0857 | - |
| 3e-4 | 0.9908 / 0.0709 | 0.9938 / 0.0622 |
| 1e-4 | **0.9942** / 0.0607 | 0.9962 / 0.0571 |
| 3e-5 | 0.9940 / 0.0714 | **0.9962** / 0.0569 |
| 1e-5 | 0.9912 / 0.0857 | 0.9954 / 0.0633 |

- ResNet-50: 1e-4 is chosen. Larger LRs damage the pretrained weights (1e-3 loses about 1 pp), and 1e-5 is too slow for 3 epochs.
- ConvNeXt: 1e-4 and 3e-5 tie on val acc. 3e-5 has the lower val loss, so it is chosen (the rule decided in advance).
- The ConvNeXt sweep was planned with 3 values (3e-4, 1e-4, 3e-5). Because 3e-5 landed on the edge of that grid, 1e-5 was added afterwards: 0.9954, so 3e-5 is an interior optimum.
- Pretrained networks need a LR 10-100x smaller than training from scratch. ConvNeXt prefers a smaller LR than ResNet-50.
- C for the sklearn models (sweeps in `runs/*/history.csv`):
  - M2a LogReg: best C = 1e-3 (0.9956) from {1e-4 ... 1}.
  - M2b SVM: best C = 1e-4 (0.9952) from {1e-5 ... 1e-2}.
  - M5b: C = 1e-4. Val acc is flat at 0.9986 for C in 1e-4 ... 1e-2, so the smallest C (strongest regularisation) is taken on the tie.
  - M0 HOG SVM: best C = 3e-6 (0.7344).
  - The first two HOG grids had their best value at the lower edge, so the grid was moved down. Larger C overfits: C = 1e-4 gives 0.7150, C = 1e-3 gives 0.6964. The earlier logs are kept in `logs/dc_hog_*_grid.log`.
- Not tuned (fixed beforehand, common defaults): weight decay 0.05, label smoothing 0.1, batch size 64, 10 fine-tuning epochs, M1 lr 1e-3, M1 dropout 0.3.
  - The 10 epochs were enough: best epochs were 4-9 for M3 and 5-8 for M4, and the val curves are flat after epoch 2 (`figures/fig_curves_finetune.png`).
- M1 epoch time was 14 s and ResNet-50 47 s on one RTX 3090. The batch size of 64 fits easily (ResNet-50 used about 4 GB).

## d) Validation accuracy (`comparison.csv`)

The 95% CI is a bootstrap over the 5000 val images (2000 resamples, seed 42) for the seed-42 run. "gap" = train acc at the best epoch minus val acc.

| model | val acc (mean ± std over seeds) | seeds | 95% CI (s42) | wrong (s42) | gap (s42) | train time |
|---|---|---|---|---|---|---|
| M0 HOG + LinearSVC | 0.7344 | 42 | 0.7222-0.7462 | 1328 | 0.0224 | 26.3 s features + 113.1 s for 5 C values |
| M1 SimpleCNN, no aug | 0.9440 | 42 | 0.9374-0.9504 | 280 | 0.0544 | 281.8 s |
| M1 SimpleCNN | 0.9522 | 42 | 0.9460-0.9580 | 239 | 0.0013 | 283.5 s |
| M2b ResNet-50 frozen + SVM | 0.9952 | 42 | 0.9932-0.9970 | 24 | 0.0033 | 31.7 s features + 31.0 s |
| M2a ResNet-50 frozen + LogReg | 0.9956 | 42 | 0.9938-0.9972 | 22 | 0.0025 | 31.7 s features + 13.9 s |
| M2c ResNet-50 frozen + linear | 0.9944 | 42 | 0.9924-0.9962 | 28 | 0.0042 | 31.7 s features + 5.9 s |
| M3 ResNet-50 fine-tune, no aug | 0.9948 | 42 | 0.9928-0.9966 | 26 | 0.0045 | 448.2 s |
| M3 ResNet-50 fine-tune | 0.99453 ± 0.00031 (0.9946, 0.9948, 0.9942) | 42, 43, 44 | 0.9924-0.9966 | 27 | 0.0048 | 450.8 s mean |
| M4 ConvNeXt-Tiny fine-tune | 0.99640 ± 0.00087 (0.9968, 0.9970, 0.9954) | 42, 43, 44 | 0.9952-0.9982 | 16 | 0.0021 | 628.9 s mean |
| M5a CLIP ViT-L/14 zero-shot | 0.9984 | deterministic | 0.9972-0.9994 | 8 | - | 208.6 s features, no training |
| M5b CLIP ViT-L/14 + LogReg | 0.9986 | deterministic | 0.9974-0.9996 | 7 | -0.0003 | 208.6 s features + 5.3 s |

Notes on the table:
- The std is the sample std (ddof = 1).
- Train acc for M1/M3/M4 is measured on augmented images in train mode (dropout on), so the gap is only indicative.
- Train times are for one run on one RTX 3090, including val evaluation every epoch. The CPU was shared with other users' jobs.

### Final configuration (`final_selection.csv`, `final_config.json`)

Candidates: the 3 best models that output real probabilities (M5b, M5a, M4; seed-42 runs). Each candidate was tried alone and as averaged-probability ensembles of 2 and 3 models, with and without flip TTA (18 configs). Selection rule: highest val acc, ties go to lower val NLL, then fewer models.

- **Final = M5a (CLIP zero-shot) + M4 (ConvNeXt-Tiny, seed 42), average of P(dog), with horizontal-flip TTA.**
  Val acc **0.9990 (5 wrong of 5000)**, val NLL 0.0286. Confusion matrix: `figures/fig_confusion_final.png`.
- Honest caveat: the final config is better than the best single model (M5b, 0.9986) by only 2 images net. It is right on 4 images M5b gets wrong and wrong on 2 images M5b gets right. McNemar p = 0.69, so the difference is not significant. Because val was used to pick 1 config out of 18, 0.9990 is slightly optimistic as an estimate of test accuracy.
- TTA per single model (seed 42, plain -> flip TTA):
  - Small gains: M5a 0.9984 -> 0.9988, M5b 0.9986 -> 0.9988, M3 0.9946 -> 0.9954.
  - No change: M4 0.9968 -> 0.9968.
  - Worse: M2c 0.9944 -> 0.9928.
  - Larger gains for the weak models: M0 +0.0058, M1 +0.0016.
- On the 500 test images, the final labels are identical to those of M4, M5a and M5b alone, so the submission does not depend on the ensemble choice. Other models differ from the final on: M3 2 images, M2a 4, M2c 3, M1 21, M0 127.
- `submission.csv` (repo root and `results/dogcat/`):
  - Columns `id,label`, ids in the same order as `sampleSubmission.csv`.
  - 500 rows, labels only 0/1 (checked in the script).
  - 259 dogs, 241 cats.
- `test_probs.csv` has P(dog) for every model (seed 42, no TTA) plus `final`.

## e) Correct and incorrect samples

Test labels are not known. The judgements below come from looking at the images. No external label source was used.

Validation errors of the final model (`figures/fig_val_errors_final.png`, `val_errors_final.csv`): 5 images.
- 4 of the 5 are ambiguous or arguably mislabeled:
  - `dog.8507`: a puppy and a kitten together. P(dog) = 0.22.
  - `cat.9882`: a cat and a dog. P(dog) = 0.64.
  - `cat.7920`: a black cat next to a large dog that fills most of the image. P(dog) = 0.61.
  - `cat.10539`: an extreme close-up of a paw. P(dog) = 0.62.
- Only `dog.12092` (a black-and-white puppy held by hands) is a clear model error. P(dog) = 0.45.
- Grad-CAM (`figures/fig_gradcam_val_errors.png`):
  - On `dog.8507`, ConvNeXt focuses on the kitten. On `cat.7920`, both CNNs focus on the dog's head and body. So the "errors" follow the more visible animal.
  - On `dog.12092`, both fine-tuned CNNs are right (P(dog) 0.85 / 0.82). The ensemble is pulled below 0.5 by CLIP.

Test images where the 5 strong models (M2a, M3, M4, M5a, M5b) disagree, or the final confidence is lowest (`figures/fig_test_uncertain.png`, `test_uncertain.csv`): only 5 of 500. The most confident ones are in `figures/fig_test_confident.png`. The maximum final P is about 0.98, because M4 was trained with label smoothing.

Selected cases:

1. **Correct, hard: test 374.** By eye, two or three kittens (orange/white) curled up inside a fabric cat cube. Final P(dog) = 0.182, predicted cat, which is correct.
   - The ImageNet-feature models lean towards dog: M2a 0.70, M2c 0.70.
   - The fine-tuned CNNs are less sure: M3 0.24, M4 0.47.
   - CLIP is very sure it is a cat: M5a 0.001, M5b 0.018.
   - Grad-CAM (`figures/fig_gradcam_test.png`): ResNet-50 looks at the kittens' bodies and the white paw. ConvNeXt's map is spread over the top of the cube opening.
   - Strength shown: the model works with curled-up animals and no visible face, mainly thanks to CLIP's broad pretraining.
   - Weakness shown: fur texture and colour alone (orange/white) are ambiguous for CNNs trained on 20k images.
2. **Weakness, no correct answer: tests 464 and 328.** The same "No Photo Available" placeholder: a brown dog-shaped silhouette with text and no real animal. Final P(dog) is about 0.78, predicted dog.
   - The CNNs are close to 0.5 (M3 0.41, M4 0.52).
   - CLIP says dog with high confidence (M5a 0.99, M5b about 0.90), probably from the silhouette shape.
   - Grad-CAM: ConvNeXt spreads attention over the silhouette edges and the background. ResNet-50 looks at a small patch near the top.
   - Weakness shown: the binary classifier has no "neither" option and gives confident answers to out-of-distribution inputs.
   - Context: the same placeholder appears in the training set labeled dog (`train/dog/dog.8898.jpg`). It was removed from training by the duplicate audit, so the models did not memorise it. A different placeholder (a "PHOTO UNAVAILABLE" cat drawing) is labeled dog in 3 train images and cat in val (`val/cat/cat.11565.jpg`; see `results/data_audit/duplicate_pairs.csv`), so the dataset labels placeholder images inconsistently.

Other low-confidence test images:
- test 47: puppies in a basket behind a wire fence, small in the frame. P(dog) = 0.80, 5/5 votes dog. By eye, dog.
- test 242: a black cat. P(dog) = 0.12. M2a and M2c say dog (0.59, 0.67). By eye, cat.

## f) Effect of model and data choices (comparison chain, `mcnemar.csv`)

Each step changes one thing. Accuracies are for the seed-42 runs. "Δ" is in percentage points. The McNemar test is exact (binomial on discordant images), on the same 5000 val images.

| step | change | Δ acc | discordant (A right / B right) | p | significant? |
|---|---|---|---|---|---|
| M0 -> M1 | hand-crafted HOG + SVM -> CNN trained from scratch | +21.78 | 119 / 1208 | 2.4e-227 | yes |
| M1-noaug -> M1 | augmentation on (from scratch) | +0.82 | 92 / 133 | 0.0075 | yes |
| M1 -> M2c | scratch CNN -> frozen ImageNet ResNet-50 + linear | +4.22 | 10 / 221 | 5.9e-53 | yes |
| M2b -> M2a | frozen features: SVM -> LogReg | +0.04 | 2 / 4 | 0.69 | no |
| M2a -> M2c | LogReg -> PyTorch linear layer | -0.12 | 14 / 8 | 0.29 | no |
| M2c -> M3 | frozen -> full fine-tuning (ResNet-50) | +0.02 | 17 / 18 | 1.00 | no |
| M3-noaug -> M3 | augmentation on (pretrained, fine-tuned) | -0.02 | 13 / 12 | 1.00 | no |
| M3 -> M4 | ResNet-50 -> ConvNeXt-Tiny | +0.22 | 7 / 18 | 0.043 | yes (borderline) |
| M5a -> M5b | CLIP zero-shot -> CLIP linear probe | +0.02 | 0 / 1 | 1.00 | no |
| M4 -> M5b | fine-tuned ConvNeXt-Tiny -> frozen CLIP ViT-L/14 + LogReg | +0.18 | 6 / 15 | 0.078 | no |

Main points:
- The biggest jump comes from learned features (+21.8 pp), the next from **pretraining** (+4.2 pp). After that, the differences between pretrained models are 0.0-0.4 pp, about the size of the CIs (each CI is about ±0.2 pp at 99.5%).
- Augmentation helps when training from scratch:
  - +0.82 pp, significant.
  - The train-val gap drops from 5.4 pp to 0.1 pp (`figures/fig_curves.png`: the no-aug train acc reaches 99.8% while val stays at 94.4%).
- Augmentation does not help the fine-tuned ResNet-50 here: 0.9948 without vs 0.9945 ± 0.0003 with. The pretrained model barely overfits with 20k images in 10 epochs.
- Fine-tuning the whole ResNet-50 gives no measurable gain over a linear classifier on frozen features, at about 10x the compute. M3 0.9945 over 3 seeds vs M2a 0.9956 / M2c 0.9944.
- ConvNeXt-Tiny is a modern architecture with a similar parameter count. It is better than ResNet-50:
  - Seed means: 0.99640 ± 0.00087 vs 0.99453 ± 0.00031.
  - Every ConvNeXt seed (0.9954-0.9970) is at or above every ResNet seed (0.9942-0.9948).
  - McNemar p = 0.043 for seed 42.
- CLIP ViT-L/14 is the best model overall:
  - Zero-shot already reaches 0.9984 with no training at all.
  - A linear probe adds 1 image (0.9986).
  - It is much larger (428M params) and was pretrained on 400M image-text pairs. Its advantage over fine-tuned ConvNeXt is not significant (p = 0.078). It is significant over the ResNet-50 feature models (p = 0.003-0.016).
- Choice of linear classifier on frozen features (SVM / LogReg / linear layer) does not matter (p >= 0.29).
- Seed variance of the fine-tuned models is 0.03-0.09 pp, smaller than the differences between some models but not all.

## Hardware, software, time

- GPU: NVIDIA GeForce RTX 3090 24 GB, driver 580.173.02. The server has 4 GPUs shared with other users. Runs used GPUs 1-3, one job per GPU.
- CPU: Intel Xeon W-2295. The data loaders used 12 workers.
- Software:
  - Python 3.10.21, torch 2.5.1+cu121, torchvision 0.20.1+cu121, open_clip 2.32.0.
  - scikit-learn 1.7.2, scikit-image 0.25.2, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, Pillow 12.3.0, matplotlib 3.10.9.
- Wall-clock for all experiments: about 53 min (13:26-14:19, with GPUs 1 and 2 running in parallel), plus 3 min for the extra ConvNeXt lr = 1e-5 sweep point.
  - Sweep runs: 2m10s-3m20s each.
  - M3 runs: about 7m45s each.
  - M4 runs: about 10m50s each.
  - M1 runs: about 4m50s each.
- Logs are in `logs/` (not in git).

## Files per question

- a) `figures/fig_train_samples.png`, `figures/fig_augmentation.png`, `results/data_audit/audit.md`
- b) `figures/fig_arch_simplecnn.png`, `figures/fig_arch_final.png`, `arch_*.txt`, `src/`, `scripts/`
- c) `figures/fig_lr_sweep.png`, `lr_sweep.csv`, `runs/*/history.csv`
- d) `comparison.csv`, `figures/fig_model_comparison.png`, `figures/fig_confusion_final.png`, `final_selection.csv`, `final_config.json`, `submission.csv`, `test_probs.csv`
- e) `figures/fig_val_errors_final.png`, `figures/fig_test_uncertain.png`, `figures/fig_test_confident.png`, `figures/fig_gradcam_test.png`, `figures/fig_gradcam_val_errors.png`, `test_uncertain.csv`, `val_errors_final.csv`
- f) `mcnemar.csv`, `comparison.csv`, `figures/fig_curves.png`, `figures/fig_curves_finetune.png`, `figures/fig_model_comparison.png`
