# Data audit

## Image counts

| folder | images |
|---|---|
| train/cat | 10000 |
| train/dog | 10000 |
| val/cat | 2500 |
| val/dog | 2500 |
| test | 500 |

Broken images (PIL failed to open/decode): 0

## Image sizes (pixels)

| split | n | width min/median/max | height min/median/max |
|---|---|---|---|
| train | 20000 | 42 / 448 / 1050 | 32 / 374 / 768 |
| val | 5000 | 51 / 447 / 500 | 37 / 374 / 500 |
| test | 500 | 37 / 435.5 / 500 | 50 / 374 / 500 |

Test file names match ids 1..500 in sampleSubmission.csv: yes

## Near-duplicates across splits

Perceptual hash (imagehash.phash, 64-bit), pair counted if Hamming distance <= 5.

| pair | count |
|---|---|
| train-val | 12 |
| train-test | 8 |
| val-test | 0 |

Train-val pairs with different labels: 3

Train images excluded (duplicate of a val or test image): 19
Train images remaining: 19981 of 20000 (cat 9987, dog 9994)

Files: `duplicate_pairs.csv`, `excluded_train.txt`, `duplicate_examples.png`. Original files are not deleted.
