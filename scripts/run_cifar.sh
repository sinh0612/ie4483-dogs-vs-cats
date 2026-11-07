#!/bin/bash
# CIFAR-10 experiments (part g). Usage: bash scripts/run_cifar.sh <queue> [seed]
#   g1       : ResNet-18 lr sweep (no erasing) -> + RandomErasing at best lr -> no-aug -> seeds 43, 44 of the best
#   g2sweep  : ConvNeXt-Tiny lr sweep, 3 epochs on the first 15k train images
#   g2 SEED  : ConvNeXt-Tiny full run (8 epochs) with the best sweep lr, waits for g2sweep
# Set the GPU with CUDA_VISIBLE_DEVICES.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-/usr1/home/s126mdg24_05/miniconda3/envs/ie4483/bin/python}
RUNS=results/cifar/runs
G2_LRS="3e-4 1e-4 3e-5"
G2_ARGS="--model convnext_tiny --optimizer adamw --wd 0.05 --batch-size 128 --warmup-epochs 1 --label-smoothing 0.1 --img-size 224"

# run name with the highest val acc (then lowest val loss) among run dirs matching a glob
best_run() {
    $PY -c "
import json, glob, os
infos = [(json.load(open(f)), os.path.basename(os.path.dirname(f))) for f in glob.glob('$RUNS/$1/run_info.json')]
best = max(infos, key=lambda d: (d[0]['best_val_acc'], -d[0]['best_val_loss']))
print(best[1] if '$2' == 'name' else best[0]['$2'])"
}

case $1 in
g1)
    for lr in 0.05 0.1 0.2; do
        $PY scripts/train_cifar.py --model resnet18_cifar --run-name G1_resnet18_lr${lr}_s42 --lr $lr
    done
    LR=$(best_run "G1_resnet18_lr*_s42" lr)
    echo "G1 best lr: $LR"
    $PY scripts/train_cifar.py --model resnet18_cifar --run-name G1_resnet18_lr${LR}_erase_s42 --lr $LR --erase
    $PY scripts/train_cifar.py --model resnet18_cifar --run-name G1_resnet18_lr${LR}_noaug_s42 --lr $LR --no-aug
    BEST=$(best_run "G1_resnet18_lr${LR}*_s42" name)  # no-aug included: chosen by val only
    echo "G1 best config: $BEST"
    EXTRA=""; [[ $BEST == *erase* ]] && EXTRA="--erase"; [[ $BEST == *noaug* ]] && EXTRA="--no-aug"
    for seed in 43 44; do
        $PY scripts/train_cifar.py --model resnet18_cifar --run-name ${BEST%_s42}_s$seed --lr $LR $EXTRA --seed $seed
    done
    ;;
g2sweep)
    for lr in $G2_LRS; do
        $PY scripts/train_cifar.py $G2_ARGS --run-name sweep_g2_convnext_lr$lr --lr $lr --epochs 3 --limit 15000
    done
    ;;
g2)
    until [ $(ls $RUNS/sweep_g2_convnext_lr*/run_info.json 2>/dev/null | wc -l) -eq 3 ]; do sleep 60; done
    LR=$(best_run "sweep_g2_convnext_lr*" lr)
    echo "G2 best lr: $LR"
    $PY scripts/train_cifar.py $G2_ARGS --run-name G2_convnext_tiny_s$2 --lr $LR --epochs 8 --seed $2
    ;;
esac
