#!/bin/bash
# Part h: CIFAR-10 with bird, cat, dog reduced to 5% of their train images.
# Usage: bash scripts/run_imbalance.sh g1 <seed>   : ResNet-18 (G1 recipe) with every method
#        bash scripts/run_imbalance.sh g2 ce|best  : ConvNeXt-Tiny (G2 recipe), seed 42; waits for g1, best is picked on val
# Set the GPU with CUDA_VISIBLE_DEVICES.
set -e
cd "$(dirname "$0")/.."
source scripts/config.sh
RUNS=results/cifar/runs
METHODS="ce cb inv resample la focal"
IMB="--minority bird cat dog --keep-frac 0.05"

case $1 in
g1)
    for m in $METHODS; do
        $PY scripts/train_cifar.py --model resnet18_cifar --run-name H_resnet18_${m}_s$2 --lr 0.2 --erase \
            --method $m $IMB --seed $2
    done
    ;;
g2)
    # wait for all G1 runs, then take the method with the highest mean val acc over seeds (val is balanced)
    until [ $(ls $RUNS/H_resnet18_*_s4[234]/run_info.json 2>/dev/null | wc -l) -eq 18 ]; do sleep 60; done
    BEST=$($PY -c "
import json, glob, numpy as np
acc = {m: np.mean([json.load(open(f))['best_val_acc'] for f in glob.glob(f'$RUNS/H_resnet18_{m}_s*/run_info.json')])
       for m in '$METHODS'.split() if m != 'ce'}
print(max(acc, key=acc.get))")
    echo "best method on val: $BEST"
    M=$2; [ "$M" == best ] && M=$BEST
    LS=0.1; [ "$BEST" == focal ] && LS=0  # focal is used without label smoothing; keep both G2 runs the same
    $PY scripts/train_cifar.py --model convnext_tiny --optimizer adamw --lr 3e-4 --wd 0.05 --batch-size 128 \
        --warmup-epochs 1 --label-smoothing $LS --img-size 224 --epochs 8 --run-name H_convnext_${M}_s42 \
        --method $M $IMB --seed 42
    ;;
esac
