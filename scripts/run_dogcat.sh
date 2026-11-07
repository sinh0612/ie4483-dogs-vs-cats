#!/bin/bash
# Dogs vs cats experiments. Usage: bash scripts/run_dogcat.sh <queue>
#   resnet   : M3 lr sweep -> M3 x3 seeds -> M3-noaug
#   convnext : M1, M1-noaug -> M4 lr sweep -> M4 x3 seeds
#   features : frozen features (ResNet-50, CLIP) -> zero-shot + probes
# Set the GPU with CUDA_VISIBLE_DEVICES.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-/usr1/home/s126mdg24_05/miniconda3/envs/ie4483/bin/python}
RUNS=results/dogcat/runs

# best lr among sweep runs: highest val acc, then lowest val loss
best_lr() {
    $PY -c "
import json, glob
infos = [json.load(open(f)) for f in glob.glob('$RUNS/$1_lr*/run_info.json')]
best = max(infos, key=lambda d: (d['best_val_acc'], -d['best_val_loss']))
print(best['lr'])"
}

case $1 in
resnet)
    for lr in 1e-3 3e-4 1e-4 3e-5 1e-5; do
        $PY scripts/train_dogcat.py --model resnet50 --run-name sweep_resnet50_lr$lr --lr $lr --epochs 3 --sweep
    done
    LR=$(best_lr sweep_resnet50)
    echo "ResNet-50 best lr: $LR"
    for seed in 42 43 44; do
        $PY scripts/train_dogcat.py --model resnet50 --run-name M3_resnet50_ft_s$seed --lr $LR --epochs 10 --seed $seed
    done
    $PY scripts/train_dogcat.py --model resnet50 --run-name M3_resnet50_ft_noaug_s42 --lr $LR --epochs 10 --no-aug
    ;;
convnext)
    $PY scripts/train_dogcat.py --model simplecnn --run-name M1_simplecnn_s42 --lr 1e-3 --epochs 20
    $PY scripts/train_dogcat.py --model simplecnn --run-name M1_simplecnn_noaug_s42 --lr 1e-3 --epochs 20 --no-aug
    for lr in 3e-4 1e-4 3e-5; do
        $PY scripts/train_dogcat.py --model convnext_tiny --run-name sweep_convnext_tiny_lr$lr --lr $lr --epochs 3 --sweep
    done
    LR=$(best_lr sweep_convnext_tiny)
    echo "ConvNeXt-Tiny best lr: $LR"
    for seed in 42 43 44; do
        $PY scripts/train_dogcat.py --model convnext_tiny --run-name M4_convnext_tiny_ft_s$seed --lr $LR --epochs 10 --seed $seed
    done
    ;;
features)
    $PY scripts/extract_features.py --backbone resnet50
    $PY scripts/extract_features.py --backbone clip_vitl14
    $PY scripts/probe_features.py --features resnet50
    $PY scripts/probe_features.py --features clip_vitl14 --methods logreg --run-names M5b_clip_vitl14_logreg
    ;;
*)
    echo "unknown queue $1"; exit 1 ;;
esac
echo "queue $1 done"
