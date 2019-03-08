#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES='2'
WEIGHTS='./snapshots_101_imagenet/resnet101_traffic_signs_01.h5'
TRAIN_PATH='/home/adrianm/work/data/traffic_signs_04_2018/train'
VALIDATION_PATH='/data/traffic_signs_04_2018/test'

echo 'Parameters:'
echo 'TRAIN_PATH:' $TRAIN_PATH
echo 'VALIDATION_PATH:' $VALIDATION_PATH
echo 'WEIGHTS:' $WEIGHTS
echo '------------------------------------'
PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u ../train.py \
    --steps 100 --multi-gpu 1 --batch-size 1 --evaluate_score_threshold 0.5 \
    --imagenet-weights \
    --snapshot-path 'snapshots_101_imagenet' \
    --backbone 'resnet101' \
    traffic_signs $TRAIN_PATH $VALIDATION_PATH
