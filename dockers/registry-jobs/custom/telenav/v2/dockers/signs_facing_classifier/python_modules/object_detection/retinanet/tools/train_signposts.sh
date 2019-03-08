#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES='3'
WEIGHTS='./snapshots/resnet101_traffic_signs_145.h5'
WEIGHTS='./snapshots_signpost_components_101/resnet101_traffic_signs_08.h5'
TRAIN_PATH='/data/datasets/traffic_sign_components_09_2018/v_0_1/filtered_dataset_2018_12/train'
VALIDATION_PATH='/data/datasets/traffic_sign_components_09_2018/v_0_1/filtered_dataset_2018_12/test'

echo 'Parameters:'
echo 'TRAIN_PATH:' $TRAIN_PATH
echo 'VALIDATION_PATH:' $VALIDATION_PATH
echo 'WEIGHTS:' $WEIGHTS
echo '------------------------------------'
PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u ../train.py \
    --steps 100 --multi-gpu 1 --batch-size 1 --evaluate_score_threshold 0.5 \
    --weights $WEIGHTS \
    --snapshot-path 'snapshots_signpost_components_101' \
    --backbone 'resnet101' \
    --is_for_signpost_components true \
    traffic_signs $TRAIN_PATH $VALIDATION_PATH