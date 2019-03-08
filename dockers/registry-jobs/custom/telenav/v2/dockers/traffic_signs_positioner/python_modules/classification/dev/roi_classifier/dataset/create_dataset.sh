#!/bin/sh

CUDA_VISIBLE_DEVICES="0"
DATASET_CONFIG_JSON="./dataset_config.json"

echo 'Parameters:'
echo 'DATASET_CONFIG_JSON = ' $DATASET_CONFIG_JSON
echo 'CUDA_VISIBLE_DEVICES = ' $CUDA_VISIBLE_DEVICES

set -e
PYTHONPATH=../../../../:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2.

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ../../../../classification/run/dataset_creator.py \
    --dataset_config_json $DATASET_CONFIG_JSON
    