#!/bin/sh

CUDA_VISIBLE_DEVICES="0"
TRAIN_CONFIG_JSON="./train_config.json"

echo 'Parameters:'
echo 'TRAIN_CONFIG_JSON = ' $TRAIN_CONFIG_JSON
echo 'CUDA_VISIBLE_DEVICES = ' $CUDA_VISIBLE_DEVICES

set -e
PYTHONPATH=../../../../:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2.

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ../../../../classification/run/network_trainer.py \
    --train_config_json $TRAIN_CONFIG_JSON
    