#!/bin/sh

CUDA_VISIBLE_DEVICES="0"
CONFIG_JSON="./train_config.json"

echo 'Parameters:'
echo 'CONFIG_JSON = ' $CONFIG_JSON
echo 'CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES

set -e
PYTHONPATH=../../../:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ../../scripts/trainer.py \
    --config_json $CONFIG_JSON
    