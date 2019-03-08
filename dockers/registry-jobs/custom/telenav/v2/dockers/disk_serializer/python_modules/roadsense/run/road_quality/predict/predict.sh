#!/bin/sh

CUDA_VISIBLE_DEVICES="0"
CONFIG_JSON="./predict_config.json"

echo 'Parameters:'
echo 'CONFIG_JSON = ' $CONFIG_JSON
echo 'CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES

set -e
PYTHONPATH=../../../../:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ../../../scripts/road_quality/road_quality_predictor.py \
    --config_json $CONFIG_JSON
    