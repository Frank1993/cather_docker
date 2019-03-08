#!/bin/sh

CUDA_VISIBLE_DEVICES="0"
CONFIG_JSON="./evaluate_config.json"

echo 'Parameters:'
echo 'CONFIG_JSON = ' $CONFIG_JSON
echo 'CUDA_VISIBLE_DEVICES = ' $CUDA_VISIBLE_DEVICES

set -e
PYTHONPATH=../../../../:../../../../ocr/attention_ocr/:../../../../ocr/attention_ocr/datasets/$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ../../../scripts/prediction/traffic_signs_folder_ocr_predictor.py \
    --config_json $CONFIG_JSON
    