#!/bin/sh

CUDA_VISIBLE_DEVICES="0"
PREDICT_CONFIG_JSON="./predict_config.json"

echo 'Parameters:'
echo 'PREDICT_CONFIG_JSON = ' $PREDICT_CONFIG_JSON
echo 'CUDA_VISIBLE_DEVICES = ' $CUDA_VISIBLE_DEVICES

set -e
PYTHONPATH=../../../../:../../../../apollo_python_common/protobuf:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2.

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ../../../../classification/run/signs_facing_classifier/sf_predictor.py \
    --predict_config_json $PREDICT_CONFIG_JSON
    