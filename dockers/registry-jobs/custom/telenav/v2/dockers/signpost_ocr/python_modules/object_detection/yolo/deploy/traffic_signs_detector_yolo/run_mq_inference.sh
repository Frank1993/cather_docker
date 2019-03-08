#!/usr/bin/env bash

set -e

CUDA_VISIBLE_DEVICES="0"
CONFIG_FILE="./config/traffic_signs_detector_yolo_mq.json"

echo 'Parameters:'
echo 'CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

YOLO_FOLDER=./../../
cd $YOLO_FOLDER

PYTHONPATH="${PYTHONPATH}:../../:../../apollo_python_common/protobuf/:./"
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 ./mq/predictor.py \
    --config_files $CONFIG_FILE
