#!/usr/bin/env bash

set -e

CUDA_VISIBLE_DEVICES="0"
CONFIG_FILE="../../../traffic_signs_detector_yolo_mq.json"

echo 'Parameters:'
echo 'CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'
echo $PATH
echo $LD_LIBRARY_PATH

YOLO_FOLDER=./python_modules/object_detection/yolo/
cd $YOLO_FOLDER

PYTHONPATH=../../../python_modules/:../../../python_modules/apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 ./mq/predictor.py \
    --config_files $CONFIG_FILE
