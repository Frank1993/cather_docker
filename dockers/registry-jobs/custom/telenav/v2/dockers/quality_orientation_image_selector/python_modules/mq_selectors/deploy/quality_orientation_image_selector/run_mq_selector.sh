#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES="0"
CONFIG_FILE="config.json"

echo 'Parameters:'
echo 'CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u ./quality_orientation_selector.py \
    --config_file $CONFIG_FILE
