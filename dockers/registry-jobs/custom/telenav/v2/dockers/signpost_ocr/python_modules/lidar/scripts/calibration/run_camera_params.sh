#!/usr/bin/env bash
set -e
CONFIG_FILE="./camera_params_cfg.json"

echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u ../../calibration/camera_params.py \
    --config $CONFIG_FILE
