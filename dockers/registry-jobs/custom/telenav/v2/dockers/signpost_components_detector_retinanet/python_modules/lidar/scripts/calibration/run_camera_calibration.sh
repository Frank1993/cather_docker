#!/usr/bin/env bash
set -e
CONFIG_FILE="./camera_calibration_cfg.json"

echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u ../../calibration/camera_calibrator.py \
    --config $CONFIG_FILE
