#!/usr/bin/env bash
set -e
CONFIG_FILE="./image_points_cfg.json"

echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u ../../calibration/get_image_points.py \
    --config $CONFIG_FILE
