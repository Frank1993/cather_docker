#!/usr/bin/env bash
set -e
CONFIG_FILE="./lidar_camera_calibration_cfg.json"

echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u ../../calibration/lidar3d_to_camera2d.py \
    --config $CONFIG_FILE