#!/usr/bin/env bash
set -e
CONFIG_FILE="./lidar_camera_overlay_cfg.json"

echo 'Parameters:'
echo 'CONFIG_FILE='$CONFIG_FILE
echo '----------------------------------------'

PYTHONPATH=../../../:../../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

python -u .././../visualization/lidar_camera_overlay.py \
    --config $CONFIG_FILE